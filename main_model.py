import os
import math
import json
import re
import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI, TokenDistiller
from utils.prepare4llm import get_llm


def _extract_final_if_any(text: str) -> str:
    marker = "<|channel|>final<|message|>"
    if text and marker in text:
        return text.split(marker, 1)[-1].strip()
    return (text or "").strip()

def _parse_points_json(raw: str):
    raw = _extract_final_if_any(str(raw))
    if not raw.strip():
        return []
    try:
        j = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return []
        try:
            j = json.loads(m.group(0))
        except Exception:
            return []

    pts = j.get("points", j if isinstance(j, list) else [])
    if not isinstance(pts, list):
        return []
    return [p for p in pts if isinstance(p, dict)]

def build_band_points_from_llm(
    future_points,                  
    observed_data,                  
    lookback_len: int,
    pred_len: int,
    k: int = 5,
    time_jitter: int = 0,
    clamp_w: float = 10.0,
    seed: int | None = None,
    means: torch.Tensor | None = None,   
    stdev: torch.Tensor | None = None,   
    data_mean=None,                      
    data_std=None,                       
):
    device = observed_data.device
    dtype  = observed_data.dtype
    B, K, L_total = observed_data.shape
    Lh, H = lookback_len, pred_len

    t_out  = torch.zeros(B, k, device=device, dtype=torch.long)
    f_out  = torch.zeros(B, k, device=device, dtype=torch.long)
    lo_out = torch.zeros(B, k, device=device, dtype=dtype)
    hi_out = torch.zeros(B, k, device=device, dtype=dtype)
    w_out  = torch.zeros(B, k, device=device, dtype=dtype)

    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)

    for b in range(B):
        raw = ""
        if isinstance(future_points, (list, tuple)) and b < len(future_points):
            raw = future_points[b] or ""
        elif isinstance(future_points, str):
            raw = future_points
        pts = _parse_points_json(raw)[:k]

        for j, p in enumerate(pts):
            f = int(p.get("f", 0))
            if K == 1:
                f = 0
            f = max(0, min(K - 1, f))

            tt = int(p.get("t", 1))
            tt = max(1, min(H, tt))
            t_abs = Lh + (tt - 1)

            if time_jitter > 0:
                jit = torch.randint(-time_jitter, time_jitter + 1, (1,), device=device, generator=g).item()
                t_abs = max(Lh, min(Lh + H - 1, t_abs + int(jit)))

            v_lo = p.get("v_lo", None)
            v_hi = p.get("v_hi", None)

            if v_lo is None and v_hi is None:
                if "v" in p:
                    vv = float(p["v"])
                    v_lo, v_hi = vv, vv
                elif "z" in p:
                    zz = float(p["z"])
                    v_lo, v_hi = zz, zz
                else:
                    continue

            v_lo = float(v_lo) if v_lo is not None else float(v_hi)
            v_hi = float(v_hi) if v_hi is not None else float(v_lo)
            if not (math.isfinite(v_lo) and math.isfinite(v_hi)):
                continue
            if v_lo > v_hi:
                v_lo, v_hi = v_hi, v_lo

            
            
            if data_mean is not None and data_std is not None:
                try:
                    dm = float(data_mean[f].item() if hasattr(data_mean[f], 'item') else data_mean[f])
                    ds = max(float(data_std[f].item() if hasattr(data_std[f], 'item') else data_std[f]), 1e-6)
                except (IndexError, TypeError):
                    dm = float(data_mean) if not isinstance(data_mean, torch.Tensor) else float(data_mean.item())
                    ds = max(float(data_std) if not isinstance(data_std, torch.Tensor) else float(data_std.item()), 1e-6)
                v_lo = (v_lo - dm) / ds
                v_hi = (v_hi - dm) / ds

            
            if means is not None and stdev is not None:
                mu = float(means[b, f, 0].item())
                sd = float(stdev[b, f, 0].item())
                sd = max(sd, 1e-6)
                v_lo = (v_lo - mu) / sd
                v_hi = (v_hi - mu) / sd

            if not (math.isfinite(v_lo) and math.isfinite(v_hi)):
                continue

            conf = float(p.get("confidence", 0.7))
            if not math.isfinite(conf):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            band_w = abs(v_hi - v_lo)
            w = (0.5 + 0.5 * conf) * (1.0 + 1.0 / (1e-3 + band_w))
            if not math.isfinite(w):
                continue
            w = max(0.0, min(clamp_w, w))

            t_out[b, j]  = int(t_abs)
            f_out[b, j]  = int(f)
            lo_out[b, j] = torch.tensor(v_lo, device=device, dtype=dtype)
            hi_out[b, j] = torch.tensor(v_hi, device=device, dtype=dtype)
            w_out[b, j]  = torch.tensor(w, device=device, dtype=dtype)

    if (w_out.sum() <= 0).item():
        return None

    return {"t": t_out, "f": f_out, "lo": lo_out, "hi": hi_out, "w": w_out}


def build_target_amplitude_scales(
    observed_data: torch.Tensor,
    cond_mask: torch.Tensor,
    enabled: bool = False,
    strength: float = 0.15,
):
    if not enabled:
        return None

    device = observed_data.device
    dtype = observed_data.dtype
    B, K, _ = observed_data.shape
    scales = torch.empty(B, K, 1, device=device, dtype=dtype).uniform_(
        1.0 - strength,
        1.0 + strength,
    )
    hist_count = (cond_mask > 0.5).sum(dim=-1, keepdim=True)
    scales = torch.where(hist_count > 1, scales, torch.ones_like(scales))
    return scales


def apply_target_amplitude_modulation(
    x: torch.Tensor,
    observed_data: torch.Tensor,
    cond_mask: torch.Tensor,
    amplitude_scales: torch.Tensor | None,
):
    if amplitude_scales is None:
        return x

    hist_mask = (cond_mask > 0.5).to(dtype=x.dtype)
    future_mask = 1.0 - hist_mask
    hist_count = hist_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    hist_mean = (observed_data * hist_mask).sum(dim=-1, keepdim=True) / hist_count
    mod_future = hist_mean + amplitude_scales * (x - hist_mean)
    return hist_mask * x + future_mask * mod_future


def build_anchor_edit_mask(
    points: dict | None,
    num_features: int,
    series_length: int,
    time_radius: int = 0,
    dtype: torch.dtype | None = None,
):
    if points is None:
        return None

    device = points["t"].device
    if dtype is None:
        dtype = points["w"].dtype

    t_idx = points["t"]
    f_idx = points["f"].clamp(0, num_features - 1)
    valid = points["w"] > 0

    B, k = t_idx.shape
    edit_mask = torch.zeros(B, num_features, series_length, device=device, dtype=dtype)
    offsets = torch.arange(-time_radius, time_radius + 1, device=device)
    expanded_t = (t_idx[..., None] + offsets).clamp(0, series_length - 1)

    b_idx = torch.arange(B, device=device)[:, None, None].expand(B, k, offsets.numel())
    f_idx = f_idx[..., None].expand(B, k, offsets.numel())
    valid = valid[..., None].expand(B, k, offsets.numel())

    edit_mask[
        b_idx[valid],
        f_idx[valid],
        expanded_t[valid],
    ] = 1.0

    return edit_mask


def band_points_energy(
    x,                      
    points: dict,
    time_radius: int = 2,
    softmin_tau: float = 0.05,
):
    device = x.device
    B, K, L = x.shape

    t  = points["t"]
    f  = points["f"]
    lo = torch.nan_to_num(points["lo"], nan=0.0, posinf=0.0, neginf=0.0)
    hi = torch.nan_to_num(points["hi"], nan=0.0, posinf=0.0, neginf=0.0)
    w  = torch.nan_to_num(points["w"], nan=0.0, posinf=0.0, neginf=0.0)
    k = t.size(1)

    offsets = torch.arange(-time_radius, time_radius + 1, device=device)
    idx = (t[..., None] + offsets[None, None, :]).clamp(0, L - 1)

    x_blk = x.permute(0, 2, 1)
    b = torch.arange(B, device=device)[:, None, None]
    vals = x_blk[b, idx, f[:, :, None]]
    vals = torch.nan_to_num(vals, nan=0.0, posinf=1e6, neginf=-1e6)

    below = torch.relu(lo[:, :, None] - vals)
    above = torch.relu(vals - hi[:, :, None])
    dist  = below + above
    dist2 = (dist ** 2).clamp_max(1e12)

    if softmin_tau is None or softmin_tau <= 0:
        best = dist2.min(dim=-1).values
    else:
        best = -softmin_tau * torch.logsumexp(-dist2 / softmin_tau, dim=-1)

    E = (w * best).sum(dim=1).mean()
    return E


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=25):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class CSDI_series_decomp(nn.Module):
    def __init__(self, lookback_len, pred_len, kernel_size=25):
        super(CSDI_series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.lookback_len = lookback_len
        self.pred_len = pred_len

    def forward(self, x):
        x = x.permute(0, 2, 1)
        lookback = x[:, :self.lookback_len, :]

        moving_mean = self.moving_avg(lookback)
        res = lookback - moving_mean
        
        moving_mean = moving_mean.permute(0, 2, 1)
        res = res.permute(0, 2, 1)

        moving_mean = nn.functional.pad(moving_mean, (0, self.pred_len), "constant", 0)
        res = nn.functional.pad(res, (0, self.pred_len), "constant", 0)
        return res, moving_mean
        

    

class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device, window_lens):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.timestep_branch = config["model"]["timestep_branch"]
        self.timestep_emb_cat = config["model"]["timestep_emb_cat"]
        self.with_texts = config["model"]["with_texts"]
        self.with_intrinsic = bool(config["model"].get("with_intrinsic", self.with_texts))
        self.with_future_hint = bool(config["model"].get("with_future_hint", self.with_texts))
        self.noise_esti = config["diffusion"]["noise_esti"]
        self.relative_size_emb_cat = config["model"]["relative_size_emb_cat"]
        self.decomp = config["model"]["decomp"]
        self.ddim = config["diffusion"]["ddim"]
        self.sample_steps = config["diffusion"]["sample_steps"]
        self.sample_method = config["diffusion"]["sample_method"]

        self.lookback_len = config["model"]["lookback_len"]
        self.pred_len = config["model"]["pred_len"]
        self.diff_channels = config["diffusion"]["channels"]
        self.cfg = config["diffusion"]["cfg"]
        self.c_mask_prob = config["diffusion"]["c_mask_prob"]
        self.context_dim = config["model"]["context_dim"]
        self.llm = config["model"]["llm"]
        self.domain = config["model"]["domain"]
        self.save_attn = config["model"]["save_attn"]
        self.save_token = config["model"]["save_token"]
        
        self.use_token_distill = bool(config["model"].get("use_token_distill", self.with_texts))
        self.use_token_distill = False

        self.intrinsic_distill_tokens = int(config["model"].get("intrinsic_distill_tokens", max(1, self.lookback_len // 4)))
        self.future_distill_tokens = int(config["model"].get("future_distill_tokens", 8))
        self.distill_heads = int(config["model"].get("distill_heads", config["diffusion"]["nheads"]))

        self.use_text_mixer = False  

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1 

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
            
        if self.decomp:
            self.decomposition = CSDI_series_decomp(self.lookback_len, self.pred_len, kernel_size=25)
        
        if self.timestep_emb_cat:
            self.timestep_emb = nn.Sequential(nn.Linear(config["model"]["timestep_dim"], self.diff_channels//8), 
                                      nn.LayerNorm(self.diff_channels//8),
                                      nn.ReLU(),
                                      nn.Linear(self.diff_channels//8, self.diff_channels//4), 
                                      nn.LayerNorm(self.diff_channels//4),
                                      nn.ReLU())
        
        if self.relative_size_emb_cat:
            self.relative_size_emb = nn.Sequential(nn.Linear(self.lookback_len, self.lookback_len), 
                                                   nn.LayerNorm(self.lookback_len),
                                                   nn.ReLU(),
                                                   nn.Linear(self.lookback_len, self.diff_channels),
                                                   nn.LayerNorm(self.diff_channels),
                                                   nn.ReLU(),)

        if self.with_texts:
            self.text_encoder, self.tokenizer = get_llm(self.llm, config["model"]["llm_layers"])
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()
            self.text_max_length = self._resolve_text_max_length(config["model"])
            if self.text_max_length is not None:
                self.tokenizer.model_max_length = self.text_max_length
            self._text_mean_cache = {}
            self._future_token_cache = {}
            if self.llm != 'bert':
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    pad_token = '[PAD]'
                    self.tokenizer.add_special_tokens({'pad_token': pad_token})
                    self.tokenizer.pad_token = pad_token
            if self.use_token_distill and self.with_intrinsic:
                self.intrinsic_distiller = TokenDistiller(
                    token_dim=self.context_dim,
                    num_queries=self.intrinsic_distill_tokens,
                    nheads=self.distill_heads,
                    dropout=config["diffusion"]["attn_drop"],
                )
            if self.use_token_distill and self.with_future_hint:
                self.future_distiller = TokenDistiller(
                    token_dim=self.context_dim,
                    num_queries=self.future_distill_tokens,
                    nheads=self.distill_heads,
                    dropout=config["diffusion"]["attn_drop"],
                )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        
        
        config_diff["decomp"] = self.decomp
        config_diff["lookback_len"] = self.lookback_len
        config_diff["pred_len"] = self.pred_len
        config_diff["with_timestep"] = True if self.timestep_emb_cat else False
        config_diff["context_dim"] = self.context_dim
        config_diff["with_texts"] = self.with_texts
        config_diff["time_weight"] = config["diffusion"]["time_weight"]
        config_diff["save_attn"] = config["model"]["save_attn"]
        
        
        
        
        
        


        input_dim = 1 if self.is_unconditional == True else 2
        mode_num = 1

        if self.decomp:
            self.diffmodel_trend = diff_CSDI(config_diff, input_dim, mode_num=mode_num)
            self.diffmodel_sesonal = diff_CSDI(config_diff, input_dim, mode_num=mode_num)
        else:
            self.diffmodel = diff_CSDI(config_diff, input_dim, mode_num=mode_num)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
    
    def _cond_time_mask(self, cond_mask: torch.Tensor) -> torch.Tensor:
        """Internal helper."""
        if cond_mask.dim() != 3:
            raise ValueError(f"cond_mask must be 3D (B,K,L), got {tuple(cond_mask.shape)}")
        B, K, L = cond_mask.shape
        cond_bool = (cond_mask > 0.5) if cond_mask.dtype != torch.bool else cond_mask
        cond_t = cond_bool.any(dim=1)           
        return cond_t

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i-1] 
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask


    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim) 
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1) 
        side_info = side_info.permute(0, 3, 2, 1) 

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1) 
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, timesteps=None, timestep_emb=None, size_emb=None, future_context=None, intrinsic_texts=None, future_kpm=None
    ):
        loss_sum = 0
        for t in range(self.num_steps): 
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, future_context=future_context, intrinsic_texts=intrinsic_texts, future_kpm=future_kpm
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, timesteps=None, timestep_emb=None, size_emb=None, future_context=None, intrinsic_texts=None, future_kpm=None, set_t=-1
    ):  
        
        B, K, L = observed_data.shape

        
        if not self.noise_esti:
            means = torch.sum(observed_data*cond_mask, dim=2, keepdim=True) / torch.sum(cond_mask, dim=2, keepdim=True)
            stdev = torch.sqrt(torch.sum((observed_data - means) ** 2 * cond_mask, dim=2, keepdim=True) / (torch.sum(cond_mask, dim=2, keepdim=True) - 1) + 1e-5)
            observed_data = (observed_data - means) / stdev
        
        t = (torch.ones(B) * set_t).long().to(self.device) if (is_train != 1) else torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask) 
        intrinsic_tokens, intrinsic_kpm = None, None
        if self.with_intrinsic:
            
            intrinsic_tokens_raw = self.encode_intrinsic_texts(
                intrinsic_texts, cond_mask=cond_mask, L_total=L, B=B, align="strict"
            ).to(device=self.device, dtype=observed_data.dtype)
            _ = self.assert_intrinsic_mapping(
                intrinsic_texts, cond_mask, L_total=L, B=B, align="strict",
                atol=0.0, encoded_out=intrinsic_tokens_raw
            )
            intrinsic_tokens = intrinsic_tokens_raw
            intrinsic_tokens, intrinsic_kpm = self.distill_intrinsic_context(intrinsic_tokens, cond_mask)

        
        if self.with_future_hint:
            future_context, future_kpm = self.distill_future_context(future_context, future_kpm)
        else:
            future_context, future_kpm = None, None

        
        if self.cfg:
            cfg_mask = torch.bernoulli(torch.ones((B, )) - self.c_mask_prob).to(self.device) 
        else:
            cfg_mask = None

        
        if self.decomp:
            
            

            predicted_seasonal, _ = self.diffmodel_sesonal(total_input[0], side_info, t, cfg_mask, timestep_emb, size_emb)
            predicted_trend = self.diffmodel_trend(total_input[1], side_info, t, cfg_mask, timestep_emb, size_emb)
            predicted, _ = predicted_seasonal + predicted_trend
        else:
            if self.save_attn:
                predicted, _ = self.diffmodel(total_input, side_info, t, cfg_mask, timestep_emb, size_emb, future_context=future_context, intrinsic_tokens=intrinsic_tokens, future_kpm=future_kpm, intrinsic_kpm=intrinsic_kpm, cond_mask=cond_mask) 
            else:
                predicted = self.diffmodel(total_input, side_info, t, cfg_mask, timestep_emb, size_emb, future_context=future_context, intrinsic_tokens=intrinsic_tokens, future_kpm=future_kpm, intrinsic_kpm=intrinsic_kpm, cond_mask=cond_mask) 

        
        
        
        

        
        target_mask = observed_mask - cond_mask
        if self.noise_esti:
            residual = (noise - predicted) * target_mask 
        else:
            residual = (observed_data - predicted) * target_mask 
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  
        else:
            cond_obs = cond_mask * observed_data
            noisy_target = noisy_data.unsqueeze(1) 
            if self.decomp:
                res, moving_mean = self.decomposition(cond_obs) 
                res, moving_mean = res.unsqueeze(1), moving_mean.unsqueeze(1) 
                res_input = torch.cat([res, noisy_target], dim=1)  
                moving_mean_input = torch.cat([moving_mean, noisy_target], dim=1) 
                total_input = [res_input, moving_mean_input]
            else:
                cond_obs = cond_obs.unsqueeze(1) 
                total_input = torch.cat([cond_obs, noisy_target], dim=1) 

        return total_input

    
    def impute(self, observed_data, cond_mask, side_info, n_samples, guide_w,
               timesteps=None, timestep_emb=None, size_emb=None, future_context=None,
               intrinsic_texts=None,
               future_points=None,
               edit_guidance_method: str = "langevin",
               guidance_scale: float = 0.0,
               time_radius: int = 1,
               num_iter: int = 0,
               energy_softmin_tau: float = 0.05,
               langevin_step_size: float | None = None,
               langevin_alpha: float = 1.0,
               langevin_noise_scale: float | None = None,
               blended_amplitude_modulation: bool = False,
               blended_amplitude_strength: float = 0.15,
               data_mean=0, data_std=1):
        device = observed_data.device
        dtype  = observed_data.dtype

        B, K, L = observed_data.shape
        S = int(n_samples)
        BS = B * S

        intrinsic_tokens, intrinsic_kpm = None, None
        if self.with_intrinsic:
            
            intrinsic_tokens = self.encode_intrinsic_texts(
                intrinsic_texts, cond_mask=cond_mask, L_total=L, B=B, align="strict"
            ).to(device=device, dtype=dtype)
            intrinsic_tokens, intrinsic_kpm = self.distill_intrinsic_context(intrinsic_tokens, cond_mask)
        

        
        future_ctx, future_kpm = None, None
        if self.with_future_hint:
            future_ctx, future_kpm = self.encode_future_hint_tokens(future_context)  
            future_ctx, future_kpm = self.distill_future_context(future_ctx, future_kpm)

        
        
        def repS(x, dim=0):
            return None if x is None else x.repeat_interleave(S, dim=dim)

        timestep_emb_orig = timestep_emb  
        observed_bs   = observed_data.repeat_interleave(S, dim=0)          
        cond_mask_bs  = cond_mask.repeat_interleave(S, dim=0)              
        side_info_bs  = repS(side_info, 0)                                  
        timestep_emb  = repS(timestep_emb, 0)
        size_emb_bs   = repS(size_emb, 0)
        intrinsic_bs  = repS(intrinsic_tokens, 0)                           
        intrinsic_kpm_bs = repS(intrinsic_kpm, 0)                           
        future_ctx_bs = repS(future_ctx, 0)                                 
        future_kpm_bs = repS(future_kpm, 0)                                 

        
        
        
        
        
        
        
        
        
        
        
        
        

        
        if self.ddim:
            if self.sample_method == 'linear':
                a = max(1, self.num_steps // self.sample_steps)
                tau_indices = np.arange(0, self.num_steps, a, dtype=int)   
                tau_indices = tau_indices[:self.sample_steps]               
            elif self.sample_method == "quad":
                tau_indices = (np.linspace(0, self.num_steps - 1, self.sample_steps) ** 0.5)
                tau_indices = np.round((tau_indices / tau_indices.max()) * (self.num_steps - 1)).astype(int)
            else:
                raise NotImplementedError(f"sampling method {self.sample_method} is not implemented!")
            
            tau_prev_indices = np.concatenate([[-1], tau_indices[:-1]])
        else:
            self.sample_steps = self.num_steps  

        
        means = stdev = None
        if not self.noise_esti:
            means = torch.sum(observed_data*cond_mask, dim=2, keepdim=True) / torch.sum(cond_mask, dim=2, keepdim=True)
            stdev = torch.sqrt(torch.sum((observed_data - means) ** 2 * cond_mask, dim=2, keepdim=True) / (torch.sum(cond_mask, dim=2, keepdim=True) - 1) + 1e-5)
            observed_data = (observed_data - means) / stdev

        
        if langevin_step_size is None:
            langevin_step_size = 5e-4 if (not self.noise_esti) else 1e-6
        if langevin_noise_scale is None:
            langevin_noise_scale = 0.3 if (not self.noise_esti) else 0.05

        
        pseudo_points = None
        edit_mask = None
        blended_amplitude_scales = build_target_amplitude_scales(
            observed_data=observed_data,
            cond_mask=cond_mask,
            enabled=(edit_guidance_method == "blended") and blended_amplitude_modulation,
            strength=blended_amplitude_strength,
        )
        needs_anchor_guidance = (
            (future_points is not None)
            and (guidance_scale > 0)
            and (num_iter > 0)
            and (edit_guidance_method in ("langevin", "blended"))
        )
        if needs_anchor_guidance:
            pseudo_points = build_band_points_from_llm(
                future_points=future_points,
                observed_data=observed_data,
                lookback_len=self.lookback_len,
                pred_len=self.pred_len,
                k=5,
                time_jitter=2,
                clamp_w=10.0,
                seed=2025,
                means=means if (not self.noise_esti) else None,
                stdev=stdev if (not self.noise_esti) else None,
                data_mean=data_mean,
                data_std=data_std,
            )
            if edit_guidance_method == "blended":
                edit_mask = build_anchor_edit_mask(
                    pseudo_points,
                    num_features=K,
                    series_length=L,
                    time_radius=time_radius,
                    dtype=observed_data.dtype,
                )
                if edit_mask is not None:
                    edit_mask = edit_mask * (1.0 - cond_mask)

        
        if self.cfg:
            
            side_info_bs = side_info_bs.repeat(2, 1, 1, 1)
            if timestep_emb is not None:
                timestep_emb = timestep_emb.repeat(2, 1, 1, 1)
            if future_ctx_bs is not None:
                future_ctx_bs = future_ctx_bs.repeat(2, 1, 1)
                future_kpm_bs = future_kpm_bs.repeat(2, 1)
            if intrinsic_bs is not None:
                intrinsic_bs = intrinsic_bs.repeat(2, 1, 1)
                intrinsic_kpm_bs = intrinsic_kpm_bs.repeat(2, 1)
            cond_mask_2bs = cond_mask_bs.repeat(2, 1, 1)   
            cfg_mask = torch.zeros((2*BS,), device=self.device); cfg_mask[:BS] = 1.
        else:
            cfg_mask = None
            cond_mask_2bs = cond_mask_bs

        current_sample = torch.randn_like(observed_bs)  

        for s in range(self.sample_steps - 1, -1, -1):
            
            if self.ddim:
                tau = int(tau_indices[s]); tau_prev = int(tau_prev_indices[s])
                model_step = torch.full((BS if not self.cfg else 2*BS,), tau, device=self.device, dtype=torch.long)
            else:
                model_step = torch.full((BS if not self.cfg else 2*BS,), s, device=self.device, dtype=torch.long)

            
            cond_obs  = (cond_mask_bs * observed_bs).unsqueeze(1)       
            noisy_tgt = ((1 - cond_mask_bs) * current_sample).unsqueeze(1)

            diff_input = torch.cat([cond_obs, noisy_tgt], dim=1)        
            if self.cfg:
                diff_input = diff_input.repeat(2, 1, 1, 1)              

            if self.save_attn:
                predicted, attn = self.diffmodel(
                    diff_input, side_info_bs, model_step, cfg_mask,
                    timestep_emb, size_emb_bs, future_ctx_bs, intrinsic_bs, future_kpm_bs, intrinsic_kpm_bs, cond_mask_2bs
                )
            else:
                predicted = self.diffmodel(
                    diff_input, side_info_bs, model_step, cfg_mask,
                    timestep_emb, size_emb_bs, future_ctx_bs, intrinsic_bs, future_kpm_bs, intrinsic_kpm_bs, cond_mask_2bs
                )

            
            if self.cfg:
                pred_c, pred_u = predicted[:BS], predicted[BS:]
                predicted = pred_u + guide_w * (pred_c - pred_u)

            
            if self.noise_esti:
                if not self.ddim:
                    coeff1 = 1 / self.alpha_hat[s] ** 0.5
                    coeff2 = (1 - self.alpha_hat[s]) / (1 - self.alpha[s]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)
                    if s > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = (((1.0 - self.alpha[s - 1]) / (1.0 - self.alpha[s])) * self.beta[s]) ** 0.5
                        current_sample += sigma * noise
                else:
                    
                    
                    ap = 1.0 if tau_prev < 0 else self.alpha[tau_prev]
                    a  = self.alpha[tau]
                    current_sample = (
                        math.sqrt(ap / a) * current_sample +
                        (math.sqrt(1 - ap) - math.sqrt((ap * (1 - a)) / a)) * predicted
                    )
            else:
                if not self.ddim:
                    if s > 1:
                        coeff1 = (self.alpha_hat[s] ** 0.5 * (1 - self.alpha[s-1])) / (1 - self.alpha[s])
                        coeff2 = (self.alpha[s-1] ** 0.5 * self.beta[s]) / (1 - self.alpha[s])
                        current_sample = coeff1 * current_sample + coeff2 * predicted
                        if s > 2:
                            noise = torch.randn_like(current_sample)
                            sigma = (((1.0 - self.alpha[s - 1]) / (1.0 - self.alpha[s])) * self.beta[s]) ** 0.5
                            current_sample += sigma * noise
                else:
                    
                    tau_i, tau_prev_i = tau, tau_prev
                    alpha_prev = 1.0 if tau_prev_i < 0 else self.alpha[tau_prev_i]
                    alpha_t = self.alpha[tau_i]
                    aaa_ = ((1 - alpha_prev) / (1 - alpha_t)) ** 0.5
                    current_sample = (
                        aaa_ * current_sample +
                        (alpha_prev ** 0.5 - alpha_t ** 0.5 * aaa_) * predicted
                    )
        samples_bs = current_sample.detach().view(B, S, K, L)

        
        if self.timestep_branch and timesteps is not None:
            pred_ts = self.timestep_pred(timesteps).unsqueeze(1)        
            samples_bs = 0.9 * samples_bs + 0.1 * pred_ts

        
        if pseudo_points is not None and num_iter > 0:
            device = observed_data.device
            dtype  = observed_data.dtype
            BS_g = B * S  

            
            def _repBS(t, dim=0):
                return None if t is None else t.repeat_interleave(S, dim=dim)

            obs_g    = observed_data.repeat_interleave(S, dim=0)   
            cmask_g  = cond_mask.repeat_interleave(S, dim=0)        
            emask_g  = (_repBS(edit_mask)
                        if edit_mask is not None else None)           
            amp_g    = (_repBS(blended_amplitude_scales)
                        if blended_amplitude_scales is not None else None)  
            pp_g     = {k: v.repeat_interleave(S, dim=0)
                        for k, v in pseudo_points.items()}            

            
            if self.cfg:
                _g_cfg   = torch.zeros((2 * BS_g,), device=device)
                _g_cfg[:BS_g] = 1.0
                _g_side  = side_info.repeat_interleave(S, dim=0).repeat(2, 1, 1, 1)
                _g_te    = (_repBS(timestep_emb_orig).repeat(2, 1, 1, 1)
                            if timestep_emb_orig is not None else None)
                _g_se    = (_repBS(size_emb).repeat(2, 1, 1, 1)
                            if size_emb is not None else None)
                _g_fctx  = (_repBS(future_ctx).repeat(2, 1, 1)
                            if future_ctx is not None else None)
                _g_fkpm  = (_repBS(future_kpm).repeat(2, 1)
                            if future_kpm is not None else None)
                _g_itok  = (_repBS(intrinsic_tokens).repeat(2, 1, 1)
                            if intrinsic_tokens is not None else None)
                _g_ikpm  = (_repBS(intrinsic_kpm).repeat(2, 1)
                            if intrinsic_kpm is not None else None)
                _g_cmask = cmask_g.repeat(2, 1, 1)
            else:
                _g_cfg   = None
                _g_side  = side_info.repeat_interleave(S, dim=0)
                _g_te    = _repBS(timestep_emb_orig)
                _g_se    = _repBS(size_emb)
                _g_fctx  = _repBS(future_ctx)
                _g_fkpm  = _repBS(future_kpm)
                _g_itok  = _repBS(intrinsic_tokens)
                _g_ikpm  = _repBS(intrinsic_kpm)
                _g_cmask = cmask_g

            def _gdiff(diff_in, step):
                out = self.diffmodel(
                    diff_in, _g_side, step, _g_cfg,
                    _g_te, _g_se, _g_fctx, _g_itok, _g_fkpm, _g_ikpm, _g_cmask
                )
                return out[0] if isinstance(out, (tuple, list)) else out

            def _mk_step(t_val):
                sz = 2 * BS_g if self.cfg else BS_g
                return torch.full((sz,), t_val, device=device, dtype=torch.long)

            def _x0_from_eps_g(x_t, eps_pred, ab):
                return (x_t - torch.sqrt(1.0 - ab) * eps_pred) / torch.sqrt(ab + 1e-12)

            def _eps_from_x0_g(x_t, x0_pred, ab):
                return (x_t - torch.sqrt(ab) * x0_pred) / torch.sqrt(1.0 - ab + 1e-12)

            def _clip_g(x, max_norm=1.0):
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                n = torch.sqrt((x ** 2).sum(dim=(1, 2)) + 1e-8)
                s = (max_norm / n).clamp_max(1.0).view(-1, 1, 1)
                return x * s

            def _predict_score_g(x_full, step, ab):
                if self.is_unconditional:
                    din = (cmask_g * obs_g + (1 - cmask_g) * x_full).unsqueeze(1)
                else:
                    din = torch.cat([(cmask_g * obs_g).unsqueeze(1),
                                        ((1 - cmask_g) * x_full).unsqueeze(1)], dim=1)
                if self.cfg:
                    pr = _gdiff(din.repeat(2, 1, 1, 1), step)
                    pred = pr[BS_g:] + guide_w * (pr[:BS_g] - pr[BS_g:])
                else:
                    pred = _gdiff(din, step)
                eps = pred if self.noise_esti else _eps_from_x0_g(x_full, pred, ab)
                return -(eps / torch.sqrt(1.0 - ab + 1e-12))

            def _q_from_src(x0, step_idx):
                if step_idx <= 0:
                    return x0.detach()
                ab = torch.tensor(self.alpha[step_idx], device=device, dtype=dtype)
                return (torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * torch.randn_like(x0)).detach()

            def _rev_update(x_t, pred_out, cur_t, prv_t):
                if self.noise_esti:
                    if not self.ddim:
                        c1 = 1 / self.alpha_hat[cur_t] ** 0.5
                        c2 = (1 - self.alpha_hat[cur_t]) / (1 - self.alpha[cur_t]) ** 0.5
                        xp = c1 * (x_t - c2 * pred_out)
                        if cur_t > 0:
                            sig = (((1 - self.alpha[cur_t - 1]) / (1 - self.alpha[cur_t])) * self.beta[cur_t]) ** 0.5
                            xp = xp + sig * torch.randn_like(xp)
                        return xp
                    ap, a = self.alpha[prv_t], self.alpha[cur_t]
                    return (torch.sqrt(torch.tensor(ap / a, device=device, dtype=dtype)) * x_t +
                            (torch.sqrt(torch.tensor(1 - ap, device=device, dtype=dtype)) -
                                torch.sqrt(torch.tensor(ap * (1 - a) / a, device=device, dtype=dtype))) * pred_out)
                if not self.ddim:
                    if cur_t > 1:
                        c1 = (self.alpha_hat[cur_t] ** 0.5 * (1 - self.alpha[cur_t - 1])) / (1 - self.alpha[cur_t])
                        c2 = (self.alpha[cur_t - 1] ** 0.5 * self.beta[cur_t]) / (1 - self.alpha[cur_t])
                        xp = c1 * x_t + c2 * pred_out
                        if cur_t > 2:
                            sig = (((1 - self.alpha[cur_t - 1]) / (1 - self.alpha[cur_t])) * self.beta[cur_t]) ** 0.5
                            xp += sig * torch.randn_like(xp)
                        return xp
                    return x_t
                aaa = ((1 - self.alpha[prv_t]) / (1 - self.alpha[cur_t])) ** 0.5
                return (torch.tensor(aaa, device=device, dtype=dtype) * x_t +
                        (torch.sqrt(torch.tensor(self.alpha[prv_t], device=device, dtype=dtype)) -
                            torch.sqrt(torch.tensor(self.alpha[cur_t], device=device, dtype=dtype)) *
                            torch.tensor(aaa, device=device, dtype=dtype)) * pred_out)

            
            x_flat = samples_bs.reshape(BS_g, K, L)
            if edit_guidance_method == "blended":
                
                if emask_g is not None and num_iter > 0 and guidance_scale > 0:
                    if self.ddim:
                        sched = np.array(tau_indices[:min(num_iter, len(tau_indices))], dtype=int)
                    else:
                        sched = np.arange(min(num_iter, self.num_steps), dtype=int)
                    if sched.size > 0:
                        src_base = (cmask_g * obs_g + (1 - cmask_g) * x_flat).detach()
                        pres_mask = 1.0 - emask_g
                        x = _q_from_src(src_base, int(sched[-1]))
                        for ii in range(len(sched) - 1, -1, -1):
                            ct = int(sched[ii])
                            pt = int(sched[ii - 1]) if ii > 0 else 0
                            ms = _mk_step(ct)
                            ab_ct = torch.tensor(self.alpha[ct], device=device, dtype=dtype)
                            src_prev = _q_from_src(src_base, pt) if pt > 0 else src_base
                            with torch.enable_grad():
                                x = x.clone().detach().requires_grad_()
                                if self.is_unconditional:
                                    din = (cmask_g * obs_g + (1 - cmask_g) * x).unsqueeze(1)
                                else:
                                    din = torch.cat([(cmask_g * obs_g).unsqueeze(1),
                                                        ((1 - cmask_g) * x).unsqueeze(1)], dim=1)
                                if self.cfg:
                                    pr = _gdiff(din.repeat(2, 1, 1, 1), ms)
                                    pred = pr[BS_g:] + guide_w * (pr[:BS_g] - pr[BS_g:])
                                else:
                                    pred = _gdiff(din, ms)
                                x0_pred = pred
                                x0_pred = apply_target_amplitude_modulation(
                                    x0_pred, observed_data=obs_g, cond_mask=cmask_g,
                                    amplitude_scales=amp_g,
                                )
                                E = band_points_energy(x0_pred, pp_g,
                                                        time_radius=time_radius, softmin_tau=energy_softmin_tau)
                                if E.dim() != 0:
                                    E = E.sum()
                                grad_E = torch.autograd.grad(E, x, retain_graph=False, create_graph=False)[0]
                                grad_E = torch.nan_to_num(grad_E, nan=0.0, posinf=0.0, neginf=0.0)
                                grad_log_prob = _clip_g((-grad_E) * emask_g, max_norm=1.0)
                                if self.noise_esti:
                                    eps_guided = pred - torch.sqrt(1.0 - ab_ct + 1e-12) * guidance_scale * grad_log_prob
                                else:
                                    eps_g2 = _eps_from_x0_g(x.detach(), x0_pred.detach(), ab_ct)
                                    eps_guided = eps_g2 - torch.sqrt(1.0 - ab_ct + 1e-12) * guidance_scale * grad_log_prob
                                eps_guided = torch.nan_to_num(eps_guided, nan=0.0, posinf=1e6, neginf=-1e6)
                                guided_pred = (x.detach() - torch.sqrt(1.0 - ab_ct + 1e-12) * eps_guided) / torch.sqrt(ab_ct)
                                guided_pred = torch.nan_to_num(guided_pred, nan=0.0, posinf=1e6, neginf=-1e6)
                                x_prev = _rev_update(x, guided_pred, ct, pt)
                                x_prev = torch.nan_to_num(x_prev, nan=0.0, posinf=1e6, neginf=-1e6)
                                x = (pres_mask * src_prev + emask_g * x_prev).detach()
                        x_flat = x

            elif edit_guidance_method == "langevin":
                t_idx = 1 if self.num_steps > 2 else 0
                ab = torch.tensor(self.alpha[t_idx], device=device, dtype=dtype)
                step_t = _mk_step(t_idx)
                x = x_flat.detach()
                for _ in range(num_iter):
                    x = cmask_g * obs_g + (1.0 - cmask_g) * x
                    with torch.enable_grad():
                        x_leaf = x.detach().clone().requires_grad_(True)
                        E = band_points_energy(x_leaf, pp_g,
                                                time_radius=time_radius,
                                                softmin_tau=energy_softmin_tau)
                        if E.dim() != 0:
                            E = E.sum()
                        grad_E = torch.autograd.grad(E, x_leaf, retain_graph=False, create_graph=False)[0]
                    grad_E = _clip_g(grad_E * (1.0 - cmask_g), max_norm=1.0)
                    with torch.no_grad():
                        score = _predict_score_g(x_leaf.detach(), step_t, ab)
                        score = _clip_g(score * (1.0 - cmask_g), max_norm=1.0)
                    noise = torch.randn_like(x_leaf) * (math.sqrt(2.0 * langevin_step_size) * langevin_noise_scale)
                    x = x_leaf.detach() + langevin_step_size * (langevin_alpha * score - guidance_scale * grad_E) + noise
                x_flat = (cmask_g * obs_g + (1.0 - cmask_g) * x).detach()

            
            samples_bs = x_flat.reshape(B, S, K, L)

        
        if not self.noise_esti:
            samples_bs = samples_bs * stdev.unsqueeze(1) + means.unsqueeze(1)

        imputed_samples = samples_bs  
        return (imputed_samples, attn) if self.save_attn else imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)): 
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim, window_lens):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device, window_lens)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        
        observed_data = batch["observed_data"].to(self.device).float()  
        observed_mask = batch["observed_mask"].to(self.device).float()  
        observed_tp = batch["timepoints"].to(self.device).float()       
        gt_mask = batch["gt_mask"].to(self.device).float()              
        
        
        
        intrinsic_texts = batch.get("intrinsic_texts", None)   
        future_context  = batch.get("future_context", None)       

        if self.timestep_emb_cat or self.timestep_branch:
            timesteps = batch["timesteps"].to(self.device).float()  
            timesteps = timesteps.permute(0, 2, 1)                  
        else:
            timesteps = None

        
        observed_data = observed_data.permute(0, 2, 1) 
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        future_points  = batch.get("future_points", None)  

        return (
            observed_data, 
            observed_mask,
            observed_tp, 
            gt_mask,
            for_pattern_mask, 
            cut_length,
            feature_id,
            timesteps, 
            
            
            intrinsic_texts,
            future_context,
            future_points,
        )        

    
    def _safe_text_length(self, value):
        try:
            length = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        
        if 0 < length < 1_000_000:
            return length
        return None

    def _resolve_text_max_length(self, model_config):
        explicit = self._safe_text_length(model_config.get("text_max_length"))

        config_obj = getattr(self.text_encoder, "config", None)
        candidates = []
        for name in ("max_position_embeddings", "n_positions", "max_sequence_length"):
            candidates.append(self._safe_text_length(getattr(config_obj, name, None)))
        candidates.append(self._safe_text_length(getattr(self.tokenizer, "model_max_length", None)))
        candidates = [length for length in candidates if length is not None]

        model_limit = min(candidates) if candidates else None
        if explicit is not None and model_limit is not None:
            return min(explicit, model_limit)
        if explicit is not None:
            return explicit
        if model_limit is not None:
            return model_limit
        if self.llm == "bert":
            return 512
        return None

    def _tokenize_texts(self, texts, padding=True):
        kwargs = {
            "padding": padding,
            "truncation": True,
            "return_tensors": "pt",
        }
        if getattr(self, "text_max_length", None) is not None:
            kwargs["max_length"] = self.text_max_length
        return self.tokenizer(texts, **kwargs).to(self.device)

    def _encode_texts_chunked(self, texts):
        """Internal helper."""

        if texts is None or len(texts) == 0:
            return torch.zeros(0, self.context_dim, device=self.device)

        texts = [text if isinstance(text, str) else "" for text in texts]
        cached = []
        missing = []
        missing_indices = []
        cache = getattr(self, "_text_mean_cache", None)

        if cache is None:
            cache = self._text_mean_cache = {}

        for idx, text in enumerate(texts):
            value = cache.get(text)
            if value is None:
                cached.append(None)
                missing.append(text)
                missing_indices.append(idx)
            else:
                cached.append(value)

        if missing:
            self.text_encoder.eval()
            tok = self._tokenize_texts(missing, padding=True)
            with torch.no_grad():
                out = self.text_encoder(**tok).last_hidden_state          
            mask = tok['attention_mask'].unsqueeze(-1).float()            
            mean = (out * mask).sum(1) / mask.sum(1).clamp_min(1e-6)      

            for local_idx, text in enumerate(missing):
                value = mean[local_idx].detach().cpu()
                cache[text] = value
                cached[missing_indices[local_idx]] = value

        return torch.stack(cached, dim=0).to(self.device)

    def encode_intrinsic_texts(self,
                            intrinsic_texts,
                            cond_mask,
                            L_total,
                            B=None,
                            align="strict",
                            layout="time"):
        """Internal helper."""
        import torch
        device = self.device
        E = getattr(self, "context_dim", None)
        assert E is not None, "self.context_dim is not initialized"
        
        if cond_mask is None:
            raise ValueError("encode_intrinsic_texts requires cond_mask")
        if B is None:
            B = len(intrinsic_texts) if intrinsic_texts is not None else cond_mask.size(0)
        if cond_mask.dim() != 3 or cond_mask.size(0) != B or cond_mask.size(-1) != L_total:
            raise ValueError(f"cond_mask must match (B,K,L={L_total}), got {tuple(cond_mask.shape)}")

        out = torch.zeros(B, L_total, E, device=device)

        if intrinsic_texts is None:
            return out

        if not isinstance(intrinsic_texts, (list, tuple)):
            raise TypeError(f"intrinsic_texts must be a nested list, got {type(intrinsic_texts)}")

        
        def _to_sample_major_from_time_major(texts_tm, cond_t_bool):
            """Internal helper."""
            if not isinstance(texts_tm, (list, tuple)) or len(texts_tm) == 0:
                return [[] for _ in range(B)]
            Tm = len(texts_tm)
            
            check_ok = sum(1 for x in texts_tm if isinstance(x, (list, tuple)) and len(x) == B)
            if check_ok < max(1, int(0.8 * Tm)):
                raise ValueError("[time->sample] input is not a valid time-major sequence.")

            sample_major = [[] for _ in range(B)]
            
            for t in range(Tm):
                row_t = texts_tm[t]
                if not isinstance(row_t, (list, tuple)) or len(row_t) != B:
                    continue
                if t >= L_total:
                    
                    continue
                for b in range(B):
                    s = row_t[b] if isinstance(row_t[b], str) else ""
                    if cond_t_bool[b, t]:
                        
                        
                        sample_major[b].append(s)
                    elif s.strip() != "":
                        
                        raise ValueError(f"[time->sample] future text found at b={b}, t={t}: {s[:60]}...")
            return sample_major

        
        cond_t = self._cond_time_mask(cond_mask)  
        if layout not in ("auto", "sample", "time"):
            raise ValueError(f"layout must be 'auto', 'sample', or 'time', got {layout}")

        if layout == "time":
            
            intrinsic_texts = _to_sample_major_from_time_major(intrinsic_texts, cond_t)

        elif layout == "auto":
            
            
            lengths = []
            if isinstance(intrinsic_texts, (list, tuple)) and len(intrinsic_texts) > 0:
                
                
                
                probe = intrinsic_texts[:min(5, len(intrinsic_texts))]
                inner_is_list = [isinstance(x, (list, tuple)) for x in probe]
                if all(inner_is_list) and all(isinstance(x, (list, tuple)) for x in intrinsic_texts):
                    
                    inner_lens = [len(x) for x in intrinsic_texts if isinstance(x, (list, tuple))]
                    lengths = inner_lens
                    majority_is_B = sum(1 for Lx in inner_lens if Lx == B) >= max(1, int(0.8 * len(inner_lens)))
                    looks_time_major = (len(intrinsic_texts) != B) and majority_is_B
                else:
                    
                    looks_time_major = False
            else:
                looks_time_major = False

            if looks_time_major:
                
                raise ValueError(
                    f"[encode_intrinsic_texts] input appears time-major "
                    f"(outer len={len(intrinsic_texts)}, average inner len="
                    f"{(sum(lengths)/len(lengths)) if lengths else 'unknown'}, B={B}); "
                    "use sample-major input or layout='time'."
                )
            

        
        intr_list = list(intrinsic_texts)[:B] + [[] for _ in range(max(0, B - len(intrinsic_texts)))]

        
        positions, texts_to_encode = [], []
        for b in range(B):
            idx_hist = torch.nonzero(cond_t[b], as_tuple=False).squeeze(-1).tolist()  
            row = intr_list[b]
            if isinstance(row, str):
                row = [row]
            row = [s if isinstance(s, str) else "" for s in (row or [])]

            n_hist = len(idx_hist)
            n_txt  = len(row)

            if n_hist == 0 and n_txt == 0:
                continue

            if align == "strict":
                if n_txt != n_hist:
                    raise ValueError(
                        f"[strict] sample b={b}: #texts ({n_txt}) != #history_steps ({n_hist}). "
                        f"history idx={idx_hist}"
                    )
                idx_sel = idx_hist[:]  
            elif align == "right":
                if n_txt == 0 or n_hist == 0:
                    continue
                if n_txt > n_hist:
                    row = row[-n_hist:]   
                idx_sel = idx_hist[-len(row):]
            elif align == "left":
                if n_txt == 0 or n_hist == 0:
                    continue
                if n_txt > n_hist:
                    row = row[:n_hist]
                idx_sel = idx_hist[:len(row)]
            else:
                raise ValueError(f"align must be 'left', 'right', or 'strict', got {align}")

            for t_pos, s in zip(idx_sel, row):
                if s.strip() == "":
                    continue
                positions.append((b, t_pos, s))
                texts_to_encode.append(s)

        if not positions:
            
            return out

        
        unique_texts = list(dict.fromkeys(texts_to_encode))
        emb = self._encode_texts_chunked(unique_texts)  
        out = out.to(dtype=emb.dtype)
        lookup = {txt: emb[i] for i, txt in enumerate(unique_texts)}

        for (b, t, s) in positions:
            out[b, t] = lookup[s]

        
        out = out * cond_t.unsqueeze(-1).to(dtype=out.dtype)
        return out

    @torch.no_grad()
    def assert_intrinsic_mapping(self, intrinsic_texts, cond_mask, L_total, B=None, align="strict", atol=0.0, encoded_out=None):
        """Internal helper."""
        if encoded_out is None:
            out = self.encode_intrinsic_texts(intrinsic_texts, cond_mask, L_total, B=B, align=align)  
        else:
            out = encoded_out
        B_, L_, E = out.shape
        cond_t = self._cond_time_mask(cond_mask)  
        future = ~cond_t                           

        
        fut_vals = out[future]                    
        
        
        

        max_abs = fut_vals.abs().max().item()
        
        
        fut_viol = (fut_vals.abs().sum(dim=1) > atol).sum().item()

        
        

        if fut_viol > 0:
            
            where_b, where_l = torch.nonzero(future, as_tuple=True)
            
            nz_mask = (out[future].abs().sum(dim=1) > atol)
            bad_idx = torch.nonzero(nz_mask, as_tuple=False).squeeze(1)
            top = bad_idx[:5].tolist()
            examples = [(int(where_b[i].item()), int(where_l[i].item())) for i in top]
            raise AssertionError(f"{fut_viol} future steps have nonzero vectors (atol={atol}); examples: {examples}")

        
        return out


    def encode_future_hint_tokens(self, future_texts):
        """Internal helper."""
        if future_texts is None:
            return None, None
        
        fut = []
        for x in future_texts:
            if isinstance(x, (list, tuple)):
                fut.append(" ".join([s for s in x if isinstance(s, str)]))
            elif isinstance(x, str):
                fut.append(x)
            else:
                fut.append("")

        cache = getattr(self, "_future_token_cache", None)
        if cache is None:
            cache = self._future_token_cache = {}

        encoded = []
        missing = []
        missing_indices = []
        for idx, text in enumerate(fut):
            value = cache.get(text)
            if value is None:
                encoded.append(None)
                missing.append(text)
                missing_indices.append(idx)
            else:
                encoded.append(value)

        if missing:
            self.text_encoder.eval()
            tok = self._tokenize_texts(missing, padding=True)
            with torch.no_grad():
                hidden = self.text_encoder(**tok).last_hidden_state       
            attn_mask = tok["attention_mask"].to(torch.bool)              
            key_padding_mask = ~attn_mask                                 

            for local_idx, text in enumerate(missing):
                real_len = int(attn_mask[local_idx].sum().item())
                value = (
                    hidden[local_idx, :real_len].detach().cpu(),
                    key_padding_mask[local_idx, :real_len].detach().cpu(),
                )
                cache[text] = value
                encoded[missing_indices[local_idx]] = value

        max_len = max(tokens.size(0) for tokens, _ in encoded)
        out_rows = []
        kpm_rows = []
        for tokens, kpm in encoded:
            pad_len = max_len - tokens.size(0)
            if pad_len > 0:
                tokens = torch.cat(
                    [tokens, torch.zeros(pad_len, tokens.size(-1), dtype=tokens.dtype)],
                    dim=0,
                )
                kpm = torch.cat(
                    [kpm, torch.ones(pad_len, dtype=torch.bool)],
                    dim=0,
                )
            out_rows.append(tokens)
            kpm_rows.append(kpm)

        out = torch.stack(out_rows, dim=0).to(self.device)
        key_padding_mask = torch.stack(kpm_rows, dim=0).to(self.device)
        
        if out.size(-1) != self.context_dim:
            if not hasattr(self, "_ctx_proj"):
                self._ctx_proj = nn.Linear(out.size(-1), self.context_dim, bias=False).to(self.device)
            out = self._ctx_proj(out)
        return out, key_padding_mask

    def distill_future_context(self, future_tokens, future_kpm):
        """
        Apply token distillation on future context tokens.
        Input:
          future_tokens: (B, T_ctx, E) or None
          future_kpm: (B, T_ctx) bool or None
        Output:
          distilled_tokens: (B, K_future, E) or None
          distilled_kpm: (B, K_future) bool or None
        """
        if future_tokens is None:
            return None, None
        B, T_ctx, _ = future_tokens.shape
        if future_kpm is None:
            future_kpm = torch.zeros(B, T_ctx, dtype=torch.bool, device=future_tokens.device)
        elif future_kpm.dtype != torch.bool:
            future_kpm = future_kpm.to(torch.bool)

        if not self.use_token_distill:
            return future_tokens, future_kpm

        distilled, distilled_kpm = self.future_distiller(future_tokens, future_kpm)
        return distilled, distilled_kpm

    def distill_intrinsic_context(self, intrinsic_tokens, cond_mask):
        """
        Distill history intrinsic tokens using learnable prototype queries.
        Input:
          intrinsic_tokens: (B, L, E)
          cond_mask: (B, K, L), 1=history
        Output:
          distilled_tokens: (B, K_intr, E)
          distilled_kpm: (B, K_intr) bool
        """
        if intrinsic_tokens is None:
            return None, None
        if cond_mask is None:
            raise ValueError("distill_intrinsic_context needs cond_mask to build history mask.")

        cond_t = self._cond_time_mask(cond_mask)  
        nonzero_tok = intrinsic_tokens.abs().sum(dim=-1) > 0
        valid_mask = cond_t & nonzero_tok
        key_padding_mask = ~valid_mask

        if not valid_mask.any():
            return None, None

        if not self.use_token_distill:
            return intrinsic_tokens, key_padding_mask

        distilled, distilled_kpm = self.intrinsic_distiller(intrinsic_tokens, key_padding_mask)
        return distilled, distilled_kpm

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask
    
    def get_timestep_info(self, timesteps):
        timestep_emb = self.timestep_emb(timesteps.transpose(1, 2)).transpose(1, 2)
        timestep_emb = timestep_emb.unsqueeze(2).expand(-1, -1, self.target_dim, -1) 
        return timestep_emb
    
    
    
    
    
    
    
    

    
    
    
    
    

    
    
    

    
    
    

    
    
    
    


    def get_relative_size_info(self, observed_data):
        B, K, L = observed_data.shape

        size_emb = observed_data[:, :, :self.lookback_len].clone().unsqueeze(3).expand(-1, -1, -1, self.lookback_len) - \
            observed_data[:, :, :self.lookback_len].clone().unsqueeze(2).expand(-1, -1, self.lookback_len, -1) 
        size_emb = self.relative_size_emb(size_emb)
        size_emb = size_emb.permute(0, 3, 1, 2)
        size_emb = torch.cat([size_emb, torch.zeros((B, self.diff_channels, K, self.pred_len)).to(observed_data.device)], dim=-1) 
        return size_emb
    
    def get_text_info(self, text, text_mask):
        token_input = self._tokenize_texts(text, padding='max_length')
        context = self.text_encoder(**token_input).last_hidden_state
        context = context * text_mask.unsqueeze(1).unsqueeze(1)
        context = context.permute(0, 2, 1) 
        if self.save_token:
            tokens_str = self.tokenizer.batch_decode(token_input['input_ids'])
            return context, tokens_str
        else:
            return context

    def get_side_info(self, observed_tp, cond_mask, feature_id=None, timesteps=None, texts=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim) 
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1) 

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            ) 
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else: 
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1) 

        side_info = torch.cat([time_embed, feature_embed], dim=-1) 
        side_info = side_info.permute(0, 3, 2, 1) 

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1) 
            side_info = torch.cat([side_info, side_mask], dim=1) 
    

        return side_info

    def forward(self, batch, is_train=1):
        (
        observed_data,            
        observed_mask,            
        observed_tp,              
        gt_mask,                  
        for_pattern_mask,         
        cut_length,               
        feature_id,               
        timesteps,                
        intrinsic_texts,          
        future_context,           
        _future_points,           
        ) = self.process_data(batch)
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else: 
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id, timesteps)

        if self.timestep_emb_cat:
            timestep_emb = self.get_timestep_info(timesteps)
        else:
            timestep_emb = None

        if self.relative_size_emb_cat:
            size_emb = self.get_relative_size_info(observed_data)
        else:
            size_emb = None

        future_ctx, future_kpm = None, None
        if self.with_future_hint:
            future_ctx, future_kpm = self.encode_future_hint_tokens(future_context)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
  
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        
        return loss_func(
                observed_data, cond_mask, observed_mask, side_info, is_train,
                timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb,
                future_context=future_ctx,          
                intrinsic_texts=intrinsic_texts,
                future_kpm=future_kpm               
            )
    
    def evaluate(self, batch, n_samples, guide_w,
                edit_guidance_method: str = "langevin",
                guidance_scale: float = 0.0,
                time_radius: int = 1,
                energy_softmin_tau: float = 0.05,
                num_iter: int = 0,
                langevin_step_size: float | None = None,
                langevin_alpha: float = 1.0,
                langevin_noise_scale: float | None = None,
                blended_amplitude_modulation: bool = False,
                blended_amplitude_strength: float = 0.15,
                data_mean=0, data_std=1):
        (
            observed_data,            
            observed_mask,            
            observed_tp,              
            gt_mask,                  
            for_pattern_mask,         
            cut_length,               
            feature_id,               
            timesteps,                
            intrinsic_texts,          
            future_context,           
            future_points,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask, timesteps=timesteps)

            if self.timestep_emb_cat:
                timestep_emb = self.get_timestep_info(timesteps)
            else:
                timestep_emb = None

            if self.relative_size_emb_cat:
                size_emb = self.get_relative_size_info(observed_data)
            else:
                size_emb = None

            
            
            
            
            
            
            
            if self.save_attn:
                samples, attn = self.impute(observed_data, cond_mask, side_info, n_samples, guide_w,
                                            timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb,
                                            intrinsic_texts=intrinsic_texts, future_context=future_context,
                                            future_points=future_points,
                                            edit_guidance_method=edit_guidance_method,
                                            guidance_scale=guidance_scale,
                                            time_radius=time_radius,
                                            energy_softmin_tau=energy_softmin_tau,
                                            num_iter=num_iter,
                                            langevin_step_size=langevin_step_size,
                                            langevin_alpha=langevin_alpha,
                                            langevin_noise_scale=langevin_noise_scale,
                                            blended_amplitude_modulation=blended_amplitude_modulation,
                                            blended_amplitude_strength=blended_amplitude_strength,
                                            data_mean=data_mean,
                                            data_std=data_std)
            else:
                samples = self.impute(observed_data, cond_mask, side_info, n_samples, guide_w,
                                      timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb,
                                      intrinsic_texts=intrinsic_texts, future_context=future_context,
                                      future_points=future_points,
                                      edit_guidance_method=edit_guidance_method,
                                      guidance_scale=guidance_scale,
                                      time_radius=time_radius,
                                      energy_softmin_tau=energy_softmin_tau,
                                      num_iter=num_iter,
                                      langevin_step_size=langevin_step_size,
                                      langevin_alpha=langevin_alpha,
                                      langevin_noise_scale=langevin_noise_scale,
                                      blended_amplitude_modulation=blended_amplitude_modulation,
                                      blended_amplitude_strength=blended_amplitude_strength,
                                      data_mean=data_mean,
                                      data_std=data_std)
        
        if self.save_attn:
            
            
            
            return samples, observed_data, target_mask, observed_mask, observed_tp, attn
        else:
            return samples, observed_data, target_mask, observed_mask, observed_tp
