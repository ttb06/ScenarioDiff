import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
from utils.SelfAttention_Family import FullAttention, TV_AttentionLayer, TV_AttentionEncoderLayer, AttentionLayer, AttentionEncoderLayer

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def get_custom_tv_trans(heads=8, layers=1, channels=64, dropout=0.0, pre_norm=False):
    encoder_layer = TV_AttentionLayer(
        FullAttention(mask_flag=False),
        d_t=channels//4,
        d_v=channels,
        n_heads=heads
        )
    return TV_AttentionEncoderLayer(encoder_layer, d_t=channels//4, d_v=channels, d_ff=64, activation="gelu", dropout=dropout, pre_norm=pre_norm)

def get_cross_trans(heads=8, layers=1, channels=64, dropout=0.0, pre_norm=False):
    encoder_layer = AttentionLayer(
        FullAttention(mask_flag=False),
        d_model=channels,
        n_heads=heads
    )
    return AttentionEncoderLayer(encoder_layer, d_model=channels, d_ff=64, activation="gelu", dropout=dropout, pre_norm=pre_norm)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def safe_cross_attention(mha, query, key, value, key_padding_mask):
    """
    Run MultiheadAttention when some batch rows have every key masked.
    PyTorch returns NaN for all-masked rows, so expose one zero key there
    and zero the corresponding attention output afterwards.
    """
    if key_padding_mask is None:
        return mha(query=query, key=key, value=value, key_padding_mask=None)

    if key_padding_mask.dtype != torch.bool:
        key_padding_mask = key_padding_mask.to(torch.bool)

    if key_padding_mask.size(1) == 0:
        out = torch.zeros_like(query)
        attn = torch.zeros(
            query.size(0), query.size(1), 0,
            dtype=query.dtype, device=query.device,
        )
        return out, attn

    all_masked = key_padding_mask.all(dim=1)
    attn_kpm = key_padding_mask
    safe_key = key
    safe_value = value

    if all_masked.any():
        attn_kpm = key_padding_mask.clone()
        attn_kpm[all_masked, 0] = False
        safe_key = key.clone()
        safe_value = value.clone()
        safe_key[all_masked, 0] = 0.0
        safe_value[all_masked, 0] = 0.0

    out, attn = mha(query=query, key=safe_key, value=safe_value, key_padding_mask=attn_kpm)

    if all_masked.any():
        row_mask = all_masked.view(-1, 1, 1)
        out = out.masked_fill(row_mask, 0.0)
        if attn is not None:
            attn = attn.masked_fill(row_mask, 0.0)

    return out, attn


class AdaptiveLayerNorm(nn.Module):
    """Internal helper."""
    def __init__(self, channels: int, emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.proj = nn.Linear(emb_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        s = self.proj(emb)                        
        scale, shift = s.chunk(2, dim=-1)         
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNAttentionBlock(nn.Module):
    """Internal helper."""
    def __init__(self, channels: int, nheads: int, emb_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.ada_ln1 = AdaptiveLayerNorm(channels, emb_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=nheads,
            dropout=dropout, batch_first=True,
        )
        self.ada_ln2 = AdaptiveLayerNorm(channels, emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * channels, channels),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        
        h = self.ada_ln1(x, emb)
        attn_out, attn_w = self.attn(h, h, h,
                                     need_weights=True,
                                     average_attn_weights=False)
        x = x + self.drop(attn_out)
        
        h = self.ada_ln2(x, emb)
        x = x + self.drop(self.ff(h))
        return x, attn_w


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x) 
        x = F.silu(x)
        x = self.projection2(x) 
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1) 
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0) 
        table = steps * frequencies 
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  
        return table


class TokenDistiller(nn.Module):
    """
    Token distillation with learnable queries (paper-style):
      X_distilled = CrossAttn(R, X_tokens),
    where R are learnable prototype queries.
    """
    def __init__(self, token_dim: int, num_queries: int, nheads: int, dropout: float = 0.0):
        super().__init__()
        if num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {num_queries}")
        self.num_queries = int(num_queries)
        self.token_dim = int(token_dim)
        self.query_tokens = nn.Parameter(torch.randn(self.num_queries, self.token_dim) * 0.02)
        self.norm_q = nn.LayerNorm(self.token_dim)
        self.norm_kv = nn.LayerNorm(self.token_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        """
        tokens: (B, N, D)
        key_padding_mask: (B, N) bool, True means PAD/masked.
        returns:
          distilled: (B, Q, D)
          distilled_kpm: (B, Q) bool
        """
        if tokens.dim() != 3:
            raise ValueError(f"tokens must be (B,N,D), got {tuple(tokens.shape)}")
        B, N, D = tokens.shape
        if D != self.token_dim:
            raise ValueError(f"token dim mismatch: expected {self.token_dim}, got {D}")
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        else:
            if key_padding_mask.dim() != 2 or key_padding_mask.shape != (B, N):
                raise ValueError(
                    f"key_padding_mask must be (B,N)=({B},{N}), got {tuple(key_padding_mask.shape)}"
                )
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(torch.bool)

        q = self.norm_q(self.query_tokens).unsqueeze(0).expand(B, -1, -1)   
        kv = self.norm_kv(tokens)                                            
        distilled, _ = safe_cross_attention(self.cross_attn, q, kv, kv, key_padding_mask)
        all_masked = key_padding_mask.all(dim=1, keepdim=True)              
        distilled = torch.where(all_masked.unsqueeze(-1), torch.zeros_like(distilled), distilled)
        distilled_kpm = all_masked.expand(-1, self.num_queries).contiguous()
        return distilled, distilled_kpm

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads,
                 is_linear=False, with_text=False, context_dim=None,
                 dropout=0., attn_drop=0., pre_norm=False, pred_len=-1):
        super().__init__()
        self.channels = channels
        self.side_dim = side_dim
        self.diffusion_embedding_dim = diffusion_embedding_dim
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)   
        self.cond_projection      = Conv1d_with_init(side_dim, 2 * channels, 1)   
        self.mid_projection       = Conv1d_with_init(channels, 2 * channels, 1)   
        self.output_projection    = Conv1d_with_init(channels, 2 * channels, 1)   
        self.dropout  = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_drop)

        self.is_linear = is_linear
        self.with_text = with_text  
        
        
        self.time_condition_proj = nn.LazyLinear(channels)

        
        
        
        
        if is_linear:
            self.time_layer    = get_linear_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer    = AdaLNAttentionBlock(channels, nheads, diffusion_embedding_dim, dropout)
            self.feature_layer = AdaLNAttentionBlock(channels, nheads, diffusion_embedding_dim, dropout)

        
        if with_text:
            assert context_dim is not None, "context_dim is required when with_text=True"
            self.ctx_proj = nn.Linear(context_dim, channels, bias=False)   
            self.cross_attn_future = nn.MultiheadAttention(
                embed_dim=channels, num_heads=nheads, dropout=attn_drop, batch_first=True
            )
            self.gamma_future = nn.Parameter(torch.tensor(0.5))            

            
            self.intr_proj  = nn.Linear(context_dim, channels, bias=False)  
            self.cross_attn_intrinsic = nn.MultiheadAttention(
                embed_dim=channels, num_heads=nheads, dropout=attn_drop, batch_first=True
            )
            self.gamma_intr = nn.Parameter(torch.tensor(0.1))

    
    def forward_time(self, y, base_shape, diffusion_emb=None, timestep_emb=None):
        """Internal helper."""
        B, C, K, L = base_shape
        if L == 1:
            return y, None
        
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B * K, C, L)
        y_seq = y.permute(0, 2, 1)                                       

        if timestep_emb is not None:
            if timestep_emb.dim() != 4:
                raise ValueError(f"timestep_emb must have shape (B,C_t,K,L), got {tuple(timestep_emb.shape)}")
            if timestep_emb.size(0) != B or timestep_emb.size(2) != K or timestep_emb.size(3) != L:
                raise ValueError(
                    f"timestep_emb must match (B,*,K,L)=({B},*,{K},{L}), got {tuple(timestep_emb.shape)}"
                )
            time_cond = timestep_emb.permute(0, 2, 3, 1).reshape(B * K, L, -1)   
            y_seq = y_seq + self.time_condition_proj(time_cond)                   

        if self.is_linear:
            
            y_seq = self.time_layer(y_seq)                                       
            y = y_seq.permute(0, 2, 1)                                           
            attn_w = None
        else:
            
            emb_bk = (diffusion_emb
                      .unsqueeze(1)
                      .expand(-1, K, -1)
                      .reshape(B * K, -1))                               
            y_seq, attn_w = self.time_layer(y_seq, emb_bk)              
            y = y_seq.permute(0, 2, 1)                                   
        y = y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K * L)
        return y, attn_w

    
    def forward_feature(self, y, base_shape, diffusion_emb=None):
        """Internal helper."""
        B, C, K, L = base_shape
        if K == 1:
            return y
        
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B * L, C, K)
        if self.is_linear:
            
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1) 
        else:
            
            y_seq = y.permute(0, 2, 1)                                   
            emb_bl = (diffusion_emb
                      .unsqueeze(1)
                      .expand(-1, L, -1)
                      .reshape(B * L, -1))                               
            y_seq, _ = self.feature_layer(y_seq, emb_bl)                
            y = y_seq.permute(0, 2, 1)                                   
        y = y.reshape(B, L, C, K).permute(0, 2, 3, 1).reshape(B, C, K * L)
        return y

    def _assert_same_BL(self, name, tensor, B, L):
        if tensor is None: return
        if tensor.dim() != 3:
            raise ValueError(f"{name} must be 3D, got {tuple(tensor.shape)}")
        if tensor.size(0) != B or tensor.size(1) != L:
            raise ValueError(f"{name} must match (B,L,*)=({B},{L},*), got {tuple(tensor.shape)}")
    
    def forward(self, x, cond_info, diffusion_emb,
                timestep_emb=None,      
                future_context=None,    
                intrinsic_tokens=None,  
                future_kpm=None,        
                intrinsic_kpm=None,     
                cond_mask=None,         
                ):
        """Internal helper."""
        
        if x.dim() != 4:
            raise ValueError(f"x must have shape (B,C,K,L), got {tuple(x.shape)}")
        B, C, K, L = x.shape
        if C != self.channels:
            raise ValueError(f"x channel count must be {self.channels}, got {C}")
        if cond_info.shape[:1] != (B,):
            raise ValueError(f"cond_info batch size must be {B}, got {cond_info.shape[0]}")
        if cond_info.dim() != 4 or cond_info.size(2) != K or cond_info.size(3) != L:
            raise ValueError(f"cond_info must have shape (B,side_dim,K,L), got {tuple(cond_info.shape)}")
        if cond_info.size(1) != self.side_dim:
            raise ValueError(f"cond_info side_dim={cond_info.size(1)} != {self.side_dim}")
        

        
        x_flat = x.reshape(B, C, K * L)                                  
        d_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  
        y = x_flat + d_proj                                              
        

        
        attn_series_text = None
        if self.with_text:
            if cond_mask is None:
                raise ValueError("with_text=True requires cond_mask.")
            if cond_mask.dim() != 3 or cond_mask.size(0) != B or cond_mask.size(1) != K or cond_mask.size(2) != L:
                raise ValueError(f"cond_mask must match (B,K,L)=({B},{K},{L}), got {tuple(cond_mask.shape)}")

            
            y_4d   = y.reshape(B, C, K, L)                               
            y_time = y_4d.permute(0, 3, 1, 2).mean(dim=3)                

            
            cond_bool = (cond_mask > 0.5) if cond_mask.dtype != torch.bool else cond_mask
            hist_mask = cond_bool.any(dim=1)                              
            fut_mask  = ~hist_mask                                        

            
            if future_context is not None:
                if future_context.dim() != 3 or future_context.size(0) != B:
                    raise ValueError(f"future_context must have shape (B,T_ctx,E_ctx), got {tuple(future_context.shape)}")
                T_ctx = future_context.size(1)
                
                if future_kpm is None:
                    raise ValueError("future_kpm is required with future_context.")
                if future_kpm.dim() != 2 or future_kpm.size(0) != B or future_kpm.size(1) != T_ctx:
                    raise ValueError(f"future_kpm must match (B,T_ctx)=({B},{T_ctx}), got {tuple(future_kpm.shape)}")
                if future_kpm.dtype != torch.bool:
                    raise ValueError("future_kpm must be torch.bool (True=PAD).")

                
                ctx = self.ctx_proj(future_context)                       
                
                
                attn_out, attn_w = safe_cross_attention(
                    self.cross_attn_future, y_time, ctx, ctx, future_kpm
                )                                                         

                
                fut_mask_f = fut_mask.float().unsqueeze(-1)               
                attn_out = self.attn_drop(attn_out) * fut_mask_f          

                
                
                attn_out_ckl = attn_out.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, K, -1)  
                y_4d = y_4d + self.gamma_future * attn_out_ckl
                attn_series_text = attn_w
            

            
            if intrinsic_tokens is not None:
                if intrinsic_tokens.dim() != 3 or intrinsic_tokens.size(0) != B:
                    raise ValueError(
                        f"intrinsic_tokens must have shape (B,T_intr,E_intr), got {tuple(intrinsic_tokens.shape)}"
                    )
                T_intr = intrinsic_tokens.size(1)
                if intrinsic_kpm is None:
                    intrinsic_kpm = torch.zeros(B, T_intr, dtype=torch.bool, device=intrinsic_tokens.device)
                else:
                    if intrinsic_kpm.dim() != 2 or intrinsic_kpm.shape != (B, T_intr):
                        raise ValueError(
                            f"intrinsic_kpm must match (B,T_intr)=({B},{T_intr}), got {tuple(intrinsic_kpm.shape)}"
                        )
                    if intrinsic_kpm.dtype != torch.bool:
                        intrinsic_kpm = intrinsic_kpm.to(torch.bool)

                intr_ctx = self.intr_proj(intrinsic_tokens)                 
                intr_out, _ = safe_cross_attention(
                    self.cross_attn_intrinsic, y_time, intr_ctx, intr_ctx, intrinsic_kpm
                )                                                           
                hist_mask_f = hist_mask.float().unsqueeze(-1)               
                intr_out = self.attn_drop(intr_out) * hist_mask_f
                intr_ckl = intr_out.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, K, -1)  
                y_4d = y_4d + self.gamma_intr * intr_ckl
            

            
            y = y_4d.reshape(B, C, K * L)
        

        
        y, atten_series_timestamp = self.forward_time(
            y, (B, C, K, L), diffusion_emb=diffusion_emb, timestep_emb=timestep_emb
        )
        

        
        y = self.forward_feature(y, (B, C, K, L), diffusion_emb)
        

        
        y = self.attn_drop(y)                              

        y = self.mid_projection(y)                         
        y = self.dropout(y)
        

        
        if cond_info.size(1) != self.side_dim:
            raise ValueError(f"cond_info side_dim={cond_info.size(1)} != {self.side_dim}")
        cond_flat = cond_info.reshape(B, self.side_dim, K * L)            
        cond_flat = self.cond_projection(cond_flat)                       
        y = y + cond_flat                                                 

        
        gate, filt = torch.chunk(y, 2, dim=1)                             
        y = torch.sigmoid(gate) * torch.tanh(filt)                        

        y = self.output_projection(y)                                     
        y = self.dropout(y)

        residual, skip = torch.chunk(y, 2, dim=1)                         
        x_out  = x_flat                                                   

        
        residual = residual.reshape(B, C, K, L)
        skip     = skip.reshape(B, C, K, L)
        x_out    = x_out.reshape(B, C, K, L)

        return (x_out + residual) / math.sqrt(2.0), skip, timestep_emb, atten_series_timestamp, attn_series_text

class TimeSeriesPatcher(nn.Module):
    """Internal helper."""
    def __init__(self, channels: int, side_dim: int,
                 patch_len: int, lookback_len: int, pred_len: int,
                 mode: str = 'pad'):
        super().__init__()
        if patch_len < 2:
            raise ValueError(f"patch_len must be at least 2, got {patch_len}")
        self.P    = patch_len
        self.lk   = lookback_len
        self.pl   = pred_len
        self.mode = mode

        if mode == 'pad':
            
            self.hist_pad = (-lookback_len) % patch_len   
            self.pred_pad = (-pred_len)     % patch_len
        elif mode == 'skip':
            self.hist_pad = 0
            self.pred_pad = 0
        else:
            raise ValueError(f"mode must be 'pad' or 'skip', got {mode!r}")

        self.n_hist    = (lookback_len + self.hist_pad) // patch_len
        self.n_pred    = (pred_len     + self.pred_pad) // patch_len
        self.n_patches = self.n_hist + self.n_pred

        
        
        self.proj_x  = nn.Conv1d(
            channels, channels, patch_len, stride=patch_len, bias=False, groups=channels
        )
        self.proj_ci = nn.Conv1d(
            side_dim, side_dim, patch_len, stride=patch_len, bias=False, groups=side_dim
        )
        nn.init.constant_(self.proj_x .weight, 1.0 / patch_len)
        nn.init.constant_(self.proj_ci.weight, 1.0 / patch_len)

    def _prep(self, seg_hist: torch.Tensor, seg_pred: torch.Tensor) -> torch.Tensor:
        """Internal helper."""
        P = self.P
        if self.mode == 'pad':
            
            z_h = seg_hist.new_zeros(list(seg_hist.shape[:-1]) + [self.hist_pad])
            z_p = seg_pred.new_zeros(list(seg_pred.shape[:-1]) + [self.pred_pad])
            h = torch.cat([z_h, seg_hist], dim=-1)   
            p = torch.cat([seg_pred, z_p], dim=-1)   
        else:  
            drop_h = self.lk - self.n_hist * P       
            h = seg_hist[..., drop_h:]               
            p = seg_pred[..., : self.n_pred * P]     
        return torch.cat([h, p], dim=-1)             

    def forward(self,
                x: torch.Tensor,
                cond_info: torch.Tensor,
                cond_mask: torch.Tensor,
                intrinsic_tokens: torch.Tensor | None = None,
                ):
        """Internal helper."""
        B, C, K, L = x.shape
        P, lk, pl  = self.P, self.lk, self.pl
        n           = self.n_patches
        assert L == lk + pl, (
            f"L={L} != lookback_len+pred_len={lk+pl}. "
            "Check lookback_len and pred_len."
        )

        
        x_h,  x_p_   = x[:, :, :, :lk],        x[:, :, :, lk:]
        ci_h, ci_p_  = cond_info[:, :, :, :lk], cond_info[:, :, :, lk:]
        cm_h, cm_p_  = cond_mask[:, :, :lk],    cond_mask[:, :, lk:]

        
        x_proc  = self._prep(x_h,  x_p_)    
        ci_proc = self._prep(ci_h, ci_p_)   
        cm_proc = self._prep(cm_h, cm_p_)   
        if self.mode == 'pad' and self.hist_pad > 0:
            cm_proc[..., :self.hist_pad] = 1.0

        
        sd   = cond_info.size(1)
        x_bk  = x_proc .permute(0, 2, 1, 3).reshape(B * K, C,  n * P)
        ci_bk = ci_proc.permute(0, 2, 1, 3).reshape(B * K, sd, n * P)

        x_out  = self.proj_x (x_bk ).reshape(B, K, C,  n).permute(0, 2, 1, 3)  
        ci_out = self.proj_ci(ci_bk).reshape(B, K, sd, n).permute(0, 2, 1, 3)  

        
        
        cm_out = cm_proc.reshape(B, K, n, P).min(dim=-1).values               

        
        it_out = None
        if intrinsic_tokens is not None:
            E = intrinsic_tokens.size(-1)
            it_h = intrinsic_tokens[:, :lk, :].permute(0, 2, 1)  
            it_p = intrinsic_tokens[:, lk:, :].permute(0, 2, 1)  
            it_proc = self._prep(it_h, it_p).permute(0, 2, 1)    
            it_out  = it_proc.reshape(B, n, P, E).mean(dim=2)     

        return x_out, ci_out, cm_out, it_out


class PatchExpander(nn.Module):
    """Internal helper."""
    def __init__(self, channels: int, patch_len: int,
                 lookback_len: int, pred_len: int,
                 hist_pad: int, pred_pad: int,
                 n_hist: int, n_pred: int,
                 mode: str = 'pad',
                 decode_mode: str = 'conv'):
        super().__init__()
        self.P    = patch_len
        self.lk   = lookback_len
        self.pl   = pred_len
        self.hp   = hist_pad
        self.pp   = pred_pad
        self.nh   = n_hist
        self.np_  = n_pred
        self.mode = mode
        self.decode_mode = decode_mode

        if decode_mode not in ("conv", "linear"):
            raise ValueError(f"decode_mode must be 'conv' or 'linear', got {decode_mode!r}")

        
        if decode_mode == "conv":
            self.unproj = nn.ConvTranspose1d(
                channels, channels, patch_len, stride=patch_len, bias=True, groups=channels
            )
            nn.init.zeros_(self.unproj.bias)
            nn.init.constant_(self.unproj.weight, 1.0)

    def _linear_expand_segment(self, x_seg: torch.Tensor, out_len: int) -> torch.Tensor:
        """Upsample one history/future patch segment without mixing across the boundary."""
        B, C, K, n = x_seg.shape
        if n <= 0 or out_len <= 0:
            return x_seg.new_zeros(B, C, K, max(out_len, 0))
        xbk = x_seg.permute(0, 2, 1, 3).reshape(B * K, C, n)
        if n == 1:
            out = xbk.expand(-1, -1, out_len)
        else:
            out = F.interpolate(xbk, size=out_len, mode="linear", align_corners=False)
        return out.reshape(B, K, C, out_len).permute(0, 2, 1, 3)

    def forward(self, x_p: torch.Tensor) -> torch.Tensor:
        """Internal helper."""
        B, C, K, n = x_p.shape
        P, lk, pl  = self.P, self.lk, self.pl

        
        if self.decode_mode == "conv":
            xbk = x_p.permute(0, 2, 1, 3).reshape(B * K, C, n)
            out  = self.unproj(xbk)                               
            out  = out.reshape(B, K, C, n * P).permute(0, 2, 1, 3)  
        else:
            hist = self._linear_expand_segment(x_p[:, :, :, :self.nh], self.nh * P)
            pred = self._linear_expand_segment(x_p[:, :, :, self.nh:self.nh + self.np_], self.np_ * P)
            out = torch.cat([hist, pred], dim=-1)

        if self.mode == 'pad':
            
            hist_out = out[:, :, :, self.hp : self.hp + lk]
            ps = self.nh * P                                   
            pred_out = out[:, :, :, ps : ps + pl]
        else:  
            drop_h   = lk - self.nh * P
            hist_out = out.new_zeros(B, C, K, lk)
            hist_out[:, :, :, drop_h:] = out[:, :, :, : self.nh * P]

            pred_out = out.new_zeros(B, C, K, pl)
            pred_out[:, :, :, : self.np_ * P] = out[:, :, :, self.nh*P : (self.nh + self.np_)*P]

        return torch.cat([hist_out, pred_out], dim=-1)         


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2, mode_num=0):
        super().__init__()
        self.channels = config["channels"]
        self.mode_num = mode_num
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.lookback_len = config["lookback_len"]
        self.pred_len = config["pred_len"]
        self.with_timestep = config["with_timestep"]
        dropout = config["dropout"]
        attn_drop = config["attn_drop"]
        self.pre_norm = config["pre_norm"]
        self.time_weight= config["time_weight"]
        self.save_attn = config["save_attn"]

        self.input_projection  = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.dropout = nn.Dropout(dropout)

        
        
        
        
        
        
        
        
        
        

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                    with_text=config["with_texts"],        
                    context_dim=config["context_dim"],     
                    dropout=dropout,
                    attn_drop=attn_drop,
                    pre_norm=self.pre_norm,
                    pred_len=self.pred_len
                )
                for _ in range(config["layers"])
            ]
        )

        
        
        
        
        patch_len = int(config.get("patch_len", 1))
        self.use_patching = patch_len > 1
        if self.use_patching:
            patch_mode = config.get("patch_mode", "pad")
            patch_decode_mode = config.get("patch_decode_mode", "conv")
            self.patcher = TimeSeriesPatcher(
                channels    = self.channels,
                side_dim    = config["side_dim"],
                patch_len   = patch_len,
                lookback_len= config["lookback_len"],
                pred_len    = config["pred_len"],
                mode        = patch_mode,
            )
            self.expander = PatchExpander(
                channels    = self.channels,
                patch_len   = patch_len,
                lookback_len= config["lookback_len"],
                pred_len    = config["pred_len"],
                hist_pad    = self.patcher.hist_pad,
                pred_pad    = self.patcher.pred_pad,
                n_hist      = self.patcher.n_hist,
                n_pred      = self.patcher.n_pred,
                mode        = patch_mode,
                decode_mode = patch_decode_mode,
            )

    @torch.no_grad()
    def _expand_cfg_mask_like(self, cfg_mask, *sizes):
        """Internal helper."""
        
        expand_shape = (cfg_mask.shape[0],) + tuple(1 for _ in sizes[0][1:])
        return cfg_mask.view(expand_shape).to(dtype=torch.float32)

    def forward(self, x, cond_info, diffusion_step, cfg_mask,
                timestep_emb=None, size_emb=None,
                future_context=None,    
                intrinsic_tokens=None,  
                future_kpm=None,        
                intrinsic_kpm=None,     
                cond_mask=None):        
        """Internal helper."""
        B, inputdim, K, L = x.shape
        
        x = x.reshape(B, inputdim, K * L)       
        x = self.input_projection(x)            
        x = self.dropout(x)
        x = F.relu(x)

        
        if timestep_emb is not None:
            if timestep_emb.dim() != 4:
                raise ValueError(f"timestep_emb must have shape (B,C_t,K,L), got {tuple(timestep_emb.shape)}")
            if timestep_emb.size(0) != B or timestep_emb.size(2) != K or timestep_emb.size(3) != L:
                raise ValueError(
                    f"timestep_emb must match (B,*,K,L)=({B},*,{K},{L}), got {tuple(timestep_emb.shape)}"
                )
        if size_emb is not None:
            size_emb = size_emb.reshape(B, -1, K * L)          
            x = torch.cat([x, size_emb], dim=1)                
        x = x.reshape(B, self.channels, K, L)                  

        
        diffusion_emb = self.diffusion_embedding(diffusion_step)  

        if cfg_mask is not None:
            if future_context is not None:
                m = self._expand_cfg_mask_like(cfg_mask, future_context.shape)   
                future_context = future_context * m
            if intrinsic_tokens is not None:
                m = self._expand_cfg_mask_like(cfg_mask, intrinsic_tokens.shape) 
                intrinsic_tokens = intrinsic_tokens * m

        if self.use_patching:
            if cond_mask is None:
                raise ValueError(
                    "patch_len > 1 requires cond_mask to identify the "
                    "history and forecast boundary."
                )
            intrinsic_for_patch = intrinsic_tokens
            
            
            if intrinsic_for_patch is not None and intrinsic_for_patch.size(1) != L:
                intrinsic_for_patch = None
            x, cond_info, cond_mask, intrinsic_tokens = self.patcher(
                x, cond_info, cond_mask, intrinsic_for_patch
            )
            
            
            if timestep_emb is not None and timestep_emb.size(-1) != x.size(-1):
                _, c_t, _, l_old = timestep_emb.shape
                l_new = x.size(-1)
                time_flat = timestep_emb.reshape(B * c_t * K, 1, l_old)
                time_flat = F.interpolate(time_flat, size=l_new, mode="linear", align_corners=False)
                timestep_emb = time_flat.reshape(B, c_t, K, l_new)
            
        

        
        memory_efficient_guidance = torch.is_grad_enabled() and not self.training and not self.save_attn
        skip = [] if not memory_efficient_guidance else None
        skip_sum = None
        attn = [] if self.save_attn else None

        for layer in self.residual_layers:
            x, skip_connection, timestep_emb, attn_time, attn_text = layer(
                x, cond_info, diffusion_emb,
                timestep_emb=timestep_emb,
                future_context=future_context,     
                intrinsic_tokens=intrinsic_tokens, 
                future_kpm=future_kpm,             
                intrinsic_kpm=intrinsic_kpm,       
                cond_mask=cond_mask                
            )
            if memory_efficient_guidance:
                skip_sum = skip_connection if skip_sum is None else skip_sum + skip_connection
            else:
                skip.append(skip_connection)
            if self.save_attn:
                attn.append((attn_time, attn_text))
            else:
                del attn_time, attn_text

        
        if memory_efficient_guidance:
            x = skip_sum / math.sqrt(len(self.residual_layers))
        else:
            x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        
        if self.use_patching:
            x = self.expander(x)    
        
        x = x.reshape(B, self.channels, K * L)    
        x = self.output_projection1(x)            
        x = self.dropout(x)
        x = F.relu(x)
        x = self.output_projection2(x)            
        x = x.reshape(B, K, L)                    


        if self.save_attn:
            return x, attn
        else:
            return x
