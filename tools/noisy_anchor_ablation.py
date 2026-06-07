#!/usr/bin/env python3
"""Internal helper."""

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml


DEFAULT_CHECKPOINT_DIRS = []

METRICS = ("MSE", "MAE", "NMSE", "NMAE", "CRPS", "CRPS_sum")
LOWER_IS_BETTER = set(METRICS)


def _as_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def parse_levels(values):
    levels = []
    for value in values:
        for item in str(value).replace(",", " ").split():
            if not item:
                continue
            ratio = float(item)
            if ratio > 1.0:
                ratio /= 100.0
            if ratio < 0:
                raise ValueError(f"noise level must be non-negative, got {item}")
            levels.append(ratio)
    return levels


def pct_label(ratio):
    return f"{int(round(100.0 * ratio))}pct"


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def load_first_json_object(path):
    text = Path(path).read_text()
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text.lstrip())
    return obj


def extract_final_if_any(text):
    marker = "<|channel|>final<|message|>"
    if text and marker in text:
        return text.split(marker, 1)[-1].strip()
    return (text or "").strip()


def parse_points(raw):
    raw = extract_final_if_any(str(raw or ""))
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return []
        try:
            obj = json.loads(match.group(0))
        except Exception:
            return []
    points = obj.get("points", obj if isinstance(obj, list) else [])
    if not isinstance(points, list):
        return []
    return [point for point in points if isinstance(point, dict)]


def dump_points(points):
    return json.dumps({"points": points}, ensure_ascii=False)


def tensor_scalar(value, index=0):
    if torch.is_tensor(value):
        value = value.detach().cpu().flatten()
        if value.numel() == 0:
            return 0.0
        return float(value[min(index, value.numel() - 1)].item())
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[min(index, arr.size - 1)])


def get_future_points(batch):
    values = batch.get("future_points")
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, tuple):
        return list(values)
    if isinstance(values, list):
        return values
    return list(values)


def clone_batch_with_future_points(batch, future_points):
    out = dict(batch)
    out["future_points"] = future_points
    return out


class NoisyAnchorContaminator:
    def __init__(
        self,
        level,
        seq_len,
        pred_len,
        data_mean,
        data_std,
        seed=2025,
        noisy_value_mode="tail_gaussian",
        noisy_severity_policy="level_scaled",
        noisy_confidence=0.95,
        noisy_band_std_frac=0.0,
        noisy_std_multiplier=1.0,
        noisy_min_abs_z=1.0,
        noisy_level_std_bonus=1.5,
        noisy_level_tail_bonus=1.0,
        min_std=1e-6,
        time_mode="uniform",
    ):
        self.level = float(level)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.data_mean = float(data_mean)
        self.data_std = max(float(data_std), 1e-6)
        self.seed = int(seed)
        self.noisy_value_mode = noisy_value_mode
        self.noisy_severity_policy = noisy_severity_policy
        self.noisy_confidence = float(noisy_confidence)
        self.noisy_band_std_frac = max(0.0, float(noisy_band_std_frac))
        self.noisy_std_multiplier = max(0.0, float(noisy_std_multiplier))
        self.noisy_min_abs_z = max(0.0, float(noisy_min_abs_z))
        self.noisy_level_std_bonus = max(0.0, float(noisy_level_std_bonus))
        self.noisy_level_tail_bonus = max(0.0, float(noisy_level_tail_bonus))
        self.min_std = max(float(min_std), 1e-12)
        self.time_mode = time_mode
        self.stats = self._fresh_stats()

    def _fresh_stats(self):
        return {
            "level": self.level,
            "original_anchor_count": 0,
            "noisy_anchor_count": 0,
            "windows": 0,
            "windows_with_original_anchor": 0,
            "noisy_abs_z_sum": 0.0,
            "noisy_abs_z_max": 0.0,
            "noisy_effective_std_multiplier_sum": 0.0,
            "noisy_effective_min_abs_z_sum": 0.0,
            "noisy_value_mode": self.noisy_value_mode,
            "noisy_severity_policy": self.noisy_severity_policy,
            "noisy_confidence": self.noisy_confidence,
            "noisy_band_std_frac": self.noisy_band_std_frac,
        }

    def summary_stats(self):
        stats = dict(self.stats)
        n = int(stats.get("noisy_anchor_count") or 0)
        if n > 0:
            stats["mean_abs_z"] = stats["noisy_abs_z_sum"] / n
            stats["max_abs_z"] = stats["noisy_abs_z_max"]
            stats["mean_effective_std_multiplier"] = stats["noisy_effective_std_multiplier_sum"] / n
            stats["mean_effective_min_abs_z"] = stats["noisy_effective_min_abs_z_sum"] / n
        else:
            stats["mean_abs_z"] = None
            stats["max_abs_z"] = None
            stats["mean_effective_std_multiplier"] = None
            stats["mean_effective_min_abs_z"] = None
        for key in (
            "noisy_abs_z_sum",
            "noisy_abs_z_max",
            "noisy_effective_std_multiplier_sum",
            "noisy_effective_min_abs_z_sum",
        ):
            stats.pop(key, None)
        return stats

    def _rng(self, batch_no, sample_no, original_count):
        payload = (
            self.seed
            + int(round(self.level * 10_000)) * 1_000_003
            + int(batch_no) * 10_007
            + int(sample_no) * 97
            + int(original_count) * 13
        )
        return np.random.default_rng(payload % (2**32))

    def _history_stats_original_scale(self, batch, sample_no):
        observed = batch["observed_data"]
        hist = observed[sample_no, : self.seq_len]
        hist_np = hist.detach().cpu().numpy() if torch.is_tensor(hist) else np.asarray(hist)
        hist_np = np.asarray(hist_np, dtype=np.float64).reshape(self.seq_len, -1)[:, 0]
        hist_orig = hist_np * self.data_std + self.data_mean
        mean = float(np.mean(hist_orig))
        std = float(np.std(hist_orig, ddof=0))
        if not math.isfinite(std) or std < self.min_std:
            std = self.min_std
        return mean, std

    def _sample_time(self, rng, points):
        if self.time_mode == "existing" and points:
            t_values = []
            for point in points:
                try:
                    t = int(point.get("t", 1))
                except (TypeError, ValueError):
                    continue
                if 1 <= t <= self.pred_len:
                    t_values.append(t)
            if t_values:
                return int(rng.choice(t_values))
        return int(rng.integers(1, self.pred_len + 1))

    def _effective_std_multiplier(self):
        if self.noisy_severity_policy == "level_scaled":
            return self.noisy_std_multiplier * (1.0 + self.level * self.noisy_level_std_bonus)
        return self.noisy_std_multiplier

    def _effective_min_abs_z(self):
        if self.noisy_severity_policy == "level_scaled":
            return self.noisy_min_abs_z * (1.0 + self.level * self.noisy_level_tail_bonus)
        return self.noisy_min_abs_z

    def _sample_noisy_center(self, rng, hist_mean, hist_std):
        eff_std = self._effective_std_multiplier()
        eff_min_z = self._effective_min_abs_z()

        if self.noisy_value_mode == "history_gaussian":
            z = float(rng.normal())
            abs_z = abs(z)
            signed_z = z
        elif self.noisy_value_mode == "tail_gaussian":
            sign = -1.0 if rng.random() < 0.5 else 1.0
            abs_z = max(abs(float(rng.normal())), eff_min_z)
            signed_z = sign * abs_z
        elif self.noisy_value_mode == "shifted_tail_gaussian":
            sign = -1.0 if rng.random() < 0.5 else 1.0
            abs_z = eff_min_z + abs(float(rng.normal()))
            signed_z = sign * abs_z
        else:
            raise ValueError(f"Unknown noisy_value_mode={self.noisy_value_mode!r}")

        center = float(hist_mean + signed_z * hist_std * eff_std)
        return center, abs_z, eff_std, eff_min_z

    def _make_noisy_point(self, rng, points, hist_mean, hist_std):
        center, abs_z, eff_std, eff_min_z = self._sample_noisy_center(rng, hist_mean, hist_std)
        half_width = self.noisy_band_std_frac * hist_std
        self.stats["noisy_abs_z_sum"] += abs_z
        self.stats["noisy_abs_z_max"] = max(self.stats["noisy_abs_z_max"], abs_z)
        self.stats["noisy_effective_std_multiplier_sum"] += eff_std
        self.stats["noisy_effective_min_abs_z_sum"] += eff_min_z
        return {
            "t": self._sample_time(rng, points),
            "f": 0,
            "type": "NOISY_GAUSSIAN",
            "v_lo": center - half_width,
            "v_hi": center + half_width,
            "confidence": self.noisy_confidence,
        }

    def contaminate(self, batch, batch_no):
        future_points = get_future_points(batch)
        if not future_points:
            return batch

        out_future = []
        for sample_no, raw in enumerate(future_points):
            points = parse_points(raw)
            original_count = len(points)
            self.stats["windows"] += 1
            self.stats["original_anchor_count"] += original_count
            if original_count > 0:
                self.stats["windows_with_original_anchor"] += 1

            add_count = int(math.ceil(self.level * original_count))
            if add_count <= 0:
                out_future.append(dump_points(points))
                continue

            hist_mean, hist_std = self._history_stats_original_scale(batch, sample_no)
            rng = self._rng(batch_no, sample_no, original_count)
            noisy = [
                self._make_noisy_point(rng, points, hist_mean, hist_std)
                for _ in range(add_count)
            ]
            self.stats["noisy_anchor_count"] += add_count
            out_future.append(dump_points(points + noisy))

        return clone_batch_with_future_points(batch, out_future)


class ContaminatedLoader:
    def __init__(self, loader, contaminator):
        self.loader = loader
        self.contaminator = contaminator
        self.dataset = getattr(loader, "dataset", None)

    def __iter__(self):
        self.contaminator.stats = self.contaminator._fresh_stats()
        for batch_no, batch in enumerate(self.loader, start=1):
            yield self.contaminator.contaminate(batch, batch_no)

    def __len__(self):
        return len(self.loader)


def build_args_from_summary(summary_args):
    from exe_forecasting import build_parser, normalize_text_perturb_args

    parser = build_parser()
    args = parser.parse_args([])
    for key, value in (summary_args or {}).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.enable_wandb = False
    args.anchor_sweep = False
    args.modelfolder = ""
    normalize_text_perturb_args(args)
    if args.text_len == 0:
        args.text_len = args.seq_len
    return args


def prepare_config(args, checkpoint_dir):
    config_path = Path(checkpoint_dir) / "config_results.json"
    if config_path.is_file():
        config = load_first_json_object(config_path)
    else:
        with Path("config", args.config).open() as f:
            config = yaml.safe_load(f)

    
    timestep_dim_dict = {"d": 3, "w": 2, "m": 1}
    context_dim_dict = {"bert": 768}
    if args.embed == "timeF":
        if config["model"]["timestep_branch"] or config["model"]["timestep_emb_cat"]:
            config["model"]["timestep_dim"] = timestep_dim_dict[args.freq]
        else:
            config["model"]["timestep_dim"] = 0
    else:
        config["model"]["timestep_dim"] = 4

    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["lookback_len"] = args.seq_len
    config["model"]["pred_len"] = args.pred_len
    config["model"]["domain"] = args.data_path.split("/")[0]
    config["model"]["text_len"] = args.text_len
    config["model"]["save_attn"] = args.save_attn
    config["model"]["save_token"] = args.save_token
    config["diffusion"]["dropout"] = args.dropout
    config["diffusion"]["attn_drop"] = args.attn_drop
    config["diffusion"]["time_weight"] = args.time_weight
    config["seed"] = args.seed
    config["model"]["with_intrinsic"] = args.with_intrinsic
    config["model"]["with_future_hint"] = args.with_future_hint
    config["model"]["mixer_side_channels"] = args.mixer_side_channels
    config["model"]["text_backend"] = args.text_backend
    config["model"]["text_perturbation"] = {
        "ratio": args.text_perturb_ratio,
        "targets": args.text_perturb_targets,
        "seed": args.text_perturb_seed,
    }
    config["model"]["with_texts"] = bool(
        config["model"].get("with_texts", False)
        and (args.with_intrinsic or args.with_future_hint)
    )
    config["model"]["context_dim"] = (
        context_dim_dict[config["model"]["llm"]] if config["model"]["with_texts"] else 0
    )
    args.batch_size = int(config["train"]["batch_size"])
    return config


def find_anchor_row(summary, name):
    for row in summary.get("anchor_results") or []:
        if row.get("anchor_name") == name or row.get("name") == name:
            return row
    return None


def choose_anchor_config(summary, metric, mode):
    from exe_forecasting import build_anchor_grid

    grid = build_anchor_grid()
    by_name = {item["name"]: item for item in grid}
    if mode.startswith("name:"):
        name = mode.split(":", 1)[1]
        if name not in by_name:
            raise ValueError(f"anchor config {name!r} is not in build_anchor_grid()")
        return by_name[name]

    anchor_summary = summary.get("anchor_summary") or {}
    row = None
    if mode == "best_blended":
        row = anchor_summary.get("anchor_test_best_blended")
    elif mode == "best_overall":
        row = anchor_summary.get("anchor_test_best")
    elif mode == "best_from_results":
        candidates = [
            r for r in summary.get("anchor_results") or []
            if (r.get("anchor_name") or r.get("name")) != "00_baseline_no_anchor"
            and r.get(metric) is not None
        ]
        row = min(candidates, key=lambda item: item[metric]) if candidates else None
    else:
        raise ValueError(f"unknown --anchor_config_mode {mode}")

    name = (row or {}).get("anchor_name") or (row or {}).get("name")
    if name and name in by_name and name != "00_baseline_no_anchor":
        return by_name[name]

    candidates = [item for item in grid if item["name"] != "00_baseline_no_anchor"]
    return candidates[-1]


def metric_improvement(baseline, value):
    baseline = _as_float(baseline)
    value = _as_float(value)
    if baseline is None or value is None or abs(baseline) <= 1e-12:
        return None
    return 100.0 * (baseline - value) / abs(baseline)


def metric_degradation(baseline, value):
    baseline = _as_float(baseline)
    value = _as_float(value)
    if baseline is None or value is None or abs(baseline) <= 1e-12:
        return None
    return 100.0 * (value - baseline) / abs(baseline)


def fmt_float(value, digits=6):
    value = _as_float(value)
    return f"{value:.{digits}f}" if value is not None else "NA"


def fmt_pct(value):
    value = _as_float(value)
    return f"{value:.2f}%" if value is not None else "NA"


def add_comparisons(rows, metric):
    by_horizon = {}
    for row in rows:
        key = (row["dataset"], row["seq_len"], row["pred_len"])
        by_horizon.setdefault(key, []).append(row)

    for group_rows in by_horizon.values():
        no_anchor = next((r for r in group_rows if r["condition"] == "no_anchor_baseline"), None)
        clean = next((r for r in group_rows if r["condition"] == "clean_anchor_0pct"), None)
        for row in group_rows:
            for base_name, base in (("no_anchor", no_anchor), ("clean_anchor", clean)):
                base_value = base.get(metric) if base else None
                value = row.get(metric)
                row[f"baseline_{base_name}_{metric}"] = base_value
                row[f"delta_{metric}_vs_{base_name}"] = (
                    value - base_value
                    if _as_float(value) is not None and _as_float(base_value) is not None
                    else None
                )
                row[f"gain_pct_{metric}_vs_{base_name}"] = metric_improvement(base_value, value)
                row[f"degradation_pct_{metric}_vs_{base_name}"] = metric_degradation(base_value, value)
                row[f"beats_{base_name}_{metric}"] = (
                    value < base_value
                    if _as_float(value) is not None and _as_float(base_value) is not None
                    else None
                )
    return rows


def build_markdown_report(rows, metric):
    rows = add_comparisons([dict(row) for row in rows], metric)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["seq_len"], row["pred_len"]), []).append(row)

    noisy_condition = lambda row: str(row.get("condition", "")).startswith("noisy_anchor_")
    lines = [
        "# Robustness to Anchor Contamination",
        "",
        f"Rank metric: `{metric}` (lower is better).",
        "",
        "Noise protocol: noisy anchors are sampled from each window history distribution in original scale. "
        "Default mode uses tail Gaussian samples with point-like high-confidence bands; under `level_scaled`, "
        "tail severity increases with the contamination ratio.",
        "",
    ]
    summary_rows = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0][2]):
        dataset, seq_len, pred_len = key
        candidates = [r for r in group_rows if r["condition"] != "no_anchor_baseline"]
        noisy_rows = [r for r in group_rows if noisy_condition(r)]
        best = min(candidates, key=lambda r: r[metric]) if candidates else None
        best_noisy = min(noisy_rows, key=lambda r: r[metric]) if noisy_rows else None
        max_noisy = max(noisy_rows, key=lambda r: r["noise_ratio"]) if noisy_rows else None
        no_anchor = next((r for r in group_rows if r["condition"] == "no_anchor_baseline"), None)
        clean = next((r for r in group_rows if r["condition"] == "clean_anchor_0pct"), None)
        summary_rows.append(
            {
                "H": pred_len,
                "best": best,
                "best_noisy": best_noisy,
                "max_noisy": max_noisy,
                "no_anchor": no_anchor,
                "clean": clean,
            }
        )
        lines.append(f"## {dataset} L{seq_len} H{pred_len}")
        if best_noisy is not None:
            lines.append(
                f"Best noisy: `{best_noisy['condition']}` "
                f"{metric}={fmt_float(best_noisy[metric])}, "
                f"gain vs no-anchor={fmt_pct(best_noisy.get(f'gain_pct_{metric}_vs_no_anchor'))}, "
                f"degradation vs clean={fmt_pct(best_noisy.get(f'degradation_pct_{metric}_vs_clean_anchor'))}."
            )
        if max_noisy is not None:
            lines.append(
                f"Highest contamination: `{max_noisy['condition']}` "
                f"{metric}={fmt_float(max_noisy[metric])}, "
                f"degradation vs clean={fmt_pct(max_noisy.get(f'degradation_pct_{metric}_vs_clean_anchor'))}, "
                f"degradation vs no-anchor={fmt_pct(max_noisy.get(f'degradation_pct_{metric}_vs_no_anchor'))}."
            )
        lines.append("")
        headers = [
            "condition",
            "noise",
            "anchor",
            metric,
            "gain vs no-anchor",
            "degradation vs clean",
            "mean |z|",
            "eff std x",
            "orig anchors",
            "noisy anchors",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        order = {"no_anchor_baseline": -2, "clean_anchor_0pct": -1}
        for row in sorted(group_rows, key=lambda r: (order.get(r["condition"], r["noise_ratio"]), r["condition"])):
            value = row.get(metric)
            gain_no = row.get(f"gain_pct_{metric}_vs_no_anchor")
            deg_clean = row.get(f"degradation_pct_{metric}_vs_clean_anchor")
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["condition"],
                        f"{row['noise_ratio_pct']}%",
                        row["anchor_name"],
                        f"{value:.6f}" if value is not None else "NA",
                        fmt_pct(gain_no),
                        fmt_pct(deg_clean),
                        fmt_float(row.get("mean_abs_z"), 2),
                        fmt_float(row.get("mean_effective_std_multiplier"), 2),
                        str(row.get("original_anchor_count", "")),
                        str(row.get("noisy_anchor_count", "")),
                    ]
                )
                + " |"
            )
        lines.append("")

    best_values = [item["best"][metric] for item in summary_rows if item["best"] is not None]
    best_noisy_values = [item["best_noisy"][metric] for item in summary_rows if item["best_noisy"] is not None]
    max_noisy_values = [item["max_noisy"][metric] for item in summary_rows if item["max_noisy"] is not None]
    no_anchor_values = [item["no_anchor"][metric] for item in summary_rows if item["no_anchor"] is not None]
    clean_values = [item["clean"][metric] for item in summary_rows if item["clean"] is not None]
    if best_values:
        mean_best = float(np.mean(best_values))
        lines.append("## Mean Across Horizons")
        lines.append("")
        lines.append(f"Mean best {metric}: {mean_best:.6f}")
        if no_anchor_values:
            mean_no_anchor = float(np.mean(no_anchor_values))
            gain = metric_improvement(mean_no_anchor, mean_best)
            lines.append(f"Mean no-anchor baseline {metric}: {mean_no_anchor:.6f}")
            lines.append(f"Mean gain vs no-anchor: {gain:.2f}%")
        if clean_values:
            mean_clean = float(np.mean(clean_values))
            gain = metric_improvement(mean_clean, mean_best)
            lines.append(f"Mean clean-anchor {metric}: {mean_clean:.6f}")
            lines.append(f"Mean gain vs clean-anchor: {gain:.2f}%")
        if best_noisy_values:
            mean_best_noisy = float(np.mean(best_noisy_values))
            lines.append(f"Mean best-noisy {metric}: {mean_best_noisy:.6f}")
            if no_anchor_values:
                lines.append(f"Mean best-noisy gain vs no-anchor: {metric_improvement(float(np.mean(no_anchor_values)), mean_best_noisy):.2f}%")
            if clean_values:
                lines.append(f"Mean best-noisy degradation vs clean: {metric_degradation(float(np.mean(clean_values)), mean_best_noisy):.2f}%")
        if max_noisy_values:
            mean_max_noisy = float(np.mean(max_noisy_values))
            lines.append(f"Mean highest-contamination {metric}: {mean_max_noisy:.6f}")
            if no_anchor_values:
                lines.append(f"Mean highest-contamination degradation vs no-anchor: {metric_degradation(float(np.mean(no_anchor_values)), mean_max_noisy):.2f}%")
            if clean_values:
                lines.append(f"Mean highest-contamination degradation vs clean: {metric_degradation(float(np.mean(clean_values)), mean_max_noisy):.2f}%")
        if clean_values and max_noisy_values:
            hurt_count = sum(
                1
                for item in summary_rows
                if item["clean"] is not None
                and item["max_noisy"] is not None
                and item["max_noisy"][metric] > item["clean"][metric]
            )
            lines.append(
                f"Highest contamination worse than clean anchor on {hurt_count}/{len(max_noisy_values)} horizons."
            )
        lines.append("")
    return "\n".join(lines)


def write_csv(rows, path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_condition(
    model,
    loader,
    args,
    scaler,
    mean_scaler,
    folder,
    anchor_cfg,
    condition,
    noise_ratio,
    contaminator=None,
):
    from utils.utils import evaluate

    eval_loader = ContaminatedLoader(loader, contaminator) if contaminator is not None else loader
    cfg = {key: value for key, value in anchor_cfg.items() if key != "name"}
    print(
        f"[NOISY_ANCHOR] H={args.pred_len} condition={condition} "
        f"anchor={anchor_cfg['name']} noise={noise_ratio:.2f}"
    )
    metrics = evaluate(
        model,
        eval_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        guide_w=0.8,
        foldername=str(folder) + "/",
        window_lens=[args.seq_len, args.pred_len],
        save_attn=args.save_attn,
        save_token=args.save_token,
        evaluating=False,
        return_dict=True,
        reload=False,
        epoch=0,
        isPlot=False,
        eval_name=f"NoisyAnchor-{condition}",
        **cfg,
    )
    stats = contaminator.summary_stats() if contaminator is not None else {}
    print(
        f"[NOISY_ANCHOR] done H={args.pred_len} {condition} | "
        + " | ".join(f"{key}={metrics[key]:.6f}" for key in ("MSE", "NMSE", "NMAE", "CRPS"))
    )
    return {
        "condition": condition,
        "noise_ratio": float(noise_ratio),
        "noise_ratio_pct": int(round(100.0 * noise_ratio)),
        "anchor_name": anchor_cfg["name"],
        **stats,
        **metrics,
    }


def load_checkpoint_for_inference(model, checkpoint_path, map_location):
    state = torch.load(checkpoint_path, map_location=map_location)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_non_text = [key for key in missing if not key.startswith("text_encoder.")]
    if missing_non_text or unexpected:
        print(
            f"[NOISY_ANCHOR][Checkpoint] WARNING loading {checkpoint_path}: "
            f"missing_non_text={missing_non_text}, unexpected={unexpected}"
        )
    else:
        print(f"[NOISY_ANCHOR][Checkpoint] Loaded {checkpoint_path}")
    return state


def build_summary_metric_row(summary, anchor_name, condition, noise_ratio, metric_source):
    row = find_anchor_row(summary, anchor_name)
    if row is None:
        return None
    return {
        "condition": condition,
        "noise_ratio": float(noise_ratio),
        "noise_ratio_pct": int(round(100.0 * noise_ratio)),
        "anchor_name": anchor_name,
        "metric_source": metric_source,
        **{key: row.get(key) for key in METRICS if key in row},
    }


def run_checkpoint(checkpoint_dir, cli_args, out_root):
    from dataset_forecasting import get_dataloader
    from exe_forecasting import set_seed, validate_dataset_paths
    from main_model import CSDI_Forecasting

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    summary = load_json(checkpoint_dir / "run_summary.json")
    args = build_args_from_summary(summary.get("args") or {})
    if cli_args.device is not None:
        args.device = cli_args.device
    elif str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print(f"[NOISY_ANCHOR] CUDA not available; falling back to CPU for {checkpoint_dir.name}")
        args.device = "cpu"
    if cli_args.nsample is not None:
        args.nsample = cli_args.nsample
    if cli_args.eval_batch_size is not None:
        args.eval_batch_size = cli_args.eval_batch_size
    if cli_args.num_workers is not None:
        args.num_workers = cli_args.num_workers
    validate_dataset_paths(args)
    set_seed(args.seed)
    config = prepare_config(args, checkpoint_dir)

    metric = cli_args.metric or summary.get("selection", {}).get("metric") or "MSE"
    anchor_cfg = choose_anchor_config(summary, metric, cli_args.anchor_config_mode)
    no_anchor_cfg = {
        "name": "00_baseline_no_anchor",
        "edit_guidance_method": "blended",
        "guidance_scale": 0.0,
        "time_radius": 0,
        "energy_softmin_tau": 0.0,
        "num_iter": 0,
        "langevin_step_size": 0.0,
        "langevin_alpha": 0.0,
        "langevin_noise_scale": 0.0,
        "blended_amplitude_modulation": False,
        "blended_amplitude_strength": 0.15,
    }

    run_name = f"{checkpoint_dir.name}_noisy_anchor"
    folder = out_root / run_name
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "config_results.json").open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        datatype=args.datatype,
        device=args.device,
        batch_size=config["train"]["batch_size"],
        args=args,
    )
    del train_loader, valid_loader

    target_dim = 370 if args.datatype == "electricity" else 1
    model = CSDI_Forecasting(config, args.device, target_dim, window_lens=[args.seq_len, args.pred_len]).to(args.device)
    load_checkpoint_for_inference(model, checkpoint_dir / "model.pth", map_location=args.device)
    model.target_dim = target_dim
    model.eval()

    rows = []
    base_meta = {
        "checkpoint_dir": str(checkpoint_dir),
        "dataset": args.data_path.split("/")[0],
        "data_path": args.data_path,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "nsample": args.nsample,
        "seed": args.seed,
        "device": args.device,
        "selected_anchor_name": anchor_cfg["name"],
        "rank_metric": metric,
    }

    if cli_args.recompute_clean:
        no_anchor_row = evaluate_condition(
            model, test_loader, args, scaler, mean_scaler, folder,
            no_anchor_cfg, "no_anchor_baseline", 0.0,
        )
        clean_row = evaluate_condition(
            model, test_loader, args, scaler, mean_scaler, folder,
            anchor_cfg, "clean_anchor_0pct", 0.0,
        )
    else:
        no_anchor_row = build_summary_metric_row(
            summary, "00_baseline_no_anchor", "no_anchor_baseline", 0.0, "run_summary"
        )
        clean_row = build_summary_metric_row(
            summary, anchor_cfg["name"], "clean_anchor_0pct", 0.0, "run_summary"
        )
        if no_anchor_row is None:
            no_anchor_row = evaluate_condition(
                model, test_loader, args, scaler, mean_scaler, folder,
                no_anchor_cfg, "no_anchor_baseline", 0.0,
            )
        if clean_row is None:
            clean_row = evaluate_condition(
                model, test_loader, args, scaler, mean_scaler, folder,
                anchor_cfg, "clean_anchor_0pct", 0.0,
            )

    rows.extend([{**base_meta, **no_anchor_row}, {**base_meta, **clean_row}])

    data_mean = tensor_scalar(mean_scaler)
    data_std = tensor_scalar(scaler)
    for level in cli_args.levels:
        contaminator = NoisyAnchorContaminator(
            level=level,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            data_mean=data_mean,
            data_std=data_std,
            seed=cli_args.noisy_seed if cli_args.noisy_seed is not None else args.seed,
            noisy_value_mode=cli_args.noisy_value_mode,
            noisy_severity_policy=cli_args.noisy_severity_policy,
            noisy_confidence=cli_args.noisy_confidence,
            noisy_band_std_frac=cli_args.noisy_band_std_frac,
            noisy_std_multiplier=cli_args.noisy_std_multiplier,
            noisy_min_abs_z=cli_args.noisy_min_abs_z,
            noisy_level_std_bonus=cli_args.noisy_level_std_bonus,
            noisy_level_tail_bonus=cli_args.noisy_level_tail_bonus,
            time_mode=cli_args.noisy_time_mode,
        )
        row = evaluate_condition(
            model,
            test_loader,
            args,
            scaler,
            mean_scaler,
            folder,
            anchor_cfg,
            f"noisy_anchor_{pct_label(level)}",
            level,
            contaminator=contaminator,
        )
        rows.append({**base_meta, **row})

    add_comparisons(rows, metric)
    with (folder / "noisy_anchor_ablation_summary.json").open("w") as f:
        json.dump(
            {
                "checkpoint_dir": str(checkpoint_dir),
                "args": vars(args),
                "script_args": vars(cli_args),
                "selected_anchor": anchor_cfg,
                "rows": rows,
            },
            f,
            indent=2,
            default=_json_default,
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run noisy-anchor contamination ablation from trained checkpoints.")
    parser.add_argument(
        "--checkpoint_dirs",
        nargs="+",
        default=DEFAULT_CHECKPOINT_DIRS,
        help="Checkpoint folders containing model.pth and run_summary.json.",
    )
    parser.add_argument("--levels", nargs="+", default=["20", "30", "50", "100"], help="Noisy anchor ratios, e.g. 20 30 50 100 or 0.2 0.3.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults to outputs/noisy_anchor_ablation_TIMESTAMP.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--nsample", type=int, default=None, help="Override generated samples per evaluation.")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--metric", default="MSE", choices=METRICS, help="Ranking/report metric.")
    parser.add_argument(
        "--anchor_config_mode",
        default="best_blended",
        choices=["best_blended", "best_overall", "best_from_results"],
        help="Which existing clean anchor guidance config to contaminate.",
    )
    parser.add_argument("--recompute_clean", action="store_true", help="Recompute no-anchor and clean-anchor rows instead of reusing run_summary metrics.")
    parser.add_argument("--noisy_seed", type=int, default=2025, help="Seed for Gaussian noisy anchors.")
    parser.add_argument(
        "--noisy_value_mode",
        choices=["history_gaussian", "tail_gaussian", "shifted_tail_gaussian"],
        default="tail_gaussian",
        help=(
            "history_gaussian is plain N(history_mean, history_var). "
            "tail_gaussian forces samples into distribution tails. "
            "shifted_tail_gaussian is stronger."
        ),
    )
    parser.add_argument(
        "--noisy_severity_policy",
        choices=["fixed", "level_scaled"],
        default="level_scaled",
        help="level_scaled increases tail severity as contamination ratio grows.",
    )
    parser.add_argument("--noisy_confidence", type=float, default=0.95)
    parser.add_argument("--noisy_band_std_frac", type=float, default=0.0, help="Half-width of noisy band as a fraction of history std. Use 0 for point anchors.")
    parser.add_argument("--noisy_std_multiplier", type=float, default=1.5, help="Base multiplier applied to history std for noisy values.")
    parser.add_argument("--noisy_min_abs_z", type=float, default=1.5, help="Minimum absolute Gaussian z-score for tail modes.")
    parser.add_argument("--noisy_level_std_bonus", type=float, default=1.5, help="Extra std multiplier at high contamination when policy=level_scaled.")
    parser.add_argument("--noisy_level_tail_bonus", type=float, default=1.0, help="Extra min tail z-score at high contamination when policy=level_scaled.")
    parser.add_argument("--noisy_time_mode", choices=["uniform", "existing"], default="uniform", help="Sample noisy t uniformly or from existing anchor times.")
    args = parser.parse_args()
    args.levels = parse_levels(args.levels)
    if not args.checkpoint_dirs:
        parser.error("--checkpoint_dirs requires at least one trained checkpoint directory")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir or f"outputs/noisy_anchor_ablation_{timestamp}").expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for checkpoint_dir in args.checkpoint_dirs:
        rows = run_checkpoint(checkpoint_dir, args, out_root)
        all_rows.extend(rows)

    add_comparisons(all_rows, args.metric)
    csv_path = out_root / "noisy_anchor_ablation_results.csv"
    json_path = out_root / "noisy_anchor_ablation_results.json"
    md_path = out_root / "noisy_anchor_ablation_report.md"
    write_csv(all_rows, csv_path)
    json_path.write_text(json.dumps(all_rows, indent=2, default=_json_default) + "\n")
    md_path.write_text(build_markdown_report(all_rows, args.metric) + "\n")

    print(f"[NOISY_ANCHOR] wrote {csv_path}")
    print(f"[NOISY_ANCHOR] wrote {json_path}")
    print(f"[NOISY_ANCHOR] wrote {md_path}")
    print(build_markdown_report(all_rows, args.metric))


if __name__ == "__main__":
    main()
