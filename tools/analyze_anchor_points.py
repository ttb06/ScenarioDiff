#!/usr/bin/env python3
"""Internal helper."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _as_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _as_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_numeric(root_path, data_path, target):
    csv_path = Path(root_path).expanduser().resolve() / "numerical" / data_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cannot find numerical CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(axis="index", how="any", subset=[target]).copy()
    for col in ("date", "start_date", "end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    if "date" in df.columns:
        df = df.sort_values("date", ascending=True).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    num_train = int(len(df) * 0.7)
    train_values = df[[target]].iloc[:num_train].to_numpy(dtype=np.float64)
    mean = float(train_values.mean(axis=0)[0])
    std = float(train_values.std(axis=0, ddof=0)[0])
    if std <= 0:
        raise ValueError(f"Non-positive train std in {csv_path}: {std}")
    return df, mean, std, str(csv_path), num_train


def load_anchor_jsonl(path):
    records = []
    with Path(path).expanduser().open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] skip invalid JSON line {line_no}: {exc}")
                continue
            if isinstance(obj, dict):
                obj["_line_no"] = line_no
                records.append(obj)
    return records


def normalize_points(raw_points):
    points = raw_points if isinstance(raw_points, list) else []
    out = []
    for p in points:
        if not isinstance(p, dict):
            continue
        t = _as_int(p.get("t"))
        f = _as_int(p.get("f", 0), 0)
        v_lo = _as_float(p.get("v_lo", p.get("value_lo", p.get("lo"))))
        v_hi = _as_float(p.get("v_hi", p.get("value_hi", p.get("hi"))))
        if t is None or v_lo is None or v_hi is None:
            continue
        if v_lo > v_hi:
            v_lo, v_hi = v_hi, v_lo
        out.append({
            "t": t,
            "f": 0 if f is None else f,
            "v_lo": v_lo,
            "v_hi": v_hi,
            "confidence": _as_float(p.get("confidence"), 1.0),
            "type": p.get("type", ""),
        })
    return out


def build_date_index(df):
    if "end_date" not in df.columns:
        raise ValueError("Numerical CSV must contain end_date for anchor mapping.")
    return {pd.Timestamp(row.end_date): int(idx) for idx, row in df[["end_date"]].iterrows()}


def _band_distance(value, lo, hi):
    if lo <= value <= hi:
        return 0.0
    return min(abs(value - lo), abs(value - hi))


def point_rows(records, df, date_to_idx, target, std, time_radius=0, include_history_radius=False):
    rows = []
    skipped = defaultdict(int)
    for rec in records:
        end_date_raw = rec.get("end_date")
        if end_date_raw is None:
            skipped["missing_end_date"] += 1
            continue
        end_date = pd.Timestamp(end_date_raw)
        hist_idx = date_to_idx.get(end_date)
        if hist_idx is None:
            skipped["end_date_not_in_numeric"] += 1
            continue

        anchor_points = normalize_points(rec.get("points", []))
        if not anchor_points:
            skipped["empty_points"] += 1
            continue

        last_value = float(df.iloc[hist_idx][target])
        for point_ord, p in enumerate(anchor_points):
            if p["t"] < 1:
                skipped["non_future_t"] += 1
                continue
            gt_idx = hist_idx + p["t"]
            if gt_idx >= len(df):
                skipped["t_out_of_range"] += 1
                continue

            gt_value = float(df.iloc[gt_idx][target])
            mid = 0.5 * (p["v_lo"] + p["v_hi"])
            width = p["v_hi"] - p["v_lo"]
            inside = p["v_lo"] <= gt_value <= p["v_hi"]
            dist_to_band = _band_distance(gt_value, p["v_lo"], p["v_hi"])
            radius_start = hist_idx + p["t"] - time_radius
            if not include_history_radius:
                radius_start = max(radius_start, hist_idx + 1)
            else:
                radius_start = max(radius_start, 0)
            radius_end = min(hist_idx + p["t"] + time_radius, len(df) - 1)
            radius_candidates = []
            for cand_idx in range(radius_start, radius_end + 1):
                cand_value = float(df.iloc[cand_idx][target])
                cand_dist = _band_distance(cand_value, p["v_lo"], p["v_hi"])
                radius_candidates.append((cand_dist, cand_idx, cand_value))
            if radius_candidates:
                radius_dist, radius_best_idx, radius_best_value = min(radius_candidates, key=lambda item: item[0])
            else:
                radius_dist, radius_best_idx, radius_best_value = dist_to_band, gt_idx, gt_value
            anchor_err = mid - gt_value
            baseline_err = last_value - gt_value

            rows.append({
                "line_no": rec.get("_line_no"),
                "window_end_date": str(end_date.date()),
                "gt_end_date": str(pd.Timestamp(df.iloc[gt_idx]["end_date"]).date()),
                "t": p["t"],
                "point_ord": point_ord,
                "confidence": p["confidence"],
                "type": p["type"],
                "last_value_baseline": last_value,
                "v_lo": p["v_lo"],
                "v_hi": p["v_hi"],
                "v_mid": mid,
                "band_width": width,
                "gt_value": gt_value,
                "inside_band": inside,
                "dist_to_band": dist_to_band,
                "band_abs_error": dist_to_band,
                "band_sq_error": dist_to_band ** 2,
                "band_nabs_error": dist_to_band / std,
                "band_nsq_error": (dist_to_band / std) ** 2,
                "time_radius": time_radius,
                "radius_include_history": include_history_radius,
                "radius_best_gt_end_date": str(pd.Timestamp(df.iloc[radius_best_idx]["end_date"]).date()),
                "radius_best_gt_value": radius_best_value,
                "radius_hit_band": radius_dist == 0.0,
                "radius_dist_to_band": radius_dist,
                "radius_band_abs_error": radius_dist,
                "radius_band_sq_error": radius_dist ** 2,
                "radius_band_nabs_error": radius_dist / std,
                "radius_band_nsq_error": (radius_dist / std) ** 2,
                "anchor_abs_error": abs(anchor_err),
                "anchor_sq_error": anchor_err ** 2,
                "anchor_nabs_error": abs(anchor_err) / std,
                "anchor_nsq_error": (anchor_err / std) ** 2,
                "baseline_abs_error": abs(baseline_err),
                "baseline_sq_error": baseline_err ** 2,
                "baseline_nabs_error": abs(baseline_err) / std,
                "baseline_nsq_error": (baseline_err / std) ** 2,
            })
    return rows, skipped


def weighted_mean(values, weights=None):
    values = [v for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return float("nan")
    if weights is None:
        return float(np.mean(values))
    pairs = [
        (float(v), max(float(w), 0.0))
        for v, w in zip(values, weights)
        if v is not None and w is not None and math.isfinite(float(v)) and math.isfinite(float(w))
    ]
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return float(np.mean([v for v, _ in pairs])) if pairs else float("nan")
    return float(sum(v * w for v, w in pairs) / total_w)


def summarize(rows, records, skipped, mean, std, scale_csv, train_rows):
    conf = [row["confidence"] for row in rows]
    summary = {
        "anchor_records": len(records),
        "records_with_points": sum(1 for rec in records if normalize_points(rec.get("points", []))),
        "valid_anchor_points": len(rows),
        "scale_source_csv": scale_csv,
        "scale_train_rows": train_rows,
        "pipeline_train_mean": mean,
        "pipeline_train_std": std,
        "skipped": dict(skipped),
    }
    if not rows:
        return summary

    summary.update({
        "coverage_records_pct": 100.0 * summary["records_with_points"] / max(1, len(records)),
        "band_hit_rate_pct": 100.0 * weighted_mean([float(row["inside_band"]) for row in rows]),
        "avg_band_width": weighted_mean([row["band_width"] for row in rows]),
        "avg_band_width_scaled": weighted_mean([row["band_width"] / std for row in rows]),
        "avg_dist_to_band": weighted_mean([row["dist_to_band"] for row in rows]),
        "avg_dist_to_band_scaled": weighted_mean([row["dist_to_band"] / std for row in rows]),
        "band_MAE": weighted_mean([row["band_abs_error"] for row in rows]),
        "band_MSE": weighted_mean([row["band_sq_error"] for row in rows]),
        "band_NMAE": weighted_mean([row["band_nabs_error"] for row in rows]),
        "band_NMSE": weighted_mean([row["band_nsq_error"] for row in rows]),
        "time_radius": rows[0].get("time_radius", 0),
        "radius_include_history": rows[0].get("radius_include_history", False),
        "radius_band_hit_rate_pct": 100.0 * weighted_mean([float(row["radius_hit_band"]) for row in rows]),
        "radius_avg_dist_to_band": weighted_mean([row["radius_dist_to_band"] for row in rows]),
        "radius_avg_dist_to_band_scaled": weighted_mean([row["radius_dist_to_band"] / std for row in rows]),
        "radius_band_MAE": weighted_mean([row["radius_band_abs_error"] for row in rows]),
        "radius_band_MSE": weighted_mean([row["radius_band_sq_error"] for row in rows]),
        "radius_band_NMAE": weighted_mean([row["radius_band_nabs_error"] for row in rows]),
        "radius_band_NMSE": weighted_mean([row["radius_band_nsq_error"] for row in rows]),
        "anchor_MAE": weighted_mean([row["anchor_abs_error"] for row in rows]),
        "anchor_MSE": weighted_mean([row["anchor_sq_error"] for row in rows]),
        "anchor_NMAE": weighted_mean([row["anchor_nabs_error"] for row in rows]),
        "anchor_NMSE": weighted_mean([row["anchor_nsq_error"] for row in rows]),
        "baseline_last_MAE": weighted_mean([row["baseline_abs_error"] for row in rows]),
        "baseline_last_MSE": weighted_mean([row["baseline_sq_error"] for row in rows]),
        "baseline_last_NMAE": weighted_mean([row["baseline_nabs_error"] for row in rows]),
        "baseline_last_NMSE": weighted_mean([row["baseline_nsq_error"] for row in rows]),
        "anchor_conf_weighted_NMAE": weighted_mean([row["anchor_nabs_error"] for row in rows], conf),
        "anchor_conf_weighted_NMSE": weighted_mean([row["anchor_nsq_error"] for row in rows], conf),
    })
    for metric in ("NMAE", "NMSE", "MAE", "MSE"):
        anchor = summary[f"anchor_{metric}"]
        band = summary[f"band_{metric}"]
        base = summary[f"baseline_last_{metric}"]
        summary[f"delta_{metric}_vs_last_baseline"] = anchor - base
        summary[f"improvement_pct_{metric}_vs_last_baseline"] = (
            100.0 * (base - anchor) / abs(base) if abs(base) > 1e-12 else float("nan")
        )
        summary[f"band_delta_{metric}_vs_last_baseline"] = band - base
        summary[f"band_improvement_pct_{metric}_vs_last_baseline"] = (
            100.0 * (base - band) / abs(base) if abs(base) > 1e-12 else float("nan")
        )
        radius_band = summary[f"radius_band_{metric}"]
        summary[f"radius_band_delta_{metric}_vs_last_baseline"] = radius_band - base
        summary[f"radius_band_improvement_pct_{metric}_vs_last_baseline"] = (
            100.0 * (base - radius_band) / abs(base) if abs(base) > 1e-12 else float("nan")
        )
    return summary


def summarize_by_t(rows, std):
    by_t = defaultdict(list)
    for row in rows:
        by_t[row["t"]].append(row)
    out = []
    for t in sorted(by_t):
        part = by_t[t]
        out.append({
            "t": t,
            "n": len(part),
            "band_hit_rate_pct": 100.0 * weighted_mean([float(row["inside_band"]) for row in part]),
            "band_NMAE": weighted_mean([row["band_nabs_error"] for row in part]),
            "band_NMSE": weighted_mean([row["band_nsq_error"] for row in part]),
            "radius_band_hit_rate_pct": 100.0 * weighted_mean([float(row["radius_hit_band"]) for row in part]),
            "radius_band_NMAE": weighted_mean([row["radius_band_nabs_error"] for row in part]),
            "radius_band_NMSE": weighted_mean([row["radius_band_nsq_error"] for row in part]),
            "anchor_NMAE": weighted_mean([row["anchor_nabs_error"] for row in part]),
            "anchor_NMSE": weighted_mean([row["anchor_nsq_error"] for row in part]),
            "baseline_last_NMAE": weighted_mean([row["baseline_nabs_error"] for row in part]),
            "baseline_last_NMSE": weighted_mean([row["baseline_nsq_error"] for row in part]),
            "avg_band_width": weighted_mean([row["band_width"] for row in part]),
            "avg_band_width_scaled": weighted_mean([row["band_width"] / std for row in part]),
        })
    return out


def maybe_add_model_baseline(summary, run_summary_path):
    if not run_summary_path:
        return
    run_summary = json.loads(Path(run_summary_path).expanduser().read_text())
    anchor_rows = run_summary.get("anchor_results") or []
    baseline = next((row for row in anchor_rows if row.get("anchor_name") == "00_baseline_no_anchor"), None)
    if baseline is None:
        baseline = ((run_summary.get("train_summary") or {}).get("final_test_metrics") or None)
    if not baseline:
        return
    summary["model_baseline_note"] = (
        "Model baseline metrics are full-horizon metrics from run_summary.json; "
        "anchor metrics here are computed only on valid anchor points."
    )
    for metric in ("NMAE", "NMSE", "MAE", "MSE"):
        value = _as_float(baseline.get(metric))
        if value is None:
            continue
        summary[f"model_baseline_{metric}"] = value
        anchor = _as_float(summary.get(f"anchor_{metric}"))
        if anchor is not None:
            summary[f"delta_{metric}_vs_model_baseline"] = anchor - value
            summary[f"improvement_pct_{metric}_vs_model_baseline"] = (
                100.0 * (value - anchor) / abs(value) if abs(value) > 1e-12 else float("nan")
            )
        band = _as_float(summary.get(f"band_{metric}"))
        if band is not None:
            summary[f"band_delta_{metric}_vs_model_baseline"] = band - value
            summary[f"band_improvement_pct_{metric}_vs_model_baseline"] = (
                100.0 * (value - band) / abs(value) if abs(value) > 1e-12 else float("nan")
            )
        radius_band = _as_float(summary.get(f"radius_band_{metric}"))
        if radius_band is not None:
            summary[f"radius_band_delta_{metric}_vs_model_baseline"] = radius_band - value
            summary[f"radius_band_improvement_pct_{metric}_vs_model_baseline"] = (
                100.0 * (value - radius_band) / abs(value) if abs(value) > 1e-12 else float("nan")
            )


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary):
    print("[ANCHOR_POINTS]")
    print(
        f"records={summary['anchor_records']} | records_with_points={summary['records_with_points']} "
        f"({summary.get('coverage_records_pct', 0.0):.2f}%) | valid_points={summary['valid_anchor_points']}"
    )
    print(
        f"scale mean={summary['pipeline_train_mean']:.6f} | "
        f"std={summary['pipeline_train_std']:.6f} | train_rows={summary['scale_train_rows']}"
    )
    if summary["valid_anchor_points"] == 0:
        print(f"skipped={summary['skipped']}")
        return
    print(
        f"anchor_NMSE={summary['anchor_NMSE']:.6f} | "
        f"anchor_NMAE={summary['anchor_NMAE']:.6f} | "
        f"band_NMSE={summary['band_NMSE']:.6f} | "
        f"band_NMAE={summary['band_NMAE']:.6f} | "
        f"radius={summary['time_radius']} radius_band_NMSE={summary['radius_band_NMSE']:.6f} | "
        f"radius_band_NMAE={summary['radius_band_NMAE']:.6f} | "
        f"band_hit={summary['band_hit_rate_pct']:.2f}% | "
        f"radius_hit={summary['radius_band_hit_rate_pct']:.2f}% | "
        f"avg_width_scaled={summary['avg_band_width_scaled']:.6f}"
    )
    print(
        f"last_baseline_NMSE={summary['baseline_last_NMSE']:.6f} | "
        f"last_baseline_NMAE={summary['baseline_last_NMAE']:.6f} | "
        f"radius_band_improve_NMSE={summary['radius_band_improvement_pct_NMSE_vs_last_baseline']:.2f}% | "
        f"radius_band_improve_NMAE={summary['radius_band_improvement_pct_NMAE_vs_last_baseline']:.2f}%"
    )
    if "model_baseline_NMSE" in summary:
        print(
            f"model_baseline_NMSE={summary['model_baseline_NMSE']:.6f} | "
            f"model_baseline_NMAE={summary['model_baseline_NMAE']:.6f} | "
            f"radius_band_vs_model_NMSE={summary['radius_band_improvement_pct_NMSE_vs_model_baseline']:.2f}% | "
            f"radius_band_vs_model_NMAE={summary['radius_band_improvement_pct_NMAE_vs_model_baseline']:.2f}%"
        )
    if summary.get("skipped"):
        print(f"skipped={summary['skipped']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze anchor point JSONL quality directly.")
    parser.add_argument("--root_path", default="Time-MMD")
    parser.add_argument("--data_path", required=True, help="e.g. Economy/Economy.csv")
    parser.add_argument("--anchor_path", required=True, help="*_abnormal_points*.jsonl")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--run_summary", default=None, help="Optional run_summary.json for model no-anchor baseline.")
    parser.add_argument("--time_radius", type=int, default=0, help="Analyze band distance over future t +/- radius.")
    parser.add_argument(
        "--include_history_radius",
        action="store_true",
        help="Allow radius window to include lookback/history points, matching sampling more literally.",
    )
    parser.add_argument("--output_json", default="outputs/anchor_points_analysis.json")
    parser.add_argument("--output_points_csv", default=None)
    parser.add_argument("--output_by_t_csv", default=None)
    args = parser.parse_args()

    df, mean, std, scale_csv, train_rows = load_numeric(args.root_path, args.data_path, args.target)
    records = load_anchor_jsonl(args.anchor_path)
    rows, skipped = point_rows(
        records,
        df,
        build_date_index(df),
        args.target,
        std,
        time_radius=max(0, args.time_radius),
        include_history_radius=args.include_history_radius,
    )
    summary = summarize(rows, records, skipped, mean, std, scale_csv, train_rows)
    summary.update({
        "data_path": args.data_path,
        "anchor_path": str(Path(args.anchor_path).expanduser()),
        "target": args.target,
    })
    maybe_add_model_baseline(summary, args.run_summary)

    write_json(args.output_json, summary)
    if args.output_points_csv:
        write_csv(args.output_points_csv, rows)
    if args.output_by_t_csv:
        write_csv(args.output_by_t_csv, summarize_by_t(rows, std))
    print_summary(summary)
    print(f"[ANCHOR_POINTS] wrote {args.output_json}")


if __name__ == "__main__":
    main()
