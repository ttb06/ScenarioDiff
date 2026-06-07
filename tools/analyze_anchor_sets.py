#!/usr/bin/env python3
"""Internal helper."""

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("NMSE", "NMAE", "MSE", "MAE", "CRPS", "CRPS_sum")


def _as_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _pct_improvement(baseline, value):
    baseline = _as_float(baseline)
    value = _as_float(value)
    if baseline is None or value is None or abs(baseline) <= 1e-12:
        return None
    return 100.0 * (baseline - value) / abs(baseline)


def load_pipeline_scale(root_path, data_path, target="OT"):
    csv_path = Path(root_path).expanduser().resolve() / "numerical" / data_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cannot find numerical CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(axis="index", how="any", subset=[target])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=True).reset_index(drop=True)

    num_train = int(len(df) * 0.7)
    train_values = df[[target]].iloc[:num_train].to_numpy(dtype=np.float64)
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0, ddof=0)
    if np.any(std <= 0):
        raise ValueError(f"Non-positive train std in {csv_path}: {std}")
    return float(mean[0]), float(std[0]), str(csv_path), int(num_train), int(len(df))


def find_baseline(summary):
    anchor_rows = summary.get("anchor_results") or []
    baseline = next(
        (row for row in anchor_rows if row.get("anchor_name") == "00_baseline_no_anchor"),
        None,
    )
    if baseline is not None:
        return baseline, "anchor_results/00_baseline_no_anchor"

    train_summary = summary.get("train_summary") or {}
    final_test = train_summary.get("final_test_metrics")
    if final_test:
        return final_test, "train_summary/final_test_metrics"
    return None, None


def analyze_summary(path):
    path = Path(path)
    summary = json.loads(path.read_text())
    args = summary.get("args") or {}
    root_path = args.get("root_path")
    data_path = args.get("data_path")
    target = args.get("target", "OT")
    if not root_path or not data_path:
        raise ValueError(f"{path} is missing args.root_path or args.data_path")

    mean, std, csv_path, num_train, num_total = load_pipeline_scale(root_path, data_path, target)
    baseline, baseline_source = find_baseline(summary)
    if baseline is None:
        raise ValueError(f"{path} has no anchor baseline or final_test_metrics")

    rows = []
    for rank, row in enumerate(summary.get("anchor_results") or [], start=1):
        out = {
            "summary_path": str(path),
            "exp_name": summary.get("exp_name"),
            "dataset": data_path.split("/")[0],
            "data_path": data_path,
            "seq_len": args.get("seq_len"),
            "pred_len": args.get("pred_len"),
            "target": target,
            "scale_source_csv": csv_path,
            "scale_train_rows": num_train,
            "scale_total_rows": num_total,
            "pipeline_train_mean": mean,
            "pipeline_train_std": std,
            "baseline_source": baseline_source,
            "rank_in_file": rank,
            "anchor_name": row.get("anchor_name") or row.get("name"),
            "edit_guidance_method": row.get("edit_guidance_method"),
            "guidance_scale": row.get("guidance_scale"),
            "num_iter": row.get("num_iter"),
            "time_radius": row.get("time_radius"),
            "blended_amplitude_modulation": row.get("blended_amplitude_modulation"),
            "blended_amplitude_strength": row.get("blended_amplitude_strength"),
        }

        nmse = _as_float(row.get("NMSE"))
        nmae = _as_float(row.get("NMAE"))
        out["MSE_from_pipeline_scale"] = nmse * (std ** 2) if nmse is not None else None
        out["MAE_from_pipeline_scale"] = nmae * std if nmae is not None else None

        for metric in METRICS:
            value = _as_float(row.get(metric))
            base = _as_float(baseline.get(metric))
            out[metric] = value
            out[f"baseline_{metric}"] = base
            out[f"delta_{metric}_vs_baseline"] = (
                value - base if value is not None and base is not None else None
            )
            out[f"improvement_pct_{metric}_vs_baseline"] = _pct_improvement(base, value)
            out[f"beats_baseline_{metric}"] = (
                value < base if value is not None and base is not None else None
            )

        mse = _as_float(row.get("MSE"))
        mae = _as_float(row.get("MAE"))
        out["MSE_scale_abs_error"] = (
            abs(mse - out["MSE_from_pipeline_scale"])
            if mse is not None and out["MSE_from_pipeline_scale"] is not None
            else None
        )
        out["MSE_scale_rel_error"] = (
            out["MSE_scale_abs_error"] / abs(mse)
            if mse is not None and abs(mse) > 1e-12 and out["MSE_scale_abs_error"] is not None
            else None
        )
        out["MAE_scale_abs_error"] = (
            abs(mae - out["MAE_from_pipeline_scale"])
            if mae is not None and out["MAE_from_pipeline_scale"] is not None
            else None
        )
        out["MAE_scale_rel_error"] = (
            out["MAE_scale_abs_error"] / abs(mae)
            if mae is not None and abs(mae) > 1e-12 and out["MAE_scale_abs_error"] is not None
            else None
        )
        rows.append(out)
    return rows


def collect_summary_paths(inputs):
    paths = []
    for item in inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            paths.extend(sorted(p.rglob("run_summary.json")))
        elif any(ch in str(p) for ch in "*?["):
            paths.extend(Path(match) for match in sorted(glob.glob(str(p), recursive=True)))
        else:
            paths.append(p)
    return [p for p in paths if p.name == "run_summary.json" and p.is_file()]


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top(rows, sort_metric, top_k):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["summary_path"], []).append(row)

    for summary_path, run_rows in grouped.items():
        baseline = next(
            (row for row in run_rows if row["anchor_name"] == "00_baseline_no_anchor"),
            run_rows[0],
        )
        candidates = [
            row for row in run_rows
            if row["anchor_name"] != "00_baseline_no_anchor" and row.get(sort_metric) is not None
        ]
        candidates.sort(key=lambda row: row[sort_metric])
        print(f"\n[ANCHOR_ANALYSIS] {summary_path}")
        print(
            "  baseline="
            f"{baseline['anchor_name']} | "
            f"NMSE={baseline.get('NMSE'):.6f} | "
            f"NMAE={baseline.get('NMAE'):.6f} | "
            f"std={baseline.get('pipeline_train_std'):.6f}"
        )
        for row in candidates[:top_k]:
            print(
                f"  {row['anchor_name']} | "
                f"{sort_metric}={row[sort_metric]:.6f} | "
                f"dNMSE={row['delta_NMSE_vs_baseline']:.6f} "
                f"({row['improvement_pct_NMSE_vs_baseline']:.2f}%) | "
                f"dNMAE={row['delta_NMAE_vs_baseline']:.6f} "
                f"({row['improvement_pct_NMAE_vs_baseline']:.2f}%)"
            )


def main():
    parser = argparse.ArgumentParser(description="Analyze anchor set metrics vs original baseline.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="run_summary.json path, glob, or directory containing run_summary.json files.",
    )
    parser.add_argument("--output_csv", default="outputs/anchor_analysis.csv")
    parser.add_argument("--sort_metric", default="NMSE", choices=METRICS)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    summary_paths = collect_summary_paths(args.inputs)
    if not summary_paths:
        raise SystemExit("No run_summary.json files found.")

    rows = []
    for path in summary_paths:
        rows.extend(analyze_summary(path))
    if not rows:
        raise SystemExit("No anchor_results rows found.")

    write_csv(rows, args.output_csv)
    print_top(rows, args.sort_metric, args.top_k)
    print(f"\n[ANCHOR_ANALYSIS] wrote {args.output_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
