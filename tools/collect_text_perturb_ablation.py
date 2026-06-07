#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


METRIC_KEYS = ("CRPS", "NMSE", "NMAE", "CRPS_sum", "MSE", "MAE")


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_ratio(value):
    ratio = as_float(value) or 0.0
    if ratio > 1.0 and ratio <= 100.0:
        ratio /= 100.0
    return ratio


def load_summary(path):
    with path.open() as f:
        return json.load(f)


def get_metrics(summary):
    selection = summary.get("selection") or {}
    selected = selection.get("selected_metrics") or {}
    if selected:
        return selected, selection.get("source", "selection")
    train_summary = summary.get("train_summary") or {}
    final_test = train_summary.get("final_test_metrics") or {}
    return final_test, "train_summary/final_test_metrics"


def row_from_summary(path, summary):
    args = summary.get("args") or {}
    metrics, metric_source = get_metrics(summary)
    data_path = args.get("data_path", "")
    dataset = data_path.split("/")[0] if data_path else None
    ratio = normalize_ratio(args.get("text_perturb_ratio", 0.0))
    row = {
        "dataset": dataset,
        "data_path": data_path,
        "seq_len": args.get("seq_len"),
        "pred_len": args.get("pred_len"),
        "ratio": ratio,
        "ratio_pct": int(round(ratio * 100.0)),
        "targets": args.get("text_perturb_targets", ""),
        "seed": args.get("text_perturb_seed", args.get("seed")),
        "exp_name": summary.get("exp_name"),
        "folder": summary.get("folder"),
        "summary_path": str(path),
        "metric_source": metric_source,
    }
    for key in METRIC_KEYS:
        row[key] = as_float(metrics.get(key))
    return row


def sort_key(row):
    return (
        str(row.get("dataset") or ""),
        int(row.get("seq_len") or 0),
        int(row.get("pred_len") or 0),
        float(row.get("ratio") or 0.0),
        str(row.get("exp_name") or ""),
    )


def add_clean_comparisons(rows, metric):
    clean_by_key = {}
    for row in rows:
        if normalize_ratio(row.get("ratio")) != 0.0:
            continue
        key = (row.get("dataset"), row.get("seq_len"), row.get("pred_len"), row.get("targets"))
        clean_by_key[key] = row

    for row in rows:
        key = (row.get("dataset"), row.get("seq_len"), row.get("pred_len"), row.get("targets"))
        clean = clean_by_key.get(key)
        clean_value = clean.get(metric) if clean else None
        value = row.get(metric)
        row[f"clean_{metric}"] = clean_value
        row[f"delta_{metric}_vs_clean"] = None
        row[f"degradation_pct_{metric}_vs_clean"] = None
        if clean_value is None or value is None:
            continue
        row[f"delta_{metric}_vs_clean"] = value - clean_value
        if abs(clean_value) > 1e-12:
            row[f"degradation_pct_{metric}_vs_clean"] = 100.0 * (value - clean_value) / abs(clean_value)
    return rows


def write_csv(rows, out_path, metric):
    columns = [
        "dataset",
        "data_path",
        "seq_len",
        "pred_len",
        "ratio_pct",
        "ratio",
        "targets",
        "seed",
        "exp_name",
        "metric_source",
        *METRIC_KEYS,
        f"clean_{metric}",
        f"delta_{metric}_vs_clean",
        f"degradation_pct_{metric}_vs_clean",
        "folder",
        "summary_path",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows, metric, limit=None):
    shown = rows[:limit] if limit else rows
    headers = ["dataset", "L", "H", "noise", metric, "delta", "deg%"]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in shown:
        value = row.get(metric)
        delta = row.get(f"delta_{metric}_vs_clean")
        deg = row.get(f"degradation_pct_{metric}_vs_clean")
        value_text = f"{value:.6f}" if value is not None else "NA"
        print(
            f"{row.get('dataset')} | {row.get('seq_len')} | {row.get('pred_len')} | "
            f"{row.get('ratio_pct')}% | {value_text}",
            end="",
        )
        print(
            " | "
            + (f"{delta:.6f}" if delta is not None else "NA")
            + " | "
            + (f"{deg:.2f}" if deg is not None else "NA")
        )


def main():
    parser = argparse.ArgumentParser(description="Collect source-text perturbation ablation summaries.")
    parser.add_argument("--root", required=True, help="Directory containing run_summary.json files.")
    parser.add_argument("--out_csv", default=None, help="Output CSV path. Defaults to ROOT/text_perturb_ablation_summary.csv.")
    parser.add_argument("--metric", default="CRPS", choices=METRIC_KEYS)
    parser.add_argument("--limit", type=int, default=None, help="Limit printed rows.")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    paths = sorted(root.rglob("run_summary.json"))
    if not paths:
        raise SystemExit(f"No run_summary.json files found under {root}")

    rows = []
    for path in paths:
        try:
            summary = load_summary(path)
            rows.append(row_from_summary(path, summary))
        except Exception as exc:
            print(f"[WARN] skipped {path}: {exc}")
    rows.sort(key=sort_key)
    rows = add_clean_comparisons(rows, args.metric)

    out_csv = Path(args.out_csv).expanduser() if args.out_csv else root / "text_perturb_ablation_summary.csv"
    write_csv(rows, out_csv, args.metric)
    print(f"[TEXT_ABLATION] wrote {out_csv} rows={len(rows)}")
    print_table(rows, args.metric, args.limit)


if __name__ == "__main__":
    main()
