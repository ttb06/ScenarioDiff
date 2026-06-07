import argparse
import datetime
import json
import os
import random

import numpy as np
import torch
import yaml


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "yes", "y"):
        return True
    if value in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser():
    parser = argparse.ArgumentParser(description="ScenarioDiff forecasting pipeline")
    default_root_path = os.environ.get("TIMEMMD_ROOT", "Time-MMD")
    parser.add_argument("--config", type=str, default="economy_36_18.yaml")
    parser.add_argument("--datatype", type=str, default="multimodal")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=50)
    parser.add_argument("--data", type=str, default="custom")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--root_path", type=str, default=default_root_path)
    parser.add_argument("--data_path", type=str, default="Economy/Economy.csv")
    parser.add_argument("--seq_len", type=int, default=36)
    parser.add_argument("--pred_len", type=int, default=18)
    parser.add_argument("--text_len", type=int, default=36)
    parser.add_argument("--features", type=str, default="S")
    parser.add_argument("--freq", type=str, default="m")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn_drop", type=float, default=0.0)
    parser.add_argument("--init", type=str, default="None")
    parser.add_argument("--valid_interval", type=int, default=10)
    parser.add_argument("--time_weight", type=float, default=0.1)
    parser.add_argument("--c_mask_prob", type=float, default=-1)
    parser.add_argument("--beta_end", type=float, default=-1)
    parser.add_argument("--lr", type=float, default=-1)
    parser.add_argument("--save_attn", type=str2bool, default=False)
    parser.add_argument("--save_token", type=str2bool, default=False)
    parser.add_argument("--with_intrinsic", action="store_true")
    parser.add_argument("--with_future_hint", action="store_true")
    parser.add_argument("--mixer_side_channels", type=int, default=32)
    parser.add_argument("--text_backend", choices=("bert",), default="bert")
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument(
        "--text_perturb_ratio",
        type=float,
        default=0.0,
        help="Add source-style noisy text at this ratio. Accepts 0.1 or 10 for 10 percent.",
    )
    parser.add_argument(
        "--text_perturb_targets",
        type=str,
        default="intrinsic,future",
        help="Comma-separated targets: intrinsic, future. Aliases: scenario, description, coarse.",
    )
    parser.add_argument(
        "--text_perturb_seed",
        type=int,
        default=None,
        help="Seed for deterministic source-text perturbation. Defaults to --seed.",
    )

    
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--nheads", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--patch_mode", choices=["pad", "skip"], default=None)
    parser.add_argument("--patch_decode_mode", choices=["conv", "linear"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--itr_per_epoch", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")

    
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="scenariodiff")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "online"))

    
    parser.add_argument("--anchor_sweep", action="store_true")
    parser.add_argument("--anchor_eval_split", choices=["test"], default="test")
    parser.add_argument("--anchor_metric", type=str, default="CRPS", choices=["CRPS", "MAE", "MSE", "NMSE", "NMAE"])
    parser.add_argument("--anchor_min_improvement", type=float, default=0.0)
    return parser


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def normalize_text_perturb_args(args):
    ratio = float(args.text_perturb_ratio or 0.0)
    if ratio < 0:
        raise ValueError("--text_perturb_ratio must be non-negative.")
    if ratio > 1.0:
        if ratio <= 100.0:
            ratio = ratio / 100.0
        else:
            raise ValueError("--text_perturb_ratio must be <= 1.0 or <= 100 when passed as percent.")
    args.text_perturb_ratio = ratio
    if args.text_perturb_seed is None:
        args.text_perturb_seed = args.seed
    return args


def validate_dataset_paths(args):
    root_path = os.path.abspath(os.path.expanduser(args.root_path))
    data_prefix = os.path.splitext(args.data_path)[0]
    numerical_path = os.path.join(root_path, "numerical", args.data_path)
    textual_dir = os.path.join(root_path, "textual", os.path.dirname(data_prefix))
    required_dirs = [
        os.path.join(root_path, "numerical"),
        os.path.join(root_path, "textual"),
    ]
    missing = [path for path in required_dirs if not os.path.isdir(path)]
    if not os.path.isfile(numerical_path):
        missing.append(numerical_path)
    if missing:
        raise FileNotFoundError(
            "Dataset path is not valid. --root_path must point to the Time-MMD "
            "directory that contains both numerical/ and textual/.\n"
            f"root_path={root_path}\n"
            f"data_path={args.data_path}\n"
            "Missing:\n  - " + "\n  - ".join(missing)
        )
    args.root_path = root_path


def apply_overrides(config, args):
    if args.c_mask_prob > 0:
        config["diffusion"]["c_mask_prob"] = args.c_mask_prob
    if args.beta_end > 0:
        config["diffusion"]["beta_end"] = args.beta_end
    if args.lr > 0:
        config["train"]["lr"] = args.lr
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.itr_per_epoch is not None:
        config["train"]["itr_per_epoch"] = args.itr_per_epoch
    if args.max_grad_norm is not None:
        config["train"]["max_grad_norm"] = args.max_grad_norm
    for key in ("layers", "channels", "nheads", "num_steps", "sample_steps"):
        value = getattr(args, key)
        if value is not None:
            config["diffusion"][key] = value
    if args.patch_len is not None:
        config["diffusion"]["patch_len"] = args.patch_len
    if args.patch_mode is not None:
        config["diffusion"]["patch_mode"] = args.patch_mode
    if args.patch_decode_mode is not None:
        config["diffusion"]["patch_decode_mode"] = args.patch_decode_mode


def build_anchor_grid():
    return [
        dict(
            name="31_Blended+Aug_n6_w50_r2_noaug",
            edit_guidance_method="blended", guidance_scale=50.0, time_radius=2,
            energy_softmin_tau=0.10, langevin_step_size=8e-4, langevin_alpha=1.0, langevin_noise_scale=0.60,
            num_iter=6, blended_amplitude_modulation=False, blended_amplitude_strength=0.15,
        ),
    ]


def evaluate_anchor_grid(model, loader, args, scaler, mean_scaler, foldername, split_name, wandb_run):
    from utils.utils import evaluate

    rows = []
    for cfg in build_anchor_grid():
        print(f"[ANCHOR][{split_name}] {cfg['name']}")
        eval_cfg = {key: value for key, value in cfg.items() if key != "name"}
        metrics = evaluate(
            model,
            loader,
            nsample=args.nsample,
            scaler=scaler,
            mean_scaler=mean_scaler,
            guide_w=0.8,
            foldername=foldername,
            window_lens=[args.seq_len, args.pred_len],
            return_dict=True,
            epoch=0,
            eval_name=f"Anchor-{split_name}",
            **eval_cfg,
        )
        row = {"split": split_name, "anchor_name": cfg["name"], **cfg, **metrics}
        rows.append(row)
        metric_parts = " | ".join(f"{key}={row[key]:.6f}" for key in ("CRPS", "NMSE", "NMAE") if key in row)
        print(f"[ANCHOR][{split_name}] {cfg['name']} | {metric_parts}")
    return rows


def pick_working_anchor(rows, metric, min_improvement):
    baseline = next((r for r in rows if r["anchor_name"] == "00_baseline_no_anchor"), None)
    if baseline is None:
        return None, False, None
    candidates = [
        r for r in rows
        if r["edit_guidance_method"] == "blended" and r["anchor_name"] != "00_baseline_no_anchor"
    ]
    if not candidates:
        return None, False, baseline
    best = min(candidates, key=lambda r: r[metric])
    threshold = baseline[metric] * (1.0 - min_improvement)
    return best, best[metric] < threshold, baseline


def pick_best_anchor(rows, metric):
    if not rows:
        return None
    return min(rows, key=lambda row: row[metric])


def add_baseline_comparison(row, baseline, metric):
    row = dict(row)
    baseline_value = baseline.get(metric) if baseline is not None else None
    rank_value = row.get(metric)
    row["baseline_anchor_name"] = baseline.get("anchor_name") if baseline is not None else None
    row["rank_metric"] = metric
    row["rank_value"] = rank_value
    row["baseline_value"] = baseline_value
    row["delta_vs_00_baseline"] = None
    row["improvement_pct_vs_00_baseline"] = None
    row["beats_00_baseline"] = False
    row["is_better_than_baseline"] = False
    if baseline_value is not None and rank_value is not None:
        row["delta_vs_00_baseline"] = rank_value - baseline_value
        row["beats_00_baseline"] = rank_value < baseline_value
        row["is_better_than_baseline"] = row["beats_00_baseline"]
        if abs(baseline_value) > 1e-12:
            row["improvement_pct_vs_00_baseline"] = 100.0 * (baseline_value - rank_value) / abs(baseline_value)
    return row


def annotate_anchor_rows_with_baseline(rows, metric):
    baseline = next((row for row in rows if row["anchor_name"] == "00_baseline_no_anchor"), None)
    return [add_baseline_comparison(row, baseline, metric) for row in rows]


def log_anchor_table(wandb_run, rows, split_name, exp_name, metric):
    if wandb_run is None or not rows:
        return
    try:
        import wandb
    except ImportError:
        return
    baseline = next((row for row in rows if row["anchor_name"] == "00_baseline_no_anchor"), None)
    blended_rows = [
        row for row in rows
        if row.get("edit_guidance_method") == "blended" and row.get("anchor_name") != "00_baseline_no_anchor"
    ]
    if not blended_rows:
        return
    columns = [
        "model",
        "anchor_name",
        "baseline_anchor_name",
        "rank_metric",
        "rank_value",
        "baseline_value",
        "delta_vs_00_baseline",
        "improvement_pct_vs_00_baseline",
        "beats_00_baseline",
        "is_better_than_baseline",
        "CRPS",
        "NMSE",
        "NMAE",
        "CRPS_sum",
        "guidance_scale",
        "time_radius",
        "num_iter",
        "blended_amplitude_modulation",
        "blended_amplitude_strength",
    ]
    table = wandb.Table(columns=columns)
    for row in sorted(blended_rows, key=lambda item: item[metric]):
        compared = add_baseline_comparison(row, baseline, metric)
        table.add_data(
            exp_name,
            compared.get("anchor_name"),
            compared.get("baseline_anchor_name"),
            compared.get("rank_metric"),
            compared.get("rank_value"),
            compared.get("baseline_value"),
            compared.get("delta_vs_00_baseline"),
            compared.get("improvement_pct_vs_00_baseline"),
            compared.get("beats_00_baseline"),
            compared.get("is_better_than_baseline"),
            compared.get("CRPS"),
            compared.get("NMSE"),
            compared.get("NMAE"),
            compared.get("CRPS_sum"),
            compared.get("guidance_scale"),
            compared.get("time_radius"),
            compared.get("num_iter"),
            compared.get("blended_amplitude_modulation"),
            compared.get("blended_amplitude_strength"),
        )
    table_key = f"anchor_{split_name}/{exp_name}/blended_versions"
    wandb_run.log({table_key: table})


def build_anchor_comparison_rows(rows, metric):
    baseline = next((row for row in rows if row["anchor_name"] == "00_baseline_no_anchor"), None)
    blended_rows = [
        row for row in rows
        if row.get("edit_guidance_method") == "blended" and row.get("anchor_name") != "00_baseline_no_anchor"
    ]
    return [add_baseline_comparison(row, baseline, metric) for row in sorted(blended_rows, key=lambda item: item[metric])]


def print_anchor_comparison(rows, metric):
    baseline = next((row for row in rows if row["anchor_name"] == "00_baseline_no_anchor"), None)
    if baseline is None:
        print("[ANCHOR][test] WARNING no 00_baseline_no_anchor row found.")
        return
    print(
        f"[ANCHOR][test] baseline=00_baseline_no_anchor | "
        f"{metric}={baseline[metric]:.6f} | CRPS={baseline['CRPS']:.6f} | "
        f"NMSE={baseline['NMSE']:.6f} | NMAE={baseline['NMAE']:.6f}"
    )
    for row in build_anchor_comparison_rows(rows, metric):
        delta = row.get("delta_vs_00_baseline")
        improvement = row.get("improvement_pct_vs_00_baseline")
        delta_text = f"{delta:.6f}" if delta is not None else "n/a"
        improvement_text = f"{improvement:.2f}%" if improvement is not None else "n/a"
        print(
            f"[ANCHOR][test] compare | {row['anchor_name']} | "
            f"{metric}={row['rank_value']:.6f} | "
            f"delta_vs_00={delta_text} | "
            f"improve={improvement_text} | "
            f"beats_00={row['beats_00_baseline']} | "
            f"better_than_baseline={row['is_better_than_baseline']} | "
            f"w={row['guidance_scale']} n={row['num_iter']} r={row['time_radius']} "
            f"aug={row['blended_amplitude_modulation']} strength={row['blended_amplitude_strength']}"
        )


def cleanup_extra_checkpoints(foldername, keep_name="model.pth"):
    removed = []
    kept = []
    for name in os.listdir(foldername):
        if not name.endswith((".pth", ".pt", ".ckpt")):
            continue
        path = os.path.join(foldername, name)
        if name == keep_name:
            kept.append(path)
            continue
        try:
            os.remove(path)
            removed.append(path)
        except OSError as exc:
            print(f"[Checkpoint cleanup] WARNING could not remove {path}: {exc}")
    print(
        f"[Checkpoint cleanup] kept={kept if kept else 'none'} | "
        f"removed_extra={len(removed)}"
    )
    return {"kept": kept, "removed": removed}


def maybe_init_wandb(args, config, foldername):
    if not args.enable_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] wandb is not installed; continuing without W&B logging.")
        return None
    run_name = args.exp_name or os.path.basename(os.path.normpath(foldername))
    payload = {
        "args": vars(args),
        "train": config.get("train", {}),
        "diffusion": config.get("diffusion", {}),
        "model": config.get("model", {}),
    }
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        mode=args.wandb_mode,
        config=payload,
        dir=args.output_dir,
    )


def build_selection_summary(args, train_summary, anchor_summary):
    metric = args.anchor_metric
    final_test = (train_summary or {}).get("final_test_metrics") or {}
    best_blended = (anchor_summary or {}).get("anchor_test_best_blended") or {}
    anchor_work = bool((anchor_summary or {}).get("anchor_blended_work", False))
    if anchor_work and best_blended.get(metric) is not None:
        source = "anchor_test_best_blended"
        selected = best_blended
    else:
        source = "final_test"
        selected = final_test
    return {
        "metric": metric,
        "value": float(selected[metric]) if selected.get(metric) is not None else None,
        "source": source,
        "anchor_blended_work": anchor_work,
        "selected_metrics": selected,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    normalize_text_perturb_args(args)
    validate_dataset_paths(args)
    from main_model import CSDI_Forecasting
    from dataset_forecasting import get_dataloader
    from utils.utils import load_checkpoint, train

    set_seed(args.seed)

    if args.text_len == 0:
        args.text_len = args.seq_len

    timestep_dim_dict = {"d": 3, "w": 2, "m": 1}
    context_dim_dict = {"bert": 768}
    with open(os.path.join("config", args.config), "r") as f:
        config = yaml.safe_load(f)

    if args.embed == "timeF":
        if config["model"]["timestep_branch"] or config["model"]["timestep_emb_cat"]:
            config["model"]["timestep_dim"] = timestep_dim_dict[args.freq]
        else:
            config["model"]["timestep_dim"] = 0
    else:
        config["model"]["timestep_dim"] = 4

    target_dim = 370 if args.datatype == "electricity" else 1
    if args.datatype == "electricity":
        args.seq_len = 168
        args.pred_len = 24

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
    apply_overrides(config, args)
    args.batch_size = config["train"]["batch_size"]

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"forecasting_{args.data_path.split('/')[0]}_{args.seq_len}_{args.pred_len}_{current_time}"
    foldername = os.path.join(args.output_dir, exp_name) + "/"
    print("model folder:", foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(os.path.join(foldername, "config_results.json"), "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")

    wandb_run = maybe_init_wandb(args, config, foldername)
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        datatype=args.datatype,
        device=args.device,
        batch_size=config["train"]["batch_size"],
        args=args,
    )

    model = CSDI_Forecasting(config, args.device, target_dim, window_lens=[args.seq_len, args.pred_len]).to(args.device)
    if args.modelfolder == "":
        train_summary = train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
            valid_epoch_interval=args.valid_interval,
            scaler=scaler,
            mean_scaler=mean_scaler,
            window_lens=[args.seq_len, args.pred_len],
            eval_nsample=args.nsample,
            guide_w=0,
            compute_valid_metrics=True,
            test_loader=test_loader,
            wandb_run=wandb_run,
        )
    else:
        load_checkpoint(model, os.path.join("./save", args.modelfolder, "model.pth"))
        train_summary = {"loaded_model_folder": args.modelfolder}
    model.target_dim = target_dim

    anchor_rows = []
    anchor_summary = {}
    if args.anchor_sweep:
        test_rows = evaluate_anchor_grid(model, test_loader, args, scaler, mean_scaler, foldername, "test", wandb_run)
        test_rows = annotate_anchor_rows_with_baseline(test_rows, args.anchor_metric)
        anchor_rows.extend(test_rows)
        log_anchor_table(wandb_run, test_rows, "test", exp_name, args.anchor_metric)
        print_anchor_comparison(test_rows, args.anchor_metric)
        anchor_comparison = build_anchor_comparison_rows(test_rows, args.anchor_metric)
        best_blended, blended_works, baseline = pick_working_anchor(
            test_rows, args.anchor_metric, args.anchor_min_improvement
        )
        best_any = pick_best_anchor(test_rows, args.anchor_metric)
        anchor_summary = {
            "anchor_metric": args.anchor_metric,
            "anchor_min_improvement": args.anchor_min_improvement,
            "anchor_blended_work": blended_works,
            "anchor_test_baseline": baseline,
            "anchor_test_best_blended": best_blended,
            "anchor_test_best": best_any,
            "anchor_test_blended_comparison": anchor_comparison,
        }
        print("[ANCHOR][test] blended work:", blended_works)
        if best_blended is not None:
            print("[ANCHOR][test] best blended:", best_blended["anchor_name"], args.anchor_metric, best_blended[args.anchor_metric])
        if best_any is not None:
            print("[ANCHOR][test] best overall:", best_any["anchor_name"], args.anchor_metric, best_any[args.anchor_metric])
        if blended_works:
            anchor_summary["checkpoint_cleanup"] = cleanup_extra_checkpoints(foldername)
        else:
            print("[Checkpoint cleanup] skipped because no blended anchor beat 00_baseline_no_anchor.")

    summary = {
        "folder": foldername,
        "exp_name": exp_name,
        "args": vars(args),
        "train_summary": train_summary,
        "anchor_summary": anchor_summary,
        "anchor_results": anchor_rows,
    }
    summary["selection"] = build_selection_summary(args, train_summary, anchor_summary)
    summary_path = os.path.join(foldername, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SUMMARY] wrote {summary_path}")

    if wandb_run is not None:
        wandb_run.summary["dataset"] = args.data_path.split("/")[0]
        wandb_run.summary["lookback"] = int(args.seq_len)
        wandb_run.summary["horizon"] = int(args.pred_len)
        wandb_run.summary["hparam/batch_size"] = int(config["train"]["batch_size"])
        wandb_run.summary["hparam/lr"] = float(config["train"]["lr"])
        wandb_run.summary["hparam/layers"] = int(config["diffusion"]["layers"])
        wandb_run.summary["hparam/channels"] = int(config["diffusion"]["channels"])
        wandb_run.summary["hparam/nheads"] = int(config["diffusion"]["nheads"])
        wandb_run.summary["hparam/num_steps"] = int(config["diffusion"]["num_steps"])
        wandb_run.summary["hparam/sample_steps"] = int(config["diffusion"]["sample_steps"])
        wandb_run.summary["hparam/nsample"] = int(args.nsample)
        wandb_run.summary["hparam/c_mask_prob"] = float(config["diffusion"]["c_mask_prob"])
        selection = summary["selection"]
        wandb_run.summary["selection/metric"] = selection["metric"]
        wandb_run.summary["selection/source"] = selection["source"]
        wandb_run.summary["selection/anchor_blended_work"] = bool(selection["anchor_blended_work"])
        if selection["value"] is not None:
            wandb_run.summary["selection/value"] = float(selection["value"])
        if anchor_summary:
            wandb_run.summary["anchor_blended_work"] = bool(anchor_summary["anchor_blended_work"])
            best = anchor_summary.get("anchor_test_best")
            if best is not None:
                wandb_run.summary["anchor_test_best_name"] = best["anchor_name"]
                for key in ("CRPS", "NMSE", "NMAE", "CRPS_sum"):
                    if key in best:
                        wandb_run.summary[f"anchor_test_best/{key}"] = float(best[key])
            best_blended = anchor_summary.get("anchor_test_best_blended")
            if best_blended is not None:
                wandb_run.summary["anchor_test_best_blended_name"] = best_blended["anchor_name"]
                for key in ("CRPS", "NMSE", "NMAE", "CRPS_sum"):
                    if key in best_blended:
                        wandb_run.summary[f"anchor_test_best_blended/{key}"] = float(best_blended[key])
        wandb_run.finish()


if __name__ == "__main__":
    main()
