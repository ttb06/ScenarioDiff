import numpy as np
import torch
from torch.optim import Adam, AdamW
import os
import json
import random, numpy as np, torch
from torch.nn.parameter import is_lazy


def log_wandb(wandb_run, metrics, prefix=None, step=None):
    if wandb_run is None or metrics is None:
        return
    payload = {}
    for key, value in metrics.items():
        if key in ("MSE", "MAE"):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            metric_key = f"{prefix}/{key}" if prefix else key
            payload[metric_key] = float(value)
    if payload:
        wandb_run.log(payload, step=step)


def compact_state_dict(model):
    """Return trainable/project weights only; frozen HF text encoders dominate checkpoint size."""
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("text_encoder.")
    }


def save_checkpoint(model, output_path):
    torch.save(compact_state_dict(model), output_path)


def load_checkpoint(model, checkpoint_path, map_location=None):
    state = torch.load(checkpoint_path, map_location=map_location)
    model_state = model.state_dict()
    compatible_state = {}
    mismatched = []
    for key, value in state.items():
        target = model_state.get(key)
        if target is not None and not is_lazy(target) and target.shape != value.shape:
            mismatched.append((key, tuple(value.shape), tuple(target.shape)))
        else:
            compatible_state[key] = value
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    missing_non_text = [key for key in missing if not key.startswith("text_encoder.")]
    if mismatched:
        print(
            f"[Checkpoint] WARNING skipped {len(mismatched)} shape-mismatched weights "
            f"from {checkpoint_path}: {mismatched}"
        )
    if missing_non_text or unexpected:
        print(
            f"[Checkpoint] WARNING loading {checkpoint_path}: "
            f"missing_non_text={missing_non_text}, unexpected={unexpected}"
        )
    elif missing:
        print(
            f"[Checkpoint] Loaded compact checkpoint from {checkpoint_path} "
            f"(skipped frozen text_encoder weights)."
        )
    else:
        print(f"[Checkpoint] Loaded checkpoint from {checkpoint_path}.")
    return state


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=10,
    foldername="",
    scaler=1,
    mean_scaler=0,
    window_lens=None,
    eval_nsample=50,
    guide_w=0,
    compute_valid_metrics=True,
    test_loader=None,
    wandb_run=None,
):
    optimizer = Adam(model.parameters(), lr=float(config["lr"]), weight_decay=1e-6)
    max_grad_norm = config.get("max_grad_norm")
    max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
    if max_grad_norm is not None and max_grad_norm <= 0:
        max_grad_norm = None

    output_path = None
    if foldername != "":
        os.makedirs(foldername, exist_ok=True)
        output_path = os.path.join(foldername, "model.pth")

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    
    
    
    best_valid_score = float("inf")
    best_valid_epoch = -1
    best_valid_metrics = None

    
    
    
    
    best_test_score = float("inf")
    best_test_epoch = -1
    best_test_metrics = None

    eval_guide_w = 0.8
    window_lens = window_lens if window_lens is not None else [1, 1]

    def metric_score(metrics):
        
        
        return metrics["NMSE"] + metrics["NMAE"]

    def print_metrics(prefix, metrics):
        print(
            f"{prefix} "
            f"NMSE: {metrics['NMSE']:.6f} | "
            f"NMAE: {metrics['NMAE']:.6f} | "
            f"CRPS: {metrics['CRPS']:.6f} | "
            f"CRPS_sum: {metrics['CRPS_sum']:.6f}"
        )

    for epoch_no in range(config["epochs"]):
        avg_loss = 0.0
        batch_no = 0
        loss_batches = 0
        skipped_batches = 0
        model.train()

        for batch_no, train_batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)

            loss = model(train_batch)
            if not torch.isfinite(loss):
                skipped_batches += 1
                print(f"[Train][WARN] non-finite loss at epoch={epoch_no}, batch={batch_no}")
                bad_params = [
                    name for name, p in model.named_parameters()
                    if p is not None and not torch.isfinite(p).all()
                ][:5]
                if bad_params:
                    raise RuntimeError(
                        "Model parameters became non-finite; lower LR. "
                        f"First bad params: {bad_params}"
                    )
                continue

            loss.backward()
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if not torch.isfinite(grad_norm):
                    skipped_batches += 1
                    print(f"[Train][WARN] non-finite grad_norm at epoch={epoch_no}, batch={batch_no}")
                    optimizer.zero_grad(set_to_none=True)
                    continue
            optimizer.step()

            avg_loss += loss.item()
            loss_batches += 1

            if batch_no >= config["itr_per_epoch"]:
                break

        if loss_batches == 0:
            raise RuntimeError(
                f"No finite training batches at epoch={epoch_no}; "
                "check data/model numerics and reduce learning rate."
            )

        lr_scheduler.step()
        avg_epoch_loss = avg_loss / loss_batches
        print(
            f"[Train] epoch={epoch_no} | "
            f"loss={avg_epoch_loss:.6f} | "
            f"lr={lr_scheduler.get_last_lr()[0]:.8g}"
            + (f" | skipped={skipped_batches}" if skipped_batches else "")
        )
        log_wandb(
            wandb_run,
            {"loss": avg_epoch_loss, "lr": lr_scheduler.get_last_lr()[0], "epoch": epoch_no},
            prefix="train",
            step=epoch_no,
        )

        should_eval_epoch = (
            (epoch_no + 1) % valid_epoch_interval == 0
            and epoch_no >= 9
        )

        if not should_eval_epoch:
            continue

        
        
        
        if valid_loader is not None and compute_valid_metrics and epoch_no > 18:
            model.eval()
            print(f"\n[Evaluating][Valid] epoch={epoch_no}, guide_w={eval_guide_w}")

            valid_metrics = evaluate(
                model,
                valid_loader,
                nsample=eval_nsample,
                scaler=scaler,
                mean_scaler=mean_scaler,
                foldername=foldername,
                window_lens=window_lens,
                guide_w=eval_guide_w,
                return_dict=True,
                epoch=epoch_no,
                eval_name="Validation",
            )

            print_metrics("[Valid metrics]", valid_metrics)
            log_wandb(wandb_run, valid_metrics, prefix="valid", step=epoch_no)

            valid_score = metric_score(valid_metrics)

            if valid_score < best_valid_score:
                best_valid_score = valid_score
                best_valid_epoch = epoch_no
                best_valid_metrics = valid_metrics

                if output_path is not None:
                    save_checkpoint(model, output_path)
                    print(
                        f"[Checkpoint] Saved best VALID checkpoint "
                        f"at epoch {epoch_no}, "
                        f"valid_score={valid_score:.6f} -> {output_path}"
                    )

        
        
        
        
        if test_loader is not None and epoch_no > 18:
            model.eval()
            print(f"\n[Evaluating][Test-current] epoch={epoch_no}, guide_w={eval_guide_w}")

            test_metrics = evaluate(
                model,
                test_loader,
                nsample=eval_nsample,
                scaler=scaler,
                mean_scaler=mean_scaler,
                foldername=foldername,
                window_lens=window_lens,
                guide_w=eval_guide_w,
                return_dict=True,
                epoch=epoch_no,
                eval_name="Test-current",
            )

            print_metrics("[Test-current metrics]", test_metrics)
            log_wandb(wandb_run, test_metrics, prefix="test_current", step=epoch_no)

            test_score = metric_score(test_metrics)

            if test_score < best_test_score:
                best_test_score = test_score
                best_test_epoch = epoch_no
                best_test_metrics = test_metrics

                print(
                    f"[Best-test tracker] Updated best TEST score "
                    f"at epoch {epoch_no}, "
                    f"test_score={test_score:.6f}. "
                    f"This is for reporting only, not checkpoint selection."
                )

    
    
    
    print("\n================ Training Summary ================")

    if best_valid_metrics is not None:
        print(f"\n[Best VALID] epoch={best_valid_epoch}, score={best_valid_score:.6f}")
        print_metrics("[Best VALID metrics]", best_valid_metrics)
    else:
        print("\n[Best VALID] No validation evaluation was performed.")

    if best_test_metrics is not None:
        print(
            f"\n[Best TEST seen during training - diagnostic only] "
            f"epoch={best_test_epoch}, score={best_test_score:.6f}"
        )
        print_metrics("[Best TEST metrics]", best_test_metrics)
    else:
        print("\n[Best TEST seen during training] No test evaluation was performed.")

    
    
    
    
    if test_loader is not None:
        if output_path is not None and os.path.isfile(output_path):
            device = next(model.parameters()).device
            load_checkpoint(model, output_path, map_location=device)
            print(f"\n[Final Test] Loaded best VALID checkpoint from {output_path}")
        else:
            print(
                "\n[Final Test] WARNING: best-valid checkpoint not found. "
                "Evaluating current in-memory model instead."
            )

        model.eval()
        final_test_metrics = evaluate(
            model,
            test_loader,
            nsample=eval_nsample,
            scaler=scaler,
            mean_scaler=mean_scaler,
            foldername=foldername,
            window_lens=window_lens,
            guide_w=eval_guide_w,
            return_dict=True,
            epoch=best_valid_epoch if best_valid_epoch >= 0 else config["epochs"] - 1,
            eval_name="Test",
        )

        print("\n[Official Test Result from Best-VALID checkpoint]")
        print_metrics("[Final Test metrics]", final_test_metrics)
        log_wandb(wandb_run, final_test_metrics, prefix="test_final", step=config["epochs"])
    else:
        final_test_metrics = None

    
    
    
    if output_path is not None and best_valid_epoch < 0:
        save_checkpoint(model, output_path)
        print(f"\n[Checkpoint] No valid checkpoint was selected. Saved latest model -> {output_path}")

    summary = {
        "best_valid_epoch": best_valid_epoch,
        "best_valid_score": best_valid_score,
        "best_valid_metrics": best_valid_metrics,
        "best_test_epoch": best_test_epoch,
        "best_test_score": best_test_score,
        "best_test_metrics": best_test_metrics,
        "final_test_metrics": final_test_metrics,
    }
    if wandb_run is not None:
        for key, value in summary.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                wandb_run.summary[key] = float(value)
            elif isinstance(value, dict):
                for metric_key, metric_value in value.items():
                    if isinstance(metric_value, (int, float, np.integer, np.floating)):
                        wandb_run.summary[f"{key}/{metric_key}"] = float(metric_value)
    return summary

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="",
             window_lens=[1, 1], guide_w=0,
             return_dict=False, reload=False, epoch=0,
             eval_name=None,
             edit_guidance_method="langevin", guidance_scale=0.0,
             time_radius=1, energy_softmin_tau=0.05, num_iter=0,
             langevin_step_size=None, langevin_alpha=1.0, langevin_noise_scale=None,
             blended_amplitude_modulation=False, blended_amplitude_strength=0.15):

    if reload:
        ckpt_path = os.path.join(foldername, "model.pth")
        if os.path.isfile(ckpt_path):
            load_checkpoint(model, ckpt_path, map_location="cpu")
        else:
            print(f"[evaluate] WARNING: checkpoint not found at {ckpt_path}, using in-memory weights.")

    # Save random state so evaluation doesn't affect training reproducibility
    was_training = model.training
    py_state    = random.getstate()
    np_state    = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    param_grad_states = None
    if edit_guidance_method == "blended" and guidance_scale > 0 and num_iter > 0:
        param_grad_states = [
            (param, param.requires_grad)
            for param in model.parameters()
            if not isinstance(param, torch.nn.parameter.UninitializedParameter)
        ]
        for param, _ in param_grad_states:
            param.requires_grad_(False)

    try:
        with torch.no_grad():
            model.eval()
            mse_total = 0.0
            mae_total = 0.0
            nmse_total = 0.0
            nmae_total = 0.0
            evalpoints_total = 0.0
            CRPS = float("nan")
            CRPS_sum = float("nan")

            all_target = []
            all_observed_point = []
            all_observed_time = []
            all_evalpoint = []
            all_generated_samples = []

            for batch_no, test_batch in enumerate(test_loader, start=1):
                output = model.evaluate(
                    test_batch,
                    nsample,
                    guide_w,
                    data_mean=mean_scaler,
                    data_std=scaler,
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
                )
                # model.evaluate returns (samples, observed_data, target_mask, observed_mask, observed_tp)
                # or (samples, ..., attn) when save_attn=True — only unpack the first 5
                samples, c_target, eval_points, observed_points, observed_time = output[:5]

                if not torch.isfinite(samples).all():
                    bad = (~torch.isfinite(samples)).sum().item()
                    print(f"[evaluate][WARN] batch={batch_no}: {bad} non-finite sample values; clamping for metrics.")
                    samples = torch.nan_to_num(samples, nan=0.0, posinf=1e6, neginf=-1e6)

                samples = samples.permute(0, 1, 3, 2)
                c_target = c_target.permute(0, 2, 1)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_total  += (((samples_median.values - c_target) * eval_points) ** 2 * scaler ** 2).sum().item()
                mae_total  += (torch.abs((samples_median.values - c_target) * eval_points) * scaler).sum().item()
                nmse_total += (((samples_median.values - c_target) * eval_points) ** 2).sum().item()
                nmae_total += torch.abs((samples_median.values - c_target) * eval_points).sum().item()
                evalpoints_total += eval_points.sum().item()

            all_target            = torch.cat(all_target, dim=0)
            all_evalpoint         = torch.cat(all_evalpoint, dim=0)
            all_observed_point    = torch.cat(all_observed_point, dim=0)
            all_observed_time     = torch.cat(all_observed_time, dim=0)
            all_generated_samples = torch.cat(all_generated_samples, dim=0)

            if evalpoints_total > 0:
                CRPS     = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)
                CRPS_sum = calc_quantile_CRPS_sum(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)

            results = {
                "epoch": epoch,
                "guide_w": guide_w,
                "set": eval_name or "Evaluation",
                "NMSE": nmse_total / evalpoints_total if evalpoints_total > 0 else float("nan"),
                "NMAE": nmae_total / evalpoints_total if evalpoints_total > 0 else float("nan"),
                "CRPS": CRPS,
                "CRPS_sum": CRPS_sum,
            }
            with open(foldername + "config_results.json", "a") as f:
                json.dump(results, f, indent=4)
                f.write("\n")

            print(f"NMSE: {results['NMSE']:.6f} | NMAE: {results['NMAE']:.6f} | CRPS: {CRPS:.6f}")

    finally:
        # Always restore random state so subsequent training is unaffected by evaluation
        if param_grad_states is not None:
            for param, requires_grad in param_grad_states:
                param.requires_grad_(requires_grad)
        if was_training:
            model.train()
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    if return_dict:
        return {
            "MSE":      mse_total  / evalpoints_total if evalpoints_total > 0 else float("nan"),
            "MAE":      mae_total  / evalpoints_total if evalpoints_total > 0 else float("nan"),
            "NMSE":     nmse_total / evalpoints_total if evalpoints_total > 0 else float("nan"),
            "NMAE":     nmae_total / evalpoints_total if evalpoints_total > 0 else float("nan"),
            "CRPS":     CRPS,
            "CRPS_sum": CRPS_sum,
        }

    return nmse_total / evalpoints_total if evalpoints_total > 0 else float("nan")
