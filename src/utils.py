from datetime import datetime
import os
import re
import torch


def next_run_id(base_dir, model_name):
    run_ids = []

    for path in base_dir.glob(f"{model_name}_run*.md"):
        suffix = path.stem.replace(f"{model_name}_run", "")

        if suffix.isdigit():
            run_ids.append(int(suffix))

    return max(run_ids, default=0) + 1


def _safe_filename_part(value):
    """
    Return a filesystem-safe string fragment.

    This is mainly useful for experiment names because they may contain spaces,
    slashes, or other characters that are inconvenient in checkpoint/log names.
    """
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip("-")


def create_run_tag(runs_dir, model_name, seed=None, experiment_name=None):
    """
    Create a unique run tag.

    On SLURM clusters, several jobs can start at the same time. The old
    next_run_id-based naming could make parallel jobs all choose the same
    runXXX tag and overwrite each other's checkpoints/submissions/logs.

    When SLURM_JOB_ID exists, use it in the tag because it is unique for each
    submitted job. Outside SLURM, keep the previous runXXX behavior.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if slurm_job_id:
        parts = [model_name, f"job{slurm_job_id}"]

        if slurm_array_task_id is not None:
            parts.append(f"task{slurm_array_task_id}")

        if seed is not None:
            parts.append(f"seed{seed}")

        if experiment_name:
            safe_experiment_name = _safe_filename_part(experiment_name)
            if safe_experiment_name:
                parts.append(safe_experiment_name)

        run_tag = "_".join(parts)
    else:
        run_id = next_run_id(runs_dir, model_name)
        run_tag = f"{model_name}_run{run_id:03d}"

    return run_tag, timestamp


def save_checkpoint(
    path, model, optimizer, run_tag, model_name, args, train_stats, val_stats
):
    torch.save(
        {
            "run_tag": run_tag,
            "model_name": model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "scheduler": getattr(args, "scheduler", "none"),
            "min_lr": getattr(args, "min_lr", None),
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "val_every": getattr(args, "val_every", None),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_balanced_metric": val_stats["balanced_metric"],
            "validation_error": val_stats["error"],
            "train_error": train_stats["error"],
        },
        path,
    )


def write_log(
    path,
    run_tag,
    start_timestamp,
    end_timestamp,
    total_duration_seconds,
    model_name,
    args,
    checkpoint_path,
    submission_path,
    train_stats,
    val_stats,
    train_rows,
    val_rows,
    test_rows,
    total_params=None,
    trainable_params=None,
):
    log_lines = [
        f"run: {run_tag}",
        f"start_timestamp: {start_timestamp}",
        f"end_timestamp: {end_timestamp}",
        f"total_duration_seconds: {total_duration_seconds:.3f}",
        f"model: {model_name}",
        f"total_params: {total_params}",
        f"trainable_params: {trainable_params}",
        f"epochs: {args.epochs}",
        f"batch_size: {args.batch_size}",
        f"lr: {args.lr}",
        f"scheduler: {getattr(args, 'scheduler', 'none')}",
        f"min_lr: {getattr(args, 'min_lr', None)}",
        f"weight_decay: {args.weight_decay}",
        f"num_workers: {args.num_workers}",
        f"val_every: {getattr(args, 'val_every', None)}",
        f"checkpoint_file: {checkpoint_path.name}",
        f"train_error: {train_stats['error']:.6f}",
        f"train_female_error: {train_stats['female_error']:.6f}",
        f"train_male_error: {train_stats['male_error']:.6f}",
        f"train_gender_gap: {train_stats['gender_gap']:.6f}",
        f"validation_error: {val_stats['error']:.6f}",
        f"validation_female_error: {val_stats['female_error']:.6f}",
        f"validation_male_error: {val_stats['male_error']:.6f}",
        f"validation_gender_gap: {val_stats['gender_gap']:.6f}",
        f"validation_balanced_metric: {val_stats['balanced_metric']:.6f}",
        "competition_test_error: NA (labels unavailable)",
        f"submission_file: {submission_path.name}",
        f"train_rows: {train_rows}",
        f"validation_rows: {val_rows}",
        f"test_rows: {test_rows}",
    ]

    path.write_text("\n".join(log_lines), encoding="utf-8")
