from datetime import datetime
import torch


def next_run_id(base_dir, model_name):
    run_ids = []

    for path in base_dir.glob(f"{model_name}_run*.md"):
        suffix = path.stem.replace(f"{model_name}_run", "")

        if suffix.isdigit():
            run_ids.append(int(suffix))

    return max(run_ids, default=0) + 1


def create_run_tag(runs_dir, model_name):
    run_id = next_run_id(runs_dir, model_name)
    run_tag = f"{model_name}_run{run_id:03d}"
    timestamp = datetime.now().isoformat(timespec="seconds")

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
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
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
):
    log_lines = [
        f"run: {run_tag}",
        f"start_timestamp: {start_timestamp}",
        f"end_timestamp: {end_timestamp}",
        f"total_duration_seconds: {total_duration_seconds:.3f}",
        f"model: {model_name}",
        f"epochs: {args.epochs}",
        f"batch_size: {args.batch_size}",
        f"lr: {args.lr}",
        f"weight_decay: {args.weight_decay}",
        f"num_workers: {args.num_workers}",
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
