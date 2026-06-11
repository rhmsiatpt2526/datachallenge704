from pathlib import Path
import sys
from datetime import datetime
import argparse
import random

import numpy as np
import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.config import LOGS_DIR, SUBMISSIONS_DIR, CHECKPOINTS_DIR
from src.data import create_dataloaders
from src.model import build_model
from src.engine import train_one_epoch
from src.predict import collect_predictions, predict_test
from src.metrics import split_errors
from src.utils import create_run_tag, save_checkpoint, write_log


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--experiment-name", type=str, default="datachallenge704")
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--log-model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine"],
    )
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--val-every", type=int, default=10)

    parser.add_argument(
        "--model",
        type=str,
        default="mobilenetv3_small",
        choices=[
            "mobilenetv3_small",
            "mobilenetv3_large",
            "efficientnet_b0",
            "efficientnet_b1",
            "resnet50",
            "convnext_tiny",
            "dinov2_small",
            "dinov2_base",
            "dinov3-vits16",
            "dinov3-vitb16",
        ],
    )

    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=0)

    parser.add_argument(
        "--use-gender-gap-loss",
        action="store_true",
        help="Add a gender gap penalty to the weighted MSE loss.",
    )

    parser.add_argument(
        "--lambda-gap",
        type=float,
        default=0.1,
        help="Weight of the gender gap penalty in the training loss.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    run_started_at = datetime.now()
    run_started_at_iso = run_started_at.isoformat(timespec="seconds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"
    use_non_blocking = device.type == "cuda"

    print(f"Device utilisé : {device}")
    if device.type == "cuda":
        print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")

    MODEL_NAME = args.model
    runs_dir = LOGS_DIR / MODEL_NAME
    checkpoints_dir = CHECKPOINTS_DIR / MODEL_NAME
    submissions_dir = SUBMISSIONS_DIR

    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    run_tag, _ = create_run_tag(runs_dir, MODEL_NAME)

    if args.tracking_uri is None:
        tracking_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    else:
        tracking_uri = args.tracking_uri

    artifact_location = (ROOT / "mlartifacts").resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(
            name=args.experiment_name,
            artifact_location=artifact_location,
        )
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(run_name=run_tag, experiment_id=experiment_id):
        mlflow.log_params(
            {
                "model_name": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "scheduler": args.scheduler,
                "min_lr": args.min_lr,
                "val_every": args.val_every,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "num_workers": args.num_workers,
                "device": str(device),
                "seed": args.seed,
                "freeze_backbone": args.freeze_backbone,
                "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
                "use_gender_gap_loss": args.use_gender_gap_loss,
                "lambda_gap": args.lambda_gap,
            }
        )

        training_loader, validation_loader, test_loader = create_dataloaders(
            args,
            use_pin_memory=use_pin_memory,
        )

        model = build_model(
            args.model,
            freeze_backbone=args.freeze_backbone,
            unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        )
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        mlflow.log_param("total_params", total_params)
        mlflow.log_param("trainable_params", trainable_params)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.min_lr,
            )
        else:
            scheduler = None

        best_val_metric = float("inf")
        best_epoch = None
        best_checkpoint_path = checkpoints_dir / f"{run_tag}_best.pt"

        for epoch in range(args.epochs):
            epoch_loss = train_one_epoch(
                model,
                training_loader,
                optimizer,
                device,
                epoch,
                args.epochs,
                use_non_blocking=use_non_blocking,
                use_gender_gap_loss=args.use_gender_gap_loss,
                lambda_gap=args.lambda_gap,
            )

            if scheduler is not None:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            mlflow.log_metric("train_epoch_loss", epoch_loss, step=epoch + 1)
            mlflow.log_metric("lr", current_lr, step=epoch + 1)

            should_validate = args.val_every > 0 and (
                (epoch + 1) % args.val_every == 0 or (epoch + 1) == args.epochs
            )

            if should_validate:
                val_results_epoch = collect_predictions(
                    model,
                    validation_loader,
                    "validation",
                    device,
                    use_non_blocking=use_non_blocking,
                )

                val_stats_epoch = split_errors(val_results_epoch)

                if val_stats_epoch["balanced_metric"] < best_val_metric:
                    best_val_metric = val_stats_epoch["balanced_metric"]
                    best_epoch = epoch + 1

                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "run_tag": run_tag,
                            "model_name": MODEL_NAME,
                            "args": vars(args),
                            "epoch": epoch + 1,
                            "best_val_metric": best_val_metric,
                            "best_val_stats": val_stats_epoch,
                        },
                        best_checkpoint_path,
                    )

                    mlflow.log_metric(
                        "best_validation_balanced_metric",
                        best_val_metric,
                        step=epoch + 1,
                    )
                    mlflow.log_metric("best_epoch", best_epoch, step=epoch + 1)

                    print(
                        f"New best checkpoint saved at epoch {best_epoch}: "
                        f"{best_checkpoint_path} "
                        f"(balanced metric: {best_val_metric:.6f})"
                    )

                mlflow.log_metric(
                    "validation_error_epoch",
                    val_stats_epoch["error"],
                    step=epoch + 1,
                )
                mlflow.log_metric(
                    "validation_female_error_epoch",
                    val_stats_epoch["female_error"],
                    step=epoch + 1,
                )
                mlflow.log_metric(
                    "validation_male_error_epoch",
                    val_stats_epoch["male_error"],
                    step=epoch + 1,
                )
                mlflow.log_metric(
                    "validation_gender_gap_epoch",
                    val_stats_epoch["gender_gap"],
                    step=epoch + 1,
                )
                mlflow.log_metric(
                    "validation_balanced_metric_epoch",
                    val_stats_epoch["balanced_metric"],
                    step=epoch + 1,
                )

                print(
                    f"Epoch {epoch + 1}/{args.epochs} - "
                    f"val balanced metric: {val_stats_epoch['balanced_metric']:.6f}"
                )

        if best_checkpoint_path.exists():
            print(
                f"Loading best checkpoint from epoch {best_epoch}: "
                f"{best_checkpoint_path}"
            )
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("No best checkpoint found, using final model.")

        train_results = collect_predictions(
            model,
            training_loader,
            "train",
            device,
            use_non_blocking=use_non_blocking,
        )

        val_results = collect_predictions(
            model,
            validation_loader,
            "validation",
            device,
            use_non_blocking=use_non_blocking,
        )

        val_predictions_path = runs_dir / f"{run_tag}_val_predictions.csv"
        val_results.to_csv(val_predictions_path, index=False)

        train_stats = split_errors(train_results)
        val_stats = split_errors(val_results)

        test_predictions = predict_test(
            model,
            test_loader,
            device,
            use_non_blocking=use_non_blocking,
        )

        run_finished_at = datetime.now()
        run_finished_at_iso = run_finished_at.isoformat(timespec="seconds")
        run_duration_seconds = (run_finished_at - run_started_at).total_seconds()

        mlflow.log_metrics(
            {
                "train_error": train_stats["error"],
                "train_female_error": train_stats["female_error"],
                "train_male_error": train_stats["male_error"],
                "train_gender_gap": train_stats["gender_gap"],
                "validation_error": val_stats["error"],
                "validation_female_error": val_stats["female_error"],
                "validation_male_error": val_stats["male_error"],
                "validation_gender_gap": val_stats["gender_gap"],
                "validation_balanced_metric": val_stats["balanced_metric"],
                "run_duration_seconds": run_duration_seconds,
            }
        )

        submission_df = test_predictions.copy()
        submission_df["FaceOcclusion"] = submission_df["FaceOcclusion"].clip(0, 1)
        submission_df["gender"] = "x"

        submission_path = submissions_dir / f"{run_tag}.csv"
        submission_df.to_csv(submission_path, index=False)

        checkpoint_path = checkpoints_dir / f"{run_tag}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            run_tag,
            MODEL_NAME,
            args,
            train_stats,
            val_stats,
        )

        log_path = runs_dir / f"{run_tag}.md"
        write_log(
            log_path,
            run_tag,
            run_started_at_iso,
            run_finished_at_iso,
            run_duration_seconds,
            MODEL_NAME,
            args,
            checkpoint_path,
            submission_path,
            train_stats,
            val_stats,
            train_rows=len(train_results),
            val_rows=len(val_results),
            test_rows=len(submission_df),
            total_params=total_params,
            trainable_params=trainable_params,
        )

        mlflow.log_artifact(str(submission_path), artifact_path="submissions")
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

        if best_checkpoint_path.exists():
            mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")
            mlflow.log_param("best_checkpoint_path", str(best_checkpoint_path))
            mlflow.log_param("best_epoch_final", best_epoch)

        mlflow.log_artifact(str(log_path), artifact_path="logs")
        mlflow.log_artifact(
            str(val_predictions_path),
            artifact_path="validation_predictions",
        )

        if args.log_model:
            mlflow.pytorch.log_model(model, artifact_path="model")

        print(f"Submission saved to: {submission_path}")
        print(f"Checkpoint saved to: {checkpoint_path}")
        print(f"Run log saved to: {log_path}")
        print(f"Best epoch: {best_epoch}")
        print(f"Validation balanced metric: {val_stats['balanced_metric']:.6f}")


if __name__ == "__main__":
    main()
