from pathlib import Path
import sys
from datetime import datetime
import argparse
import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.config import MODEL_NAME, LOGS_DIR, SUBMISSIONS_DIR, CHECKPOINTS_DIR
from src.data import create_dataloaders
from src.model import build_mobilenetv3_small
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
    return parser.parse_args()


def main():
    args = parse_args()
    run_started_at = datetime.now()
    run_started_at_iso = run_started_at.isoformat(timespec="seconds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"
    use_non_blocking = device.type == "cuda"

    print(f"Device utilisé : {device}")
    if device.type == "cuda":
        print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")

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
                "model_name": MODEL_NAME,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "num_workers": args.num_workers,
                "device": str(device),
                "freeze_backbone": True,
            }
        )

        training_loader, validation_loader, test_loader = create_dataloaders(
            args,
            use_pin_memory=use_pin_memory,
        )

        model = build_mobilenetv3_small(freeze_backbone=True)
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        for epoch in range(args.epochs):
            epoch_loss = train_one_epoch(
                model,
                training_loader,
                optimizer,
                device,
                epoch,
                args.epochs,
                use_non_blocking=use_non_blocking,
            )

            mlflow.log_metric("train_epoch_loss", epoch_loss, step=epoch + 1)

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
        )

        mlflow.log_artifact(str(submission_path), artifact_path="submissions")
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(log_path), artifact_path="logs")

        if args.log_model:
            mlflow.pytorch.log_model(model, artifact_path="model")

        print(f"Submission saved to: {submission_path}")
        print(f"Checkpoint saved to: {checkpoint_path}")
        print(f"Run log saved to: {log_path}")
        print(f"Validation balanced metric: {val_stats['balanced_metric']:.6f}")


if __name__ == "__main__":
    main()
