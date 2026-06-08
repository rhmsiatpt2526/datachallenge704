from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "occlusion_datasets" / "train.csv"
TEST_CSV = ROOT / "occlusion_datasets" / "test_students.csv"
IMAGE_DIR = ROOT / "crops" / "Crop_224_5fp_100K"

LOGS_DIR = ROOT / "logs"
SUBMISSIONS_DIR = ROOT / "submissions"
CHECKPOINTS_DIR = ROOT / "checkpoints"
