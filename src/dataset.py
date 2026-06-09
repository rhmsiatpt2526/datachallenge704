import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class OcclusionDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, training=True):
        self.training = training
        self.image_dir = image_dir
        self.df = df.reset_index(drop=True)

        if self.training:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.05,
                    ),
                    transforms.RandomAffine(
                        degrees=5,
                        translate=(0.02, 0.02),
                        scale=(0.95, 1.05),
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        filename = row["filename"]

        img = Image.open(self.image_dir / filename).convert("RGB")
        x = self.transform(img)

        if self.training:
            y = np.float32(row["FaceOcclusion"])
            gender = np.float32(row["gender"])
            return x, y, gender, filename

        return x, filename
