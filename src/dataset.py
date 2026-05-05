import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])


class MVTecMultiClass(Dataset):
    def __init__(
        self,
        root: str,
        categories: List[str],
        split: str = "train",
        image_size: int = 224,
    ):
        assert split in {"train", "test"}
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.samples: List[Tuple[Path, int, str, Optional[Path]]] = []

        for cat in categories:
            cat_root = self.root / cat
            if not cat_root.exists():
                raise FileNotFoundError(f"Category not found: {cat_root}")

            if split == "train":
                good_dir = cat_root / "train" / "good"
                for img_path in _list_images(good_dir):
                    self.samples.append((img_path, 0, cat, None))
            else:
                test_root = cat_root / "test"
                gt_root = cat_root / "ground_truth"
                for defect_dir in sorted(test_root.iterdir()):
                    label = 0 if defect_dir.name == "good" else 1
                    for img_path in _list_images(defect_dir):
                        mask_path = None
                        if label == 1:
                            mask_path = gt_root / defect_dir.name / f"{img_path.stem}_mask.png"
                            if not mask_path.exists():
                                mask_path = None
                        self.samples.append((img_path, label, cat, mask_path))

        self.image_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, cat, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_t = self.image_tf(img)

        if mask_path is None:
            mask_t = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        else:
            m = Image.open(mask_path).convert("L")
            m = self.mask_tf(m)
            m_arr = np.array(m, dtype=np.float32)
            m_arr = (m_arr > 0).astype(np.float32)
            mask_t = torch.from_numpy(m_arr).unsqueeze(0)

        return {
            "image": img_t,
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask_t,
            "category": cat,
            "path": str(img_path),
        }


def build_train_loader(cfg) -> DataLoader:
    ds = MVTecMultiClass(
        root=cfg["data"]["root"],
        categories=cfg["data"]["categories"],
        split="train",
        image_size=cfg["data"]["image_size"],
    )
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )


def build_eval_loader(cfg) -> DataLoader:
    ds = MVTecMultiClass(
        root=cfg["data"]["root"],
        categories=cfg["data"]["categories"],
        split="test",
        image_size=cfg["data"]["image_size"],
    )
    return DataLoader(
        ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["eval"]["num_workers"],
        pin_memory=True,
    )
