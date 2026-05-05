from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .metrics import aggregate_per_category


class Evaluator:
    def __init__(self, model, eval_loader, cfg, device: str = "cuda"):
        self.model = model.to(device)
        self.loader = eval_loader
        self.cfg = cfg
        self.device = device
        self.pool_size = cfg["eval"]["pool_size"]

    @torch.no_grad()
    def run(self):
        self.model.eval()
        all_maps, all_labels, all_masks, all_cats = [], [], [], []
        for batch in tqdm(self.loader, desc="eval"):
            images = batch["image"].to(self.device, non_blocking=True)
            maps = self.model.anomaly_map(images).cpu().numpy()
            all_maps.append(maps)
            all_labels.append(batch["label"].numpy())
            all_masks.append(batch["mask"].squeeze(1).numpy())
            all_cats.extend(batch["category"])

        maps = np.concatenate(all_maps, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        masks = np.concatenate(all_masks, axis=0)
        return aggregate_per_category(labels, masks, maps, all_cats, self.pool_size)

    def save(self, results: dict, out_path: Path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {out_path}")
