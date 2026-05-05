import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import UniAD, build_train_loader, Trainer


def expand_env(cfg):
    if isinstance(cfg, dict):
        return {k: expand_env(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [expand_env(v) for v in cfg]
    if isinstance(cfg, str) and cfg.startswith("${env:") and cfg.endswith("}"):
        var = cfg[len("${env:"):-1]
        v = os.environ.get(var)
        if v is None:
            raise RuntimeError(f"environment variable {var} is not set")
        return v
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = expand_env(yaml.safe_load(f))

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = build_train_loader(cfg)
    print(f"train samples: {len(train_loader.dataset)}")

    model = UniAD(cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable/1e6:.2f}M")

    trainer = Trainer(model, train_loader, cfg, device=device)
    trainer.fit()


if __name__ == "__main__":
    main()
