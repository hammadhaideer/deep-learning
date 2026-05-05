import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import UniAD, build_eval_loader, Evaluator


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/results.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = expand_env(yaml.safe_load(f))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_loader = build_eval_loader(cfg)
    print(f"test samples: {len(eval_loader.dataset)}")

    model = UniAD(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

    evaluator = Evaluator(model, eval_loader, cfg, device=device)
    results = evaluator.run()
    evaluator.save(results, args.out)

    print()
    mean = results.get("mean", {})
    if mean:
        print("== mean across categories ==")
        for k, v in mean.items():
            print(f"  {k:18s} {v*100:6.2f}")


if __name__ == "__main__":
    main()
