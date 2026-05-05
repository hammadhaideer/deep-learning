# uniad-reproduced

> Clean reproduction of **UniAD** (You et al., NeurIPS 2022 Spotlight) on MVTec-AD - multi-class anomaly detection and localization with one unified transformer-based reconstruction model.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Paper: [A Unified Model for Multi-class Anomaly Detection](https://arxiv.org/abs/2206.03687) - You et al., NeurIPS 2022 Spotlight  
Official repo: <https://github.com/zhiyuanyou/UniAD>

## What this is

End-to-end reimplementation of UniAD: a frozen EfficientNet-b4 backbone, multi-stage feature concatenation at 14×14, and a transformer encoder-decoder with neighbor-masked attention, layer-wise learnable queries, and feature jittering. One model is trained jointly across all 15 MVTec-AD categories - the unified one-for-all setting that defines the paper.

This is the **third reproduction** in a series of visual anomaly detection methods I'm building toward **UniVAD** (CVPR 2025), the current state-of-the-art training-free unified VAD method. UniAD is the canonical unified-model baseline that every later "one model for all categories" paper compares against, including UniVAD.

## Status

In active development. Code lands over the coming week, results follow once GPU access returns.

## Why UniAD

UniAD is the paper that put "one model for all categories" on the map. Earlier methods (PatchCore, EfficientAD) train a separate model per category, which is fine for benchmarks but unrealistic for a factory with hundreds of products. UniAD trains a single transformer across every category at once and shows you can match or beat per-category baselines while serving the whole catalog from one checkpoint.

To get there, the paper diagnoses an "identical shortcut" failure mode in plain reconstruction networks - they trivially copy anomalies through to the output - and fixes it with three design choices:

1. **Neighbor-masked attention** stops each token from attending to itself or its immediate neighbors, forcing reconstruction from non-local context.
2. **Layer-wise learnable queries** in the decoder use a fresh query set per layer instead of propagating decoded tokens, blocking input leakage.
3. **Feature jittering** perturbs encoder input tokens during training so the model recovers the underlying normal pattern instead of memorizing exact features.

|                                | Detection AUROC | Localization AUROC |
| ------------------------------ | --------------- | ------------------ |
| UniAD (paper, unified setting) | 96.7            | 96.8               |

(Numbers from the UniAD paper, mean across 15 MVTec-AD categories under the unified one-for-all setting.)

## Goal

Match the paper's reported numbers within ±0.5 points on MVTec-AD (mean across 15 categories) under the unified setting:

| Setting | Dataset  | Detection AUROC | Localization AUROC |
| ------- | -------- | --------------- | ------------------ |
| Unified | MVTec-AD | 96.7 → TBD      | 96.8 → TBD         |

Per-category table will land here as runs complete.

## Installation

Clone and create the environment:

    git clone https://github.com/hammadhaideer/uniad-reproduced.git
    cd uniad-reproduced
    conda env create -f environment.yml
    conda activate uniad

Or with pip in an existing environment:

    pip install -r requirements.txt

## Dataset

Download MVTec-AD from the [official page](https://www.mvtec.com/company/research/datasets/mvtec-ad) (registration required, free for non-commercial use). Extract it anywhere on your machine, then point the code to it via an environment variable:

    export UNIAD_DATA_ROOT=/path/to/mvtec_ad

Expected layout:

    $UNIAD_DATA_ROOT/
    ├── bottle/
    │   ├── train/good/*.png
    │   ├── test/<defect>/*.png
    │   └── ground_truth/<defect>/*_mask.png
    ├── cable/
    └── ... (15 categories)

| Dataset  | Source                                                   | Categories |
| -------- | -------------------------------------------------------- | ---------- |
| MVTec-AD | https://www.mvtec.com/company/research/datasets/mvtec-ad | 15         |

## Run

Train a unified model across all 15 categories (default: 1000 epochs, batch size 8, single GPU):

    python scripts/train.py --config configs/default.yaml

Evaluate a checkpoint:

    python scripts/eval.py --config configs/default.yaml \
        --checkpoint checkpoints/uniad_epoch1000.pth \
        --out results/results.json

Build the comparison table after the run completes:

    python scripts/aggregate_results.py --results results/results.json

Per-category results land in `results/results.json`. The aggregated table prints to stdout, comparing your numbers against the paper's.

## Walkthrough notebook

For an interactive end-to-end pipeline check on a single category - feature shapes, neighbor mask visualization, and qualitative heatmaps before/after training:

    jupyter notebook notebooks/01_walkthrough.ipynb

Run cells top-to-bottom. The notebook reads the same config and dataset path as the scripts.

## Roadmap

- [x] Repo scaffold, configs, dataset loader
- [x] EfficientNet-b4 backbone + multi-stage feature concatenation
- [x] Transformer with neighbor-masked attention + layer-wise queries + feature jittering
- [x] Training loop with AMP, cosine schedule, warmup
- [x] Image-AUROC, pixel-AUROC, AUPRO metrics
- [x] Eval runner and results aggregation script
- [x] Walkthrough notebook with neighbor-mask and heatmap visualizations
- [ ] Empirical reproduction across all 15 categories (pending GPU)
- [ ] Qualitative heatmap figures
- [ ] Medium walkthrough post

## Reproduction series

Part of a broader series reproducing UniVAD's full comparison set:

- [x] [`patchcore-reproduced`](https://github.com/hammadhaideer/patchcore-reproduced) - PatchCore (CVPR 2022)
- [x] [`winclip-reproduced`](https://github.com/hammadhaideer/winclip-reproduced) - WinCLIP (CVPR 2023)
- [ ] **`uniad-reproduced`** - UniAD (NeurIPS 2022) ← this repo
- [ ] `anomalygpt-reproduced` - AnomalyGPT (AAAI 2024)
- [ ] `comad-reproduced` - ComAD (PR 2024)
- [ ] `medclip-reproduced` - MedCLIP (EMNLP 2022)
- [ ] `univad-reproduced` - UniVAD (CVPR 2025) ← target

## References

- You et al., *A Unified Model for Multi-class Anomaly Detection*, NeurIPS 2022 (Spotlight) - [arXiv:2206.03687](https://arxiv.org/abs/2206.03687)
- Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019
- Bergmann et al., *MVTec-AD*, CVPR 2019
- Official UniAD repo: <https://github.com/zhiyuanyou/UniAD>

## License

MIT
