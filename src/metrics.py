from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.ndimage import label as cc_label


def _avg_pool_max(maps: np.ndarray, pool_size: int) -> np.ndarray:
    t = torch.from_numpy(maps).unsqueeze(1).float()
    pooled = F.avg_pool2d(t, kernel_size=pool_size, stride=pool_size)
    pooled = pooled.view(pooled.shape[0], -1)
    return pooled.max(dim=1).values.numpy()


def image_score(maps: np.ndarray, mode: str, pool_size: int = 16) -> np.ndarray:
    if mode == "mean":
        return maps.reshape(maps.shape[0], -1).mean(axis=1)
    if mode == "max":
        return _avg_pool_max(maps, pool_size)
    if mode == "std":
        return maps.reshape(maps.shape[0], -1).std(axis=1)
    raise ValueError(f"unknown post-processing mode: {mode}")


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3, num_thresh: int = 100) -> float:
    masks = masks.astype(np.uint8)
    if masks.sum() == 0:
        return float("nan")

    flat_maps = maps.reshape(-1)
    lo, hi = float(flat_maps.min()), float(flat_maps.max())
    thresholds = np.linspace(hi, lo, num_thresh)

    pros, fprs = [], []
    neg = (masks == 0)
    neg_total = neg.sum()

    for th in thresholds:
        pred = (maps >= th)
        per_region_recalls = []
        for i in range(masks.shape[0]):
            comps, ncomp = cc_label(masks[i])
            for c in range(1, ncomp + 1):
                region = (comps == c)
                if region.sum() == 0:
                    continue
                per_region_recalls.append((pred[i] & region).sum() / region.sum())
        if not per_region_recalls:
            continue
        pro = float(np.mean(per_region_recalls))
        fpr = float((pred & neg).sum() / max(neg_total, 1))
        pros.append(pro)
        fprs.append(fpr)
        if fpr > fpr_limit:
            break

    if len(fprs) < 2:
        return float("nan")

    fprs = np.array(fprs)
    pros = np.array(pros)
    order = np.argsort(fprs)
    fprs, pros = fprs[order], pros[order]
    keep = fprs <= fpr_limit
    fprs, pros = fprs[keep], pros[keep]
    if len(fprs) < 2:
        return float("nan")
    auc = (np.trapezoid(pros, fprs) if hasattr(np, "trapezoid") else np.trapz(pros, fprs)) / fpr_limit
    return float(auc)


def compute_metrics(
    labels: np.ndarray,
    masks: np.ndarray,
    maps: np.ndarray,
    pool_size: int = 16,
) -> Dict[str, float]:
    out = {}
    out["pixel_auroc"] = float(roc_auc_score(masks.ravel(), maps.ravel()))
    out["aupro"] = aupro(masks, maps)
    for mode in ("mean", "max", "std"):
        scores = image_score(maps, mode, pool_size)
        out[f"image_auroc_{mode}"] = float(roc_auc_score(labels, scores))
    return out


def aggregate_per_category(
    labels: np.ndarray,
    masks: np.ndarray,
    maps: np.ndarray,
    categories: List[str],
    pool_size: int = 16,
) -> Dict[str, Dict[str, float]]:
    cats = np.array(categories)
    by_cat: Dict[str, Dict[str, float]] = {}
    for cat in sorted(set(categories)):
        idx = np.where(cats == cat)[0]
        by_cat[cat] = compute_metrics(labels[idx], masks[idx], maps[idx], pool_size)
    by_cat["mean"] = {
        k: float(np.mean([by_cat[c][k] for c in by_cat if c != "mean" and not np.isnan(by_cat[c][k])]))
        for k in next(iter(by_cat.values())).keys()
    }
    return by_cat
