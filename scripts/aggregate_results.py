import argparse
import json
from pathlib import Path

from tabulate import tabulate

PAPER = {
    "image_auroc_max": 96.7,
    "pixel_auroc": 96.8,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/results.json")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    cats = [c for c in results.keys() if c != "mean"]
    rows = []
    for c in cats:
        r = results[c]
        rows.append([
            c,
            f"{r['image_auroc_max']*100:.2f}",
            f"{r['pixel_auroc']*100:.2f}",
            f"{r['aupro']*100:.2f}" if r['aupro'] == r['aupro'] else "n/a",
        ])
    m = results["mean"]
    rows.append([
        "mean",
        f"{m['image_auroc_max']*100:.2f}",
        f"{m['pixel_auroc']*100:.2f}",
        f"{m['aupro']*100:.2f}" if m['aupro'] == m['aupro'] else "n/a",
    ])
    rows.append([
        "paper",
        f"{PAPER['image_auroc_max']:.2f}",
        f"{PAPER['pixel_auroc']:.2f}",
        "—",
    ])
    rows.append([
        "delta (mean - paper)",
        f"{m['image_auroc_max']*100 - PAPER['image_auroc_max']:+.2f}",
        f"{m['pixel_auroc']*100 - PAPER['pixel_auroc']:+.2f}",
        "—",
    ])

    print(tabulate(rows, headers=["category", "image AUROC (max)", "pixel AUROC", "AUPRO"], tablefmt="github"))


if __name__ == "__main__":
    main()
