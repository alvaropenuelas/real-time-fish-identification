"""
Validation gate for a YOLO-format detection dataset (A4): draw the YOLO boxes back onto a
random sample of images and save a grid PNG, so misaligned/scaled boxes are caught by eye
BEFORE any training. Also prints image/box counts and the empty-label (background) fraction.

Single class is assumed (column 0). Label files are "<cls> cx cy w h" normalized to [0,1].

Example:
    python scripts/check_yolo_labels.py --src /tmp/deepfish_test/test \\
        --split-name benchmark-official --n 30 --out outputs/label_check.png
"""

import argparse
import glob
import os
import random

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def count_split(imgs: list[str]) -> tuple[int, int, int]:
    """Return (n_images, n_boxes, n_empty_label_images)."""
    n_box = n_empty = 0
    for im in imgs:
        lab = im[:-4] + ".txt"
        lines = [ln for ln in open(lab).read().splitlines() if ln.strip()] if os.path.exists(lab) else []
        n_box += len(lines)
        n_empty += not lines
    return len(imgs), n_box, n_empty


def main():
    parser = argparse.ArgumentParser(description="Draw YOLO boxes onto sample images for a sanity check")
    parser.add_argument("--src", required=True, help="Directory of paired <name>.jpg / <name>.txt")
    parser.add_argument("--n", type=int, default=30, help="Number of random samples to draw")
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-name", default="unknown", help="Label printed for which split this is")
    parser.add_argument("--out", default="outputs/label_check.png")
    args = parser.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.src, "*.jpg")))
    if not imgs:
        raise SystemExit(f"No .jpg images found under {args.src}")
    n_img, n_box, n_empty = count_split(imgs)

    random.seed(args.seed)
    sample = random.sample(imgs, min(args.n, n_img))
    rows = (len(sample) + args.cols - 1) // args.cols
    fig, axes = plt.subplots(rows, args.cols, figsize=(args.cols * 3.7, rows * 3.2))
    for ax, im in zip(axes.ravel(), sample, strict=False):
        img = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        lab = im[:-4] + ".txt"
        nb = 0
        if os.path.exists(lab):
            for ln in open(lab).read().splitlines():
                p = ln.split()
                if len(p) != 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in p[1:])
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                nb += 1
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{os.path.basename(im)[:24]}  [{nb} box]", fontsize=7)
    for ax in axes.ravel()[len(sample) :]:
        ax.axis("off")
    fig.suptitle(f"YOLO label check — split={args.split_name}, {len(sample)} random samples", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=90)

    print("=== YOLO LABEL CHECK ===")
    print(f"split            : {args.split_name}")
    print(f"images           : {n_img}")
    print(f"boxes            : {n_box}  (mean {n_box / n_img:.2f}/img)")
    print(f"empty-label imgs : {n_empty}  ({100 * n_empty / n_img:.1f}% background)")
    print(f"wrote            : {args.out}")


if __name__ == "__main__":
    main()
