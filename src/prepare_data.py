"""
Merges Fish4Knowledge and Mediterranean datasets into a single
data/combined/<class>/ folder tree that ImageFolder can read.

Fish4Knowledge folders (fish_01…fish_23) are symlinked, not copied.
Mediterranean folders are symlinked from data/mediterranean/.

Usage:
    python src/prepare_data.py
"""

import os
import shutil
from pathlib import Path

F4K_DIR = Path("data/fish4knowledge/fish_image")
MED_DIR = Path("data/mediterranean")
OUT_DIR = Path("data/combined")


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    linked = 0

    # Fish4Knowledge
    if F4K_DIR.exists():
        for folder in sorted(F4K_DIR.iterdir()):
            if folder.is_dir():
                target = OUT_DIR / folder.name
                os.symlink(folder.resolve(), target)
                n = len(list(folder.glob("*.png"))) + len(list(folder.glob("*.jpg")))
                print(f"  F4K  {folder.name}: {n} images")
                linked += 1
    else:
        print(f"WARNING: {F4K_DIR} not found — skipping Fish4Knowledge")

    # Mediterranean
    if MED_DIR.exists():
        for folder in sorted(MED_DIR.iterdir()):
            if folder.is_dir():
                target = OUT_DIR / folder.name
                os.symlink(folder.resolve(), target)
                n = len(list(folder.glob("*.jpg")))
                print(f"  MED  {folder.name}: {n} images")
                linked += 1
    else:
        print(f"WARNING: {MED_DIR} not found — skipping Mediterranean species")

    print(f"\nCombined dataset: {linked} classes → {OUT_DIR}")


if __name__ == "__main__":
    main()
