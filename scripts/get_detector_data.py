"""
Fetch marine fish-detection datasets for fine-tuning the YOLO detector, replacing the
freshwater set. All URLs below are the datasets' official, verified locations — none are
invented. Each source is CC-licensed for research; cite per the references in the repo.

Sources
-------
deepfish : DeepFish (JCU) — ~40k underwater images, 20 tropical-Australia habitats, with
           classification / point / segmentation annotations usable for fish localization.
           Saleh et al., Sci. Rep. 2020. License: see dataset page.
           https://alzayats.github.io/DeepFish/  |  data: http://data.qld.edu.au/public/Q5842/...
ozfish   : OzFish (AIMS/UWA/Curtin) — ~80k crops, ~45k fish/no-fish bounding boxes from BRUVS.
           Distributed as Pawsey manifests (lists of object URLs). License: CC BY 3.0.
           https://github.com/open-AIMS/ozfish
noaa     : NOAA SWFSC "Labeled Fishes in the Wild" — labeled + UNLABELED frames; used here as
           BACKGROUND / hard-negative frames for the detector (reduce false positives).
           https://swfscdata.nmfs.noaa.gov/labeled-fishes-in-the-wild/

This script downloads code/data only; training runs on Kaggle (see notebooks/). Large
downloads — use --dry-run first to see what would be fetched.

Examples:
    python scripts/get_detector_data.py --source all --dry-run
    python scripts/get_detector_data.py --source deepfish
    python scripts/get_detector_data.py --source ozfish --download-images
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests

REQUEST_TIMEOUT = 60
ARCHIVE_EXTS = (".zip", ".tar", ".tar.gz", ".tgz", ".7z")

SOURCES = {
    "deepfish": {
        "desc": "DeepFish — ~40k underwater images, tropical Australia (classification/point/segmentation)",
        "license": "Research use; cite Saleh et al., Sci. Rep. 2020",
        "url": "http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/",
        "kind": "archive_index",
    },
    "ozfish": {
        "desc": "OzFish — ~45k fish bounding boxes from BRUVS (fish/no-fish)",
        "license": "CC BY 3.0; cite AIMS OzFish",
        "url": "https://data.pawsey.org.au/public/?path=/FDFML/labelled/manifests",
        "repo": "https://github.com/open-AIMS/ozfish",
        "kind": "manifest",
    },
    "noaa": {
        "desc": "NOAA Labeled Fishes in the Wild — background / hard-negative frames",
        "license": "U.S. public domain (NOAA Fisheries)",
        "url": "https://swfscdata.nmfs.noaa.gov/labeled-fishes-in-the-wild/",
        "kind": "archive_index",
        "role": "hard_negatives",
    },
}


def _hrefs(page_url: str) -> list[str]:
    """Absolute hrefs from an HTML index/landing page."""
    resp = requests.get(page_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    raw = re.findall(r'href=["\']([^"\']+)["\']', resp.text, flags=re.IGNORECASE)
    return [urljoin(page_url, h) for h in raw]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  skip (exists): {dest.name}")
        return
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.rename(dest)
    print(f"  saved: {dest.name}")


def fetch_archive_index(url: str, out: Path, dry: bool) -> None:
    links = [h for h in _hrefs(url) if h.lower().endswith(ARCHIVE_EXTS)]
    if not links:
        print(f"  no archive links found at {url}")
        print("  (the dataset may require browsing the index or accepting a license manually)")
        return
    for link in links:
        print(f"  archive: {link}")
        if not dry:
            _download(link, out / Path(link).name)


def fetch_manifests(url: str, out: Path, dry: bool, download_images: bool) -> None:
    manifests = [h for h in _hrefs(url) if h.lower().endswith((".txt", ".csv", ".json", ".manifest"))]
    print(f"  {len(manifests)} manifest file(s) discovered")
    for m in manifests:
        print(f"  manifest: {m}")
        if dry:
            continue
        dest = out / "manifests" / Path(m).name
        _download(m, dest)
        if download_images:
            urls = [ln.strip() for ln in dest.read_text().splitlines() if ln.strip().startswith("http")]
            print(f"    {len(urls)} image URLs in {dest.name}")
            for u in urls:
                _download(u, out / "images" / Path(u).name)


def main():
    parser = argparse.ArgumentParser(description="Fetch marine detector datasets (DeepFish/OzFish/NOAA)")
    parser.add_argument("--source", choices=[*SOURCES, "all"], default="all")
    parser.add_argument("--out-dir", type=Path, default=Path("data/detector"))
    parser.add_argument("--dry-run", action="store_true", help="List what would be downloaded, fetch nothing")
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="OzFish: also fetch every image URL listed in each manifest (large)",
    )
    args = parser.parse_args()

    targets = list(SOURCES) if args.source == "all" else [args.source]
    for key in targets:
        s = SOURCES[key]
        print(f"\n=== {key} — {s['desc']} ===")
        print(f"  license: {s['license']}")
        if s.get("repo"):
            print(f"  repo (download docs): {s['repo']}")
        if s.get("role"):
            print(f"  role: {s['role']}")
        out = args.out_dir / key
        try:
            if s["kind"] == "archive_index":
                fetch_archive_index(s["url"], out, args.dry_run)
            elif s["kind"] == "manifest":
                fetch_manifests(s["url"], out, args.dry_run, args.download_images)
        except Exception as exc:  # noqa: BLE001 — report and continue to next source
            print(f"  ERROR fetching {key}: {exc}", file=sys.stderr)
            print(f"  obtain manually from: {s['url']}", file=sys.stderr)

    print("\nNOTE: detector training runs on Kaggle (notebooks/train_yolo_kaggle.ipynb), not here.")


if __name__ == "__main__":
    main()
