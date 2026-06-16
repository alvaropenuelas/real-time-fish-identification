"""
Obtain MEDFISH101 — 101 Mediterranean fish species, ~69k iNaturalist research-grade
images (Front. Mar. Sci. 2026, doi:10.3389/fmars.2026.1754181).

The paper's Data Availability Statement provides NO packaged dataset download; it points
to the authors' code repository. That repository publishes the canonical image-link export
(columns: image_id, class_name, image_link) pointing at iNaturalist's open-data S3 bucket.
This script downloads that official CSV and then fetches the images from those links. No
dataset URL is invented here — the only network source is the authors' published CSV and
the iNaturalist photo URLs it already contains. Images are CC-licensed research-grade
iNaturalist observations.

Output layout (torchvision ImageFolder-compatible):
    <out-dir>/<Class_Name>/<image_id>.jpg
plus <out-dir>/classes.json (sorted folder names).

Heavy download — intended for Kaggle / local data prep, NOT the inference path.
Resumable: existing files are skipped, so re-running continues where it stopped.

Examples:
    python scripts/get_medfish101.py                      # all 101 species, all images
    python scripts/get_medfish101.py --top-n 30           # 30 most-populated species
    python scripts/get_medfish101.py --max-per-class 200  # cap per class (balance/quick)
    python scripts/get_medfish101.py --limit 100          # tiny smoke download
"""

import argparse
import csv
import io
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Official link export published by the MEDFISH101 authors (NOT a fabricated URL).
DEFAULT_CSV_URL = (
    "https://raw.githubusercontent.com/andrewkof/Mediterranean-Fish-Classification/"
    "main/MEDFISH101/all_image_links_101.csv"
)
REQUEST_TIMEOUT = 30


def _folder(class_name: str) -> str:
    return class_name.strip().replace(" ", "_")


def load_rows(csv_url: str, csv_path: Path | None) -> list[dict]:
    """Read the (image_id, class_name, image_link) rows from a local CSV if given,
    otherwise from the official remote CSV."""
    if csv_path and csv_path.exists():
        text = csv_path.read_text(encoding="utf-8")
    else:
        print(f"Fetching link CSV: {csv_url}")
        resp = requests.get(csv_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        text = resp.text
    reader = csv.DictReader(io.StringIO(text))
    rows = [r for r in reader if r.get("image_link")]
    print(f"{len(rows)} image links across {len({r['class_name'] for r in rows})} species")
    return rows


def select_rows(rows: list[dict], top_n: int, max_per_class: int, limit: int) -> list[dict]:
    counts = Counter(r["class_name"] for r in rows)
    if top_n and top_n > 0:
        keep = {c for c, _ in counts.most_common(top_n)}
        rows = [r for r in rows if r["class_name"] in keep]
        print(f"--top-n {top_n}: kept {len(keep)} species, {len(rows)} images")
    if max_per_class and max_per_class > 0:
        seen: Counter = Counter()
        capped = []
        for r in rows:
            if seen[r["class_name"]] < max_per_class:
                capped.append(r)
                seen[r["class_name"]] += 1
        rows = capped
        print(f"--max-per-class {max_per_class}: {len(rows)} images")
    if limit and limit > 0:
        rows = rows[:limit]
        print(f"--limit {limit}: {len(rows)} images")
    return rows


def _download_one(row: dict, out_dir: Path) -> str:
    dest = out_dir / _folder(row["class_name"]) / f"{row['image_id']}.jpg"
    if dest.exists():
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(row["image_link"], timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return "ok"
    except Exception as exc:  # noqa: BLE001 — log and continue; one bad link must not abort
        print(f"  FAIL {row['image_id']} ({row['class_name']}): {exc}", file=sys.stderr)
        return "fail"


def main():
    parser = argparse.ArgumentParser(description="Download MEDFISH101 from the authors' iNaturalist link export")
    parser.add_argument("--csv-url", default=DEFAULT_CSV_URL, help="Official link-export CSV URL")
    parser.add_argument("--csv", type=Path, default=None, help="Use a local CSV copy instead of downloading it")
    parser.add_argument("--out-dir", type=Path, default=Path("data/medfish101"))
    parser.add_argument("--top-n", type=int, default=0, help="Keep only the N most-populated species (0 = all 101)")
    parser.add_argument("--max-per-class", type=int, default=0, help="Cap images per species (0 = no cap)")
    parser.add_argument("--limit", type=int, default=0, help="Cap total images (0 = all; for a quick smoke run)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download workers")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.csv_url, args.csv)
    rows = select_rows(rows, args.top_n, args.max_per_class, args.limit)

    tally = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_download_one, r, args.out_dir) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            tally[fut.result()] += 1
            if i % 500 == 0:
                print(f"  {i}/{len(rows)} | ok={tally['ok']} skip={tally['skip']} fail={tally['fail']}")

    classes = sorted({_folder(r["class_name"]) for r in rows})
    (args.out_dir / "classes.json").write_text(json.dumps(classes, indent=2))

    print("=== MEDFISH101 download done ===")
    print(f"species: {len(classes)} | downloaded: {tally['ok']} | skipped: {tally['skip']} | failed: {tally['fail']}")
    print(f"images under: {args.out_dir}/<Class_Name>/  | classes.json written")


if __name__ == "__main__":
    main()
