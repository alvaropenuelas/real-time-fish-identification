"""
Downloads Mediterranean fish images from iNaturalist API.
Saves to data/mediterranean/<species_name>/<obs_id>.jpg

Usage:
    python download_med_species.py --per-species 150
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import requests

# Make imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))
from src.species_map import MED_SPECIES

SCRIPT_DIR = Path(__file__).parent

INAT_API = "https://api.inaturalist.org/v1/observations"


def fetch_observations(taxon_id: int, per_page: int, quality: str = "research") -> list:
    params = {
        "taxon_id": taxon_id,
        "quality_grade": quality,
        "photos": "true",
        "per_page": per_page,
        "order_by": "votes",
    }
    r = requests.get(INAT_API, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])


def download_image(url: str, dest: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  skip {dest.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(SCRIPT_DIR / "data/mediterranean"))
    parser.add_argument("--per-species", type=int, default=150)
    parser.add_argument("--quality", default="research")
    args = parser.parse_args()

    base = Path(args.output)

    for name, info in MED_SPECIES.items():
        species_dir = base / name
        species_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(species_dir.glob("*.jpg")))
        if existing >= args.per_species:
            print(f"{name}: {existing} images already present, skipping")
            continue

        print(f"Fetching {name} (taxon {info['taxon_id']})...")
        obs = fetch_observations(info["taxon_id"], args.per_species, args.quality)

        downloaded = 0
        for o in obs:
            photos = o.get("photos", [])
            if not photos:
                continue
            url = photos[0].get("url", "").replace("square", "medium")
            if not url:
                continue
            dest = species_dir / f"{o['id']}.jpg"
            if dest.exists():
                downloaded += 1
                continue
            if download_image(url, dest):
                downloaded += 1
            time.sleep(0.1)

        print(f"  {downloaded} images → {species_dir}")

    print("\nDone. Run: python main.py")


if __name__ == "__main__":
    main()
