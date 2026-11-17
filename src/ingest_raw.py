"""Module 02 - Raw ingestion.

Downloads the Elliptic Data Set from Kaggle into data/raw/elliptic/, extracts it
untouched and records a manifest (source, license, size and sha256 of each file).
Running it again does nothing if everything on disk still matches the manifest.

Flow: verify -> (download -> extract -> write manifest -> verify) -> done.

Run:  docker compose run --rm dev uv run python src/ingest_raw.py

The Kaggle token is never read here: the `kaggle` command picks it up from the
KAGGLE_API_TOKEN environment variable, which docker compose passes in from .env.
"""

import hashlib
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

# --- Settings ----------------------------------------------------------------

# Fixed on purpose: "elliptic2-data-set" is a different dataset of about 25 GB.
DATASET = "ellipticco/elliptic-data-set"
LICENSE = "CC BY-NC-ND 4.0"  # the data (raw or derived) must not be redistributed

RAW_DIR = Path("data/raw/elliptic")  # relative to the project root (/workspace)
ZIP_PATH = RAW_DIR / "elliptic-data-set.zip"
MANIFEST_PATH = RAW_DIR / "manifest.json"

# Files the ZIP must contain, as paths relative to RAW_DIR.
EXPECTED_FILES = [
    "elliptic_bitcoin_dataset/elliptic_txs_classes.csv",
    "elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv",
    "elliptic_bitcoin_dataset/elliptic_txs_features.csv",
]


# --- Checksums ---------------------------------------------------------------


def sha256(path):
    """Return the SHA-256 of a file as a hex string."""
    # Read in 1 MiB chunks so the 658 MB features file never sits in memory.
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path):
    """Describe one file for the manifest: relative path, size and checksum."""
    return {
        "path": str(path.relative_to(RAW_DIR)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


# --- Download and extraction -------------------------------------------------


def download():
    """Download the dataset ZIP with the Kaggle CLI."""
    # --force: without it the CLI skips a ZIP that already exists, even a corrupt
    # one. We only get here after verify() failed, so we want a fresh copy.
    # The command is a list, not a string, so no shell is involved.
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-p",
            str(RAW_DIR),
            "--force",
        ],
        check=True,
    )


def extract():
    """Extract the ZIP as-is and check that the three expected files are there."""
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(RAW_DIR)
    missing = [name for name in EXPECTED_FILES if not (RAW_DIR / name).exists()]
    if missing:
        raise SystemExit(f"Missing files after extraction: {missing}")


# --- Manifest ----------------------------------------------------------------


def write_manifest():
    """Record where the data came from and the checksum of every file."""
    manifest = {
        "dataset": DATASET,
        "url": f"https://www.kaggle.com/datasets/{DATASET}",
        "license": LICENSE,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": [file_info(ZIP_PATH)]
        + [file_info(RAW_DIR / name) for name in EXPECTED_FILES],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")


def verify():
    """Return True only if the manifest exists and every file in it matches."""
    # First run: there is no manifest yet, so there is nothing to trust.
    if not MANIFEST_PATH.exists():
        return False
    manifest = json.loads(MANIFEST_PATH.read_text())
    recorded = {info["path"]: info["sha256"] for info in manifest["files"]}
    # The manifest must cover every file we expect, not just some of them.
    if not all(name in recorded for name in EXPECTED_FILES):
        return False
    # Missing or changed on disk (deleted, corrupted, edited): not intact.
    for relative, expected_hash in recorded.items():
        path = RAW_DIR / relative
        if not path.exists() or sha256(path) != expected_hash:
            return False
    return True


# --- Entry point -------------------------------------------------------------


def main():
    """Skip if the Raw layer is intact; otherwise download, extract and record it."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if verify():
        print("Raw layer is already up to date; nothing to download.")
        return
    print(f"Downloading {DATASET} ...")
    download()
    extract()
    write_manifest()
    # Check again, so a bad download fails here and not later in the pipeline.
    if not verify():
        raise SystemExit("Verification failed right after download.")
    print(f"Done. Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
