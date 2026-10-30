"""Download the public forum dataset without accessing Kaggle credentials."""

import argparse
import csv
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATASET = "prvsiyan/prvsiyans-extended-name-that-llm-dataset"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET}"
DEFAULT_VERSION = 7
DEFAULT_DATA_DIR = Path(f"data/raw/extended-v{DEFAULT_VERSION}")
REQUIRED_FILES = {"passages.csv", "passages_clean.csv", "README.md"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_archive(archive: Path, destination: Path) -> None:
    """Validate all paths before extracting any member of an external archive."""
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or stat.S_ISLNK(mode)
            ):
                raise ValueError(f"Unsafe archive path: {member.filename!r}")
        source.extractall(destination)


def verify_download(destination: Path, version: int) -> dict:
    manifest_path = destination / "download_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{destination} already exists without a download manifest. Use --output.")
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset"] != DATASET or manifest["version"] != version:
        raise ValueError("Existing download has a different source or version. Use --output.")
    files = manifest["files_sha256"]
    if not REQUIRED_FILES.issubset(files):
        raise ValueError("Download manifest is missing required files. Use a new --output.")
    for name, digest in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"Unsafe manifest path: {name!r}")
        path = destination / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing or modified downloaded file: {path}. Use a new --output.")
    return manifest


def download_dataset(destination: Path, version: int = DEFAULT_VERSION) -> dict:
    if version < 1:
        raise ValueError("Dataset version must be positive.")
    if destination.exists():
        return verify_download(destination, version)

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}?datasetVersionNumber={version}"
    )
    # This public endpoint requires no token, cookies, or local credential files.
    request = Request(url, headers={"User-Agent": "name-that-llm-baseline/0.1"})
    with tempfile.TemporaryDirectory(prefix=".extended-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / "dataset.zip"
        with urlopen(request, timeout=120) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)
        if not zipfile.is_zipfile(archive):
            raise ValueError(
                "Kaggle did not return a ZIP archive. Check the dataset URL in README.md."
            )
        payload = temporary / "payload"
        extract_archive(archive, payload)
        missing = REQUIRED_FILES - {path.name for path in payload.iterdir() if path.is_file()}
        if missing:
            raise ValueError(f"Expanded dataset is missing required files: {sorted(missing)}")
        with (payload / "passages_clean.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"text", "model", "prompt_id", "split"}.issubset(reader.fieldnames or []):
                raise ValueError("Unexpected expanded-dataset CSV schema.")
            clean_rows = sum(1 for _ in reader)
        if not clean_rows:
            raise ValueError("The expanded dataset contains no clean passages.")
        manifest = {
            "dataset": DATASET,
            "dataset_url": DATASET_URL,
            "version": version,
            "download_url": url,
            "downloaded_at_utc": datetime.now(UTC).isoformat(),
            "archive_sha256": sha256(archive),
            "clean_rows": clean_rows,
            "files_sha256": {
                path.relative_to(payload).as_posix(): sha256(path)
                for path in sorted(payload.rglob("*"))
                if path.is_file()
            },
        }
        (payload / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        payload.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", type=int, default=DEFAULT_VERSION, help="Kaggle dataset version"
    )
    parser.add_argument("--output", type=Path, help="Default: data/raw/extended-v<VERSION>")
    args = parser.parse_args()
    output = args.output or Path(f"data/raw/extended-v{args.version}")
    try:
        manifest = download_dataset(output, args.version)
    except HTTPError as error:
        parser.exit(1, f"Kaggle download failed (HTTP {error.code}). Check {DATASET_URL}\n")
    except (URLError, TimeoutError, OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Download failed: {error}\n")
    print(f"Verified {DATASET}, version {manifest['version']}: {output}")
    print(f"Clean passages before baseline split filtering: {manifest['clean_rows']:,}")


if __name__ == "__main__":
    main()
