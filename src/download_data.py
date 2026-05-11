"""Downloads the raw datasets from the Kaggle competition."""

import os
import shutil
from pathlib import Path

import kagglehub

COMPETITION = "bda-test"
RAW_DATA_DIR = "data/raw"

# Maps the local target filename -> a substring used to find the matching
# file inside the Kaggle competition download.
FILES = {
    "dataset_train.csv": "train",
    "dataset_test.csv": "test",
}


def _find_source_file(source_dir: Path, name_substring: str) -> Path:
    """Pick the .csv in source_dir whose filename contains name_substring."""
    matches = [
        p for p in source_dir.glob("*.csv")
        if name_substring.lower() in p.name.lower()
    ]
    if not matches:
        available = sorted(p.name for p in source_dir.iterdir())
        raise FileNotFoundError(
            f"No CSV containing {name_substring!r} found in {source_dir}. "
            f"Available files: {available}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple CSVs match {name_substring!r} in {source_dir}: "
            f"{[p.name for p in matches]}"
        )
    return matches[0]


def download_raw_data(force: bool = False) -> tuple[str, str]:
    """
    Download the raw datasets from the Kaggle competition into data/raw/.

    Uses kagglehub to fetch the competition files, then copies the train and
    test CSVs into data/raw/ under the canonical names (dataset_train.csv,
    dataset_test.csv) so downstream code can use stable relative paths.

    Args:
        force: If True, re-download and re-copy even if the files already exist.

    Returns:
        (train_path, test_path) as strings, both pointing inside data/raw/.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    target_paths = {name: os.path.join(RAW_DATA_DIR, name) for name in FILES}

    if not force and all(os.path.exists(p) for p in target_paths.values()):
        for p in target_paths.values():
            print(f"Raw data already exists at {p}, skipping download.")
        return target_paths["dataset_train.csv"], target_paths["dataset_test.csv"]

    print(f"Downloading competition '{COMPETITION}' from Kaggle...")
    source_dir = Path(kagglehub.competition_download(COMPETITION))
    print(f"Kaggle files cached at: {source_dir}")

    for target_name, name_substring in FILES.items():
        source = _find_source_file(source_dir, name_substring)
        target = target_paths[target_name]
        shutil.copyfile(source, target)
        print(f"Copied {source.name} -> {target}")

    return target_paths["dataset_train.csv"], target_paths["dataset_test.csv"]


def upload_to_colab() -> tuple[str, str]:
    """
    Upload local dataset_train.csv and dataset_test.csv into a Colab runtime.

    Intended for sessions that connect to a Google Colab kernel (e.g. from
    VSCode) without Kaggle credentials available. Opens the Colab file picker,
    then writes the selected files into data/raw/ on the runtime under the
    canonical names so downstream code can use the same relative paths.

    The picker accepts both files at once; the train/test role is inferred
    from a 'train'/'test' substring in each uploaded filename.

    Returns:
        (train_path, test_path) as strings, both pointing inside data/raw/.
    """
    try:
        from google.colab import files  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "upload_to_colab() must be called from a Google Colab runtime."
        ) from e

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print(
        "Select the two CSVs to upload: one whose name contains 'train' and "
        "one whose name contains 'test'."
    )
    uploaded = files.upload()  # dict: filename -> bytes

    target_paths: dict[str, str] = {}
    for target_name, name_substring in FILES.items():
        matches = [fn for fn in uploaded if name_substring.lower() in fn.lower()]
        if not matches:
            raise FileNotFoundError(
                f"No uploaded file containing {name_substring!r}. "
                f"Uploaded: {list(uploaded)}"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple uploaded files match {name_substring!r}: {matches}"
            )
        source_name = matches[0]
        target = os.path.join(RAW_DATA_DIR, target_name)
        with open(target, "wb") as f:
            f.write(uploaded[source_name])
        target_paths[target_name] = target
        print(f"Saved {source_name} -> {target}")

    return target_paths["dataset_train.csv"], target_paths["dataset_test.csv"]


if __name__ == "__main__":
    download_raw_data()
