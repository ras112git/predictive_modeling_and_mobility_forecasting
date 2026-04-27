"""Downloads the raw datasets from Google Drive."""

import os
import gdown

# Map of output filename -> Google Drive file ID (from the shared link)
FILES = {
    "dataset_train.csv": "13p_DwfnywGNLFgovm_RWRs-ZnKsEjPdc",
    "dataset_test.csv": "1xTFvlyL-te42wvR3s-NN02qNGtz1VqOQ",
}
RAW_DATA_DIR = "data/raw"


def download_raw_data(force: bool = False) -> list[str]:
    """
    Download the raw datasets from Google Drive into data/raw/.

    Args:
        force: If True, re-download even if a file already exists.

    Returns:
        The paths to the downloaded files.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    paths = []
    for filename, file_id in FILES.items():
        path = os.path.join(RAW_DATA_DIR, filename)

        if os.path.exists(path) and not force:
            print(f"Raw data already exists at {path}, skipping download.")
            paths.append(path)
            continue

        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename} from Google Drive...")
        gdown.download(url, path, quiet=False)
        print(f"Saved to {path}")
        paths.append(path)

    return paths


if __name__ == "__main__":
    download_raw_data()