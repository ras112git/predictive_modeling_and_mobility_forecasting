"""Downloads the raw dataset from Google Drive."""

import os
import gdown

# Google Drive file ID (from the shared link)
FILE_ID = "1Lz8JSqtcUT9mBfTz0rOf9KkmRSxf5k5t"
RAW_DATA_PATH = "data/raw/dataset.csv"

def download_raw_data(force: bool = False) -> str:
    """
    Download the raw dataset from Google Drive into data/raw/.

    Args:
        force: If True, re-download even if the file already exists.

    Returns:
        The path to the downloaded file.
    """
    os.makedirs("data/raw", exist_ok=True)

    if os.path.exists(RAW_DATA_PATH) and not force:
        print(f"Raw data already exists at {RAW_DATA_PATH}, skipping download.")
        return RAW_DATA_PATH

    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print(f"Downloading raw data from Google Drive...")
    gdown.download(url, RAW_DATA_PATH, quiet=False)
    print(f"Saved to {RAW_DATA_PATH}")

    return RAW_DATA_PATH


if __name__ == "__main__":
    download_raw_data()