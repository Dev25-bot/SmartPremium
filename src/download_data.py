"""
Download the Insurance Premium Prediction dataset (public Kaggle dataset
"schran/insurance-premium-prediction") into data/insurance_premium_dataset.csv.

Usage:  python -m src.download_data
"""
import io
import zipfile
from pathlib import Path

import requests

URL = "https://www.kaggle.com/api/v1/datasets/download/schran/insurance-premium-prediction"
OUT_PATH = Path("data/insurance_premium_dataset.csv")


def download(out_path: Path = OUT_PATH) -> Path:
    if out_path.exists():
        print(f"Already present: {out_path}")
        return out_path
    print(f"Downloading {URL}")
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(zf.read(csv_name))
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    download()
