"""Download the Zomato Bangalore dataset from Kaggle into data/raw/zomato.csv."""
import shutil
from pathlib import Path

import kagglehub

from foodiemine import config

if __name__ == "__main__":
    src = Path(kagglehub.dataset_download("himanshupoddar/zomato-bangalore-restaurants")) / "zomato.csv"
    config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, config.RAW_CSV)
    print("Saved", config.RAW_CSV)
