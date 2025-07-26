import requests
from pathlib import Path
from src.constants import URL, DATA_PATH
from src.logging_config import logger


def download_data(url: str = URL, data_path: Path = DATA_PATH):
    try:
        logger.info(f"Downloading file from url : {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(data_path, "wb") as f:
            f.write(response.content)
        logger.info(f"File downloaded successfully at path : {data_path}")
    except Exception as e:
        logger.error(f"Failed to download file from url : {e}")
