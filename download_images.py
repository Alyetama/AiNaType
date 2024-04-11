#!/usr/bin/env python
# coding: utf-8

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from loguru import logger


def download_image(url, filename, _id):
    """Downloads an image, handling common errors gracefully. Returns ID on failure."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return None  # Indicating success
        else:
            logger.error(f'Download failed ({response.status_code}): {url}')
    except requests.RequestException as e:
        logger.error(f'Connection error for {url}: {e}')
    return _id  # Return the ID on failure


def process_chunk(urls, download_folder):
    """Processes a chunk of URLs synchronously using a ThreadPool for concurrency and appends failed downloads to a JSON file."""
    try:
        with open('failed_downloads.json', 'r') as f:
            failed_downloads = json.load(f)
    except FileNotFoundError:
        failed_downloads = []

    with ThreadPoolExecutor() as executor:
        futures = {}
        for _id, url in urls:
            url = url.replace('medium', 'original')
            suffix = Path(url).suffix.lower()
            if suffix not in ['.jpg', '.jpeg', '.png']:
                continue

            filename = os.path.join(download_folder, f'{_id}{suffix}')
            futures[executor.submit(download_image, url, filename, _id)] = _id

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                failed_downloads.append(result)

    with open('failed_downloads.json', 'w') as f:
        json.dump(failed_downloads, f)


if __name__ == "__main__":
    logger.add('ainaturalistype.log')

    df = pd.read_parquet('data_filtered_by_license.parquet')
    df = df.dropna(subset=['image_url'])
    obsv_urls = list(zip(df['photo_id'], df['image_url']))

    download_folder = "images"
    Path(download_folder).mkdir(exist_ok=True)

    chunk_size = 100
    total_processed = 0

    for i in range(0, len(obsv_urls), chunk_size):
        chunk = obsv_urls[i:i + chunk_size]
        process_chunk(chunk, download_folder)

        total_processed += len(chunk)
        logger.info(f"Processed {total_processed} items so far")
