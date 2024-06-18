#!/usr/bin/env python
# coding: utf-8

import argparse
import sqlite3
import time
from glob import glob
from pathlib import Path

from loguru import logger
from tqdm import tqdm
from ultralytics import YOLO


def opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--db-path',
                        help='Path to the sqlite database file',
                        type=str,
                        required=True)
    parser.add_argument('-m',
                        '--model-path',
                        help='Path to the known_unknown model file',
                        type=str,
                        required=True)
    parser.add_argument('-i',
                        '--images-dir',
                        help='Path to the images directory',
                        type=str,
                        required=True)
    return parser.parse_args()


def run(db_path: str, model_path: str, images_dir: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    table_prefix = 'ainatype'

    cursor.execute(f'''CREATE TABLE IF NOT EXISTS {table_prefix}_detections
                     (image TEXT PRIMARY KEY,
                      dead_conf REAL,
                      live_animal_conf REAL,
                      other_conf REAL,
                      scat_conf REAL,
                      tracks_conf REAL)''')

    cursor.execute(f'''CREATE TABLE IF NOT EXISTS {table_prefix}_errored
                     (image TEXT PRIMARY KEY)''')

    model = YOLO(model_path)
    imgs = glob(f'{images_dir}/*')
    len_1 = len(imgs)

    existing_images = set(row[0] for row in cursor.execute(
        f'SELECT image FROM {table_prefix}_detections'))
    imgs = [x for x in imgs if Path(x).name not in existing_images]
    len_2 = len(imgs)

    logger.info(f'Excluded {len_1 - len_2} existing images.')
    time.sleep(2)

    batch_size = 1000
    batch_count = 0

    for img in tqdm(imgs):
        try:
            res = model(img)[0]
            probs = [Path(img).name
                     ] + [round(x, 2)
                          for x in res.probs.data.tolist()]
            cursor.execute(
                f'INSERT INTO {table_prefix}_detections VALUES (?, ?, ?, ?, ?, ?)', probs)

            batch_count += 1
            if batch_count % batch_size == 0:
                conn.commit()  # Commit every 1000 iterations
                logger.info(f'Committed batch {batch_count // batch_size}')

        except Exception as e:
            logger.error(e)
            cursor.execute(f'INSERT INTO {table_prefix}_errored VALUES (?)',
                           (Path(img).name, ))

    conn.commit()
    conn.close()


def main() -> None:
    args = opts()
    run(args.db_path, args.model_path, args.images_dir)


if __name__ == '__main__':
    main()

