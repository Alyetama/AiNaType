#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def mutually_exclusive_df(df):
    mask = df[['tracks', 'scat', 'bone', 'dead']].notna().any(axis=1)
    df = df[mask]
    mutually_exclusive_mask = (
        ((df['tracks'].notna())
         & ~(df['scat'].notna() | df['bone'].notna() | df['dead'].notna())) |
        ((df['scat'].notna())
         & ~(df['tracks'].notna() | df['bone'].notna() | df['dead'].notna())) |
        ((df['bone'].notna())
         & ~(df['tracks'].notna() | df['scat'].notna() | df['dead'].notna())) |
        ((df['dead'].notna())
         & ~(df['tracks'].notna() | df['scat'].notna() | df['bone'].notna())))
    return df[mutually_exclusive_mask]


def opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dataset',
                        help='Path to the dataset',
                        type=str,
                        required=True)
    parser.add_argument(
        '-s',
        '--source-path',
        help='Path to the source folder where all images are downloaded',
        type=str,
        required=True)
    parser.add_argument('-o',
                        '--output-path',
                        help='Path to the output folder',
                        type=str,
                        required=True)
    parser.add_argument('-c',
                        '--classes-file',
                        help='Path to the classes list JSON file',
                        type=str,
                        required=True)
    return parser.parse_args()


def main():
    args = opts()

    df = pd.read_parquet(args.dataset)
    df = mutually_exclusive_df(df)

    with open(args.classes_file) as j:
        classes = json.load(j)

    for c in classes:
        Path(f'{args.output_path}/{c}').mkdir(exist_ok=True, parents=True)

    not_exists = 0

    for sign in ['tracks', 'scat', 'bone', 'dead']:
        value = sign.capitalize()
        for x in tqdm(df[df[sign] == value]['image_url'], desc=sign):
            if not x:
                continue
            img_name = f'{Path(x).parent.name}{Path(x).suffix}'
            img_path = f'{args.source_path}/{img_name}'
            if not Path(img_path).exists():
                not_exists += 1
                continue
            else:
                shutil.copy2(img_path, f'{args.output_path}/{sign}')

    print('Number of images that were not in source folder:', not_exists)


if __name__ == '__main__':
    main()

