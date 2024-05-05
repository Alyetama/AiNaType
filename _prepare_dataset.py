#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import random
import shutil
from glob import glob
from pathlib import Path

from tqdm import tqdm


def opts() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data-file-path',
                        help='The MIN JSON data file path',
                        type=str,
                        required=True)
    parser.add_argument('-f',
                        '--full-data-file-path',
                        help='The FULL JSON data file path',
                        type=str,
                        required=True)
    parser.add_argument('-s',
                        '--source-images-dir',
                        help='The source dir with inat photos',
                        type=str,
                        default='images')
    parser.add_argument('-o',
                        '--output-dir',
                        help='The output project dir',
                        type=str)
    parser.add_argument('--init-dataset',
                        action='store_true',
                        help='Copy the inat pre-annotated signs dataset')
    return parser.parse_args()


def main():
    """
    Main function to prepare the classification dataset.

    The function reads the MIN JSON data file, crops the images based on the
    bounding boxes, and saves the cropped images in the respective class
    folders.
    """
    args = opts()

    classify_by = 'choice'

    data_file = args.data_file_path
    #dataset_path = Path(data_file).stem

    classes = ['scat', 'tracks', 'dead', 'live_animal', 'other']

    with open(data_file) as j:
        data = json.load(j)

    with open(args.full_data_file_path) as fj:
        full_data = json.load(fj)
    live_animal = [x for x in full_data if x.get('cancelled_annotations') > 0]

    for t in live_animal:
        task = {
            'id': t['id'],
            'scientific_name': t['data'].get('scientific_name'),
            'observation_url': t['data'].get('observation_url'),
            'image': t['data']['image'],
            'choice': 'live_animal',
            'annotator': t['annotations'][0]['completed_by'],
            'annotation_id': t['annotations'][0]['id'],
            'created_at': t['annotations'][0]['created_at'],
            'updated_at': t['annotations'][0]['updated_at'],
            'lead_time': t['annotations'][0]['lead_time'],
        }
        data.append(task)

    data = [
        x for x in data if x.get('choice') not in
        ['exclude', 'not_loading', 'low_conf', 'tmp', None]
    ]

    proj_name = args.output_dir

    for c in classes:
        Path(f'{proj_name}/{c}').mkdir(exist_ok=True, parents=True)

    all_images = glob(f'{proj_name}/**/**/*')
    all_images = [Path(x).name for x in all_images]

    n_images = 0

    for d in tqdm(data):
        if not d.get(classify_by):
            continue

        image = d['image']
        image_name = Path(image).parent.name + Path(image).suffix.lower()
        if image_name in all_images:
            continue
        n_images += 1

        image_relative_path = f'images/{image_name}'

        _cls = d.get(classify_by)
        if isinstance(_cls, dict):
            _cls = _cls['choices'][0]

        if _cls == 'bone':
            _cls = 'dead'

        if not Path(image_relative_path).exists():
            print(f'Image does not exist! {image}; {image_relative_path}')
            continue

        img_name = Path(image_relative_path).name
        if Path(f'{proj_name}/{_cls}/{img_name}').exists():
            continue
        shutil.copy2(image_relative_path, f'{proj_name}/{_cls}/{img_name}')

    if args.init_dataset:
        const_dataset = glob('dataset_by_classes/*')

        for class_name in tqdm(const_dataset):
            class_images = glob(f'{class_name}/*')
            for img_path in tqdm(class_images):
                shutil.copy2(
                    img_path,
                    f'{proj_name}/{Path(class_name).name}/{Path(img_path).name}'
                )

    print('Number of new images:', n_images)

    train_path = os.path.join(proj_name, 'train')
    val_path = os.path.join(proj_name, 'val')

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    for class_name in classes:
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_path, class_name), exist_ok=True)

    for class_name in classes:
        images = glob(os.path.join(proj_name, class_name, '*'))
        random.shuffle(images)
        split_index = int(0.8 * len(images))
        train_images = images[:split_index]
        val_images = images[split_index:]

        for img_path in tqdm(train_images):
            shutil.copy2(img_path, os.path.join(train_path, class_name))

        for img_path in tqdm(val_images):
            shutil.copy2(img_path, os.path.join(val_path, class_name))

        shutil.rmtree(os.path.join(proj_name, class_name))
        Path(f'{proj_name}/train.cache').unlink(missing_ok=True)
        Path(f'{proj_name}/val.cache').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
