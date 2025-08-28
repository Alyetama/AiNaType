#!/usr/bin/env python
# coding: utf-8

import argparse
import hashlib
import os

import cv2
from PIL import Image
from tqdm import tqdm


def process_images(base_dir):
    file_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                file_paths.append(os.path.join(root, file))

    for file_path in tqdm(file_paths, desc="Processing images"):
        try:
            img = cv2.imread(file_path)

            if img is None:
                print(f"Corrupted image detected: {file_path}")
                continue

            if file_path.lower().endswith('.png'):
                new_file_path = file_path.rsplit('.', 1)[0] + '.jpg'
                cv2.imwrite(new_file_path, img,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                os.remove(file_path)
            else:
                cv2.imwrite(file_path, img)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def resize_images(directory, target_size=(224, 224)):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    for file_path in tqdm(image_files, desc="Resizing images"):
        try:
            with Image.open(file_path) as img:
                if img.size != target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                    img.save(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def calculate_hash(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    return hashlib.md5(img).hexdigest()


def find_and_remove_duplicate_images(base_dir):
    image_hashes = {}
    duplicates = []

    file_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(root, file))

    for file_path in tqdm(file_paths, desc='Checking for duplicates'):
        try:
            image_hash = calculate_hash(file_path)
            if image_hash in image_hashes:
                os.remove(file_path)
                duplicates.append((image_hashes[image_hash][0], file_path))
            else:
                image_hashes[image_hash] = [file_path]
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return duplicates


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess images in a directory.')
    parser.add_argument('-d',
                        '--directory',
                        required=True,
                        help='Path to the directory containing images.')
    args = parser.parse_args()

    directory_path = args.directory
    process_images(directory_path)
    resize_images(directory_path)
    duplicates = find_and_remove_duplicate_images(directory_path)

    if duplicates:
        print("Duplicate images found and removed:")
        for original, duplicate in duplicates:
            print(f"Original: {original}, Duplicate: {duplicate}")
    else:
        print("No duplicate images found.")


if __name__ == '__main__':
    main()
