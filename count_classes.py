#!/usr/bin/env python
# coding: utf-8

import os
import sys
from collections import defaultdict

from rich.console import Console
from rich.table import Table


def count_images(base_dir):
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)
    for split in ['train', 'val']:
        split_dir = os.path.join(base_dir, split)
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                image_count = len([
                    f for f in os.listdir(class_dir)
                    if os.path.isfile(os.path.join(class_dir, f))
                ])
                if split == 'train':
                    train_counts[class_name] += image_count
                elif split == 'val':
                    val_counts[class_name] += image_count
    return train_counts, val_counts


def main():
    directory_path = sys.argv[1]
    train_counts, val_counts = count_images(directory_path)

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Class")
    table.add_column("Train Count", justify="right")
    table.add_column("Val Count", justify="right")
    table.add_column("Total Count", justify="right")

    for class_name in set(train_counts.keys()).union(val_counts.keys()):
        train_count = train_counts[class_name]
        val_count = val_counts[class_name]
        total_count = train_count + val_count
        table.add_row(class_name, str(train_count), str(val_count),
                      str(total_count))

    console.print(table)


if __name__ == '__main__':
    main()
