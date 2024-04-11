import os
import shutil
from random import shuffle

def split_dataset(folder_path, train_ratio=0.8):
    # List all files in the folder
    all_files = os.listdir(folder_path)
    all_images = [f for f in all_files if os.path.isfile(os.path.join(folder_path, f))]

    # Shuffle the list of images randomly
    shuffle(all_images)

    # Calculate the split index
    split_index = int(len(all_images) * train_ratio)

    # Split the images
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    # Create subfolders if they don't exist
    train_folder = os.path.join(folder_path, 'train')
    val_folder = os.path.join(folder_path, 'val')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Move the images to their respective subfolders
    for img in train_images:
        shutil.move(os.path.join(folder_path, img), os.path.join(train_folder, img))

    for img in val_images:
        shutil.move(os.path.join(folder_path, img), os.path.join(val_folder, img))

# Example usage
folder_path = 'dataset_by_class/dead'
split_dataset(folder_path)
