import os
from PIL import Image
from tqdm import tqdm

def resize_images(directory, target_size=(224, 224)):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_files.append(os.path.join(root, file))

    for file_path in tqdm(image_files, desc="Resizing images"):
        try:
            with Image.open(file_path) as img:
                if img.size != target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                    img.save(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

resize_images('project-5-latest')

