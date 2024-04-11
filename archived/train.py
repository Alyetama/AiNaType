import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load dataset
df = pd.read_parquet('wd_lab/ainaturalistype/data_hotshot.parquet')

# Prepare label dictionary
labels = df.columns[3:]
id2label = {id: label for id, label in enumerate(labels)}

# Initialize model and processor
model_id = "google/siglip-so400m-patch14-384"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(
    model_id, problem_type="multi_label_classification", id2label=id2label)


# Define dataset
class MultiLabelDataset(Dataset):

    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image_path = os.path.join(self.root_dir, item['image_local_path'])
        image = Image.open(image_path).convert('RGB')
        labels = item[3:].values.astype(np.float32)
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(labels)
        return image, labels


# Define transformations
size = processor.size["height"]
transform = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std),
])

# Initialize dataset and dataloader
dataset = MultiLabelDataset(df, 'multi_classes_images', transform=transform)
dataloader = DataLoader(dataset,
                        batch_size=3,
                        shuffle=True,
                        collate_fn=lambda x: tuple(zip(*x)))

# Prepare the model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(1):
    for batch in tqdm(dataloader):
        pixel_values, labels = map(torch.stack,
                                   zip(*batch))  # Reconstruct batch and stack
        pixel_values, labels = pixel_values.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()


torch.save(model.state_dict(), 'ainaturalist_model.pt')
torch.save(model, 'ainaturalist_model.pt')

# image = Image.open('<img_path>')

# # Set the model to evaluation mode
# model.eval()

# # Prepare image for the model using the processor and move to device
# pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

# # Forward pass without computing gradients
# with torch.no_grad():
#     logits = model(pixel_values).logits

# # Convert logits to probabilities using sigmoid function
# probs = torch.sigmoid(logits.squeeze()).cpu()

# # Identify labels with probabilities above the threshold (e.g., 50%)
# threshold = 0.5
# #predicted_labels = [id2label[idx] for idx, prob in enumerate(probs) if prob > threshold]
# predictions = np.zeros(probs.shape)
# predictions[np.where(probs >= threshold)] = 1 # turn predicted id's into actual label names
# predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
# print(predicted_labels)
