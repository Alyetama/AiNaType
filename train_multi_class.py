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
df = pd.read_parquet('data_hotshot.parquet')

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
for epoch in range(1):  # loop over the dataset multiple times
    for idx, batch in enumerate(tqdm(dataloader)):
        # get the inputs;
        pixel_values, labels = batch

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(
            pixel_values=pixel_values.to(device),
            labels=labels.to(device),
        )

        # calculate gradients
        loss = outputs.loss
        losses.update(loss.item(), pixel_values.size(0))
        loss.backward()

        # optimization step
        optimizer.step()

        if idx % 1000 == 0:
            print('Epoch: [{0}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, loss=losses,))


torch.save(model.state_dict(), 'ainaturalist_model.pt')
torch.save(model, 'ainaturalist_model.pt')
