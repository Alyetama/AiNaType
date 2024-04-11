#!/usr/bin/env python
# coding: utf-8

import pandas as pd

df = pd.read_parquet('wd_lab/ainaturalistype/data_hotshot.parquet')

df.head()

len(df)

df.iloc[62].values

df.iloc[0][3:].values

labels = list(df.columns)[3:]
labels

id2label = {id: label for id, label in enumerate(labels)}
id2label

from transformers import AutoImageProcessor, AutoModelForImageClassification

model_id = "google/siglip-so400m-patch14-384"

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(
    model_id, problem_type="multi_label_classification", id2label=id2label)

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
class MultiLabelDataset(Dataset):

    def __init__(self, root, df, transform):
        self.root = root
        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        # get image
        image_path = os.path.join(self.root, item["image_local_path"])
        image = Image.open(image_path).convert("RGB")

        # prepare image for the model
        pixel_values = self.transform(image)

        # get labels
        labels = item[3:].values.astype(np.float32)

        # turn into PyTorch tensor
        labels = torch.from_numpy(labels)

        return pixel_values, labels

    def __len__(self):
        return len(self.df)


from torchvision.transforms import Compose, Normalize, Resize, ToTensor

# get appropriate size, mean and std based on the image processor
size = processor.size["height"]
mean = processor.image_mean
std = processor.image_std

transform = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=mean, std=std),
])

train_dataset = MultiLabelDataset(root='wd_lab/ainaturalistype/multi_classes_images',
                                  df=df,
                                  transform=transform)

pixel_values, labels = train_dataset[63]
print(pixel_values.shape)

unnormalized_image = (pixel_values.numpy() *
                      np.array(std)[:, None, None]) + np.array(mean)[:, None,
                                                                     None]
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
Image.fromarray(unnormalized_image)

labels

[torch.nonzero(labels).squeeze().tolist()]

[id2label[label] for label in [torch.nonzero(labels).squeeze().tolist()]]

from torch.utils.data import DataLoader
def collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    target = torch.stack([item[1] for item in batch])
    return data, target
train_dataloader = DataLoader(train_dataset,
                              collate_fn=collate_fn,
                              batch_size=3,
                              shuffle=True)

batch = next(iter(train_dataloader))

print(batch[0].shape)
print(batch[1].shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

model = model.to(device)

outputs = model(pixel_values=batch[0].to(device), labels=batch[1].to(device))

outputs.loss


# handy utility I found at https://github.com/wenwei202/pytorch-examples/blob/ecbb7beb0fac13133c0b09ef980caf002969d315/imagenet/main.py#L296
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
from torch.optim import AdamW
from tqdm.auto import tqdm

# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

losses = AverageMeter()

model.train()
for epoch in range(10):  # loop over the dataset multiple times
    for idx, batch in enumerate(tqdm(train_dataloader)):
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
                      epoch,
                      loss=losses,
                  ))

torch.save(model, 'ainaturalist_model.pt')
torch.save(model.state_dict(), 'ainaturalist_model_state_dict.pt')
