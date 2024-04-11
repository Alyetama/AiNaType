import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import AutoImageProcessor, AutoModelForImageClassification

DATASET_PATH = 'wd_lab/ainaturalistype/data_hotshot.parquet'
IMAGES_ROOT_DIR = 'wd_lab/ainaturalistype/multi_classes_images'
PATIENCE = 3
MAX_EPOCHS = 10
BATCH_SIZE = 4

# Load the dataset
df = pd.read_parquet(DATASET_PATH)
labels = list(df.columns)[3:]
id2label = {id: label for id, label in enumerate(labels)}


# Define the MultiLabelDataset
class MultiLabelDataset(Dataset):

    def __init__(self, root, df, transform):
        self.root = root
        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        image_path = os.path.join(self.root, item['image_local_path'])
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.transform(image)
        labels = item[3:].values.astype(np.float32)
        labels = torch.from_numpy(labels)
        return pixel_values, labels

    def __len__(self):
        return len(self.df)


# Setup the transformation
model_id = 'google/siglip-so400m-patch14-384'
processor = AutoImageProcessor.from_pretrained(model_id)
size = processor.size['height']
mean = processor.image_mean
std = processor.image_std
transform = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=mean, std=std),
])


# Define the LightningDataModule
class MultiLabelDataModule(pl.LightningDataModule):

    def __init__(self, df, batch_size=32, root_dir=IMAGES_ROOT_DIR):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.root_dir = root_dir

    def setup(self, stage=None):
        # Split df into train/val (assuming an 80/20 split)
        train_size = int(0.8 * len(self.df))
        val_size = len(self.df) - train_size
        self.train_dataset, self.val_dataset = random_split(
            self.df, [train_size, val_size])

        self.train_dataset = MultiLabelDataset(root=self.root_dir,
                                               df=self.train_dataset.dataset,
                                               transform=transform)
        self.val_dataset = MultiLabelDataset(root=self.root_dir,
                                             df=self.val_dataset.dataset,
                                             transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=32)

    def val_dataloader(self):  # Add the validation dataloader
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=32)


# Define the LightningModule
class MultiLabelClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # Optimize for Tensor Cores
        torch.set_float32_matmul_precision('medium')

        self.model = AutoModelForImageClassification.from_pretrained(
            model_id,
            problem_type='multi_label_classification',
            id2label=id2label)

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        self.log('val_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)


# Training script
def train_model():
    data_module = MultiLabelDataModule(df, batch_size=BATCH_SIZE)
    model = MultiLabelClassifier()
    checkpoint_callback = ModelCheckpoint(monitor='train_loss',
                                          dirpath='model_checkpoints',
                                          filename='best_model',
                                          save_top_k=1,
                                          mode='min')
    early_stopping = EarlyStopping(monitor='train_loss',
                                   patience=PATIENCE,
                                   mode='min')
    logger = TensorBoardLogger('tb_logs', name='multi_label_classifier')
    wandb_logger = WandbLogger(log_model="all")

    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         max_epochs=MAX_EPOCHS,
                         callbacks=[checkpoint_callback, early_stopping],
                         logger=[logger, wandb_logger])
    trainer.fit(model, data_module)


train_model()
