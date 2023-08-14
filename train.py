import os
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset import Dataset
from diceLoss import DiceLoss
from net import UNet


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 10
EPOCHS = 5
NUM_WORKERS = 8

data_folder = "./data/processed/"

train_data_dir = data_folder + "train/"
validation_data_dir = data_folder + "validation/"
test_data_dir = data_folder + "test/"

train_images_dir = train_data_dir + "images/"
train_masks_dir = train_data_dir + "masks/"
validation_images_dir = validation_data_dir + "images/"
validation_masks_dir = validation_data_dir + "masks/"
test_images_dir = test_data_dir + "images/"
test_masks_dir = test_data_dir + "masks/"


if __name__ == '__main__':

    # Get name of every image (same as corresponding mask name)
    train_images_paths = os.listdir(train_images_dir)
    validation_images_paths = os.listdir(validation_images_dir)
    #train_masks_paths = os.listdir(train_masks_dir)
    #validation_masks_paths = os.listdir(validation_masks_dir)

    # Load both images and masks data
    train_dataset = Dataset(train_images_dir, train_masks_dir, train_images_paths)
    valid_dataset = Dataset(validation_images_dir, validation_masks_dir, validation_images_paths)

    # Create PyTorch DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #Define model as UNet
    model = UNet(n_channels=1, n_classes=1)
    model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss = []
    val_loss = []
    for epoch in range(EPOCHS):
        model.train()
        train_running_loss = 0.0
        counter = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
            for batch in train_dataloader:
                counter += 1
                image = batch['image'].to(DEVICE)
                mask = batch['mask'].to(DEVICE)
                optimizer.zero_grad()
                outputs = model(image)
                outputs = outputs.squeeze(1)
                loss = DiceLoss()(outputs, mask)
                train_running_loss += loss.item()
                loss.backward()
                optimizer.step()
                pbar.update(image.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            train_loss.append(train_running_loss / counter)

        model.eval()
        valid_running_loss = 0.0
        counter = 0
        with torch.no_grad():
            for i, data in enumerate(valid_dataloader):
                counter += 1
                image = data['image'].to(DEVICE)
                mask = data['mask'].to(DEVICE)
                outputs = model(image)
                outputs = outputs.squeeze(1)
                loss = DiceLoss()(outputs, mask)
                valid_running_loss += loss.item()
            val_loss.append(valid_running_loss)

        # Save the trained model
        torch.save({
            'epoch': EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "./unet_model.pth")
