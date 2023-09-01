import os
from tqdm import tqdm

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from dataset import Dataset
from diceLoss import DiceLoss
from net import UNet

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
EPOCHS = 5

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
    # train_masks_paths = os.listdir(train_masks_dir)
    # validation_masks_paths = os.listdir(validation_masks_dir)

    # Load both images and masks data
    train_dataset = Dataset(train_images_dir, train_masks_dir, train_images_paths)
    valid_dataset = Dataset(validation_images_dir, validation_masks_dir, validation_images_paths)

    # Create PyTorch DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define model as UNet
    model = UNet(n_channels=1, n_classes=2)
    model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fun = nn.CrossEntropyLoss()  # DiceLoss()
    prev_epoch = 0
    total_epochs = EPOCHS

    # Load recent train progress
    # checkpoint = torch.load("./unet_model_s_prev.pth")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # prev_epoch = checkpoint['epoch'] + 1
    # total_epochs = prev_epoch + EPOCHS

    train_loss = []
    val_loss = []
    for epoch in range(prev_epoch, total_epochs):
        model.train()
        train_running_loss = 0.0
        counter = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{total_epochs}', unit='img') as tpbar:
            for batch in train_dataloader:
                counter += 1
                image = batch['image'].to(DEVICE)
                mask = torch.where(batch['mask'].to(DEVICE) > 0., 1, 0)  # binarize mask
                # mask = batch['mask'].long().to(DEVICE)
                optimizer.zero_grad()
                outputs = model(image)
                # outputs = outputs.squeeze(1)  # diceLoss
                loss_val = loss_fun(outputs, mask)
                train_running_loss += loss_val.item()
                loss_val.backward()
                optimizer.step()
                tpbar.update(image.shape[0])
                tpbar.set_postfix(**{'loss (batch)': loss_val.item()})
            train_loss.append(train_running_loss / counter)
            print("Train_loss: {0}".format(train_running_loss / counter))

        model.eval()
        valid_running_loss = 0.0
        counter = 0
        with torch.no_grad():
            with tqdm(total=len(valid_dataset), desc=f'Val Epoch {epoch + 1}/{total_epochs}', unit='img') as vpbar:
                for batch in valid_dataloader:
                    counter += 1
                    image = batch['image'].to(DEVICE)
                    mask = torch.where(batch['mask'].to(DEVICE) > 0., 1, 0)
                    outputs = model(image)
                    # outputs = outputs.squeeze(1)  # diceLoss
                    loss_val = loss_fun(outputs, mask)
                    valid_running_loss += loss_val.item()
                    vpbar.update(image.shape[0])
                    vpbar.set_postfix(**{'loss (batch)': loss_val.item()})
                train_loss.append(valid_running_loss / counter)
                print("Val_loss: {0}".format(valid_running_loss / counter))

        # Save the trained model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "./unet_model_s.pth")
