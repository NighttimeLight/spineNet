import os

from torch.autograd import Variable
from tqdm import tqdm

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from dataset import Dataset
from net import UNet

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 5
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

    # Define multi label model as UNet
    # model_m = UNet(n_channels=2, n_classes=29)  # Case 1, 5
    model_m = UNet(n_channels=1, n_classes=29)  # Case 2, 3, 4
    model_m.to(device=device)
    optimizer_m = optim.Adam(model_m.parameters(), lr=LEARNING_RATE)
    loss_fun = nn.CrossEntropyLoss()
    prev_epoch = 0
    total_epochs = EPOCHS

    # Define single label model as UNet
    model_s = UNet(n_channels=1, n_classes=2)
    model_s.to(device=device)
    # Load trained single label model
    checkpoint_s = torch.load("models/unet_model_s.pth")
    model_s.load_state_dict(checkpoint_s['model_state_dict'])

    # # Load recent train progress
    # checkpoint = torch.load("models/unet_model_m_prev.pth")
    # model_m.load_state_dict(checkpoint['model_state_dict'])
    # optimizer_m.load_state_dict(checkpoint['optimizer_state_dict'])
    # prev_epoch = checkpoint['epoch'] + 1
    # total_epochs = prev_epoch + EPOCHS

    train_losses_f = open("results/losses_train.txt", "a")
    val_losses_f = open("results/losses_val.txt", "a")

    train_loss = []
    val_loss = []
    for epoch in range(prev_epoch, total_epochs):
        model_m.train()
        train_running_loss = 0.0
        counter = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{total_epochs}', unit='img') as tpbar:
            for batch in train_dataloader:
                counter += 1
                image = batch['image'].to(DEVICE)
                mask = batch['mask'].long().to(DEVICE)
                outputs_s = model_s(image)
                outputs_s = torch.sigmoid(outputs_s)
                # image = torch.cat((image, outputs_s[:, 1, :, :].unsqueeze(1)), 1)  # Case 1
                thresh = Variable(torch.Tensor([0.05]).to(DEVICE))  # threshold  # Case 3, 5
                outs_b = (outputs_s[:, 1, :, :].unsqueeze(1) > thresh)  # Case 3, 5
                image = torch.where(outs_b, image, outs_b.float())  # Case 3
                # thresh = Variable(torch.Tensor([0]).to(DEVICE))  # threshold  # Case 4
                # outs_b = (mask[:, :, :].unsqueeze(1) > thresh)  # Case 4
                # image = torch.where(outs_b, image, outs_b.float())  # Case 4
                # image = torch.cat((image, outs_b.float()), 1)  # Case 5
                optimizer_m.zero_grad()
                outputs_m = model_m(image)
                loss_val = loss_fun(outputs_m, mask)
                train_running_loss += loss_val.item()
                loss_val.backward()
                optimizer_m.step()
                tpbar.update(image.shape[0])
                tpbar.set_postfix(**{'loss (batch)': loss_val.item()})
            train_loss.append(train_running_loss / counter)
            print("Train_loss: {0}".format(train_running_loss / counter))
            train_losses_f.write("{0}, ".format(train_running_loss / counter))
            train_losses_f.flush()

        model_m.eval()
        valid_running_loss = 0.0
        counter = 0
        with torch.no_grad():
            with tqdm(total=len(valid_dataset), desc=f'Val Epoch {epoch + 1}/{total_epochs}', unit='img') as vpbar:
                for batch in valid_dataloader:
                    counter += 1
                    image = batch['image'].to(DEVICE)
                    mask = batch['mask'].long().to(DEVICE)
                    outputs_s = model_s(image)
                    outputs_s = torch.sigmoid(outputs_s)
                    # image = torch.cat((image, outputs_s[:, 1, :, :].unsqueeze(1)), 1)  # Case 1
                    thresh = Variable(torch.Tensor([0.05]).to(DEVICE))  # threshold  # Case 3, 5
                    outs_b = (outputs_s[:, 1, :, :].unsqueeze(1) > thresh)  # Case 3, 5
                    image = torch.where(outs_b, image, outs_b.float())  # Case 3
                    # thresh = Variable(torch.Tensor([0]).to(DEVICE))  # threshold  # Case 4
                    # outs_b = (mask[:, :, :].unsqueeze(1) > thresh)  # Case 4
                    # image = torch.where(outs_b, image, outs_b.float())  # Case 4
                    # image = torch.cat((image, outs_b.float()), 1)  # Case 5
                    outputs_m = model_m(image)
                    loss_val = loss_fun(outputs_m, mask)
                    valid_running_loss += loss_val.item()
                    vpbar.update(image.shape[0])
                    vpbar.set_postfix(**{'loss (batch)': loss_val.item()})
                train_loss.append(valid_running_loss / counter)
                print("Val_loss: {0}".format(valid_running_loss / counter))
                val_losses_f.write("{0}, ".format(valid_running_loss / counter))
                val_losses_f.flush()

        # Save the trained model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_m.state_dict(),
            'optimizer_state_dict': optimizer_m.state_dict(),
        }, "models/unet_model_m.pth")

    train_losses_f.close()
    val_losses_f.close()
