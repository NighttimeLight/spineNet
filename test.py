import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import seaborn as sns
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from net import UNet

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
EPOCHS = 5
NUM_WORKERS = 8

v_dict = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
    21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum',
    27: 'Cocc', 28: 'T13'
}

data_folder = "./data/processed/"

train_data_dir = data_folder + "train_ctr1/"
validation_data_dir = data_folder + "validation_ctr1/"
test_data_dir = data_folder + "test_ctr1/"

train_images_dir = train_data_dir + "images/"
train_masks_dir = train_data_dir + "masks/"
validation_images_dir = validation_data_dir + "images/"
validation_masks_dir = validation_data_dir + "masks/"
test_images_dir = test_data_dir + "images/"
test_masks_dir = test_data_dir + "masks/"

def compute_IoU(cm):
    """
    Adapted from:
        https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py
        https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/metrics.py#L2716-L2844
    """

    sum_over_row = cm.sum(axis=0)
    sum_over_col = cm.sum(axis=1)
    true_positives = np.diag(cm)

    # sum_over_row + sum_over_col = 2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    iou = true_positives / denominator

    return iou, np.nanmean(iou)

if __name__ == '__main__':

    # Get name of every image (same as corresponding mask name)
    train_images_paths = os.listdir(train_images_dir)
    validation_images_paths = os.listdir(validation_images_dir)
    test_images_paths = os.listdir(test_images_dir)
    # train_masks_paths = os.listdir(train_masks_dir)
    # validation_masks_paths = os.listdir(validation_masks_dir)
    # test_masks_paths = os.listdir(test_masks_dir)

    # Load both images and masks data
    train_dataset = Dataset(train_images_dir, train_masks_dir, train_images_paths)
    valid_dataset = Dataset(validation_images_dir, validation_masks_dir, validation_images_paths)
    test_dataset = Dataset(test_images_dir, test_masks_dir, test_images_paths)

    # Create PyTorch DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define model as UNet
    # model = UNet(n_channels=2, n_classes=29)  # Case 1, 5
    model = UNet(n_channels=1, n_classes=29)  # Case 2, 3, 4
    model.to(device=device)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # checkpoint = torch.load("models/all/unet_model_m_intFx.pth")  # Case 1
    # checkpoint = torch.load("models/all/unet_model_m_s_intFx.pth")  # Case 2
    # checkpoint = torch.load("models/all/unet_model_m_sf_intFx.pth")  # Case 3
    checkpoint = torch.load("models/all/unet_model_m_ctr20p_2.pth")  # Case 3_
    # checkpoint = torch.load("models/all/unet_model_m_sf_t_intFx.pth")  # Case 4
    # checkpoint = torch.load("models/all/unet_model_m_f.pth")  # Case 5
    model.load_state_dict(checkpoint['model_state_dict'])
    model_s = UNet(n_channels=1, n_classes=2)
    model_s.to(device=device)
    checkpoint_s = torch.load("models/unet_model_s.pth")
    model_s.load_state_dict(checkpoint_s['model_state_dict'])
    loss_fun = nn.CrossEntropyLoss()

    train_loss = []
    val_loss = []
    train_running_loss = 0
    acc = []
    acc2 = []
    model.eval()
    counter = 0
    labels = np.arange(29)
    cm = torch.zeros((29, 29))
    vertebrae_dict = {val: [] for key, val in v_dict.items()}
    with torch.no_grad():
        with tqdm(total=len(test_dataset), unit='img') as pbar:
            for i, data in enumerate(test_dataloader):
                counter += 1
                image = data['image'].to(DEVICE)
                mask = data['mask'].long().to(DEVICE)
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
                outputs = model(image)
                loss_val = loss_fun(outputs, mask)
                train_running_loss += loss_val.item()
                _, masks_pred = torch.max(outputs, 1)
                colors_mask, counts_mask = np.unique(data['mask'][0][:, :], return_counts=True)
                colors, counts = np.unique(masks_pred[0].cpu(), return_counts=True)
                mask_dict = dict(zip(colors_mask, counts_mask))
                pred_dict = dict(zip(colors, counts))
                pred_dict = {key - 1: val for key, val in pred_dict.items() if val > 100 and key != 0}  # choose colors with more than 100 pixels present, and map from 1:29 to 0:28 range
                mask_list = [val for key, val in v_dict.items() if key in mask_dict]
                pred_list = [val for key, val in v_dict.items() if key in pred_dict]
                result = all(elem in mask_list for elem in pred_list)
                if result:
                    acc2.append(1)
                else:
                    acc2.append(0)
                for item in mask_list:
                    if item in pred_list:
                        vertebrae_dict[item].append(1)
                    else:
                        vertebrae_dict[item].append(0)
                loss_value = loss_fun(outputs, mask.long())
                val_loss.append(loss_value.item())
                for j in range(len(mask)):
                    cm[mask[j].long(), masks_pred[j].long()] += 1
                pbar.update(1)

    cm = np.array(cm.cpu())
    sns.heatmap(cm, annot=True)
    print("Cross-entropy loss: " + str(sum(val_loss) / len(val_loss)))
    print("Accuracy: " + str((sum(acc2) / len(acc2))))
    # print(vertebrae_dict)
    for key, value in vertebrae_dict.items():
        if len(value) > 0:
            print(key + ': ' + str(round(sum(value) / len(value), 3)))
        else:
            print(key + ': ' + str([]))
    class_iou, mean_iou = compute_IoU(cm)
    print("Class IoU:" + str(class_iou) + " Mean IoU:" + str(mean_iou))

    print("Avg loss value: {0}".format(train_running_loss / counter))

