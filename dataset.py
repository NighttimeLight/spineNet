import numpy as np
import os
from PIL import Image
import torch
import torchvision.transforms as T
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

# image size to convert to
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 250


class Dataset(Dataset):
    def __init__(self, raw_images_path, masks_path, images_name):
        self.raw_images_path = raw_images_path
        self.masks_path = masks_path
        self.images_name = images_name

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, index):
        # get image and mask for a given index
        img_path = os.path.join(self.raw_images_path, self.images_name[index])
        mask_path = os.path.join(self.masks_path, self.images_name[index])

        # read the image and mask
        image = Image.open(img_path)
        mask = Image.open(mask_path)

        # resize image and change the shape to (1, image_width, image_height)
        w, h = image.size
        image = image.resize((w, h), resample=Image.BICUBIC)
        image = T.Resize(size=(IMAGE_WIDTH, IMAGE_HEIGHT))(image)
        image_ndarray = np.array(image)
        image_ndarray = image_ndarray.reshape(1, image_ndarray.shape[0], image_ndarray.shape[1])

        # resize the mask. Mask shape is (image_width, image_height)
        mask = mask.resize((w, h), resample=Image.NEAREST)
        mask = T.Resize(size=(IMAGE_WIDTH, IMAGE_HEIGHT))(mask)
        mask_ndarray = np.array(mask)
        image_ndarray[0] = np.where(mask_ndarray == 0, 0, image_ndarray[0])
        return {
            'image': torch.as_tensor(image_ndarray.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask_ndarray.copy()).float()
        }

