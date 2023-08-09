import os
import glob
import shutil
from tqdm import tqdm

from helpers import generate_data

data_source_folder = "./data/source/"
data_dest_folder = "./data/processed/"

train_data_source = data_source_folder + "train/"
validation_data_source = data_source_folder + "validation/"
test_data_source = data_source_folder + "test/"

train_data_dest = data_dest_folder + "train/"
validation_data_dest = data_dest_folder + "validation/"
test_data_dest = data_dest_folder + "test/"

dest_train_images = train_data_dest + "images/"
dest_train_masks = train_data_dest + "masks/"
dest_validation_images = validation_data_dest + "images/"
dest_validation_masks = validation_data_dest + "masks/"
dest_test_images = test_data_dest + "images/"
dest_test_masks = test_data_dest + "masks/"


if __name__ == '__main__':

    # Check if folders already exists, remove and create new again
    if os.path.exists(train_data_dest):
        shutil.rmtree(train_data_dest)
    if os.path.exists(validation_data_dest):
        shutil.rmtree(validation_data_dest)
    if os.path.exists(test_data_dest):
        shutil.rmtree(test_data_dest)

    os.makedirs(dest_train_images)
    os.makedirs(dest_train_masks)
    os.makedirs(dest_validation_images)
    os.makedirs(dest_validation_masks)
    os.makedirs(dest_test_images)
    os.makedirs(dest_test_masks)

    # Relative paths to every scan file
    raw_train_files = glob.glob(os.path.join(train_data_source, 'rawdata\/*\/*nii.gz'))
    raw_validation_files = glob.glob(os.path.join(validation_data_source, 'rawdata\/*\/*nii.gz'))
    raw_test_files = glob.glob(os.path.join(test_data_source, 'rawdata\/*\/*nii.gz'))
    masks_train_files = glob.glob(os.path.join(train_data_source, 'derivatives\/*\/*nii.gz'))
    masks_validation_files = glob.glob(os.path.join(validation_data_source, 'derivatives\/*\/*nii.gz'))
    masks_test_files = glob.glob(os.path.join(test_data_source, 'derivatives\/*\/*nii.gz'))
    raw_train_files = [file.replace('\\', '/') for file in raw_train_files]
    raw_validation_files = [file.replace('\\', '/') for file in raw_validation_files]
    raw_test_files = [file.replace('\\', '/') for file in raw_test_files]
    masks_train_files = [file.replace('\\', '/') for file in masks_train_files]
    masks_validation_files = [file.replace('\\', '/') for file in masks_validation_files]
    masks_test_files = [file.replace('\\', '/') for file in masks_test_files]

    # print("Train images count: {0}, masks: {1}".format(len(raw_train_files), len(masks_train_files)))
    # print("Validation images count: {0}, masks: {1}".format(len(raw_validation_files), len(masks_validation_files)))
    # print("Test images count: {0}, masks: {1}".format(len(raw_test_files), len(masks_test_files)))

    with tqdm(total=len(raw_train_files), desc=f'Train files', unit='files') as pbar:
        for raw_file in raw_train_files:
            file_name = raw_file.split("/")[-1].split("_ct.nii.gz")[0]  # Unique case name
            for mask_file in masks_train_files:
                if file_name in mask_file.split("/")[-1]:
                    generate_data(raw_file,
                                  mask_file,
                                  file_name,
                                  dest_train_images,
                                  dest_train_masks)
                    pbar.update()
                    pbar.set_postfix(file_name=file_name)
        pbar.write("Processing train data done.")

    with tqdm(total=len(raw_validation_files), desc=f'Validation files', unit='files') as pbar:
        for raw_file in raw_validation_files:
            file_name = raw_file.split("/")[-1].split("_ct.nii.gz")[0]  # Unique case name
            for mask_file in masks_validation_files:
                if file_name in mask_file.split("/")[-1]:
                    generate_data(raw_file,
                                  mask_file,
                                  file_name,
                                  dest_validation_images,
                                  dest_validation_masks)
                    pbar.update()
                    pbar.set_postfix(file_name=file_name)
        pbar.write("Processing validation data done.")

    with tqdm(total=len(raw_test_files), desc=f'Test files', unit='files') as pbar:
        for raw_file in raw_test_files:
            file_name = raw_file.split("/")[-1].split("_ct.nii.gz")[0]  # Unique case name
            for mask_file in masks_test_files:
                if file_name in mask_file.split("/")[-1]:
                    generate_data(raw_file,
                                  mask_file,
                                  file_name,
                                  dest_test_images,
                                  dest_test_masks)
                    pbar.update()
                    pbar.set_postfix(file_name=file_name)
        pbar.write("Processing test data done.")
