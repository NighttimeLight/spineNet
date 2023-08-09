import numpy as np
from PIL import Image
import nibabel as nib


def read_file(nii_file):
    """
    Read .nii.gz file.

    Args:
      nii_file (str): a file path.

    Return:
      3D numpy array of CT image data.
    """
    return np.asanyarray(nib.load(nii_file).dataobj)


def save_file(raw_data, label_data, file_name, index, output_raw_file_path, output_label_file_path):
    """
    Save file into npz format.

    Args:
      raw_data (array): 2D numpy array of raw image data.
      label_data (array): 2D numpy array of label image data.
      file_name (str): file name.
      index (int): slice of CT image.
      output_raw_file_path (str): Path to all raw files.
      output_label_file_path (str): Path to all mask files.
    """

    # replace all non-zero pixels to 1
    # label_data = np.where(label_data > 0, 1, label_data)
    # unique_values = np.unique(label_data)
    # if data has pixel with value of 1 means it is a positive datapoint
    # if len(unique_values) > 1:
    raw_file_name = "{0}{1}_{2}.png".format(output_raw_file_path, file_name, index)
    im = Image.fromarray(raw_data)
    im = im.convert("L")
    im.save(raw_file_name)

    label_file_name = "{0}{1}_{2}.png".format(output_label_file_path, file_name, index)
    im = Image.fromarray(label_data)
    im = im.convert("L")
    im.save(label_file_name)


def save_png(raw_data, file_name, index, output_raw_file_path):
    """
    Save file into npz format.

    Args:
      raw_data (array): 2D numpy array of raw image data.
      file_name (str): file name.
      index (int): slice of CT image.
      output_raw_file_path (str): Path to all raw files.
    """

    raw_file_name = "{0}{1}_{2}.png".format(output_raw_file_path, file_name, index)
    im = Image.fromarray(raw_data)
    im = im.convert("L")
    im.save(raw_file_name)


def is_diagonal(matrix):
    '''
    Check if givem matrix is diagonal or not.

    Args:
        matrix (np array): numpy array
    '''

    for i in range(0, 3):
        for j in range(0, 3):
            if (i != j) and (matrix[i][j] != 0):
                return False
    return True


def generate_data(raw_file, label_file, file_name, output_raw_file_path, output_label_file_path):
    """
    Main function to read both raw and label file and generate series of images from slices.

    Args:
      raw_file (str): path to raw file.
      label_file (str): path to label file.
      file_name (str): file name.
      output_raw_file_path (str): Path to all raw files.
      output_label_file_path (str): Path to all mask files.
    """
    # If skip every 2 slice. Adjacent slices can be very similar to each other and
    # will generate redundant data
    skip_slice = 3
    continue_it = True
    raw_data = read_file(raw_file)
    label_data = read_file(label_file)

    if "split" in raw_file:
        continue_it = False

    affine = nib.load(raw_file).affine

    if is_diagonal(affine[:3, :3]):
        transposed_raw_data = np.transpose(raw_data, [2, 1, 0])
        transposed_raw_data = np.flip(transposed_raw_data)
        transposed_label_data = np.transpose(label_data, [2, 1, 0])
        transposed_label_data = np.flip(transposed_label_data)

    else:
        transposed_raw_data = np.rot90(raw_data)
        transposed_raw_data = np.flip(transposed_raw_data)

        transposed_label_data = np.rot90(label_data)
        transposed_label_data = np.flip(transposed_label_data)

    if continue_it:
        if transposed_raw_data.shape:
            slice_count = transposed_raw_data.shape[-1]
            # print("File name: ", file_name, " - Slice count: ", slice_count)

            # skip some slices
            for each_slice in range(1, slice_count, skip_slice):
                save_file(transposed_raw_data[:, :, each_slice],
                          transposed_label_data[:, :, each_slice],
                          file_name,
                          each_slice,
                          output_raw_file_path,
                          output_label_file_path)


def generate_png(raw_file, file_name, output_raw_file_path):
    """
    Main function to read only raw file and generate series of images from slices.

    Args:
      raw_file (str): path to raw file.
      file_name (str): file name.
      output_raw_file_path (str): Path to all raw files.
    """
    # If skip every 2 slice. Adjacent slices can be very similar to each other and
    # will generate redundant data
    skip_slice = 3
    continue_it = True
    raw_data = read_file(raw_file)

    if "split" in raw_file:
        continue_it = False

    affine = nib.load(raw_file).affine

    if is_diagonal(affine[:3, :3]):
        transposed_raw_data = np.transpose(raw_data, [2, 1, 0])
        transposed_raw_data = np.flip(transposed_raw_data)

    else:
        transposed_raw_data = np.rot90(raw_data)
        transposed_raw_data = np.flip(transposed_raw_data)

    if continue_it:
        if transposed_raw_data.shape:
            slice_count = transposed_raw_data.shape[-1]
            # print("File name: ", file_name, " - Slice count: ", slice_count)

            # skip some slices
            for each_slice in range(1, slice_count, skip_slice):
                save_png(transposed_raw_data[:, :, each_slice],
                         file_name,
                         each_slice,
                         output_raw_file_path)
