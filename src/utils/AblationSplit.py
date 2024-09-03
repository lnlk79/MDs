"""
Module Name: AblationSplit
Description: This module handles the creation of the ablation dataset by copying a percentage of image-label pairs 
             from existing dataset splits (train, val, test) to a new location.

Created: 14-08-2024

Usage:
    Specify the source directory containing the original data and the destination directory where the 
    ablation data will be copied.

    Example:
        perform_ablation_copying("/path/to/original/data", "/path/to/ablation/data")

Dependencies:
    - os (built-in)
    - random (built-in)
    - shutil (built-in)

Notes:
    - The ablation process randomly selects 10% of the data by default, but this can be adjusted by modifying 
      the `percentage` parameter in the `copy_percentage_of_data` function.
"""


import os
import random
import shutil


def create_ablation_folders(base_dir, split_names = ["train", "val", "test"]):
    """
    Create directory structure for ablation dataset splits.

    Parameters:
    base_dir (str): The base directory where the split folders will be created.
    split_names (list): List of split names (default is ["train", "val", "test"]).

    Returns:
    None
    """
    for split_name in split_names:
        os.makedirs(os.path.join(base_dir, split_name), exist_ok = True)
    print(f"[INFO] Created ablation folders in: {base_dir}")


def get_image_label_pairs(directory):
    """
    Get pairs of image and label paths from a given directory.

    Parameters:
    directory (str): Directory containing image and label files.

    Returns:
    list: List of tuples, each containing an image path and a corresponding label path.
    """
    image_files = sorted([f for f in os.listdir(directory) if f.endswith(".png")])
    label_files = sorted([f for f in os.listdir(directory) if f.endswith(".txt")])

    # Checking that the length of image files and label files matches.
    if len(image_files) != len(label_files):
        print(f"[WARNING] The number of image files and label files do not match in: {directory}")
    
    image_label_pairs = [(os.path.join(directory, img), os.path.join(directory, lbl)) for img, lbl in zip(image_files, label_files)]
    
    return image_label_pairs


def copy_percentage_of_data(image_label_pairs, dest_folder, percentage = 0.1):
    """
    Copy a percentage of image-label pairs to the destination folder.

    Parameters:
    image_label_pairs (list): List of tuples containing image and label paths.
    dest_folder (str): Destination folder where the files will be copied.
    percentage (float): Percentage of files to copy (default is 10%).

    Returns:
    None
    """
    num_files_to_copy = int(len(image_label_pairs) * percentage)
    files_to_copy = random.sample(image_label_pairs, num_files_to_copy)

    for img_path, lbl_path in files_to_copy:
        img_dest = os.path.join(dest_folder, os.path.basename(img_path))
        lbl_dest = os.path.join(dest_folder, os.path.basename(lbl_path))

        shutil.copy(img_path, img_dest)
        shutil.copy(lbl_path, lbl_dest)

    print(f"[INFO] Copied {len(files_to_copy)} files to :{dest_folder}")


def perform_ablation_copying(base_src_dir, base_dest_dir):
    """
    Perform the ablation copying process by randomly copying 10% of data from each split.

    Parameters:
    base_src_dir (str): Base directory containing the original data (with test, training, and val subdirectories).
    base_dest_dir (str): Base directory where the ablation data will be copied.

    Returns:
    None
    """
    splits = ["train", "val", "test"]

    create_ablation_folders(base_dest_dir, splits)

    for split in splits:
        src_folder = os.path.join(base_src_dir, split)
        dest_folder = os.path.join(base_dest_dir, split)
        
        image_label_pairs = get_image_label_pairs(src_folder)
        copy_percentage_of_data(image_label_pairs, dest_folder)


if __name__ == "__main__":
    # Source directories (where original test, training, and val data are located).
    base_src_dir = "/data/final"
    
    # Destination directory for ablation data.
    base_dest_dir = "/data/ablation"
    
    perform_ablation_copying(base_src_dir, base_dest_dir)
    print("[INFO] Ablation data copying completed.")