"""
Module Name: SplitMakerXview
Description: This module prepares the xView2 dataset by splitting images and labels from multiple source directories 
             into train, validation, and test sets with specified ratios, then consolidates these splits 
             into single train, validation, and test folders.
             
Created: 16-07-2024

Usage:
    To prepare the dataset with the necessary splits and configuration ensure that the `source_dirs` 
    list contains paths to the directories with "input" and "labels" subdirectories.

Dependencies:
    - os (built-in)
    - shutil (built-in)
    - random (built-in)

Notes:
    - Used to create the splits for the roads data, and pooling the data for buldings before creating splits.
"""


import os
import shutil
import random


def create_split_folders(base_dir, split_names = ["train", "val", "test"]):
    """
    Create directory structure for dataset splits.

    Parameters:
    base_dir (str): The base directory where the split folders will be created.
    split_names (list): List of split names (default is ["train", "val", "test"]).

    Returns:
    None
    """
    for split_name in split_names:
        os.makedirs(os.path.join(base_dir, split_name), exist_ok = True)
    print(f"[INFO] Created split folders in {base_dir}")


def get_image_label_pairs(source_dir):
    """
    Get pairs of image and label paths.

    Parameters:
    source_dir (str): Directory containing the images and labels subdirectories.

    Returns:
    list: List of tuples, each containing an image path and a corresponding label path.
    """
    images_dir = os.path.join(source_dir, "images")
    labels_dir = os.path.join(source_dir, "yolo_labels")
    
    image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".tif")])
    label_paths = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith(".txt")])
    
    if not image_paths:
        print(f"[WARNING] No image files found in: {images_dir}")
    if not label_paths:
        print(f"[WARNING] No label files found in: {labels_dir}")
    
    image_label_pairs = [(img, lbl) for img, lbl in zip(image_paths, label_paths)]
    print(f"[INFO] Found {len(image_label_pairs)} image-label pairs in: {images_dir} and {labels_dir}")
    
    return image_label_pairs


def split_data(image_label_pairs, train_ratio = 0.8, val_ratio = 0.1):
    """
    Split image-label pairs into train, validation, and test sets.

    Parameters:
    image_label_pairs (list): List of tuples containing image and label paths.
    train_ratio (float): Ratio of the training set (default is 0.8).
    val_ratio (float): Ratio of the validation set (default is 0.1).

    Returns:
    dict: Dictionary containing the split data.
    """
    random.shuffle(image_label_pairs)
    total = len(image_label_pairs)
    train_cutoff = int(total * train_ratio)
    val_cutoff = int(total * (train_ratio + val_ratio))
    
    splits = {"train": image_label_pairs[:train_cutoff],
              "val": image_label_pairs[train_cutoff:val_cutoff],
              "test": image_label_pairs[val_cutoff:]}
    
    print(f"[INFO] Split data into {len(splits['train'])} train, {len(splits['val'])} val, and {len(splits['test'])} test pairs.")
    
    return splits


def move_data(splits, output_base_dir):
    """
    Move split data into respective directories.

    Parameters:
    splits (dict): Dictionary containing the split data.
    output_base_dir (str): Base directory to save the split data.

    Returns:
    None
    """
    for split_name, pairs in splits.items():
        for img_path, lbl_path in pairs:
            try:
                img_basename = os.path.basename(img_path)
                lbl_basename = os.path.basename(lbl_path)
                img_dest = os.path.join(output_base_dir, split_name, img_basename)
                lbl_dest = os.path.join(output_base_dir, split_name, lbl_basename)
                
                print(f"[INFO] Moving image {img_path} to {img_dest}")
                print(f"[INFO] Moving label {lbl_path} to {lbl_dest}")
                
                if os.path.exists(img_path) and os.path.exists(lbl_path):
                    shutil.move(img_path, img_dest)
                    shutil.move(lbl_path, lbl_dest)
                    
                else:
                    print(f"[ERROR] File does not exist: {img_path} or {lbl_path}")
                    
            except Exception as e:
                print(f"[ERROR] Failed to move {img_path} or {lbl_path}: {e}")


def prepare_final_dataset(source_dir, final_output_dir, train_ratio = 0.8, val_ratio = 0.1):
    """
    Prepare the final dataset by creating split folders and splitting image-label pairs.

    Parameters:
    source_dir (str): Directory containing "images" and "labels" subdirectories.
    final_output_dir (str): Directory to save the final dataset.
    train_ratio (float): Ratio of the training set (default is 0.8).
    val_ratio (float): Ratio of the validation set (default is 0.1).

    Returns:
    None
    """
    create_split_folders(final_output_dir)
    
    image_label_pairs = get_image_label_pairs(source_dir)
    splits = split_data(image_label_pairs, train_ratio, val_ratio)
    move_data(splits, final_output_dir)


if __name__ == "__main__":
    source_dir = "./data/processed/"  # Directory containing "images" and "yolo_labels" subdirectories.
    final_output_dir = "./data/final/"

    prepare_final_dataset(source_dir, final_output_dir)

    print("[INFO] Completed dataset splitting and preparation.")
