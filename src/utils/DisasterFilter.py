"""
Module Name: DisasterFilter
Description: This module moves "post_disaster" images and their corresponding labels from multiple source directories 
             to a specified destination directory, organising them into separate "images" and "labels" folders.

Created: 16-07-2024

Usage:
    To move post-disaster data, specify the list of base directories and the destination directory.

    Example:
        move_post_disaster_data(base_dirs, destination_dir)

Dependencies:
    - os (built-in)
    - shutil (built-in)

Notes:
    - The script assumes that each base directory contains "images" and "labels" subdirectories.
    - Only files with "post_disaster" in their filenames will be moved.
"""

import os
import shutil

def move_post_disaster_data(base_dirs, destination_dir):
    """
    Move "post_disaster" images and labels from the specified base directories to the destination directory.

    Parameters:
    base_dirs (list): List of base directories containing "images" and "labels" subdirectories.
    destination_dir (str): The directory where "post_disaster" images and labels will be moved.

    Returns:
    None
    """
    images_folder = "images"
    labels_folder = "labels"
    
    # Creating destination folders.
    dest_images_path = os.path.join(destination_dir, images_folder)
    dest_labels_path = os.path.join(destination_dir, labels_folder)
    
    os.makedirs(dest_images_path, exist_ok = True)
    os.makedirs(dest_labels_path, exist_ok = True)

    for base_dir in base_dirs:
        img_path = os.path.join(base_dir, images_folder)
        lbl_path = os.path.join(base_dir, labels_folder)
        
        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            print(f"[WARNING] {base_dir} does not contain both 'images' and 'labels' folders.")
            continue
        
        # Moving "post_disaster" images.
        for file_name in os.listdir(img_path):
            if "post_disaster" in file_name:
                src_file = os.path.join(img_path, file_name)
                dst_file = os.path.join(dest_images_path, file_name)
                shutil.move(src_file, dst_file)
        
        # Moving "post_disaster" labels.
        for file_name in os.listdir(lbl_path):
            if "post_disaster" in file_name:
                src_file = os.path.join(lbl_path, file_name)
                dst_file = os.path.join(dest_labels_path, file_name)
                shutil.move(src_file, dst_file)


if __name__ == "__main__":
    base_dirs = ["/data/raw/geotiffs/hold", "/data/raw/geotiffs/test", "/data/raw/geotiffs/tier1", "/data/raw/geotiffs/tier3"] 
    destination_dir = "/data/processed"
    
    move_post_disaster_data(base_dirs, destination_dir)
    
    print(f"[INFO] Moving post_disaster data to {destination_dir} completed.")