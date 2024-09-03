"""
Module Name: LabelCleaner
Description: This module checks label files within a directory and deletes any label files (and their corresponding images) 
             that contain only class 0 annotations. 

Created: 17-07-2024

Usage:
    To clean up a directory of images and labels, specify the directory path and the image file extensions to consider.

    Example:
        check_and_delete_labels("/path/to/dataset", image_exts = [".jpg", ".png"])

Dependencies:
    - os (built-in)

Notes:
    - The script will remove both the label file and the corresponding image file if only class 0 is found in the label file.
    - The image file extensions considered by default are ".jpg" and ".png", but this can be adjusted.
"""


import os


def check_and_delete_labels(source_dir, image_exts = [".jpg", ".png"]):
    """
    Check label files for classes 1, 2, or 3. If only class 0 is found, delete the label file and corresponding image file.

    Parameters:
    source_dir (str): The directory to search for label files.
    image_exts (list): List of image file extensions to consider.

    Returns:
    None
    """
    label_files = [f for f in os.listdir(source_dir) if f.endswith(".txt")]
    
    for label_file in label_files:
        label_path = os.path.join(source_dir, label_file)
        keep_file = False
        
        with open(label_path, "r") as file:
            lines = file.readlines()
            
            for line in lines:
                class_id = int(line.split()[0])
                if class_id in [1, 2, 3]:
                    keep_file = True
                    break
        
        if not keep_file:
            try:
                os.remove(label_path)
                print(f"[INFO] Deleted label: {label_path}")
                
                base_name = os.path.splitext(label_file)[0]
                
                for ext in image_exts:
                    image_path = os.path.join(source_dir, base_name + ext)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"[INFO] Deleted image: {image_path}")
                        
            except Exception as e:
                print(f"[WARNING] Failed to delete {label_path} or its image: {e}")


if __name__ == "__main__":
    # Directories to clean up.
    directories = ["/data/final/train", "/data/final/val", "/data/final/test"]

    for directory in directories:
        check_and_delete_labels(directory)

    print(f"[INFO] Label cleaning completed.")