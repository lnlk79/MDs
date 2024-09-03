"""
Module Name: AnnotationVisualiser
Description: This module visualises annotations for images by overlaying polygon annotations on the images. 
             It supports both raster (GeoTIFF) and standard image formats and allows for visual inspection of 
             annotations by blending them with the original images.

Created: 16-07-2024

Usage:
    To visualise annotations, specify the directories containing the images and their corresponding label files. 
    The script will randomly select a specified number of images and display them with their annotations overlaid.

    Example:
        visualise_annotations(image_dir, label_dir, sample_size = 5, show = True)

Dependencies:
    - os (built-in)
    - cv2 
    - numpy
    - matplotlib
    - random (built-in)
    - rasterio

Notes:
    - The label files should be in YOLO format, containing polygon coordinates relative to the image dimensions.
"""


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import rasterio


# Defining the colour mappings for each class.
class_colours = {0: (0, 255, 0),     # no-damage: green
                1: (255, 255, 0),    # minor-damage: yellow
                2: (255, 165, 0),    # major-damage: orange
                3: (255, 0, 0)}      # destroyed: red


def load_annotations(label_path, img_w, img_h):
    """
    Load ground truth annotations from a label file.

    Parameters:
    label_path (str): Path to the label file.
    img_w (int): Width of the image.
    img_h (int): Height of the image.

    Returns:
    list: A list of tuples where each tuple contains:
        - class_id (int): The class identifier.
        - polygon (list): A list of (x, y) coordinates representing the polygon.
    """
    annotations = []
    
    with open(label_path, "r") as file:
        lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            polygon = []
            
            for i in range(1, len(parts), 2):
                x = float(parts[i]) * img_w
                y = float(parts[i + 1]) * img_h
                polygon.append((int(x), int(y)))
                
            annotations.append((class_id, polygon))
            
    return annotations

def read_image_with_rasterio(img_path):
    """
    Read a GeoTIFF image using rasterio and convert it to a format suitable for display.

    Parameters:
    img_path (str): The file path to the GeoTIFF image.

    Returns:
    np.ndarray: The image data as a 3-channel (RGB) uint8 numpy array.
    """
    with rasterio.open(img_path) as src:
        image_data = src.read()
        
    image_data = np.transpose(image_data, (1, 2, 0))

    # Ensuring the image is in uint8 format.
    if image_data.dtype != np.uint8:
        image_data = (255 * (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))).astype(np.uint8)
    
    # Ensuring the image has 3 channels.
    if image_data.shape[2] == 1:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
        
    elif image_data.shape[2] > 3:
        image_data = image_data[:, :, :3]
    
    return np.ascontiguousarray(image_data)


def visualise_annotations(image_dir, label_dir, sample_size = 5, show = False):
    """
    Visualise annotations by overlaying them on a set of randomly selected images.

    Parameters:
    image_dir (str): The directory containing the images.
    label_dir (str): The directory containing the corresponding label files.
    sample_size (int): The number of images to sample and display (default: 5).
    show (bool): Whether to display the images (default: False).

    Returns:
    None
    """
    image_files = [f for f in os.listdir(image_dir) if f.endswith((".tif", ".tiff", ".png"))]
    random.shuffle(image_files)
    sample_files = image_files[:sample_size]

    for image_file in sample_files:
        img_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")
        
        if not os.path.isfile(label_path):
            continue

        # Checking if the file is a GeoTIFF and read it with rasterio if it is.
        if img_path.endswith((".tif", ".tiff")):
            img = read_image_with_rasterio(img_path)
            
        else:
            img = cv2.imread(img_path)
        
        img_h, img_w = img.shape[:2]

        annotations = load_annotations(label_path, img_w, img_h)

        overlay = img.copy()

        for class_id, polygon in annotations:
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            color = class_colours.get(class_id, (255, 255, 255))  # Default to white if class_id not found.
            cv2.fillPoly(overlay, [pts], color)

        # Blending the overlay with the original image.
        alpha = 0.3  # Transparency setting.
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        if show:
            plt.figure(figsize = (10, 10))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"[INFO] Annotations for {image_file}")
            plt.axis("off")
            plt.show()

if __name__ == "__main__":
    building_image_dir = "./data/final/train"
    building_label_dir = "./data/final/train"

    visualise_annotations(building_image_dir, building_label_dir, sample_size = 10, show = True)