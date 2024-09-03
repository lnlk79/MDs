"""
Module Name: TifConverter
Description: This module converts TIFF images to PNG format using rasterio for reading the images and OpenCV for saving them. 
             It processes all TIFF files in a specified directory and its subdirectories.

Created: 16-07-2024

Usage:
    To convert TIFF images to PNG format, specify the source directory containing the TIFF files.

    Example:
        convert_tif_to_png(source_dir)

Dependencies:
    - os (built-in)
    - cv2 
    - numpy
    - rasterio

Notes:
    - The module ensures that the output PNG images have 3 channels (RGB) and are in uint8 format.
    - If the TIFF image has more than 3 channels, only the first 3 channels are used.
"""


import os
import cv2
import numpy as np
import rasterio


def read_image_with_rasterio(img_path):
    """
    Read a TIFF image using rasterio and convert it to a format suitable for saving as a PNG.

    Parameters:
    img_path (str): The file path to the TIFF image.

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


def convert_tif_to_png(source_dir):
    """
    Convert all TIFF images in a directory and its subdirectories to PNG format.

    Parameters:
    source_dir (str): The directory containing TIFF files to convert.

    Returns:
    None
    """
    for root, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".tif"):
                tif_path = os.path.join(root, file)
                png_path = os.path.splitext(tif_path)[0] + ".png"
                
                try:
                    img = read_image_with_rasterio(tif_path)
                    cv2.imwrite(png_path, img)
                    print(f"[INFO] Converted {tif_path} to {png_path}")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to convert {tif_path}: {e}")


if __name__ == "__main__":
    # Defining paths.
    source_dir = "/data/final/train"
    source_dir_val = "/data/final/val"
    source_dir_test = "/data/final/test"
    
    # Converting from .tif to .png.
    convert_tif_to_png(source_dir)
    convert_tif_to_png(source_dir_val)
    convert_tif_to_png(source_dir_test)

    print("[INFO] Conversion from .tig to .png completed.")