"""
Module Name: ImageCropAndReassemble
Description: This module crops large images into smaller sections, processes them using the YOLO model 
             to generate predictions, and then reassembles the processed sections back into the original image. 
             
Created: 18-08-2024

Usage:
    To crop an image, specify the image path, crop size, and the output directory for the cropped sections.

    Example:
        crop_and_save(image_path, crop_size, output_crop_folder)
        process_and_reassemble(output_crop_folder, model, output_folder, crop_size, image_size)

Dependencies:
    - PIL 
    - os (built-in)
    - ultralytics 
    - numpy 
    - cv2 

Notes:
    - The script assumes that the YOLO model has been trained and is available for loading.
    - The module supports saving both the individual processed crops and the reassembled full images.
"""

from PIL import Image
import os
from ultralytics import YOLO
import numpy as np
import cv2

# Defining the colour mappings for each class.
class_colours = {0: (0, 255, 0),     # no-damage: green
                1: (255, 255, 0),    # minor-damage: yellow
                2: (255, 165, 0),    # major-damage: orange
                3: (255, 0, 0)}      # destroyed: red


def crop_and_save(image_path, crop_size, output_folder):
    """
    Crop a large image into smaller square sections and save each crop.

    Parameters:
    image_path (str): Path to the input image.
    crop_size (int): Size of each square crop (width and height in pixels).
    output_folder (str): Directory where the cropped images will be saved.

    Returns:
    None
    """
    # Opening the image and gettings its size.
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Calculating how many crops can be made horizontally and vertically.
    x_crops = (img_width + crop_size - 1) // crop_size
    y_crops = (img_height + crop_size - 1) // crop_size
    
    # Iterating over the grid.
    for i in range(x_crops):
        for j in range(y_crops):
            # Calculating the coordinates of the crop box.
            left = i * crop_size
            top = j * crop_size
            right = min(left + crop_size, img_width)
            bottom = min(top + crop_size, img_height)
            
            # Cropping and saving the image.
            crop = img.crop((left, top, right, bottom))
            crop.save(f"{output_folder}/crop_{i}_{j}.png")


def get_outputs(image, model, threshold = 0.1, max_det = 1000):
    """
    Get model predictions (masks, bounding boxes, and labels) for a given image.

    Parameters:
    image (PIL.Image.Image): The input image for which to generate predictions.
    model (YOLO): The YOLO model used for generating predictions.
    threshold (float): Confidence threshold for predictions (default is 0.1).
    max_det (int): Maximum number of detections that can be made (default is 1000).

    Returns:
    tuple: A tuple containing:
        - masks (list): A list of masks (polygon coordinates) for the detected objects.
        - boxes (list): A list of bounding boxes for the detected objects.
        - labels (list): A list of labels for the detected objects.
    """
    outputs = model.predict(image, imgsz = 640, conf = threshold, max_det = max_det)
    scores = outputs[0].boxes.conf.detach().cpu().numpy()
    thresholded_indices = [idx for idx, score in enumerate(scores) if score > threshold]

    if len(thresholded_indices) > 0:
        masks = [outputs[0].masks.xy[idx] for idx in thresholded_indices]
        boxes = outputs[0].boxes.xyxy.detach().cpu().numpy()[thresholded_indices]
        boxes = [[(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))] for box in boxes]
        labels = [int(outputs[0].boxes.cls[idx].cpu().numpy()) for idx in thresholded_indices]
        
    else:
        masks, boxes, labels = [], [], []

    return masks, boxes, labels


def draw_segmentation_map(image, masks, labels):
    """
    Draw a segmentation map over the original image using detected masks and bounding boxes.

    Parameters:
    image (PIL.Image.Image): The original image on which the segmentation map will be drawn.
    masks (list): List of masks (polygon coordinates) for the detected objects.
    boxes (list): List of bounding boxes for the detected objects.
    labels (list): List of labels for the detected objects.

    Returns:
    PIL.Image.Image: The image with the segmentation map overlaid.
    """
    alpha = 1.0
    beta = 0.55  # Transparency for the segmentation map.
    gamma = 0  # Scalar added to each sum.
    
    # Converting the original PIL image into a NumPy format.
    image = np.array(image)  
    segmentation_map = np.zeros_like(image)

    for mask, label in zip(masks, labels):
        colour = class_colours[label]  # Using the class colour.

        if mask is not None and len(mask) > 0:
            poly = np.array(mask, dtype=np.int32)
            cv2.fillPoly(segmentation_map, [poly], colour)

    # Combining the original image with the segmentation map.
    combined_image = cv2.addWeighted(image, alpha, segmentation_map, beta, gamma)

    return Image.fromarray(combined_image)


def draw_predicted_mask(masks, labels, height, width):
    """
    Draw predicted masks for detected objects on a blank image of given dimensions.

    Parameters:
    masks (list): List of masks (polygon coordinates) for the detected objects.
    labels (list): List of labels for the detected objects.
    height (int): Height of the output mask image.
    width (int): Width of the output mask image.

    Returns:
    PIL.Image.Image: The image with the predicted masks drawn.
    """
    mask_image = np.zeros((height, width, 3), dtype = np.uint8)

    for mask, label in zip(masks, labels):
        color = class_colours[label]  # Using the class colour.
        if mask is not None and len(mask) > 0:
            poly = np.array(mask, dtype = np.int32)
            cv2.fillPoly(mask_image, [poly], color)

    return Image.fromarray(mask_image)


def process_and_reassemble(crop_folder, model, output_folder, crop_size, image_size):
    """
    Process cropped images using the YOLO model and reassembles them into the original image size.

    Parameters:
    crop_folder (str): Directory containing the cropped images.
    model (YOLO): The YOLO model used for generating predictions.
    output_folder (str): Directory where the processed images and reassembled images will be saved.
    crop_size (int): The size of each cropped image (width and height in pixels).
    image_size (tuple): The original full image size (width, height) before cropping.

    Returns:
    None
    """
    crop_images = [f for f in os.listdir(crop_folder) if f.endswith(".png")]
    crop_images.sort()  # Ensures the order is consistent.

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Placeholder for reassembling images.
    reassembled_mask = Image.new("RGB", image_size)
    reassembled_overlay = Image.new("RGB", image_size)

    for crop_image_name in crop_images:
        crop_image_path = os.path.join(crop_folder, crop_image_name)
        image = Image.open(crop_image_path).convert("RGB")
        img_w, img_h = image.size

        # Extracting row and column from filename.
        base_filename = os.path.splitext(crop_image_name)[0]
        _, col, row = base_filename.split("_")
        row, col = int(row), int(col)

        # Getting model predictions.
        masks, boxes, labels = get_outputs(image, model)
        predicted_mask_overlay = draw_segmentation_map(image, masks, boxes, labels)
        predicted_mask = draw_predicted_mask(masks, labels, img_h, img_w)

        # Saving individual images.
        predicted_mask_overlay.save(os.path.join(output_folder, f"{base_filename}_overlay.png"))
        predicted_mask.save(os.path.join(output_folder, f"{base_filename}_mask.png"))

        # Calculating position in the reassembled image grid.
        x_offset = col * crop_size
        y_offset = row * crop_size

        # Pasting each processed crop into the correct location in the full image.
        reassembled_mask.paste(predicted_mask, (x_offset, y_offset))
        reassembled_overlay.paste(predicted_mask_overlay, (x_offset, y_offset))

    # Saving the reassembled images.
    reassembled_mask.save(os.path.join(output_folder, "reassembled_predicted_mask.png"))
    reassembled_overlay.save(os.path.join(output_folder, "reassembled_predicted_overlay.png"))


if __name__ == "__main__":      
    crop_size = 640  # Selecting the size the model was trained on.
    
    # Defining paths.
    image_path = "/turkey .jpeg"  
    output_crop_folder = "/data/external/cropped/input" 
    output_folder = "/data/external/cropped/output"

    # Croping the image first.
    crop_and_save(image_path, crop_size, output_crop_folder)

    # Process and reassembling the image.
    image_size = (2250, 1526)  # Original full image size (width, height) before cropping.
    model = YOLO("/results/yolov9e_640/final/weights/best.pt")

    process_and_reassemble(output_crop_folder, model, output_folder, crop_size, image_size)
    print(f"[INFO] Cropping main image and generating predictions completed.")