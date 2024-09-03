"""
Module Name: YOLOVisualisationTool
Description: This module visualises the predictions made by the YOLO model alongside the ground truth masks. 
             It includes functionality to load images and labels, generate predictions, create segmentation maps, and 
             create composite images for comparison.

Created: 19-07-2024

Usage:
    To visualise predictions and ground truth, specify the paths to the images and their corresponding label files. 
    The script will generate and display composite images comparing the original image, ground truth, and predicted masks.

    Example:
        display_images(image_path, label_path, inference_model, output_dir, threshold = 0.1)

Dependencies:
    - matplotlib 
    - PIL 
    - numpy
    - cv2 
    - ultralytics 
    - os (built-in)

Notes:
    - The script assumes that the YOLO model has been trained and is available for loading.
    - The ground truth and predicted masks are saved alongside the original image for easy comparison.
"""


import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from ultralytics import YOLO
import os


# Defining the colour mappings for each class.
class_colours = {0: (0, 255, 0),     # no-damage: green
                1: (255, 255, 0),    # minor-damage: yellow
                2: (255, 165, 0),    # major-damage: orange
                3: (255, 0, 0)}      # destroyed: red


def get_outputs(image, model, threshold = 0.1):
    """
    Get model predictions (masks, bounding boxes, and labels) for a given image.

    Parameters:
    image (PIL.Image.Image): The input image for which to generate predictions.
    model (YOLO): The YOLO model used for generating predictions.
    threshold (float): Confidence threshold for predictions (default is 0.01).

    Returns:
    tuple: A tuple containing:
        - masks (list): A list of masks (polygon coordinates) for the detected objects.
        - boxes (list): A list of bounding boxes for the detected objects.
        - labels (list): A list of labels for the detected objects.
    """
    outputs = model.predict(image, imgsz = 640, conf = threshold)
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


def draw_segmentation_map(image, masks, boxes, labels):
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

    for mask, label in zip(masks, boxes, labels):
        color = class_colours[label]  # Using the class colour.

        if mask is not None and len(mask) > 0:
            poly = np.array(mask, dtype = np.int32)
            cv2.fillPoly(segmentation_map, [poly], color)

    # Combining the original image with the segmentation map.
    combined_image = cv2.addWeighted(image, alpha, segmentation_map, beta, gamma)

    return Image.fromarray(combined_image)


def draw_predicted_mask(masks, labels, height, width):
    """
    Draw predicted masks for detected objects in a blank image of given dimensions.

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


def load_annotations(label_path, img_w, img_h):
    """
    Load ground truth annotations.

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


def create_ground_truth_mask(image, label_path):
    """
    Create a ground truth mask image based on the label file.

    Parameters:
    image (PIL.Image.Image): The original image.
    label_path (str): Path to the label file.

    Returns:
    PIL.Image.Image: The ground truth mask image.
    """
    img_h, img_w = image.size[1], image.size[0]
    annotations = load_annotations(label_path, img_w, img_h)
    overlay = np.zeros((img_h, img_w, 3), dtype = np.uint8)

    for class_id, polygon in annotations:
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        colour = class_colours.get(class_id, (255, 255, 255))  # Default to white if class_id not found.
        cv2.fillPoly(overlay, [pts], colour)

    return Image.fromarray(overlay)


def save_image(image, filename, directory):
    """
    Save the image to a specified directory with a given filename.

    Parameters:
    image (PIL.Image.Image): The image to save.
    filename (str): The filename to use when saving the image.
    directory (str): The directory where the image will be saved.

    Returns:
    None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    image_path = os.path.join(directory, filename)
    image.save(image_path)


def create_composite_image(original, ground_truth, predicted, labels = ("Original Image", "Ground Truth", "Predicted Mask")):
    """
    Create a composite image that displays the original image, ground truth, and predicted masks side by side.

    Parameters:
    original (PIL.Image.Image): The original image.
    ground_truth (PIL.Image.Image): The ground truth mask image.
    predicted (PIL.Image.Image): The predicted mask image.
    labels (tuple): A tuple of strings representing the labels for each image (default: ("Original Image", "Ground Truth", "Predicted Mask")).

    Returns:
    PIL.Image.Image: The composite image.
    """
    # Ensuring all images are the same size.
    widths, heights = zip(*(i.size for i in [original, ground_truth, predicted]))

    # Setting the spacing between images and padding.
    padding = 20  
    label_height = 60 

    # Calculating the total width and height for the composite image.
    total_width = sum(widths) + (len(widths) - 1) * padding + 2 * padding  # Adding padding on both sides.
    max_height = max(heights) + label_height + 2 * padding  # Adding padding on top and bottom.

    # Creating a new blank image with a white background.
    composite_image = Image.new("RGB", (total_width, max_height), (255, 255, 255))

    # Setting font for captions.
    try:
        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
        font = ImageFont.truetype(font_path, 50)  
        
    except IOError:
        print("[ERROR] Arial font not found.")
        font = ImageFont.load_default()  # Fallback to default font.

    draw = ImageDraw.Draw(composite_image)

    # Pasting each image side by side with captions.
    x_offset = padding
    y_offset = padding  
    
    for i, img in enumerate([original, ground_truth, predicted]):
        composite_image.paste(img, (x_offset, y_offset))
        
        # Calculateingthe text size using textbbox.
        bbox = draw.textbbox((0, 0), labels[i], font = font)
        text_width = bbox[2] - bbox[0]
        
        label_x = x_offset + (img.width - text_width) // 2
        label_y = y_offset + img.height + padding // 2
        draw.text((label_x, label_y), labels[i], fill = "black", font = font)
        x_offset += img.width + padding

    return composite_image


def display_images(image_path, label_path, inference_model, output_dir, threshold = 0.1):
    """
    Display the original image, predicted masks, and ground truth side by side, and save the results.

    Parameters:
    image_path (str): Path to the original image.
    label_path (str): Path to the ground truth label file.
    inference_model (YOLO): The YOLO model used for inference.
    output_dir (str): Directory where the output images will be saved.
    threshold (float): Confidence threshold for the model predictions (default is 0.1).

    Returns:
    None
    """
    image = Image.open(image_path).convert("RGB")
    orig_image = image.copy()  # Keeping a copy of the original image for OpenCV functions and applying masks.
    img_w, img_h = image.size

    # Getting model predictions.
    masks, boxes, labels = get_outputs(image, inference_model, threshold)
    predicted_mask_overlay = draw_segmentation_map(orig_image, masks, boxes, labels)
    predicted_mask = draw_predicted_mask(masks, labels, img_h, img_w)

    # Creating ground truth mask
    ground_truth_mask = create_ground_truth_mask(image, label_path)

    # Saving individual images.
    save_image(image, "original_image.png", output_dir)
    save_image(predicted_mask_overlay, "predicted_mask_overlay.png", output_dir)
    save_image(predicted_mask, "predicted_mask.png", output_dir)
    save_image(ground_truth_mask, "ground_truth_mask.png", output_dir)

    # Creating and saving composite image.
    composite_image = create_composite_image(image, ground_truth_mask, predicted_mask)
    save_image(composite_image, "composite_image.png", output_dir)

    # Plotting all images
    fig, axes = plt.subplots(2, 2, figsize = (8, 10))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(predicted_mask_overlay)
    axes[0, 1].set_title("Predicted Mask Overlay")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(ground_truth_mask)
    axes[1, 0].set_title("Ground Truth Mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(predicted_mask)
    axes[1, 1].set_title("Predicted Mask")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Defining image paths.
    image_label_paths = [("/data/final/test/tuscaloosa-tornado_00000068_post_disaster.png", 
                          "/data/final/test/tuscaloosa-tornado_00000068_post_disaster.txt"),
        
                         ("/data/final/test/santa-rosa-wildfire_00000077_post_disaster.png", 
                          "/data/final/test/santa-rosa-wildfire_00000077_post_disaster.txt"),
        
                         ("/data/final/test/santa-rosa-wildfire_00000121_post_disaster.png", 
                          "/data/final/test/santa-rosa-wildfire_00000121_post_disaster.txt")]

    # Initialising the model.
    inference_model = YOLO("/results/yolov9e_640/final/weights/best.pt")

    # Getting composite images.
    for image_path, label_path in image_label_paths:
        output_dir = os.path.join("/results/ErrorAnalysis", os.path.splitext(os.path.basename(image_path))[0])
        display_images(image_path, label_path, inference_model, output_dir, threshold = 0.1)

        print(f"[INFO] Composite image created for: {image_path}")