MDs
==============================

## Overview

The table below provides a description of each script. To reproduce the results, follow the steps outlined below.

## Reproducing the Results

1. **Download the Data**:  
   Download the necessary data from the [xView2 website](https://xview2.org/dataset).

2. **Install Dependencies**:  
   Ensure you have all required dependencies installed by using the `requirements.txt` file provided in the GitHub repository. You can install these dependencies by running the following command:
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the Scripts**:  
   Execute the scripts in the order they appear in the table below. Be sure to adjust the paths within each script as needed to match your environment.

4. **Expected Outputs**:  
   The scripts will process the data and generate the corresponding outputs as described in the dissertation.

## Acknowledgments

Acknowledgment is given to xView2 for providing the necessary data, Maxar for their satellite imagery used for predictions, and Ultralytics for their contributions to the YOLO model.

## Description of Scripts Used for Data Processing, Model Training, and Visualisations

| Python Script          | Description |
|------------------------|-------------|
| **DisasterFilter**      | Moves “post_disaster” images and their corresponding labels from multiple source directories to a specified destination directory, organising them into separate “images” and “labels” folders. |
| **SplitMakerXview**     | Prepares the xView2 dataset by splitting images and labels from multiple source directories into train, validation, and test sets with specified ratios, then consolidates these splits into single train, validation, and test folders. |
| **xView2toYOLOConverter** | Converts xView2 dataset annotations from JSON format to YOLO format. The conversion normalises the polygon coordinates and maps the damage classification categories to YOLO class indices. |
| **AnnotationVisualiser** | Visualises annotations for images by overlaying polygon annotations on the images. It supports both raster (GeoTIFF) and standard image formats and allows for visual inspection of annotations by blending them with the original images. |
| **LabelCleaner**        | Checks label files within a directory and deletes any label files (and their corresponding images) that contain only class 0 annotations. |
| **TifConverter**        | Converts TIF images to PNG format using rasterio for reading the images and OpenCV for saving them. It processes all TIF files in a specified directory and its subdirectories. |
| **YoloYamlMaker**       | Creates a YAML configuration file for the YOLO model. The YAML file specifies the paths to the training, validation, and test datasets, as well as the number of classes and their respective names. |
| **YOLOv9e-train**       | This script trains the model and contains a seed setting function for reproduction. |
| **AblationSplit**       | Handles the creation of the ablation dataset by copying a percentage of image-label pairs from existing dataset splits (train, val, test) to a new location. |
| **YOLOVisualisationTool** | Visualises the predictions made by the YOLO model alongside the ground truth masks. It includes functionality to load images and labels, generate predictions, create segmentation maps, and create composite images for comparison. |
| **ImageCropAndReassemble** | Crops large images into smaller sections, processes them using the YOLO model to generate predictions, and then reassembles the processed sections back into the original image. |

---
