"""
Module Name: xView2toYOLOConverter
Description: This module converts xView2 dataset annotations from JSON format to YOLO format. The conversion normalises 
             the polygon coordinates and maps the damage classification categories to YOLO class indices.

Created: 16-07-2024

Usage:
    To convert xView2 annotations to YOLO format, specify the directory containing the xView2 JSON files and 
    the output directory where YOLO-format text files will be saved.

    Example:
        convert_annotations(annotation_dir, output_dir)

Dependencies:
    - json (built-in)
    - glob (built-in)
    - os (built-in)

Notes:
    - The script assumes that the input JSON files contain metadata with image dimensions and feature annotations 
      as polygons in Well-Known Text (WKT) format.
    - The damage classes are mapped to YOLO class indices according to the `class_to_index_mappings` dictionary.
"""


import json
import glob
import os

# Defining mappings from xView2 classes to YOLO class indices.
class_to_index_mappings = {"no-damage": 0,
                           "minor-damage": 1,
                           "major-damage": 2,
                           "destroyed": 3}


def convert_single_annotation(json_data, output_dir):
    """
    Convert a single xView2 annotation (in JSON format) to YOLO format.

    Parameters:
    json_data (dict): The xView2 annotation data loaded from a JSON file.
    output_dir (str): The directory where the converted YOLO annotation file will be saved.

    Returns:
    None
    """
    width = int(json_data["metadata"]["width"])
    height = int(json_data["metadata"]["height"])
    image_name = json_data["metadata"]["img_name"]
    xy_features = json_data["features"]["xy"]
    
    yolo_annotations = []
    
    for xy_feature in xy_features:
        if "subtype" in xy_feature["properties"] and xy_feature["properties"]["subtype"] != "un-classified":
            category_id = class_to_index_mappings[xy_feature["properties"]["subtype"]]
            polygon_text = xy_feature["wkt"]
            
            polygon_values = polygon_text[
                polygon_text.find("((") + 2:-2].replace(",", "").split(" ")
            
            polygon_coords = []
            for i in range(0, len(polygon_values), 2):
                x, y = float(polygon_values[i]), float(polygon_values[i+1])
                normalised_x = x / width
                normalised_y = y / height
                polygon_coords.extend([normalised_x, normalised_y])
            
            polygon_annotation = f"{category_id} " + " ".join(map(str, polygon_coords))
            yolo_annotations.append(polygon_annotation)
    
    yolo_filename = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")
    with open(yolo_filename, "w") as yolo_file:
        yolo_file.write("\n".join(yolo_annotations))


def convert_annotations(annotation_dir, output_dir):
    """
    Convert all xView2 JSON annotation files in a directory to YOLO format.

    Parameters:
    annotation_dir (str): The directory containing xView2 JSON annotation files.
    output_dir (str): The directory where the converted YOLO annotation files will be saved.

    Returns:
    None
    """
    annotation_files = glob.glob(os.path.join(annotation_dir, "*.json"))
    
    for filename in annotation_files:
        with open(filename, "r") as file:
            json_data = json.load(file)
        convert_single_annotation(json_data, output_dir)


if __name__ == "__main__":
    # Paths to the xView2 annotations and YOLO annotations.
    annotation_dir = "data/processed/labels/"  
    output_dir = "data/processed/yolo_labels/"  
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    convert_annotations(annotation_dir, output_dir)

    print(f"[INFO] Converting JSON to YOLO completed.")