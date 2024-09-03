"""
Module Name: YoloYamlMaker
Description: This module creates a YAML configuration file for the YOLO model. The YAML file specifies the paths 
             to the training, validation, and test datasets, as well as the number of classes and their respective names.

Created: 16-07-2024

Usage:
    To create a YOLO YAML configuration file, specify the output path for the YAML file, paths to the training, validation, 
    and test datasets, the number of classes, and a list of class names.

    Example:
        create_yolo_yaml(output_path, train_path, val_path, test_path, nc, names)

Dependencies:
    - yaml (PyYAML)

Notes:
    - The created YAML file is typically used in YOLO model training to define dataset paths and class information.
    - The `path` field in the YAML is left empty to allow flexibility in specifying paths.
"""


import yaml


def create_yolo_yaml(output_path, train_path, val_path, test_path, nc, names):
    """
    Creates a YAML file for YOLO model configuration.

    Parameters:
    output_path (str): Path where the YAML file will be saved.
    train_path (str): Path to the training dataset.
    val_path (str): Path to the validation dataset.
    test_path (str): Path to the test dataset.
    nc (int): Number of classes.
    names (list): List of class names.

    Returns:
    None
    """
    data = {"path": "",  
            "train": train_path,
            "val": val_path,
            "test": test_path,
            "nc": nc,
            "names": names}

    with open(output_path, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style = False)


if __name__ == "__main__":
    # Defining paths.
    output_path = "data/final/dataset.yaml"
    train_path = "data/final/images/train"
    val_path = "data/final/images/val"
    test_path = "data/final/images/test"
    
    # Defining class names.
    names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    nc = len(names)  
    
    create_yolo_yaml(output_path, train_path, val_path, test_path, nc, names)
    print(f"[INFO] YAML file created at: {output_path}")