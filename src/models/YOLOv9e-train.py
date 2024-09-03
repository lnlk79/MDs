from ultralytics import YOLO
import os
from pathlib import Path
import random
import numpy as np
import torch


def set_seeds(value):
    """
    Set the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Parameters:
    value (int): The seed value to set.

    Returns:
    None
    """
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(value)
        torch.cuda.manual_seed_all(value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

set_seeds(42)

# Setting the working directory.
desired_path = "/MDS_Dissertation"
os.chdir(desired_path)
print(f"Current working directory: {os.getcwd()}")

# Paths to dataset and model configurations.
yaml_file_path = "./data/final/dataset.yaml"
model_yaml_path = "yolov9e-seg.yaml"  
pretrained_model_path = "yolov9e-seg.pt"

# Destination directory for results.
project_base = Path(desired_path) / "results/yolov9e_640"
project_base.mkdir(parents = True, exist_ok = True) 

# Creating subdirectory for this specific training.
name = "45-epochs"
project = project_base / name

# Creating model instance from YAML and loading the pre-trained weights.
model = YOLO(model_yaml_path)  
model = YOLO(pretrained_model_path) 

# Training the model
results = model.train(
    data = yaml_file_path,  
    project = str(project),
    name = name,
    epochs = 45,             # Number of epochs
    patience = 15,           # Setting patience for early stopping
    batch = 1,               # Batch number
    imgsz = 640,             # Image size
    lr0 = 0.001,             # Learning rate
    optimizer = "Adam",      # Optimiser
    weight_decay = 0.0005,   # Weight decay
    hsv_h = 0.015,           # Hue augmentation
    hsv_s = 0.7,             # Saturation augmentation
    hsv_v = 0.4,             # Value augmentation
    degrees = 10,            # Rotation
    translate = 0.1,         # Translation
    scale = 0.5,             # Scaling
    shear = 0.1,             # Shear
    label_smoothing = 0.1,   # Label smoothing
    multi_scale = False,     
    device = "cuda:0")

print("[INFO] Training completed!")
