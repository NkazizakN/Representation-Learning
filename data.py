import os
import cv2
import numpy as np
from tqdm import tqdm
import zipfile
import kagglehub

BASE_DIR = "datasets"
os.makedirs(BASE_DIR, exist_ok=True)

DATASETS = [
    ("eyepacs", "ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy"),
    ("rfmid", "ozlemhakdagli/retinal-fundus-multi-disease-image-dataset-rfmid"),
    ("drive", "andrewmvd/drive-digital-retinal-images-for-vessel-extraction"),
    ("origa", "ferencjuhsz/origa-retinal-fundus-image-dataset")
]

for folder_name, kaggle_id in DATASETS:
    dataset_path = os.path.join(BASE_DIR, folder_name)
    os.makedirs(dataset_path, exist_ok=True)

    print(f"Downloading {folder_name}...")
    path = kagglehub.dataset_download(kaggle_id)
    print(f"Path to {folder_name} dataset files:", path)
