# cli/predict.py

import os
import numpy as np
import nibabel as nib
from model.cnn3d import build_3dcnn
from tensorflow.keras.models import load_model
from data_pipelines.preprocessing import rescale_volume, resize_volume

TARGET_SHAPE = (64, 64, 64)

def load_and_prepare_volume(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    data = rescale_volume(data)
    data = resize_volume(data, TARGET_SHAPE)
    data = data[..., None]  # add channel dimension
    return np.expand_dims(data, axis=0)  # batch dimension

def main():
    # === Path to model and volume ===
    model_path = "/project/your_username/models/asd_3dcnn.h5"
    nii_path = "/project/your_username/data/new_subject.nii.gz"

    # === Load and prepare volume ===
    volume = load_and_prepare_volume(nii_path)
    print(f"Loaded volume from: {nii_path} | Shape: {volume.shape}")

    # === Load model and predict ===
    model = load_model(model_path)
    pred = model.predict(volume)[0][0]
    label = "ASD" if pred >= 0.5 else "TC"
    
    print(f"\nPrediction: {label} (confidence = {pred:.4f})")

if __name__ == "__main__":
    main()