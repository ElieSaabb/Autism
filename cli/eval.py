import os
from glob import glob

from data_pipelines.dataloaders import prepare_dataset
from helpers.utils import load_phenotypic_labels, get_labels_from_filenames
from model.cnn3d import build_3dcnn
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def main():
    # === Paths ===
    data_dir = "/project/your_username/data/func_mean_fcm_resized"
    pheno_csv = "/project/your_username/data/ABIDE_I_Phenotypic.csv"
    model_path = "/project/your_username/models/asd_3dcnn.h5"

    # === Load test data ===
    filepaths = sorted(glob(os.path.join(data_dir, "*.nii.gz")))
    label_dict = load_phenotypic_labels(pheno_csv, site_filter="NYU")
    labels = get_labels_from_filenames(filepaths, label_dict)

    # Only load the test set from prepare_dataset
    _, X_test, _, y_test = prepare_dataset(filepaths, labels)

    # === Load trained model ===
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    # === Evaluate ===
    print("Evaluating...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f} | Loss: {loss:.4f}")

    # === Detailed metrics ===
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["TC", "ASD"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()