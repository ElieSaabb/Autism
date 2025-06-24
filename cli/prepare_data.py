# prepare_data.py

import os
import numpy as np
from glob import glob

from data_pipelines.dataloaders import prepare_dataset
from helpers.utils import load_phenotypic_labels, get_labels_from_filenames

def main():
    # === PATHS ===
    data_dir = "/lustre04/scratch/linah03/Datasets/ABIDE/Outputs/cpac/filt_global/func_preproc/"
    pheno_csv = "/lustre04/scratch/linah03/Datasets/ABIDE/phenotypic_NYU.csv"
    out_dir = "/lustre04/scratch/linah03/Datasets/ABIDE/Outputs/cpac/filt_global/preprocessed_images"
    os.makedirs(out_dir, exist_ok=True)

    # === Load data file paths ===
    filepaths = sorted(glob(os.path.join(data_dir, "*.nii.gz")))
    filepaths = [p for p in filepaths if os.path.basename(p).startswith('NYU_')]

    # === Load labels ===
    label_dict = load_phenotypic_labels(pheno_csv, site_filter="NYU")
    labels = get_labels_from_filenames(filepaths, label_dict)

    # === Load and preprocess data ===
    print("Preparing dataset...")
    X_train, X_test, y_train, y_test = prepare_dataset(filepaths, labels)

    # === Save preprocessed data ===
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    print(f"Preprocessed data saved to {out_dir}")

if __name__ == "__main__":
    main()
