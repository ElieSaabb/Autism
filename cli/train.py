# cli/train.py

import os
from glob import glob

from data_pipelines.dataloaders import prepare_dataset
from helpers.utils import load_phenotypic_labels, get_labels_from_filenames
from helpers.generators import DataGenerator3D
from model.cnn3d import build_3dcnn

def main():
    # === PATHS ===
    data_dir = "/home/linah03/Projects/Autism/dataset/func_mean_fcm_resized"
    pheno_csv = "/home/linah03/Projects/Autism/dataset/phenotypic_NYU.csv"

    # === Load data file paths ===
    filepaths = sorted(glob(os.path.join(data_dir, "*.nii.gz")))

    # === Load labels ===
    label_dict = load_phenotypic_labels(pheno_csv, site_filter="NYU")
    labels = get_labels_from_filenames(filepaths, label_dict)

    # === Load and preprocess data ===
    print("Preparing dataset...")
    X_train, X_test, y_train, y_test = prepare_dataset(filepaths, labels)

    # === Generators with augmentation ===
    train_gen = DataGenerator3D(X_train, y_train, batch_size=8, augment=True)
    val_gen = DataGenerator3D(X_test, y_test, batch_size=8, augment=False)

    # === Build and compile model ===
    print("Building model...")
    model = build_3dcnn(input_shape=(64, 64, 64, 1))
    model.summary()

    # === Train model ===
    print("Training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50
    )

    # === Save model ===
    out_model_path = "/project/your_username/models/asd_3dcnn.h5"
    model.save(out_model_path)
    print(f"Model saved to {out_model_path}")


if __name__ == "__main__":
    main()
