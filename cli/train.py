# cli/train.py

import os
import numpy as np

from helpers.generators import DataGenerator3D
from model.cnn3d import build_3dcnn

def main():
    # === PATHS ===
    preprocessed_dir = "/lustre04/scratch/linah03/Datasets/ABIDE/Outputs/cpac/filt_global/preprocessed_images"
    out_model_path   = "/lustre04/scratch/linah03/Autism/model/asd_3dcnn.h5"

    # === Load preprocessed data ===
    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(preprocessed_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(preprocessed_dir, "X_test.npy"))
    y_train = np.load(os.path.join(preprocessed_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(preprocessed_dir, "y_test.npy"))
    print(f"  • X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  • X_test : {X_test.shape}, y_test : {y_test.shape}")

    # === Generators with augmentation ===
    train_gen = DataGenerator3D(X_train, y_train, batch_size=8, augment=True)
    val_gen   = DataGenerator3D(X_test,  y_test,  batch_size=8, augment=False)

    # === Build and compile model ===
    print("Building model...")
    # assuming data are 64×64×64 volumes with 1 channel
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
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    model.save(out_model_path)
    print(f"Model saved to {out_model_path}")

if __name__ == "__main__":
    main()
