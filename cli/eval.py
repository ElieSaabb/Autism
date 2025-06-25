# cli/eval.py
# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def main():
    # === PATHS ===
    preprocessed_dir = "/lustre04/scratch/linah03/Datasets/ABIDE/Outputs/cpac/filt_global/preprocessed_images"
    model_path = "/lustre04/scratch/linah03/WorkSpace_ELI/Autism/model/asd_3dcnn.h5"

    # === Load model ===
    print("Loading model from", model_path, flush=True)
    model = tf.keras.models.load_model(model_path)

    # === Load test data ===
    print("Loading test data...", flush=True)
    X_test = np.load(os.path.join(preprocessed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(preprocessed_dir, "y_test.npy"))

    # add channel axis if needed
    if X_test.ndim == 4:
        X_test = X_test[..., np.newaxis]  # now (N_test, D, H, W, 1)

    # === Evaluate ===
    print("Evaluating on test set...", flush=True)
    loss, acc = model.evaluate(X_test, y_test, batch_size=8, verbose=1)
    print(f"\nTest loss: {loss:.4f}", flush=True)
    print(f"Test accuracy: {acc:.4f}", flush=True)

    # === Predictions & Metrics ===
    print("\nGenerating predictions...", flush=True)
    y_prob = model.predict(X_test, batch_size=8, verbose=1)[:, 0]
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nClassification Report:", flush=True)
    print(classification_report(y_test, y_pred, digits=4), flush=True)

    print("Confusion Matrix:", flush=True)
    print(confusion_matrix(y_test, y_pred), flush=True)


if __name__ == "__main__":
    main()