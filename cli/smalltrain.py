#!/usr/bin/env python3
import os
# ── Force single‐thread TF and tf.data to avoid pthread_create OOM ──
os.environ["TF_DATA_THREADPOOL_SIZE"]      = "1"
os.environ["TF_NUM_INTRAOP_THREADS"]       = "1"
os.environ["TF_NUM_INTEROP_THREADS"]       = "1"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import numpy as np
from model.cnn3d import build_3dcnn

def main():
    # === PATHS ===
    preprocessed_dir = (
        "/lustre04/scratch/linah03/Datasets/ABIDE/"
        "Outputs/cpac/filt_global/preprocessed_images"
    )
    out_model_path = "/lustre04/scratch/linah03/WorkSpaceELI/Autism/model/small_asd_3dcnn.h5"

    # === Load & expand data ===
    print("Loading preprocessed data...", flush=True)
    X_train = np.load(os.path.join(preprocessed_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(preprocessed_dir, "X_test.npy"))
    y_train = np.load(os.path.join(preprocessed_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(preprocessed_dir, "y_test.npy"))

    # add channel dim
    X_train = X_train[..., np.newaxis]   # (N_train, D, H, W, 1)
    X_test  = X_test[...,  np.newaxis]   # (N_test,  D, H, W, 1)

    print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape, flush=True)

    # === DEBUG SUBSAMPLING ===
    DEBUG = True
    DEBUG_SIZE   = 20   # number of subjects to use in debug mode
    DEBUG_EPOCHS = 2    # fewer epochs
    DEBUG_BATCH  = 4    # smaller batch size

    if DEBUG:
        print(f"🔍 Debug mode: using first {DEBUG_SIZE} samples", flush=True)
        X_train, y_train = X_train[:DEBUG_SIZE], y_train[:DEBUG_SIZE]
        X_test,  y_test  = X_test[:DEBUG_SIZE],  y_test[:DEBUG_SIZE]
        epochs     = DEBUG_EPOCHS
        batch_size = DEBUG_BATCH
    else:
        epochs     = 50
        batch_size = 8

    # === tf.data pipeline (single‐threaded) ===
    def augment(volume, label):
        axis = tf.random.uniform([], 0, 3, dtype=tf.int32)
        return tf.reverse(volume, axis=[axis]), label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=len(X_train))
        .map(augment, num_parallel_calls=1)
        .batch(batch_size)
        .prefetch(1)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(batch_size)
        .prefetch(1)
    )

    # === Build & compile model ===
    print("Building model...", flush=True)
    model = build_3dcnn(input_shape=X_train.shape[1:])  # e.g. (64,64,64,1)
    model.summary()

    # === Train ===
    print(f"Starting training for {epochs} epochs, batch size {batch_size}...", flush=True)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # === Save ===
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    model.save(out_model_path)
    print("Small‐run model saved to", out_model_path, flush=True)


if __name__ == "__main__":
    main()
