# cli/train.py

import os
#  Limit TFs own thread pools to avoid pthread_create errors 
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import numpy as np
from model.cnn3d import build_3dcnn

def main():
    # === PATHS ===
    preprocessed_dir = "/lustre04/scratch/linah03/Datasets/ABIDE/Outputs/cpac/filt_global/preprocessed_images"
    out_model_path   = "/lustre04/scratch/linah03/Autism/model/asd_3dcnn.h5"

    # === Load preprocessed data ===
    print("Loading preprocessed data...", flush=True)
    X_train = np.load(os.path.join(preprocessed_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(preprocessed_dir, "X_test.npy"))
    y_train = np.load(os.path.join(preprocessed_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(preprocessed_dir, "y_test.npy"))

    print(f"Shapes before expand: X_train={X_train.shape}, X_test={X_test.shape}", flush=True)

    # === Expand to 5-D (add channel axis) ===
    X_train = X_train[..., np.newaxis]  # now (N_train, D, H, W, 1)
    X_test  = X_test[...,  np.newaxis]  # now (N_test,  D, H, W, 1)

    print(f"Shapes after expand: X_train={X_train.shape}, X_test={X_test.shape}", flush=True)
    print(f"Labels: y_train={y_train.shape}, y_test={y_test.shape}", flush=True)

    # === Build tf.data.Dataset pipelines ===
    BATCH_SIZE = 8
    AUTOTUNE   = tf.data.AUTOTUNE

    # 1) Base Datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds   = tf.data.Dataset.from_tensor_slices((X_test,  y_test))

    # 2) Optional augmentation
    def augment(volume, label):
        # Random flip along one of the three axes
        axis = tf.random.uniform([], 0, 3, dtype=tf.int32)
        volume = tf.reverse(volume, axis=[axis])
        return volume, label

    # 3) Training pipeline: shuffle, augment, batch, prefetch
    train_ds = (
        train_ds
        .shuffle(buffer_size=len(X_train))
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    # 4) Validation pipeline: batch, prefetch
    val_ds = (
        val_ds
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    # === Build and compile model ===
    print("Building 3D CNN model...", flush=True)
    model = build_3dcnn(input_shape=X_train.shape[1:])  # e.g. (64,64,64,1)
    model.summary()

    # === Train ===
    print("Starting training...", flush=True)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50
    )

    # === Save model ===
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    model.save(out_model_path)
    print(f"Model saved to {out_model_path}", flush=True)


if __name__ == "__main__":
    main()
