# data_pipelines/dataloaders.py

import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from data_pipelines.preprocessing import rescale_volume, resize_volume

TARGET_SHAPE = (64, 64, 64)


def load_volume(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    data = rescale_volume(data)
    data = resize_volume(data, TARGET_SHAPE)
    return data[..., None]  # add channel dim


def load_volumes(paths):
    X = np.zeros((len(paths),) + TARGET_SHAPE + (1,), dtype=np.float32)
    for i, p in enumerate(paths):
        X[i] = load_volume(p)
    return X


def prepare_dataset(filepaths, labels, test_size=0.2, random_state=42):
    subjects = np.array([os.path.basename(f).split('_')[1] for f in filepaths])
    X_paths = np.array(filepaths)
    y_labels = np.array(labels)

    train_idx, test_idx = train_test_split(
        np.arange(len(subjects)),
        test_size=test_size,
        stratify=y_labels,
        random_state=random_state
    )

    X_train = load_volumes(X_paths[train_idx])
    y_train = y_labels[train_idx]
    X_test = load_volumes(X_paths[test_idx])
    y_test = y_labels[test_idx]

    return X_train, X_test, y_train, y_test