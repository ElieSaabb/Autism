# data_pipelines/dataloaders.py

import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from data_pipelines.preprocessing import rescale_volume, resize_volume
from data_pipelines.preprocessing import collapse_and_fcm


TARGET_SHAPE = (64, 64, 64)


COLLAPSE_OUT_DIR = "/lustre04/scratch/linah03/Datasets/ABIDE/Outputs/cpac/filt_global/3d_data"
os.makedirs(COLLAPSE_OUT_DIR, exist_ok=True)

def load_volume(path):
    # 1) Load the 4D image
    img4d  = nib.load(path)


    # 2) Build output filename for the collapsed 3D NIfTI
    base      = os.path.basename(path)
    subj      = base.replace("_func_preproc.nii.gz", "")  # adjust suffix if needed
    out_fname = f"{subj}_collapsed_fcm.nii.gz"
    out_path  = os.path.join(COLLAPSE_OUT_DIR, out_fname)

    # 3) Collapse & save: pass both in_4d and out_path
    data3d = collapse_and_fcm(img4d, out_path)  # returns the 3D NumPy array

    # 4) Resize for your network
    data_resized = resize_volume(data3d, TARGET_SHAPE)
    return data_resized

def load_volumes(paths):
    X = np.zeros((len(paths),) + TARGET_SHAPE, dtype=np.float32)
    for i, p in enumerate(paths):
        X[i] = load_volume(p)
    return X


def prepare_dataset(filepaths, labels, test_size=0.2, random_state=42):

    print(f"[DEBUG] prepare_dataset got {len(filepaths)} files, e.g.: {filepaths[:5]}")


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
