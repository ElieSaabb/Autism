# helpers/utils.py

import os
import pandas as pd

def load_phenotypic_labels(csv_path, site_filter="NYU"):
    """
    Load labels from the ABIDE phenotype CSV.

    Returns a dictionary: {subject_id: label}
    Where label = 1 for ASD, 0 for TC
    """
    pheno = pd.read_csv(csv_path)

    if site_filter:
        pheno = pheno[pheno["SITE_ID"] == site_filter]

    subid_to_label = dict(
        zip(pheno["SUB_ID"], pheno["DX_GROUP"].map({1: 1, 2: 0}))
    )

    return subid_to_label


def get_labels_from_filenames(filepaths, label_dict):
    """
    Extract subject ID from each filename and return a list of corresponding labels.
    """
    labels = []
    for path in filepaths:
        fname = os.path.basename(path)
        try:
            subject_id = int(fname.split('_')[1])  # '0050995' -> 50995
            label = label_dict.get(subject_id)
            if label is None:
                raise ValueError(f"No label found for subject ID {subject_id}")
            labels.append(label)
        except Exception as e:
            raise RuntimeError(f"Failed to parse label from {fname}") from e

    return labels