# data_pipelines/preprocessing.py

import os
import numpy as np
import nibabel as nib
from nilearn.image import mean_img
import skfuzzy as fuzz
from scipy.ndimage import zoom

def fcm_tissue_normalize(in_nii, out_nii, n_clusters=3, m=2., error=0.005, maxiter=1000):
    img = nib.load(in_nii)
    data = img.get_fdata()
    mask = data > 0
    vox = data[mask].ravel()

    cntr, u, *_ = fuzz.cluster.cmeans(vox[np.newaxis, :], n_clusters, m, error, maxiter)
    labels = np.argmax(u, axis=0)
    means = [vox[labels == i].mean() for i in range(n_clusters)]

    norm_vox = np.zeros_like(vox)
    for i, μ in enumerate(means):
        norm_vox[labels == i] = vox[labels == i] / μ

    norm_data = data.copy()
    norm_data[mask] = norm_vox

    os.makedirs(os.path.dirname(out_nii), exist_ok=True)
    nib.save(nib.Nifti1Image(norm_data, img.affine, img.header), out_nii)


def collapse_and_fcm(in_4d, out_3d, n_clusters=3, m=2., error=0.005, maxiter=1000):
    mean_img_3d = mean_img(in_4d)
    data = mean_img_3d.get_fdata()
    mask = data > 0
    vox = data[mask].ravel()

    cntr, u, *_ = fuzz.cluster.cmeans(vox[np.newaxis, :], n_clusters, m, error, maxiter)
    labels = np.argmax(u, axis=0)
    means = [vox[labels == i].mean() for i in range(n_clusters)]

    norm_vox = np.zeros_like(vox)
    for i, μ in enumerate(means):
        norm_vox[labels == i] = vox[labels == i] / μ

    norm_data = data.copy()
    norm_data[mask] = norm_vox

    nib.save(nib.Nifti1Image(norm_data, mean_img_3d.affine, mean_img_3d.header), out_3d)

    return norm_data


def rescale_volume(data):
    mask = data > 0
    vox = data[mask]
    vmin, vmax = vox.min(), vox.max()
    norm_data = np.zeros_like(data)
    norm_data[mask] = (vox - vmin) / (vmax - vmin + 1e-8)
    return norm_data


def resize_volume(data, target_shape=(64, 64, 64)):
    zoom_factors = [t / float(s) for t, s in zip(target_shape, data.shape)]
    return zoom(data, zoom_factors, order=1)
