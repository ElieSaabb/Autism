# helpers/generators.py

import numpy as np
import random
from tensorflow.keras.utils import Sequence
from scipy.ndimage import rotate

class DataGenerator3D(Sequence):
    """
    Keras-style data generator for 3D volumes with on-the-fly augmentation.
    Expects:
      - X: shape (N, D, H, W, 1)
      - y: shape (N,)
    """

    def __init__(self, X, y, batch_size=8, shuffle=True, augment=True,
                 max_rot_deg=10, flip_prob=0.5):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.max_rot = max_rot_deg
        self.flip_prob = flip_prob
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        Xb = self.X[batch_idx].copy()
        yb = self.y[batch_idx]

        if self.augment:
            for i in range(len(Xb)):
                vol = Xb[i, ..., 0]  # shape: (D, H, W)

                # Random flip
                if random.random() < self.flip_prob:
                    axis = random.choice([0, 1, 2])
                    vol = np.flip(vol, axis=axis)

                # Random rotation
                if random.random() < self.flip_prob:
                    angle = random.uniform(-self.max_rot, self.max_rot)
                    plane = random.choice([(0, 1), (0, 2), (1, 2)])
                    vol = rotate(vol, angle, axes=plane, reshape=False, order=1, mode='nearest')

                Xb[i, ..., 0] = vol

        return Xb, yb

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)