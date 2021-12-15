import os

import cv2
import numpy as np

from torch.utils.data import Dataset
from .helpers import ls


class RGBDRealDataset(Dataset):
    """Dataset class for loading data from memory."""

    def __init__(self, path, transform=None):
        """
        Args:
            path (string): Path to the dataset.
            labels (list): List of labels.
        """
        self.transform = transform
        self.images = []
        self.dmaps = []
        self.nmaps = []
        self.masks = []

        objects = sorted(os.listdir(path))
        for o in objects:
            obj_dir = os.path.join(path, o)
            if os.path.isdir(obj_dir):
                sequences = os.listdir(obj_dir)
                for s in sequences:
                    seq_dir = os.path.join(obj_dir, s)
                    if os.path.isdir(seq_dir):
                        self.images += [f'{seq_dir}/images/{p}' for p in ls(f'{seq_dir}/images/', '.tiff')]
                        self.dmaps += [f'{seq_dir}/depth_maps/{p}' for p in ls(f'{seq_dir}/depth_maps/', '.npy')]
                        self.nmaps += [f'{seq_dir}/normals/{p}' for p in ls(f'{seq_dir}/normals/', '.npy')]
                        self.masks += [f'{seq_dir}/masks/{p}' for p in ls(f'{seq_dir}/masks/', '.png')]


    def __len__(self):
        """Return the size of dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get the item at index idx."""

        # Get the data and label
        data = cv2.imread(self.images[idx])
        dmap = np.load(self.dmaps[idx]).astype(np.float32)
        nmap = np.load(self.nmaps[idx]).astype(np.float32)
        nmap /= nmap.max()
        mask = cv2.imread(self.masks[idx], 0) < 255

        # Apply transformation if any
        if self.transform:
            data = self.transform(data)

        # Return the data and label
        return data, (dmap, nmap, mask)
