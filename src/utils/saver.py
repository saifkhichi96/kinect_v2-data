import os

import numpy as np
from cv2 import cv2


def create_save_directories(path):
    os.makedirs(f'{path}/images/', exist_ok=True)
    os.makedirs(f'{path}/depth_maps/', exist_ok=True)
    os.makedirs(f'{path}/normals/', exist_ok=True)
    os.makedirs(f'{path}/masks/', exist_ok=True)


def save_frame(path, item_id, frame):
    """Save current frame of the RGB-D dataset.

    Color image is a BGR image with three channels, each with values ranging
    from 0-255. Depth map is a single-channel array with values in range 0-1
    and the surface normals have values in range 0-1, where each value is a 3D
    vector.

    :param path:
    :param item_id:
    :param frame:
    :return:
    """
    color, depth, norms, mask = frame
    cv2.imwrite(f'{path}/images/rgb_{item_id:04}.tiff', color)
    np.save(f'{path}/depth_maps/depth_{item_id:04}.npy', depth)
    np.save(f'{path}/normals/normals_{item_id:04}.npy', norms)
    cv2.imwrite(f'{path}/masks/mask_{item_id:04}.png', np.logical_not(mask).astype('uint8') * 255)
