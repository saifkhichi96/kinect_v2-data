import numpy as np
from cv2 import cv2

from .depth3d import dmap2norm
from .saver import create_save_directories, save_frame
from .segmentation import segment


def create_view(frame):
    """Show current frame of the RGB-D dataset as images.

    :param frame:
    :return:
    """
    color, depth, norms, mask = frame

    # apply a colormap on grayscale depth map, makes easier to see depth changes
    depth = cv2.cvtColor(depth * 255, cv2.COLOR_GRAY2BGR)

    # noinspection PyPep8Naming
    WINDOW_BG = 0  # gray window background
    depth[mask] = WINDOW_BG
    color[mask] = WINDOW_BG
    norms[mask] = WINDOW_BG / 255  # (divide by 255 because normal values are in range 0-1)

    mask = cv2.cvtColor(np.logical_not(mask).astype('uint8') * 255, cv2.COLOR_GRAY2BGR)
    dst1 = np.hstack((color / 255, mask * 255))
    dst2 = np.hstack((depth / 255, norms))
    return (np.vstack((dst1, dst2)) * 255).astype(np.uint8)
