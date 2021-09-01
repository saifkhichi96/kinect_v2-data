import numpy as np
from cv2 import cv2


def dmap2norm(dmap):
    """Computes surface normals from a depth map.

    :param dmap: A grayscale depth map image as a numpy array of size (H,W).
    :return: The corresponding surface normals map as numpy array of size (H,W,3).
    """
    zx = cv2.Sobel(dmap, cv2.CV_64F, 1, 0, ksize=5)
    zy = cv2.Sobel(dmap, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(dmap)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-1
    normal += 1
    normal /= 2
    return normal[:, :, ::-1]
