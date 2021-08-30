# coding: utf-8

import sys

import numpy as np
from cv2 import cv2

from kinect import start_capture


def depth2normals(d_map):
    """Computes surface normals from a depth map.

    :param d_map: A grayscale depth map image as a numpy array of size (H,W).
    :return: The corresponding surface normals map as numpy array of size (H,W,3).
    """
    zx = cv2.Sobel(d_map, cv2.CV_64F, 1, 0, ksize=5)
    zy = cv2.Sobel(d_map, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(d_map)))[:, :, ::-1]
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    normal[:, :, 0] /= 2
    return normal


def normalize_brightness(im_color):
    hsv = cv2.cvtColor(im_color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    result = cv2.equalizeHist(v)
    hsv = cv2.merge((h, s, result))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def save_frame(color, depth, normals):
    """Save current frame of the RGB-D dataset.

    Color image is a BGR image with three channels, each with values ranging
    from 0-255. Depth map is a single-channel array with values in range 700-1250
    and the surface normals have values in range 0-1, where each value is a 3D
    vector.

    :param color:
    :param depth:
    :param normals:
    :return:
    """
    pass


def show_frame(color, depth, norms, mask):
    """Show current frame of the RGB-D dataset as images.

    :param color:
    :param depth:
    :param norms:
    :param mask:
    :return:
    """
    # apply a colormap on grayscale depth map, makes easier to see depth changes
    depth -= 700
    depth *= 255 / (1250-700)
    depth = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_BONE)

    # noinspection PyPep8Naming
    WINDOW_BG = 128  # gray window background
    depth[mask] = WINDOW_BG
    color[mask] = WINDOW_BG
    norms[mask] = WINDOW_BG / 255  # (divide by 255 because normal values are in range 0-1)

    dst = np.hstack((color / 255, depth / 255, norms))
    cv2.imshow('RGB \t-\t Depth \t-\t Surface Normals', dst)


def frame_listener(color, depth):
    # Crop extra spacing around the object of interest (depends on relative placement of
    # camera and the subject in real world).
    color = color[60:-15, 200:-14]
    depth = depth[60:-15, 200:-14]

    # Compute surface normals from depth map
    norms = depth2normals(depth)

    # Keep foreground only
    mask = np.logical_or(depth > 1250, depth < 700)  # 0.7m < foreground < 1.25m
    color[mask] = 0
    depth[mask] = 0
    norms[mask] = 0

    # (optional) normalize brightness of foreground in RGB image
    color = normalize_brightness(color)

    save_frame(color, depth, norms)
    show_frame(color, depth, norms, mask)

    if cv2.waitKey(delay=1) == ord('q'):
        raise KeyboardInterrupt


def main():
    try:
        start_capture(callback=frame_listener)
    except Exception as ex:
        print(ex)
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
