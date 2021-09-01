import numpy as np
from cv2 import cv2


def normalize_brightness(im_color):
    hsv = cv2.cvtColor(im_color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    result = cv2.equalizeHist(v)
    hsv = cv2.merge((h, s, result))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def mask_skin(im_color):
    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    lower = np.array([0, 0, 0], dtype="uint8")
    upper = np.array([50, 255, 255], dtype="uint8")

    # convert frame to the HSV color space, and determine the HSV pixel
    # intensities that fall into the specified upper and lower boundaries
    converted = cv2.cvtColor(im_color, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    skin_mask = skin_mask != 0
    return skin_mask


def artefact_mask(depth, mask):
    try:
        copy = np.copy(depth * 255).astype(np.uint8)
        copy[mask] = 0
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(copy, connectivity=4)

        # Find the largest non background component.
        # Note: range() starts from 1 since 0 is the background label.
        max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
                                  key=lambda x: x[1])
        return output != max_label
    except Exception as ex:
        print(ex)
        return mask


def segment(color, depth, min_depth=500, max_depth=1500, skin=True, artefacts=True):
    # Get background mask (i.e. keep objects 0.5-1.5 meter away from camera)
    mask = np.logical_or(depth > max_depth, depth < min_depth)

    # Get skin mask (i.e. keep clothes only)
    if skin:
        mask = np.logical_or(mask, mask_skin(color))

    # Normalize depth values between 0-1 and apply mask
    depth = (depth - min_depth) / (max_depth - min_depth)
    depth[depth < 0] = 0

    # Get mask for small artefacts caused by skin removal
    if artefacts:
        mask = np.logical_or(mask, artefact_mask(depth, mask))

    # Normalize color image and apply mask
    # color = normalize_brightness(color)
    color[mask] = 0

    # Fill holes in depth map and apply mask
    holes = (depth == 0).astype(np.uint8)
    depth = cv2.inpaint(depth, holes, 7, cv2.INPAINT_NS)
    depth[mask] = 0

    return color, depth, mask
