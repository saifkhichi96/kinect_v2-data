import math

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


def dmap2pcloud(dmap, K):
    """ Generates the point cloud from given depth map `dm_gt` using intrinsic
    camera matrix `K` with perspective projection.

    Args:
        dmap (np.array): Depth map, shape (H, W).
        K (np.array): Camera intrinsic matrix, shape (3, 3).

    Returns:
        np.array: Point cloud, shape (P, 3), P is # non-zero depth values.
    """
    Kinv = np.linalg.inv(K)

    y, x = np.where(dmap != 0.0)
    N = y.shape[0]
    z = dmap[y, x]

    pts_proj = np.vstack((x[None, :], y[None, :], np.ones((1, N))) * z[None, :])
    pcloud = (Kinv @ pts_proj).T

    return pcloud.astype(np.float32)


def tex2mtl(infile, outfile):
    with open(outfile, "w") as f:
        f.write("newmtl colored\n")
        f.write("Ns 10.0000\n")
        f.write("d 1.0000\n")
        f.write("Tr 0.0000\n")
        f.write("illum 2\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("map_Ka " + infile + "\n")
        f.write("map_Kd " + infile + "\n")


def vete(v, vt):
    return str(v) + "/" + str(vt)


def dmap2obj(dmap, outfile, texture=None):
    if texture is not None:
        tex2mtl(texture, outfile + ".mtl")

    h, w = dmap.shape

    fov = 70.6
    D = (h / 2) / math.tan(fov / 2)

    with open(outfile + ".obj", "w") as f:
        if texture is not None:
            f.write("mtllib " + texture + "\n")
            f.write("usemtl " + "colored" + "\n")

        ids = np.zeros((dmap.shape[1], dmap.shape[0]), int)
        vid = 1

        for u in range(0, w):
            for v in range(h - 1, -1, -1):

                d = dmap[v, u]

                ids[u, v] = vid
                if d == 0.0:
                    ids[u, v] = 0
                vid += 1

                x = u - w / 2
                y = v - h / 2
                z = -D

                norm = 1 / math.sqrt(x * x + y * y + z * z)

                t = d / (z * norm)

                x = -t * x * norm
                y = t * y * norm
                z = -t * z * norm

                f.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")

        for u in range(0, dmap.shape[1]):
            for v in range(0, dmap.shape[0]):
                f.write("vt " + str(u / dmap.shape[1]) + " " + str(v / dmap.shape[0]) + "\n")

        for u in range(0, dmap.shape[1] - 1):
            for v in range(0, dmap.shape[0] - 1):

                v1 = ids[u, v]
                v2 = ids[u + 1, v]
                v3 = ids[u, v + 1]
                v4 = ids[u + 1, v + 1]

                if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                    continue

                f.write("f " + vete(v1, v1) + " " + vete(v2, v2) + " " + vete(v3, v3) + "\n")
                f.write("f " + vete(v3, v3) + " " + vete(v2, v2) + " " + vete(v4, v4) + "\n")
