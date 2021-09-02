# coding: utf-8
"""Export a depth map as a 3D mesh on .obj format.

Creates a sparse 3D mesh from a single depth map image. A colored texture can also
be optionally mapped on to the 3D object if provided. The material and object are
stored with same name as input depth image and .obj and .mtl extensions respectively.

usage: export3d.py [-h] [-t TEXTURE] input output

positional arguments:
  input                         path of the input depth image.
  output                        path of output directory to save exported object.

optional arguments:
  -h, --help                    show this help message and exit
  -t TEXTURE, --texture TEXTURE path of the texture to map on to the mesh.
"""

import argparse
import os

import numpy as np

from utils import dmap2obj


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="path of the input depth image.")
    parser.add_argument("output", type=str, help="path of output directory to save exported object.")
    parser.add_argument("-t", "--texture", help='path of the texture to map on to the mesh.', default=None, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    infile = args.input
    outfile = os.path.join(out_dir, "export_" + os.path.splitext(os.path.basename(os.path.normpath(infile)))[0])

    dmap2obj(dmap=np.load(infile),
             outfile=outfile,
             texture=args.texture)
