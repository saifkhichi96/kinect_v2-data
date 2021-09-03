# coding: utf-8
"""Visualize samples from dataset."""

import argparse
import os
import random
import time

import numpy as np
from cv2 import cv2


def read_dataset(path):
    data = {}
    size = 0
    clothes = filter(lambda x: os.path.isdir(f'{path}/{x}'), os.listdir(path))
    print("Enumerating dataset...")
    for cloth in clothes:
        items = []
        cloth_dir = f'{path}/{cloth}'
        sequences = filter(lambda x: os.path.isdir(f'{cloth_dir}/{x}'), os.listdir(cloth_dir))
        for s in sequences:
            seq_dir = f'{cloth_dir}/{s}'
            images = os.listdir(f'{seq_dir}/images/')
            for im in images:
                index = im.split('.')[0].split('_')[-1]
                rgb = f'{seq_dir}/images/rgb_{index}.tiff'
                mask = f'{seq_dir}/masks/mask_{index}.png'
                depth = f'{seq_dir}/depth_maps/depth_{index}.npz'
                normals = f'{seq_dir}/normals/normals_{index}.npz'
                items.append((rgb, mask, depth, normals))

        data[cloth] = items
        print(f"  {cloth.capitalize()}: {len(items)} samples")
        size += len(items)

    print(f'Dataset contains {len(data.keys())} surfaces containing {size} total samples.')
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Root location of the dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.path  # ""
    data = read_dataset(path)
    while True:
        rows = []
        for cloth in random.sample(list(data.keys()), 3) if len(data.keys()) > 3 else data.keys():
            items = data[cloth]
            random.shuffle(items)

            color, _, depth, normals = random.choice(items)
            color = cv2.imread(color)
            depth = np.load(depth)
            norms = np.load(normals)

            depth = cv2.cvtColor(depth * 255, cv2.COLOR_GRAY2BGR)
            rows.append(np.hstack((color / 255, depth / 255, norms)))

        cv2.imshow('Dataset', np.vstack(rows))
        if cv2.waitKey(delay=1) == ord('q'):
            raise KeyboardInterrupt

        time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Visualization interrupted.")
