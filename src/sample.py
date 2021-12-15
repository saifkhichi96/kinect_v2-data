# coding: utf-8
"""Visualize samples from dataset."""

import argparse
import cv2
import numpy as np
import time

from torch.utils.data import DataLoader
from utils.data import RGBDRealDataset


def read_dataset(path, b):
    dataset = RGBDRealDataset(path)
    print(len(dataset), "samples in dataset.")

    return DataLoader(dataset, batch_size=b, shuffle=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', '-d', type=str, default='../out')
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    return parser.parse_args()


def main(args):
    data = read_dataset(args.dataset_dir, args.batch_size)
    for it in data:
        image, label = it
        dmap, nmap, mask = label

        rows = []
        for i in range(image.shape[0]):
            im = image[i].numpy()
            dm = dmap[i].numpy()
            nm = nmap[i].numpy()
            ma = mask[i].numpy()

            rows.append(create_view(frame=(im, dm, nm, ma)))

        cv2.imshow('Dataset', np.vstack(rows))
        if cv2.waitKey(delay=1) == ord('q'):
            raise KeyboardInterrupt

        time.sleep(1)


if __name__ == '__main__':
    try:
        args = parse_args()
        main(args)
    except KeyboardInterrupt:
        print("Visualization interrupted.")
