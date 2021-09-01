# coding: utf-8
"""A command-line program to collect RGB-D data using Kinect V2.

usage: main.py [-h] [-l DELAY] [-d DURATION] [-r RATE] [-s] [-n] [-x X] [-X X] [-y Y] [-Y Y] [-z DEPTH] path

positional arguments:
  path                  Output directory for saving data.

optional arguments:
  -h, --help            show this help message and exit
  -l DELAY, --delay DELAY
                        Start recording with delay in seconds. Default is 0.
  -d DURATION, --duration DURATION
                        Duration in seconds for how long to record. Default is 0, which recordsindefinitely until stopped manually.
  -r RATE, --rate RATE  Frame rate of the recording in frames per second. Default is 0, whichrecords as many frames as possible.
  -s, --skin            Remove skin in images.
  -n, --noise           Remove small artefacts in images.
  -x X, --x X           Number of pixels to crop viewport on left. Default is 0.
  -X X, --X X           Number of pixels to crop viewport on right. Default is 0.
  -y Y, --y Y           Number of pixels to crop viewport on top. Default is 0.
  -Y Y, --Y Y           Number of pixels to crop viewport on bottom. Default is 0.
  -z DEPTH, --depth DEPTH
                        Maximum range of depth to capture. Default is 4500. Must be 500 < value <= 4500.
"""
import argparse
import os

from cv2 import cv2

from models import KinectV2
from utils import create_view, create_save_directories, save_frame


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Output directory for saving data.")

    parser.add_argument("-l", "--delay", type=int, default=0,
                        help="Start recording with delay in seconds. Default is 0.")
    parser.add_argument("-d", "--duration", type=int, default=0,
                        help="Duration in seconds for how long to record. Default is 0, which records"
                             "indefinitely until stopped manually.")
    parser.add_argument("-r", "--rate", type=float, default=0,
                        help="Frame rate of the recording in frames per second. Default is 0, which"
                             "records as many frames as possible.")

    parser.add_argument('-s', '--skin', action='store_true', help="Remove skin in images.")
    parser.add_argument('-n', '--noise', action='store_true', help="Remove small artefacts in images.")

    parser.add_argument("-x", "--x", type=int, default=0,
                        help="Number of pixels to crop viewport on left. Default is 0.")
    parser.add_argument("-X", "--X", type=int, default=0,
                        help="Number of pixels to crop viewport on right. Default is 0.")
    parser.add_argument("-y", "--y", type=int, default=0,
                        help="Number of pixels to crop viewport on top. Default is 0.")
    parser.add_argument("-Y", "--Y", type=int, default=0,
                        help="Number of pixels to crop viewport on bottom. Default is 0.")

    parser.add_argument('-z', '--depth', type=int, default=4500,
                        help="Maximum range of depth to capture. Default is 4500. "
                             "Must be 500 < value <= 4500.")
    return parser.parse_args()


def init_sequence():
    """Gets sequence metadata from user for recording.

    This metadata includes:

    - Surface name
    -- User-provided name.

    -- Can be anything, e.g., shirt, hoodie, t-shirt, jacket, coat, sweater, shorts, pants, etc.

    - Surface material type
    -- Colorful/Patterned (C)

    -- Plain-dark (D)

    -- Plain-light (W)

    - Lighting
    -- Daylight (N)

    -- Indoor lighting (A)

    - Viewing direction
    -- Front-only (front)

    -- Back-only (back)

    -- Random (rot)

    :return: Sequence metadata encoded as a path string.
    """
    # Surface type
    surface = str(input("Enter surface name: "))

    # Select surface material
    material = None
    while material not in [1, 2, 3]:
        material = int(input("Select surface material type:\n"
                             "  1: Single-colored (Dark shades)\n"
                             "  2: Single-colored (Light shades)\n"
                             "  3: Multi-colored/Patterns ? "))

        if material not in [1, 2, 3]:
            print("Invalid choice! Please try again.")
    material = "D" if material == 1 else "W" if material == 2 else "C"

    # Lighting source
    lighting = None
    while lighting not in [1, 2]:
        lighting = int(input("Select lighting source:\n"
                             "  1: Daylight\n"
                             "  2: Indoor lighting only ? "))

        if lighting not in [1, 2]:
            print("Invalid choice! Please try again.")
    lighting = "N" if lighting == 1 else "A"

    # Select surface direction
    view = None
    while view not in [1, 2, 3]:
        view = int(input("Select viewing direction:\n"
                         "  1: Front-only\n"
                         "  2: Back-only\n"
                         "  3: Random ? "))

        if view not in [1, 2, 3]:
            print("Invalid choice! Please try again.")
    view = "front" if view == 1 else "back" if view == 2 else "rot"

    return f'{surface.lower()}/{lighting}{material}_{view}'


def main(args):
    """The main function.

    :param args The command-line arguments."""

    # Get sequence details from user
    sequence = init_sequence()
    path = os.path.join(args.path, sequence)
    print(f"Sequence: {sequence}\n"
          f"Location: {args.path}")

    # Make directories for saving data
    create_save_directories(path)
    item_id = 0  # id of the current item in sequence, incremented at each iteration

    def callback(frame):
        """Callback function where new frames from camera are received.

        :param frame The current frame, containing RGB-D + Normals data and a foreground mask."""

        # View the frame in an OpenCV window
        cv2.imshow('Kinect Scanner', create_view(frame))

        # Save frame data
        nonlocal item_id
        save_frame(path, item_id, frame)
        item_id += 1

        if cv2.waitKey(delay=1) == ord('q'):
            raise KeyboardInterrupt

    KinectV2.record(callback,
                    config=KinectV2.Config(
                        duration=args.duration,
                        delay=args.delay,
                        rate=args.rate
                    ),
                    filters=KinectV2.Filters(
                        skin=args.skin,
                        noise=args.noise
                    ),
                    viewport=KinectV2.Viewport(
                        left=args.x,
                        right=args.X,
                        top=args.y,
                        bottom=args.Y,
                        near=500,
                        far=args.depth if 500 < args.depth <= 4500 else 4500
                    ))


if __name__ == '__main__':
    main(args=parse_arguments())
