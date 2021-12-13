import time
import traceback

import numpy as np
from pylibfreenect2 import LoggerLevel, createConsoleLogger, setGlobalLogger
from pylibfreenect2.libfreenect2 import Freenect2, Freenect2Device, Frame, FrameMap, FrameType
from pylibfreenect2.libfreenect2 import Registration, SyncMultiFrameListener
from utils import segment, dmap2norm


class Config:
    """Recording configurations."""

    def __init__(self, duration: int, delay: int = 0, rate: float = 0):
        """Initializer.

        :param duration: Time in seconds for recording length. To record indefinitely, set duration to 0. Default is 0.
        :param delay: Time to delay the start of recording by. Default is 0, i.e. recording starts immediately.
        :param rate: Limit capture frame rate per second. Default is 0, which captures as many frames as possible.
        """
        self.duration: int = duration
        self.delay: int = delay
        self.rate: float = rate


class Filters:
    """Filters to apply on the data."""

    def __init__(self, skin: bool = True, noise: bool = True):
        """Initializer

        :param skin
        :param noise
        """
        self.skin: bool = skin
        self.noise: bool = noise


class Viewport:
    """Camera viewport."""

    def __init__(self, left: int = 0, right: int = 0,
                 top: int = 0, bottom: int = 0,
                 near: float = 500.0, far: float = 4500.0):
        """Initializer.

        :param left: Number of pixels to crop on the left side. Default is 0.
        :param right: Number of pixels to crop on the right side. Default is 0.
        :param top: Number of pixels to crop on the top side. Default is 0.
        :param bottom: Number of pixels to crop on the bottom side. Default is 0.
        :param near: The minimum depth to capture. Must be 500 <= near < far. Default is 500.
        :param far: The maximum depth to capture. Must be near < far <= 4500. Default is 4500.
        """
        self.left: int = left if left >= 0 else 0
        self.right: int = right if right >= 0 else 0
        self.top: int = top if top >= 0 else 0
        self.bottom: int = bottom if bottom >= 0 else 0
        self.near: float = near if 500 <= near < far else 500
        self.far: float = far if near < far <= 4500 else 4500


# noinspection PyArgumentList,PyBroadException
def record(callback,
           config: Config,
           filters: Filters,
           viewport: Viewport):
    """Records a sequence of RGB-D images.

    Each datapoint in the sequence is a set of four values, i.e. an RGB image, a depth map,
    surface normals, and a binary mask, each of them given as a numpy array.

    :param callback: A callback function to handle captured frames.
    :param config: Configurations for recording the sequence.
    :param filters
    :param viewport
    """
    try:
        from pylibfreenect2.libfreenect2 import OpenGLPacketPipeline

        pipeline = OpenGLPacketPipeline()
    except:
        try:
            from pylibfreenect2.libfreenect2 import OpenCLPacketPipeline

            pipeline = OpenCLPacketPipeline()
        except:
            from pylibfreenect2.libfreenect2 import CpuPacketPipeline

            pipeline = CpuPacketPipeline()

    logger = createConsoleLogger(LoggerLevel.NONE)
    setGlobalLogger(logger)

    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        raise RuntimeError("No device connected!")

    serial = fn.getDeviceSerialNumber(0)
    device: Freenect2Device = fn.openDevice(serial, pipeline=pipeline)

    listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)

    print(f"Configuration:"
          f"\n  Filters: "
          f"Skin={filters.skin}, "
          f"Noise={filters.noise}"
          f"\n  Viewport: "
          f"x=({viewport.left},W-{viewport.right}), "
          f"y=({viewport.top},H-{viewport.bottom}), "
          f"z=({viewport.near},{viewport.far})")

    # Wait specified number of seconds before starting image capture
    print(f"Starting in {config.delay} seconds")
    if config.delay > 0:
        time.sleep(config.delay)

    device.start()
    print("Recording", f"for {config.duration} seconds" if config.duration > 0 else "until interrupted",
          f"at <={config.rate} fps" if config.rate > 0 else "")

    start_time = time.time()
    last_time = start_time + 0.0001

    # must be called after device.start()
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())

    count = 0
    while True:
        try:
            frames = FrameMap()
            listener.waitForNewFrame(frames)

            color = frames[FrameType.Color]  # Dimensions: 1920 x 1080, FoV: 84.1° x 53.8°
            depth = frames[FrameType.Depth]  # Dimensions: 512 x 424, FoV: 70.6° x 60°
            undistorted = Frame(512, 424, 4)
            registered = Frame(512, 424, 4)

            # Combine frames of depth and color camera
            registration.apply(color, depth, undistorted, registered, enable_filter=False)

            color = registered.asarray(dtype=np.uint8)[:, :, :3]
            depth = undistorted.asarray(dtype=np.float32)

            # Crop viewport along x and y axes
            if viewport.left > 0:
                color = color[:, viewport.left:, :]
                depth = depth[:, viewport.left:]

            if viewport.right > 0:
                color = color[:, :-viewport.right, :]
                depth = depth[:, :-viewport.right]

            if viewport.top > 0:
                color = color[viewport.top:, :, :]
                depth = depth[viewport.top:, :]

            if viewport.bottom > 0:
                color = color[:-viewport.bottom, :, :]
                depth = depth[:-viewport.bottom, :]

            # Remove undesired surfaces
            color, depth, mask = segment(color, depth,
                                         min_depth=viewport.near, max_depth=viewport.far,
                                         skin=filters.skin, artefacts=filters.noise)

            # Compute surface normals from depth map
            norms = dmap2norm(depth)
            norms[mask] = 0

            callback((color, depth, norms, mask))  # RGB-D+Normals data + Foreground mask
            listener.release(frames)
            count += 1

            # Stop capturing after specified duration, if applicable
            now = time.time()
            if config.duration > 0:
                elapsed_since_epoch = now - start_time
                if elapsed_since_epoch > config.duration:
                    print(f"Recording completed")
                    break

            # Limit by frame rate (only capture a maximum of `fps` images per second)
            now = time.time()
            if config.rate > 0:
                time_between_frames = 1. / config.rate
                time_since_prev_frame = now - last_time
                wait = time_between_frames - time_since_prev_frame
                if wait > 0:
                    time.sleep(wait)

            last_time = now
        except KeyboardInterrupt:
            print(f"Recording interrupted by user ")
            break
        except Exception as err:
            print(f"Recording interrupted by an error: {err}")
            traceback.print_exc()
            break

    print(
        f"Processed {count} frames in {(last_time - start_time):.2f}s at {(count / (last_time - start_time)):.1f} fps.")
    print("Closing device")
    device.stop()
    device.close()
