import numpy as np
from cv2 import cv2
from pylibfreenect2 import LoggerLevel
from pylibfreenect2.libfreenect2 import Frame, FrameMap, FrameType
from pylibfreenect2.libfreenect2 import Freenect2, Registration, SyncMultiFrameListener
from pylibfreenect2.libfreenect2 import createConsoleLogger, setGlobalLogger


def __align_frames(color, depth):
    """Align color and depth images to each other.

    Due to different fields of view of the two camera streams, points in the depth
    and color images cannot be directly mapped to each other as they have different
    dimensions and aspect ratios. The color sensor has a wider horizontal field of
    view, whereas the depth sensor has a wider vertical field of view. This means
    that, to align the two images so that their corresponding points are at same
    (x,y) coordinates in the image, the width of the color image and the height of
    the depth image needs to be cropped.

    First, we find the length of the baseline (distance between camera lens and the
    image plane) for the color sensor using the following formula:
          baseline = height/2 ÷ tan(FoV_vert/2)
                   = 1080/2 ÷ tan(53.8/2)
                   = 1064.398px               (note: width and FoV_horiz gives same value)

    Next, we calculate the corresponding height of the depth sensor's image plane
    in the color sensor's coordinates (i.e. using color sensor's baseline length)
          height = 2 x baseline x tan(FoV_vert/2)
                 = 2 x 1064.398 x tan(60°/2)
                 = 1229.061px

    Similarly, depth sensor's image plane width in color sensor's coordinates is:
          width = 2 x 1064.398 x tan(69.73°/2)       NOTE: 70.6° FoV gives inconsistent results
                = 1483.136px                              so approximating with 69.73°

    To make the depth and color images have same width, crop the color image on
    horizontally by (1920-1483.136) ≈ 437px (~218.5px on left/right sides)

    To make the depth and color images have the same height, crop the depth image
    vertically by:
          = 424 - 1080/1229.061 x 424
          = 424 - 372.577
          ≈ 51px (~25.5px on top/bottom sides)

    Depth image will have dimensions:  512 x  373 (Aspect Ratio: 1.373)
    Color image will have dimensions: 1483 x 1080 (Aspect Ratio: 1.373)

    (Probably) because of different center lines, cropping equally on both sizes
    isn't working, so the calculated amount is divided into left/right and top/bottom
    sides by trial and error until and approximate alignment achieved.

    :param color: The color image as received from Kinect device
    :param depth: The depth image as received from Kinect device
    :return: A 2-tuple containing aligned color and depth images
    """

    # NOTE: 482px != 437px as calculated!!!. This adjustment is by trial-and-error. Calculated
    # alignment value didn't align width properly.
    color_aligned = color[:, 295:-187, :]  # 295+187=482px
    color_aligned = cv2.resize(color_aligned, (512, 373))

    depth_aligned = depth[19:-32, :]  # 19+32=51px
    return color_aligned, depth_aligned


# noinspection PyBroadException
def start_capture(callback, verbose=False):
    try:
        from pylibfreenect2.libfreenect2 import OpenGLPacketPipeline

        pipeline = OpenGLPacketPipeline()
    except Exception:
        try:
            from pylibfreenect2.libfreenect2 import OpenCLPacketPipeline

            pipeline = OpenCLPacketPipeline()
        except Exception:
            from pylibfreenect2.libfreenect2 import CpuPacketPipeline

            pipeline = CpuPacketPipeline()

    # Set logging level
    if verbose:
        logger = createConsoleLogger(LoggerLevel.INFO)
    else:
        logger = createConsoleLogger(LoggerLevel.NONE)
    setGlobalLogger(logger)

    print("Connecting to Kinect with", type(pipeline).__name__, "pipeline")
    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        raise RuntimeError("No device connected!")

    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial, pipeline=pipeline)

    print("Registering listeners")
    listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)

    print("Capturing RGB+D images (press CTRL+C in console or 'Q' key in the GUI window to exit)")
    device.start()

    # must be called after device.start()
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())

    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)

    while True:
        try:
            frames = FrameMap()
            listener.waitForNewFrame(frames)

            color = frames["color"]  # Dimensions: 1920 x 1080, FoV: 84.1° x 53.8°
            depth = frames["depth"]  # Dimensions: 512 x 424, FoV: 70.6° x 60°

            registration.apply(color, depth, undistorted, registered)

            color = color.asarray()[:, :, :3]
            depth = depth.asarray().astype(np.float32)

            # Alignment needed because of different fields of view
            # BEFORE: color.shape = (1080,1920,3), depth.shape = (424,512)
            color, depth = __align_frames(color, depth)
            # AFTER: color.shape = (373,512,3), depth.shape = (373,512)

            callback(color, depth)

            listener.release(frames)
        except KeyboardInterrupt:
            break

    print("\nDisconnecting")
    device.stop()
    device.close()
