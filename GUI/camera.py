try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path

    configure_path()
except ImportError:
    configure_path = None

import os
import time

import tifffile
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

NUMBER_OF_IMAGES = 1  # Number of TIFF images to be saved
BASE_DIRECTORY = os.path.abspath(
    r"/home/chose/Documenti/Codes"
)  # Directory the TIFFs will be saved to

TAG_BITDEPTH = 32768
TAG_EXPOSURE = 32769


def acquisition_PL(exposure_time, frames_counted, output_dir):
    OUTPUT_FOLDER = output_dir
    with TLCameraSDK() as sdk:
        cameras = sdk.discover_available_cameras()
        if len(cameras) == 0:
            print("Error: no cameras detected!")

        with sdk.open_camera(cameras[0]) as camera:
            #  setup the camera for continuous acquisition
            camera.frames_per_trigger_zero_for_unlimited = 1
            camera.image_poll_timeout_ms = 20000  # 2 second timeout
            camera.exposure_time_us = exposure_time  # exposure time in microseconds
            camera.arm(2)

            # save these values to place in our custom TIFF tags later
            bit_depth = camera.bit_depth
            exposure = camera.exposure_time_us

            # need to save the image width and height for color processing
            # image_width = camera.image_width_pixels
            # image_height = camera.image_height_pixels

            # begin acquisition
            # time.sleep(waiting_time)
            camera.issue_software_trigger()
            frame = camera.get_pending_frame_or_null()
            if frame is None:
                raise TimeoutError(
                    "Timeout was reached while polling for a frame, program will now exit"
                )

            print(f"Acquired frame {frames_counted}")

            image_data = frame.image_buffer

            # delete image if it exists
            if os.path.exists(OUTPUT_FOLDER + os.sep + str(frames_counted) + ".tiff"):
                os.remove(OUTPUT_FOLDER + os.sep + str(frames_counted) + ".tiff")

            with tifffile.TiffWriter(
                OUTPUT_FOLDER + os.sep + str(frames_counted) + ".tiff", append=True
            ) as tiff:
                tiff.write(
                    data=image_data,  # np.ushort image data array from the camera
                    # compress=0,   # amount of compression (0-9), by default it is uncompressed (0)
                    extratags=[
                        (
                            TAG_BITDEPTH,
                            "I",
                            1,
                            bit_depth,
                            False,
                        ),  # custom TIFF tag for bit depth
                        (TAG_EXPOSURE, "I", 1, exposure, False),
                    ],  # custom TIFF tag for exposure
                )

            camera.disarm()
            camera.dispose()


def acquisition_EL(exposure_time, frames_counted, output_dir):
    OUTPUT_FOLDER = output_dir
    with TLCameraSDK() as sdk:
        cameras = sdk.discover_available_cameras()
        if len(cameras) == 0:
            print("Error: no cameras detected!")

        with sdk.open_camera(cameras[0]) as camera:
            #  setup the camera for continuous acquisition
            camera.frames_per_trigger_zero_for_unlimited = 1
            camera.image_poll_timeout_ms = 20000  # 2 second timeout
            camera.exposure_time_us = exposure_time  # exposure time in microseconds
            camera.arm(2)

            # save these values to place in our custom TIFF tags later
            bit_depth = camera.bit_depth
            exposure = camera.exposure_time_us

            # need to save the image width and height for color processing
            # image_width = camera.image_width_pixels
            # image_height = camera.image_height_pixels

            # begin acquisition
            # time.sleep(waiting_time)
            camera.issue_software_trigger()
            frame = camera.get_pending_frame_or_null()
            if frame is None:
                raise TimeoutError(
                    "Timeout was reached while polling for a frame, program will now exit"
                )

            print(f"Acquired frame {frames_counted}")

            image_data = frame.image_buffer

            # delete image if it exists
            if os.path.exists(OUTPUT_FOLDER + os.sep + str(frames_counted) + ".tiff"):
                os.remove(OUTPUT_FOLDER + os.sep + str(frames_counted) + ".tiff")

            with tifffile.TiffWriter(
                OUTPUT_FOLDER + os.sep + str(frames_counted) + ".tiff", append=True
            ) as tiff:
                tiff.write(
                    data=image_data,  # np.ushort image data array from the camera
                    # compress=0,   # amount of compression (0-9), by default it is uncompressed (0)
                    extratags=[
                        (
                            TAG_BITDEPTH,
                            "I",
                            1,
                            bit_depth,
                            False,
                        ),  # custom TIFF tag for bit depth
                        (TAG_EXPOSURE, "I", 1, exposure, False),
                    ],  # custom TIFF tag for exposure
                )

            camera.disarm()
            camera.dispose()
