import json
import os
import threading
import time

from arkeo_api import MeasurementAPI
from camera import acquisition_EL, acquisition_PL
from RPi import GPIO
from tqdm import tqdm


def trigger_jv(api, ch, n_iter):
    api.set_active_channel(ch)

    # Start measurement
    try:
        print(f"[Channel {ch}] Starting measurement")
        if n_iter == 0:
            api.start_channel()
        else:
            api.force_jv_measurement()

    except Exception as e:
        print(f"[Channel {ch}] Error: {e}")


def switch_tracking_open(api, ch):
    api.set_active_channel(ch)

    settings_str = api.get_channel_settings()
    data = json.loads(settings_str)

    # Modify settings
    data["Tracking"]["Algorithm"] = "Open circuit"

    api.set_channel_settings(json.dumps(data))
    print(f"[Channel {ch}] Switched tracking algorithm to Open Circuit.")


def switch_tracking_short(api, ch):
    api.set_active_channel(ch)

    settings_str = api.get_channel_settings()
    data = json.loads(settings_str)

    # Modify settings
    data["Tracking"]["Algorithm"] = "Short circuit"

    api.set_channel_settings(json.dumps(data))
    print(f"[Channel {ch}] Switched tracking algorithm to Short Circuit.")


# ==== WHITE LED (JV) ====
def run_JV(api, num_channels, JV_time, n_iter, GPIO_PIN_WHITE):
    GPIO.output(GPIO_PIN_WHITE, GPIO.HIGH)

    for ch in range(num_channels):
        trigger_jv(api, ch, n_iter)

    for _ in tqdm(range(int(JV_time)), desc="White light soaking...."):
        time.sleep(1)

    # stop white light soaking
    GPIO.output(GPIO_PIN_WHITE, GPIO.LOW)


# ==== BLUE LED (PL) ====
def run_PL(
    api,
    num_channels,
    PL_time,
    exposure_time,
    exposure_time_sc,
    output_dir,
    batch_name,
    acquire,
    USE_CAMERA,
    GPIO_PIN_BLUE,
):
    for ch in range(num_channels):
        switch_tracking_open(api, ch)

    GPIO.output(GPIO_PIN_BLUE, GPIO.HIGH)
    for _ in tqdm(range(int(PL_time)), desc="Blue light soaking (OC)..."):
        time.sleep(1)

    # PL image acquisition
    if acquire and USE_CAMERA and int(exposure_time) != 0:
        output_dir = output_dir + "_oc"
        try:
            os.makedirs(output_dir, exist_ok=True)
            acquisition_PL(int(exposure_time), batch_name, output_dir)

        except Exception as e:
            print(f"\n[Error during PL_oc acquisition: {e}")

    for ch in range(num_channels):
        switch_tracking_short(api, ch)

    GPIO.output(GPIO_PIN_BLUE, GPIO.HIGH)
    for _ in tqdm(range(10), desc="Blue light soaking (SC)..."):
        time.sleep(1)

    # PL image acquisition
    if acquire and USE_CAMERA and int(exposure_time_sc) != 0:
        output_dir = output_dir + "_sc"
        try:
            os.makedirs(output_dir, exist_ok=True)
            acquisition_PL(int(exposure_time_sc), batch_name, output_dir)

        except Exception as e:
            print(f"\n[Error during PL_sc acquisition: {e}")

    # stop PL bias
    GPIO.output(GPIO_PIN_BLUE, GPIO.LOW)


# ==== EL Bias (LEDs OFF) ====
def run_EL(EL_time, exposure_time, output_dir, batch_name, acquire, USE_CAMERA):
    for _ in tqdm(range(int(EL_time)), desc="Applying electric bias..."):
        time.sleep(1)

    # EL image acquisition
    if acquire and USE_CAMERA and int(exposure_time) != 0:
        try:
            os.makedirs(output_dir, exist_ok=True)
            acquisition_EL(int(exposure_time), batch_name, output_dir)

        except Exception as e:
            print(f"\n[Error during EL acquisition: {e}")
