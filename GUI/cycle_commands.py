import json
import os
import threading
import time

from arkeo_api import MeasurementAPI
from camera import acquisition_EL, acquisition_PL
from RPi import GPIO
from tqdm import tqdm

stopped_channels = set()
_stop_strikes = {}  # ch -> int
STOP_STRIKES_THRESHOLD = 3


def check_running(api, ch):
    api.set_active_channel(ch)
    state = api.get_channel_state()

    # If the API didn’t answer, don’t penalize the channel
    if not state:  # None or empty string
        return True

    # Normalize once
    s = str(state)

    if "Stopped" in s:
        _stop_strikes[ch] = _stop_strikes.get(ch, 0) + 1
        if _stop_strikes[ch] >= STOP_STRIKES_THRESHOLD:
            stopped_channels.add(ch)
            print(
                f"[Channel {ch}] blacklisted after {STOP_STRIKES_THRESHOLD} consecutive 'Stopped'"
            )
            return False
        else:
            # give it another chance before blacklisting
            print(
                f"[Channel {ch}] 'Stopped' ({_stop_strikes[ch]}/{STOP_STRIKES_THRESHOLD}); will recheck"
            )
            return True
    else:
        # any non-Stopped state resets the strike counter
        if ch in _stop_strikes:
            _stop_strikes[ch] = 0
        return True


def _get_settings_json(api):
    s = api.get_channel_settings()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def trigger_jv(api, ch, n_iter):
    api.set_active_channel(ch)
    time.sleep(0.1)
    data = _get_settings_json(api)
    if not data:
        print(f"[Channel {ch}] API offline; skipping JV trigger.")
        return
    api.set_channel_settings(json.dumps(data))
    print(f"[Channel {ch}] Settings updated.")
    try:
        print(f"[Channel {ch}] Starting measurement")
        if n_iter == 0:
            api.start_channel()
        else:
            api.force_jv_measurement()
    except Exception as e:
        print(f"[Channel {ch}] Error: {e}")


# --- add/replace helpers at top of cycle_commands.py ---
def switch_tracking_open(api, ch) -> bool:
    api.set_active_channel(ch)
    time.sleep(0.1)
    data = _get_settings_json(api)
    if not data:
        print(f"[Channel {ch}] API offline; cannot switch to OC.")
        return False
    data["Tracking"]["Algorithm"] = "Open circuit"
    try:
        r = api.set_channel_settings(json.dumps(data))
        if not r:
            print(f"[Channel {ch}] set_channel_settings returned empty/None (OC).")
            return False
    except Exception as e:
        print(f"[Channel {ch}] OC switch failed: {e}")
        return False
    time.sleep(0.03)
    print(f"[Channel {ch}] Switched tracking algorithm to Open Circuit.")
    return True


def check_tracking_open(api, ch) -> bool:
    api.set_active_channel(ch)
    time.sleep(0.1)
    data = _get_settings_json(api)
    if not data:
        print(f"[Channel {ch}] API offline; cannot check tracking algorithm.")
        return False
    if data["Tracking"]["Algorithm"] == "Open circuit":
        return True
    else:
        time.sleep(0.03)
        print(f"[Channel {ch}] has not switched to Open Circuit.")
        return False


def switch_tracking_short(api, ch) -> bool:
    api.set_active_channel(ch)
    time.sleep(0.1)
    data = _get_settings_json(api)
    if not data:
        print(f"[Channel {ch}] API offline; cannot switch to SC.")
        return False
    data["Tracking"]["Algorithm"] = "Short circuit"
    try:
        r = api.set_channel_settings(json.dumps(data))
        if not r:
            print(f"[Channel {ch}] set_channel_settings returned empty/None (SC).")
            return False
    except Exception as e:
        print(f"[Channel {ch}] SC switch failed: {e}")
        return False
    time.sleep(0.03)
    print(f"[Channel {ch}] Switched tracking algorithm to Short Circuit.")
    return True


def set_fixed_voltage(api, ch, voltage: float) -> bool:
    api.set_active_channel(ch)
    time.sleep(0.1)
    data = _get_settings_json(api)
    if not data:
        print(f"[Channel {ch}] API offline; cannot set Fixed Voltage.")
        return False
    data["Tracking"]["Algorithm"] = "Fixed Voltage"
    data["Tracking"]["ConstantOutput"] = float(voltage)
    try:
        r = api.set_channel_settings(json.dumps(data))
        if not r:
            print(f"[Channel {ch}] set_channel_settings returned empty/None (Fixed V).")
            return False
    except Exception as e:
        print(f"[Channel {ch}] Fixed-Voltage update failed: {e}")
        return False
    time.sleep(0.03)
    print(f"[Channel {ch}] Fixed-Voltage set to {voltage:.3f} V.")
    return True


# ==== WHITE LED (JV) ====
def run_JV(api, num_channels, JV_time, n_iter, GPIO_PIN_WHITE):
    GPIO.output(GPIO_PIN_WHITE, GPIO.HIGH)

    for _ in tqdm(range(20), desc="Recovering under white light soaking..."):
        time.sleep(1)

    for ch in range(num_channels):
        if ch not in stopped_channels:
            trigger_jv(api, ch, n_iter)

    for _ in tqdm(range(int(JV_time)), desc="White light soaking - JC sweep"):
        time.sleep(1)

    # stop white light soaking
    GPIO.output(GPIO_PIN_WHITE, GPIO.LOW)

    for ch in range(num_channels):
        check_running(api, ch)


# ==== BLUE LED (PL) ====
def run_PL(
    api,
    num_channels,
    t_recover,
    PL_time,
    exposure_time,
    exposure_time_sc,
    output_dir,
    batch_name,
    acquire,
    USE_CAMERA,
    GPIO_PIN_BLUE,
):
    # Try OC on all active channels; if any fails → skip PL_oc image
    oc_ok_all = True
    for ch in range(num_channels):
        if ch not in stopped_channels:
            if not switch_tracking_open(api, ch):
                oc_ok_all = False

    # Force-stop blacklisted (unchanged)
    for ch in range(num_channels):
        if ch in stopped_channels:
            api.set_active_channel(ch)
            time.sleep(0.1)
            api.stop_channel()
            print(f"[Channel {ch}] forced to stop (blacklisted)")

    for _ in tqdm(
        range(int(t_recover)), desc="Recovering at OC after white light soaking"
    ):
        time.sleep(1)

    GPIO.output(GPIO_PIN_BLUE, GPIO.HIGH)
    for _ in tqdm(range(int(PL_time)), desc="Blue light soaking (OC)..."):
        time.sleep(1)

    # Acquire PL_oc only if OC switches all succeeded
    if oc_ok_all and acquire and USE_CAMERA and int(exposure_time) != 0:
        output_dir_oc = output_dir + "_oc"
        try:
            os.makedirs(output_dir_oc, exist_ok=True)
            acquisition_PL(int(exposure_time), batch_name, output_dir_oc)
        except Exception as e:
            print(
                f"\n[Warning] Skipped PL_oc acquisition (reason: {e}). The cycle will continue."
            )
    elif not oc_ok_all:
        print("[PL_oc] Skipped: API update failed on at least one channel.")

    # Switch to SC on all; if any fails → skip PL_sc image
    sc_ok_all = True
    for ch in range(num_channels):
        if ch not in stopped_channels:
            if not switch_tracking_short(api, ch):
                sc_ok_all = False

    # Force-stop blacklisted (unchanged)
    for ch in range(num_channels):
        if ch in stopped_channels:
            api.set_active_channel(ch)
            time.sleep(0.1)
            api.stop_channel()
            print(f"[Channel {ch}] forced to stop (blacklisted)")

    for _ in tqdm(range(int(PL_time)), desc="Blue light soaking (SC)..."):
        time.sleep(1)

    # Acquire PL_sc only if SC switches all succeeded
    if sc_ok_all and acquire and USE_CAMERA and int(exposure_time_sc) != 0:
        output_dir_sc = output_dir + "_sc"
        try:
            os.makedirs(output_dir_sc, exist_ok=True)
            acquisition_PL(int(exposure_time_sc), batch_name, output_dir_sc)
        except Exception as e:
            print(
                f"\n[Warning] Skipped PL_sc acquisition (reason: {e}). The cycle will continue."
            )
    elif not sc_ok_all:
        print("[PL_sc] Skipped: API update failed on at least one channel.")

    GPIO.output(GPIO_PIN_BLUE, GPIO.LOW)


# ==== EL Bias (LEDs OFF) ====
def run_EL(
    api,
    num_channels,
    t_relax,
    EL_time,
    exposure_time,
    output_dir,
    batch_name,
    acquire,
    USE_CAMERA,
    el_voltage,
):

    for _ in tqdm(
        range(int(t_relax)), desc="Recovering at SC after blue light soaking"
    ):
        time.sleep(1)

    # All channels must accept Fixed-Voltage update to acquire EL image
    fv_ok_all = True
    for ch in range(num_channels):
        if ch not in stopped_channels:
            if not set_fixed_voltage(api, ch, el_voltage):
                fv_ok_all = False

    # Force-stop blacklisted (unchanged)
    for ch in range(num_channels):
        if ch in stopped_channels:
            api.set_active_channel(ch)
            time.sleep(0.1)
            api.stop_channel()
            print(f"[Channel {ch}] forced to stop (blacklisted)")

    for _ in tqdm(range(int(EL_time)), desc="Applying electric bias..."):
        time.sleep(1)

    if fv_ok_all and acquire and USE_CAMERA and int(exposure_time) != 0:
        try:
            os.makedirs(output_dir, exist_ok=True)
            acquisition_EL(int(exposure_time), batch_name, output_dir)
        except Exception as e:
            print(
                f"\n[Warning] Skipped EL acquisition (reason: {e}). The cycle will continue."
            )
    elif not fv_ok_all:
        print("[EL] Skipped: API update failed on at least one channel.")

    for ch in range(num_channels):
        if ch in stopped_channels:
            # keep it down
            api.set_active_channel(ch)
            time.sleep(0.1)
            api.stop_channel()
            print(f"[Channel {ch}] remains stopped (blacklisted).")
            continue

        # active channels: switch to OC and verify
        switch_tracking_open(api, ch)
        for i in range(3):
            if check_tracking_open(api, ch):
                break
            print(
                f"[Channel {ch}] switching tracking algorithm failed, trying again ({i+1}/3)..."
            )
            # try again
            switch_tracking_open(api, ch)
