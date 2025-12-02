import datetime
import json
import os
import shutil
import time
from pathlib import Path

from camera import acquisition_EL, acquisition_PL
from RPi import GPIO
from tqdm import tqdm

stopped_channels = set()
_stop_strikes = {}  # ch -> int
STOP_STRIKES_THRESHOLD = 3
LOG_ALL = False  # flip to True if you ever want every event


def _ctx_from_output_dir(output_dir):
    """
    Infer date_root from an output_dir like:
    <base>/<res_name>/<YYYY-MM-DD>/<EL or PL[_sc]/PL_oc>
    """
    p = Path(output_dir)
    date_root = p.parent.name
    return date_root


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
    print(f"[Channel {ch}] Switching tracking algorithm to Open Circuit.")
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
    print(f"[Channel {ch}] Switching tracking algorithm to Short Circuit.")
    return True


def set_fixed_current(api, ch) -> bool:
    api.set_active_channel(ch)
    time.sleep(0.1)
    data = _get_settings_json(api)
    if not data:
        print(f"[Channel {ch}] API offline; cannot set Fixed Current.")
        return False
    data["Tracking"]["Algorithm"] = "Fixed Current"
    data["Tracking"]["ConstantOutput"] = float(0.0)
    try:
        r = api.set_channel_settings(json.dumps(data))
        if not r:
            print(f"[Channel {ch}] set_channel_settings returned empty/None (OC).")
            return False
    except Exception as e:
        print(f"[Channel {ch}] Fixed-Current update failed: {e}")
        return False
    time.sleep(0.03)
    print(f"[Channel {ch}] Setting Fixed-Current to 0.0 A.")
    return True


def check_fixed_current(api, ch) -> bool:
    api.set_active_channel(ch)
    time.sleep(0.1)
    data = _get_settings_json(api)
    if not data:
        print(f"[Channel {ch}] API offline; cannot check tracking algorithm.")
        return False
    if data["Tracking"]["Algorithm"] == "Fixed Current" and data["Tracking"][
        "ConstantOutput"
    ] == float(0.0):
        return True
    else:
        time.sleep(0.03)
        print(f"[Channel {ch}] has not switched to Open Circuit.")
        return False


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
    print(f"[Channel {ch}] Setting Fixed-Voltage to {voltage:.3f} V.")
    return True


# ==== WHITE LED (JV) ====
def run_JV(
    api,
    num_channels,
    JV_time,
    t_recover,
    n_iter,
    GPIO_PIN_WHITE,
    date_root,
    cycle_counter,
):
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

    # --- log newly blacklisted channels (diff-based) ---
    prev_blacklisted = set(stopped_channels)  # snapshot before checks

    for ch in range(num_channels):
        check_running(api, ch)

    newly_blacklisted = sorted(stopped_channels - prev_blacklisted)
    if newly_blacklisted:
        for ch in newly_blacklisted:
            log_event(
                date_root=date_root,
                cycle_counter=cycle_counter,
                image_type="BLACKLIST",
                status="INFO",
                reason=f"Channel {ch} blacklisted after JV",
                stopped_channels=stopped_channels,
            )

    for _ in tqdm(
        range(int(t_recover)), desc="Recovering at OC after white light soaking"
    ):
        time.sleep(1)


# ==== BLUE LED (PL) ====
def run_PL(
    api,
    num_channels,
    PL_time,
    t_relax,
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

    GPIO.output(GPIO_PIN_BLUE, GPIO.HIGH)
    for _ in tqdm(range(int(PL_time)), desc="Blue light soaking (OC)..."):
        time.sleep(1)

    # Acquire PL_oc only if OC switches all succeeded
    if oc_ok_all and acquire and USE_CAMERA and int(exposure_time) != 0:
        output_dir_oc = output_dir + "_oc"
        try:
            os.makedirs(output_dir_oc, exist_ok=True)
            acquisition_PL(int(exposure_time), batch_name, output_dir_oc)
            # LOG success (no-op unless you set log_all=True)
            date_root = _ctx_from_output_dir(output_dir_oc)
            ploc_path = Path(output_dir_oc) / f"{batch_name}.tiff"
            log_event(
                date_root=date_root,
                cycle_counter=datetime.datetime.now().strftime(
                    "%H:%M:%S"
                ),  # or pass your real self.cycle_counter in via args if you prefer
                image_type="PLoc",
                status="OK",
                filepath=ploc_path,
                stopped_channels=stopped_channels,
            )
        except Exception as e:
            print(
                f"\n[Warning] Skipped PL_oc acquisition (reason: {e}). The cycle will continue."
            )
            date_root = _ctx_from_output_dir(output_dir + "_oc")
            log_event(
                date_root=date_root,
                cycle_counter="-",
                image_type="PLoc",
                status="ERROR",
                reason=str(e),
                stopped_channels=stopped_channels,
            )
    elif not oc_ok_all:
        print("[PL_oc] Skipped: API update failed on at least one channel.")
        date_root = _ctx_from_output_dir(output_dir + "_oc")
        log_event(
            date_root=date_root,
            cycle_counter="-",
            image_type="PLoc",
            status="SKIPPED",
            reason="API update failed on at least one channel",
            stopped_channels=stopped_channels,
        )

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
            date_root = _ctx_from_output_dir(output_dir_sc)
            plsc_path = Path(output_dir_sc) / f"{batch_name}.tiff"
            log_event(
                date_root=date_root,
                cycle_counter="-",
                image_type="PLsc",
                status="OK",
                filepath=plsc_path,
                stopped_channels=stopped_channels,
            )
        except Exception as e:
            print(
                f"\n[Warning] Skipped PL_sc acquisition (reason: {e}). The cycle will continue."
            )
            date_root = _ctx_from_output_dir(output_dir_sc)
            log_event(
                date_root=date_root,
                cycle_counter="-",
                image_type="PLsc",
                status="ERROR",
                reason=str(e),
                stopped_channels=stopped_channels,
            )
    elif not sc_ok_all:
        print("[PL_sc] Skipped: API update failed on at least one channel.")
        date_root = _ctx_from_output_dir(output_dir + "_sc")
        log_event(
            date_root=date_root,
            cycle_counter="-",
            image_type="PLsc",
            status="SKIPPED",
            reason="API update failed on at least one channel",
            stopped_channels=stopped_channels,
        )
    GPIO.output(GPIO_PIN_BLUE, GPIO.LOW)

    for _ in tqdm(
        range(int(t_relax)), desc="Recovering at SC after blue light soaking"
    ):
        time.sleep(1)


# ==== EL Bias (LEDs OFF) ====
def run_EL(
    api,
    num_channels,
    EL_time,
    exposure_time,
    output_dir,
    batch_name,
    acquire,
    USE_CAMERA,
    el_voltage,
):
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
            date_root = _ctx_from_output_dir(output_dir)
            el_path = Path(output_dir) / f"{batch_name}.tiff"
            log_event(
                date_root=date_root,
                cycle_counter="-",
                image_type="EL",
                status="OK",
                filepath=el_path,
                stopped_channels=stopped_channels,
            )
        except Exception as e:
            print(
                f"\n[Warning] Skipped EL acquisition (reason: {e}). The cycle will continue."
            )
            date_root = _ctx_from_output_dir(output_dir)
            log_event(
                date_root=date_root,
                cycle_counter="-",
                image_type="EL",
                status="ERROR",
                reason=str(e),
                stopped_channels=stopped_channels,
            )
    elif not fv_ok_all:
        print("[EL] Skipped: API update failed on at least one channel.")
        date_root = _ctx_from_output_dir(output_dir)
        log_event(
            date_root=date_root,
            cycle_counter="-",
            image_type="EL",
            status="SKIPPED",
            reason="API update failed on at least one channel",
            stopped_channels=stopped_channels,
        )

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

    # Force-stop blacklisted (unchanged)
    for ch in range(num_channels):
        if ch in stopped_channels:
            api.set_active_channel(ch)
            time.sleep(0.1)
            api.stop_channel()
            print(f"[Channel {ch}] forced to stop (blacklisted)")


def log_event(
    *,
    date_root,
    cycle_counter,
    image_type,
    status="OK",
    reason="",
    filepath=None,
    stopped_channels=None,
    log_all=False,
):
    """Append a short line to <base>/<res>/<date>/acquisition_log.txt.
    Writes only when status != OK (unless log_all=True)."""
    if status == "OK" and not log_all:
        return

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fpath = Path(filepath) if filepath else None
    try:
        fsize = fpath.stat().st_size if fpath and fpath.exists() else 0
    except Exception:
        fsize = 0
    try:
        base_dir = date_root.parent.parent
        _, _, free = shutil.disk_usage(str(base_dir))
        free_mb = round(free / (1024 * 1024))
    except Exception:
        free_mb = -1

    bl = sorted(stopped_channels) if isinstance(stopped_channels, set) else "-"

    log_dir = Path(date_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    line = (
        f"{ts} | cycle={cycle_counter} | step={image_type} | status={status}"
        f" | reason={reason or '-'} | file={fpath if fpath else '-'}"
        f" | bytes={fsize} | freeMB={free_mb} | blacklisted={bl}\n"
    )
    try:
        with open(log_dir / "acquisition_log.txt", "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
