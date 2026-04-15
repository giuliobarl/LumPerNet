import csv
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

sample_lum_settings = {
    "integration_time": 100,
    "averages": 1,
    "pixel_smoothing": 1,
}


class LuminescenceAPI:
    def __init__(
        self,
        host="192.168.0.250",
        port=6360,
        timeout=30.0,
        retries=2,
        reconnect_backoff=(0.3, 1.0, 2.0),
        enable_keepalive=True,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retries = retries
        self.reconnect_backoff = reconnect_backoff
        self.enable_keepalive = enable_keepalive
        self.sock = None  # type: Optional[socket.socket]
        self.request_id = 1000

    # -------------------------------
    # Connection
    # -------------------------------
    def _apply_keepalive(self, s):
        if not self.enable_keepalive:
            return
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if sys.platform.startswith("linux"):
                s.setsockopt(socket.IPPROTO_TCP, 0x10, 60)  # TCP_KEEPIDLE
                s.setsockopt(socket.IPPROTO_TCP, 0x12, 5)  # TCP_KEEPCNT
                s.setsockopt(socket.IPPROTO_TCP, 0x11, 10)  # TCP_KEEPINTVL
        except Exception:
            pass

    def connect(self):
        for delay in (0.0,) + tuple(self.reconnect_backoff):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(self.timeout)
                self._apply_keepalive(s)
                s.connect((self.host, self.port))
                self.sock = s
                return True
            except (TimeoutError, ConnectionError, OSError):
                self.sock = None
                time.sleep(delay)
        return False

    def disconnect(self):
        try:
            if self.sock:
                self.sock.close()
        finally:
            self.sock = None

    # -------------------------------
    # Core communication
    # -------------------------------
    def _recv_exact(self, size):
        # type: (int) -> Optional[bytes]
        if not self.sock:
            return None

        data = bytearray()
        end = time.time() + self.timeout
        try:
            while len(data) < size and time.time() < end:
                chunk = self.sock.recv(size - len(data))
                if not chunk:
                    return None
                data.extend(chunk)
            if len(data) != size:
                return None
            return bytes(data)
        except (TimeoutError, ConnectionError, OSError):
            return None

    def _send_recv_once(self, payload):
        # type: (bytes) -> Optional[Dict[str, Any]]
        if not self.sock and not self.connect():
            return None

        try:
            msg_len = len(payload).to_bytes(4, byteorder="big")
            self.sock.sendall(msg_len)
            self.sock.sendall(payload)

            raw_length = self._recv_exact(4)
            if not raw_length:
                self.disconnect()
                return None

            response_length = int.from_bytes(raw_length, byteorder="big")
            body = self._recv_exact(response_length)
            if not body:
                self.disconnect()
                return None

            return json.loads(body.decode("utf-8", errors="replace"))
        except (
            TimeoutError,
            BrokenPipeError,
            ConnectionError,
            OSError,
            json.JSONDecodeError,
        ):
            self.disconnect()
            return None

    def _next_request_id(self):
        self.request_id += 1
        return self.request_id

    def send_command(self, target, command, parameter=None):
        # type: (str, str, Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]
        payload = {
            "target": target,
            "command": command,
            "request_id": self._next_request_id(),
        }
        if parameter is not None:
            payload["parameter"] = parameter

        message = json.dumps(payload).encode("utf-8")

        resp = self._send_recv_once(message)
        if resp is not None:
            return resp

        for _ in range(self.retries):
            if not self.connect():
                continue
            resp = self._send_recv_once(message)
            if resp is not None:
                return resp

        return None

    # -------------------------------
    # Response helpers
    # -------------------------------
    @staticmethod
    def is_ok(response):
        if not response:
            return False
        status = str(response.get("status", "")).strip().lower()
        return status == "ok"

    @staticmethod
    def get_data(response, default=None):
        if not response:
            return default
        return response.get("data", default)

    @staticmethod
    def get_error(response):
        if not response:
            return {"message": "No response from server"}
        return response.get("error")

    # -------------------------------
    # MAIN
    # -------------------------------
    def start_routine(self, routine="Luminescence"):
        return self.send_command("MAIN", "StartRoutine", {"routine": routine})

    def get_active_routine(self):
        return self.send_command("MAIN", "GetActiveRoutine")

    def get_available_routines(self):
        return self.send_command("MAIN", "GetAvailableRoutines")

    # -------------------------------
    # ROUTINE
    # -------------------------------
    def start_measurement(self):
        return self.send_command("ROUTINE", "StartMeasurement")

    def stop_measurement(self):
        return self.send_command("ROUTINE", "StopMeasurement")

    def get_status(self):
        return self.send_command("ROUTINE", "GetTestStatus")

    def get_test_data(self):
        return self.send_command("ROUTINE", "GetTestData")

    def apply_settings(self, settings):
        return self.send_command("ROUTINE", "ApplySettings", settings)

    def get_settings(self):
        return self.send_command("ROUTINE", "GetSettings")

    def clear_errors(self):
        return self.send_command("ROUTINE", "ClearErrors")

    def get_custom_commands(self):
        return self.send_command("ROUTINE", "GetCustomCommands")

    def acquire_single(self):
        return self.send_command("ROUTINE", "AcquireSingle")

    def acquire_dark(self):
        return self.send_command("ROUTINE", "AcquireDark")

    def acquire_reference(self):
        return self.send_command("ROUTINE", "AcquireReference")

    def auto_exposure(self):
        return self.send_command("ROUTINE", "AutoExposure", {})

    def close_routine(self):
        return self.send_command("ROUTINE", "CloseRoutine")

    # -------------------------------
    # High-level workflow helpers
    # -------------------------------
    def ensure_ready(self, timeout_s=30.0, poll_s=0.2):
        t0 = time.time()
        last_resp = None

        while time.time() - t0 < timeout_s:
            resp = self.get_status()
            last_resp = resp

            if resp and str(resp.get("status", "")).lower() == "error":
                err = resp.get("error", {}) or {}
                if err.get("code") == 4006:  # No active routine
                    time.sleep(poll_s)
                    continue
                return False

            data = self.get_data(resp, {}) or {}
            if data.get("routine_status") == "Ready":
                return True
            if data.get("routine_status") == "Error":
                return False

            time.sleep(poll_s)

        # critical final check at timeout boundary
        resp = self.get_status()
        if resp is not None:
            last_resp = resp
            data = self.get_data(resp, {}) or {}
            if data.get("routine_status") == "Ready":
                return True

        print("ensure_ready timeout, last status:", last_resp)
        return False

    def initialize_luminescence(self, settings=None, ready_timeout_s=30.0):
        # 1. If Luminescence is already active and ready, use it
        status = self.get_status()
        data = self.get_data(status, {})
        if (
            data.get("routine_name") == "Luminescence"
            and data.get("routine_status") == "Ready"
        ):
            if settings is not None:
                resp = self.apply_settings(settings)
                if not self.is_ok(resp):
                    print("apply_settings failed on already-active routine:", resp)
                    return False
            return True

        # 2. Check active routine using the MAIN endpoint, not ROUTINE status polling
        active = self.get_active_routine()
        active_data = self.get_data(active, {})
        if active_data:
            active_name = active_data.get("routine")
            if active_name == "Luminescence":
                if self.ensure_ready(timeout_s=ready_timeout_s):
                    if settings is not None:
                        resp = self.apply_settings(settings)
                        if not self.is_ok(resp):
                            print(
                                "apply_settings failed after active-routine check:",
                                resp,
                            )
                            return False
                    return True

        # 3. Otherwise start the routine
        resp = self.start_routine("Luminescence")
        if not self.is_ok(resp):
            print("start_routine failed:", resp)
            return False

        if not self.ensure_ready(timeout_s=ready_timeout_s):
            status = self.get_status()
            data = self.get_data(status, {}) or {}
            if (
                data.get("routine_name") == "Luminescence"
                and data.get("routine_status") == "Ready"
            ):
                return True
            print("ensure_ready failed after StartRoutine, last status:", status)
            return False

        if settings is not None:
            resp = self.apply_settings(settings)
            if not self.is_ok(resp):
                print("apply_settings failed:", resp)
                return False

        return True

    # -------------------------------
    # Saving helpers
    # -------------------------------
    @staticmethod
    def _compute_summary_from_data(data):
        """
        Compute scalar indicators for quick drift checks.

        Supported layouts:
        1. flat single-spectrum:
        data["wavelengths"] = [...]
        data["spectrum"]    = [...]
        2. flat dark/reference:
        data["wavelengths"] = [...]
        data["dark"]        = [...]
        or
        data["reference"]   = [...]
        3. nested legacy layout:
        data["spectrum"]    = [wavelengths, signal, ...]
        """
        wavelengths_raw = None
        signal_raw = None

        wavelengths = data.get("wavelengths", None)
        spectrum = data.get("spectrum", None)
        dark = data.get("dark", None)
        reference = data.get("reference", None)

        def _is_1d_numeric_list(x):
            return (
                isinstance(x, (list, tuple))
                and len(x) > 0
                and not isinstance(x[0], (list, tuple))
            )

        def _looks_nonempty_signal(x):
            if not _is_1d_numeric_list(x):
                return False
            # accept if at least one element is meaningfully nonzero
            for v in x:
                try:
                    if abs(float(v)) > 1e-15:
                        return True
                except (TypeError, ValueError):
                    continue
            return False

        # Case 1: standard flat spectrum layout
        if _is_1d_numeric_list(wavelengths) and _is_1d_numeric_list(spectrum):
            if _looks_nonempty_signal(spectrum):
                wavelengths_raw = wavelengths
                signal_raw = spectrum

        # Case 2: dark acquisition
        if (wavelengths_raw is None or signal_raw is None) and _is_1d_numeric_list(
            wavelengths
        ):
            if _looks_nonempty_signal(dark):
                wavelengths_raw = wavelengths
                signal_raw = dark

        # Case 3: reference acquisition
        if (wavelengths_raw is None or signal_raw is None) and _is_1d_numeric_list(
            wavelengths
        ):
            if _looks_nonempty_signal(reference):
                wavelengths_raw = wavelengths
                signal_raw = reference

        # Case 4: legacy nested layout
        if wavelengths_raw is None or signal_raw is None:
            if (
                isinstance(spectrum, (list, tuple))
                and len(spectrum) >= 2
                and isinstance(spectrum[0], (list, tuple))
                and isinstance(spectrum[1], (list, tuple))
            ):
                wavelengths_raw = spectrum[0]
                signal_raw = spectrum[1]

        if wavelengths_raw is None or signal_raw is None:
            return {
                "peak_wavelength_nm": None,
                "peak_signal": None,
                "integrated_signal": None,
            }

        cleaned = []
        for wl, sig in zip(wavelengths_raw, signal_raw):
            try:
                cleaned.append((float(wl), float(sig)))
            except (TypeError, ValueError):
                continue

        if not cleaned:
            return {
                "peak_wavelength_nm": None,
                "peak_signal": None,
                "integrated_signal": None,
            }

        wavelengths = [x[0] for x in cleaned]
        signal = [x[1] for x in cleaned]

        max_idx = max(range(len(signal)), key=lambda i: signal[i])
        peak_wavelength = wavelengths[max_idx]
        peak_signal = signal[max_idx]

        integrated_signal = 0.0
        if len(signal) >= 2:
            for i in range(len(signal) - 1):
                dw = wavelengths[i + 1] - wavelengths[i]
                integrated_signal += 0.5 * (signal[i] + signal[i + 1]) * dw

        return {
            "peak_wavelength_nm": peak_wavelength,
            "peak_signal": peak_signal,
            "integrated_signal": integrated_signal,
        }

    @staticmethod
    def append_jsonl_record(filepath, record):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(str(filepath), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def append_summary_csv(filepath, record):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        file_exists = filepath.exists()

        with open(str(filepath), "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "cycle",
                        "modality",
                        "status",
                        "reason",
                        "peak_wavelength_nm",
                        "peak_signal",
                        "integrated_signal",
                    ]
                )

            writer.writerow(
                [
                    record.get("cycle"),
                    record.get("modality"),
                    record.get("status"),
                    record.get("reason"),
                    record.get("peak_wavelength_nm"),
                    record.get("peak_signal"),
                    record.get("integrated_signal"),
                ]
            )

    @staticmethod
    def save_spectrum_csv(data_dict, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        wavelengths = data_dict.get("wavelengths", None)
        spectrum = data_dict.get("spectrum", None)
        dark = data_dict.get("dark", None)
        reference = data_dict.get("reference", None)

        def _is_1d_numeric_list(x):
            return (
                isinstance(x, (list, tuple))
                and len(x) > 0
                and not isinstance(x[0], (list, tuple))
            )

        def _looks_nonempty_signal(x):
            if not _is_1d_numeric_list(x):
                return False
            for v in x:
                try:
                    if abs(float(v)) > 1e-15:
                        return True
                except (TypeError, ValueError):
                    continue
            return False

        signal_name = None
        signal = None

        if _is_1d_numeric_list(wavelengths):
            if _looks_nonempty_signal(spectrum):
                signal_name = "spectrum"
                signal = spectrum
            elif _looks_nonempty_signal(dark):
                signal_name = "dark"
                signal = dark
            elif _looks_nonempty_signal(reference):
                signal_name = "reference"
                signal = reference

        with open(str(filepath), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if signal is not None:
                writer.writerow(["wavelength_nm", signal_name])
                for wl, sig in zip(wavelengths, signal):
                    writer.writerow([wl, sig])
                return

            # nested legacy fallback
            if (
                isinstance(spectrum, (list, tuple))
                and len(spectrum) >= 2
                and isinstance(spectrum[0], (list, tuple))
                and isinstance(spectrum[1], (list, tuple))
            ):
                wavelengths = spectrum[0]
                y1 = spectrum[1]
                y2 = spectrum[2] if len(spectrum) > 2 else None

                if y2 is None:
                    writer.writerow(["wavelength_nm", "signal"])
                    for wl, a in zip(wavelengths, y1):
                        writer.writerow([wl, a])
                else:
                    writer.writerow(["wavelength_nm", "raw_signal", "irradiance"])
                    for wl, a, b in zip(wavelengths, y1, y2):
                        writer.writerow([wl, a, b])
                return

            raise ValueError("Unsupported spectrum layout in response data")

    def prepare_output_files(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        files = {
            "white_jsonl": save_dir / "white_spectra.jsonl",
            "white_csv": save_dir / "white_summary.csv",
            "blue_jsonl": save_dir / "blue_spectra.jsonl",
            "blue_csv": save_dir / "blue_summary.csv",
            "dark_jsonl": save_dir / "dark_spectra.jsonl",
            "dark_csv": save_dir / "dark_summary.csv",
        }

        return files

    def _acquire_and_append(
        self,
        command_name,
        modality,
        cycle,
        jsonl_path,
        summary_csv_path,
        ready_timeout_s=1.5,
    ):
        """
        Best-effort acquisition:
        - briefly checks readiness
        - skips if not ready
        - appends full spectrum to JSONL
        - appends summary metrics to CSV
        """

        if not self.ensure_ready(timeout_s=ready_timeout_s):
            summary = {
                "cycle": cycle,
                "modality": modality,
                "status": "skipped",
                "reason": "routine_not_ready",
                "peak_wavelength_nm": None,
                "peak_signal": None,
                "integrated_signal": None,
            }
            self.append_jsonl_record(jsonl_path, summary)
            self.append_summary_csv(summary_csv_path, summary)
            return None

        if command_name == "AcquireSingle":
            resp = self.acquire_single()
        elif command_name == "AcquireDark":
            resp = self.acquire_dark()
        elif command_name == "AcquireReference":
            resp = self.acquire_reference()
        else:
            raise ValueError(f"Unsupported command: {command_name}")

        if not self.is_ok(resp):
            summary = {
                "cycle": cycle,
                "modality": modality,
                "status": "failed",
                "reason": "api_error",
                "peak_wavelength_nm": None,
                "peak_signal": None,
                "integrated_signal": None,
            }
            self.append_jsonl_record(jsonl_path, summary)
            self.append_summary_csv(summary_csv_path, summary)
            return None

        data = self.get_data(resp)

        if data is None:
            summary = {
                "cycle": cycle,
                "modality": modality,
                "status": "failed",
                "reason": "empty_data",
                "peak_wavelength_nm": None,
                "peak_signal": None,
                "integrated_signal": None,
            }
            self.append_jsonl_record(jsonl_path, summary)
            self.append_summary_csv(summary_csv_path, summary)
            return None

        # print(type(data.get("wavelengths")), len(data.get("wavelengths", [])))
        # print(type(data.get("spectrum")), len(data.get("spectrum", [])))
        # print(data.get("wavelengths", [])[:5])
        # print(data.get("spectrum", [])[:5])

        record = {
            "cycle": cycle,
            "modality": modality,
            "status": "ok",
            "reason": None,
            "data": data,
        }

        summary_metrics = self._compute_summary_from_data(data)
        summary = {
            "cycle": cycle,
            "modality": modality,
            "status": "ok",
            "reason": None,
            "peak_wavelength_nm": summary_metrics["peak_wavelength_nm"],
            "peak_signal": summary_metrics["peak_signal"],
            "integrated_signal": summary_metrics["integrated_signal"],
        }

        self.append_jsonl_record(jsonl_path, record)
        self.append_summary_csv(summary_csv_path, summary)

        return {
            "record": record,
            "summary": summary,
        }

    # -------------------------------
    # DegImage-oriented aliases
    # -------------------------------
    def acquire_white_spectrum(
        self, cycle, jsonl_path, summary_csv_path, ready_timeout_s=1.5
    ):
        return self._acquire_and_append(
            command_name="AcquireSingle",
            modality="white",
            cycle=cycle,
            jsonl_path=jsonl_path,
            summary_csv_path=summary_csv_path,
            ready_timeout_s=ready_timeout_s,
        )

    def acquire_blue_spectrum(
        self, cycle, jsonl_path, summary_csv_path, ready_timeout_s=1.5
    ):
        return self._acquire_and_append(
            command_name="AcquireSingle",
            modality="blue",
            cycle=cycle,
            jsonl_path=jsonl_path,
            summary_csv_path=summary_csv_path,
            ready_timeout_s=ready_timeout_s,
        )

    def acquire_dark_spectrum(
        self, cycle, jsonl_path, summary_csv_path, ready_timeout_s=1.5
    ):
        return self._acquire_and_append(
            command_name="AcquireDark",
            modality="dark",
            cycle=cycle,
            jsonl_path=jsonl_path,
            summary_csv_path=summary_csv_path,
            ready_timeout_s=ready_timeout_s,
        )


if __name__ == "__main__":
    api = LuminescenceAPI(host="192.168.0.250", port=6360)

    if not api.connect():
        raise RuntimeError("Could not connect to Arkeo luminescence API")

    try:
        ok = api.initialize_luminescence(sample_lum_settings, ready_timeout_s=2.0)
        if not ok:
            raise RuntimeError("Could not initialize Luminescence routine")

        outdir = Path("lum_monitor")
        files = api.prepare_output_files(outdir)

        # Example cycle
        cycle = 1

        white = api.acquire_white_spectrum(
            cycle=cycle,
            jsonl_path=files["white_jsonl"],
            summary_csv_path=files["white_csv"],
            ready_timeout_s=1.0,
        )
        print("WHITE:", white)

        blue = api.acquire_blue_spectrum(
            cycle=cycle,
            jsonl_path=files["blue_jsonl"],
            summary_csv_path=files["blue_csv"],
            ready_timeout_s=1.0,
        )
        print("BLUE:", blue)

        dark = api.acquire_dark_spectrum(
            cycle=cycle,
            jsonl_path=files["dark_jsonl"],
            summary_csv_path=files["dark_csv"],
            ready_timeout_s=1.0,
        )
        print("DARK:", dark)

    finally:
        try:
            api.close_routine()
        except Exception:
            pass
        api.disconnect()
