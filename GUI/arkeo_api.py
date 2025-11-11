import json
import socket
import sys
import time

sampleSettings = {
    "Enable": True,
    "User": "User",
    "Device": "Sample",
    "JV": {
        "Vmin (V)": -0.1,
        "Vmax (V)": 2,
        "VocDetect": True,
        "Overvoltage": 0,
        "Step (mV)": 20,
        "ScanRate (mV/s)": 100,
        "ScanOrder": "FW then RV",
        "VoltageLimit": "10 V",
        "CurrentLimit": 0,
        "InvertedStructure": False,
    },
    "Tracking": {
        "TrackEnable": True,
        "Algorithm": "MPPT",
        "dV (V)": 0.01,
        "jvInterval": 1,
        "jvIntervalUnit": 1,
        "TestDuration": 100,
        "DurationUnit": 1,
        "ConstantOutput": 0.2,
        "SaveDecimation": 10,
    },
    "Cell": {
        "Type": "Cell",
        "Area (cm²)": 1,
        "NrCells": 1,
        "NrW cells": 1,
        "W cell area": 1,
    },
    "Note": "",
}


class MeasurementAPI:
    def __init__(
        self,
        host="192.168.0.250",  # your Arkeo PC IP
        port=6340,
        timeout=3.0,  # per-IO timeout
        retries=2,  # resend after reconnect
        reconnect_backoff=(0.3, 1.0, 2.0),  # progressive backoff
        enable_keepalive=True,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retries = retries
        self.reconnect_backoff = reconnect_backoff
        self.enable_keepalive = enable_keepalive
        self.connection: socket.socket | None = None

    # ---------- socket helpers ----------
    def _apply_keepalive(self, s: socket.socket) -> None:
        if not self.enable_keepalive:
            return
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if sys.platform.startswith("linux"):
                # Linux TCP keepalive tuning (best-effort)
                s.setsockopt(socket.IPPROTO_TCP, 0x10, 60)  # TCP_KEEPIDLE
                s.setsockopt(socket.IPPROTO_TCP, 0x12, 5)  # TCP_KEEPCNT
                s.setsockopt(socket.IPPROTO_TCP, 0x11, 10)  # TCP_KEEPINTVL
        except Exception:
            pass

    def connect(self) -> bool:
        for delay in (0.0, *self.reconnect_backoff):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(self.timeout)
                self._apply_keepalive(s)
                s.connect((self.host, self.port))
                self.connection = s
                return True
            except (TimeoutError, ConnectionError, OSError):
                self.connection = None
                time.sleep(delay)
        return False

    def disconnect(self):
        try:
            if self.connection:
                self.connection.close()
        finally:
            self.connection = None

    def _recv_exact(self, n: int) -> bytes | None:
        """Read exactly n bytes (with timeout). Return None on failure."""
        if not self.connection:
            return None
        buf = bytearray()
        end = time.time() + self.timeout
        try:
            while len(buf) < n and time.time() < end:
                chunk = self.connection.recv(n - len(buf))
                if not chunk:
                    return None
                buf.extend(chunk)
            return bytes(buf) if len(buf) == n else None
        except (TimeoutError, ConnectionError, OSError):
            return None

    def _send_recv_once(self, payload: bytes) -> str | None:
        """One IO attempt on current socket; returns decoded string or None."""
        if not self.connection and not self.connect():
            return None
        try:
            # send length-prefixed JSON line
            self.connection.sendall(len(payload).to_bytes(4, "big"))
            self.connection.sendall(payload)
            # read 4-byte length, then body
            hdr = self._recv_exact(4)
            if not hdr:
                self.disconnect()
                return None
            resp_len = int.from_bytes(hdr, "big")
            body = self._recv_exact(resp_len)
            if not body:
                self.disconnect()
                return None
            return body.decode("utf-8", errors="replace")
        except (TimeoutError, BrokenPipeError, ConnectionError, OSError):
            self.disconnect()
            return None

    def send_command(self, command: str, parameter="") -> str | None:
        """
        High-reliability send: try current socket, then reconnect+retry up to `retries`.
        Never raises; returns None if all attempts fail.
        """
        message = json.dumps({"command": command, "parameter": parameter}) + "\r\n"
        payload = message.encode("utf-8")

        # first try on current socket
        resp = self._send_recv_once(payload)
        if resp is not None:
            return resp

        # retries with reconnect
        for _ in range(self.retries):
            if not self.connect():
                continue
            resp = self._send_recv_once(payload)
            if resp is not None:
                return resp
        return None

    def set_active_channel(self, channel_id):
        """Set the active channel."""
        return self.send_command("SetActiveChannel", str(channel_id))

    def get_active_channel(self):
        """Get the active channel."""
        return self.send_command("GetActiveChannel")

    def set_channel_settings(self, settings_json):
        """Set the JSON settings for a specific channel."""
        # data = json.dumps({'channel_id': channel_id, 'settings': settings_json})
        return self.send_command("SetChannelSettings", settings_json)

    def get_channel_settings(self):
        """Get the JSON settings for a specific channel."""
        return self.send_command(
            "GetChannelSettings",
        )

    def start_channel(self):
        """Start measurement on a specific channel."""
        return self.send_command("StartChannel")

    def stop_channel(self):
        """Stop measurement on a specific channel."""
        return self.send_command("StopChannel")

    def force_jv_measurement(self):
        """Force a JV measurement on a specific channel."""
        return self.send_command("ForceJV")

    def get_channel_state(self):
        """Get the state of a specific channel."""
        return self.send_command("GetChannelState")


# Example of how to use the API
if __name__ == "__main__":
    api = MeasurementAPI("localhost", 6340)
    api.connect()

    print(api.set_active_channel(1))
    print(api.get_active_channel())  # this should match the channel ID set above
    ExampleSettings = api.get_channel_settings()
    # print(ExampleSettings)

    # Parse the JSON string into a Python dictionary
    data = json.loads(ExampleSettings)

    # Update the fields as needed
    data["Enable"] = True
    data["Tracking"]["Algorithm"] = "MPPT"

    # Convert the Python dictionary back to a JSON string
    sampleSettings = json.dumps(data)
    print(sampleSettings)

    print(api.set_channel_settings(sampleSettings))
    print(api.start_channel())
    time.sleep(2)
    print(api.force_jv_measurement())
    print(api.get_channel_state())
    # print(api.stop_channel())

    api.disconnect()
