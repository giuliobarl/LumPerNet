import json
import socket
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
    def __init__(self, host="localhost", port=6340):
        self.host = host
        self.port = port
        self.connection = None

    def connect(self):
        """Establish a TCP connection to the LabVIEW server."""
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.connect((self.host, self.port))

    def disconnect(self):
        """Close the TCP connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def send_command(self, command, parameter=""):
        """Send a command with optional data to the server and receive a response."""
        try:
            # Ensure the connection is alive
            if not self.connection:
                self.connect()

            # Prepare and send data
            message = json.dumps({"command": command, "parameter": parameter}) + "\r\n"

            # Prepare the length of the string
            length = len(message)
            length_bytes = length.to_bytes(
                4, byteorder="big"
            )  # 4 bytes to represent the length

            # Send the length and the string
            self.connection.sendall(length_bytes)
            self.connection.sendall(message.encode("utf-8"))

            # Wait for the response
            response_length_bytes = self.connection.recv(4)
            response_length = int.from_bytes(response_length_bytes, byteorder="big")
            if not response_length_bytes:
                print("No response received.")
                return None
            response = self.connection.recv(response_length)
            if not response:
                print("No response received.")
                return None
            return response.decode("utf-8")
        except Exception as e:
            print(f"An error occurred: {e}")
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
