import datetime
import fcntl
import json
import os
import sys
import threading
import time
from pathlib import Path
from tkinter import END, Scrollbar, Text

import customtkinter
import numpy as np
from PIL import Image
from RPi import GPIO
from tqdm import tqdm

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path

    configure_path()
except ImportError:
    configure_path = None

import faulthandler

from arkeo_api import MeasurementAPI
from camera import acquisition_PL
from cycle_commands import run_EL, run_JV, run_PL

faulthandler.enable(all_threads=True)

customtkinter.set_appearance_mode(
    "Light"
)  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme(
    "blue"
)  # Themes: "blue" (standard), "green", "dark-blue"

HEARTBEAT = "/tmp/degimage.heartbeat"


def _hb(stop_event: threading.Event):
    """Touch a heartbeat file every 2 s until told to stop."""
    while not stop_event.is_set():
        try:
            # create if missing, else update mtime
            try:
                os.utime(HEARTBEAT, None)
            except FileNotFoundError:
                open(HEARTBEAT, "w").close()
        except Exception:
            pass
        stop_event.wait(2.0)  # sleeps but wakes fast on stop


def _acquire_single_instance_lock(path="/tmp/degimage.lock"):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd  # keep it open for process lifetime
    except OSError:
        raise SystemExit("Another instance is already running.")


class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.terminal = sys.stdout  # keep mirroring to real stdout if you want

    def write(self, message):
        try:
            self.terminal.write(message)
            self.terminal.flush()
        except Exception:
            pass
        # schedule UI update on Tk main loop
        try:
            self.text_widget.after(0, self._append, message)
        except Exception:
            pass

    def _append(self, message):
        try:
            self.text_widget.insert("end", message)
            self.text_widget.see("end")
        except Exception:
            pass

    def flush(self):
        pass


def gpio_safe_cleanup(pins_off=()):
    try:
        for pin in pins_off:
            try:
                GPIO.output(pin, GPIO.LOW)
            except Exception:
                pass
    finally:
        try:
            GPIO.cleanup()
        except Exception:
            pass


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.base_dir = Path("/home/chose/Documenti")
        self.folder_path = Path("/home/chose/Documenti")
        self.PL_path = Path("/home/chose/Documenti")
        self.EL_path = Path("/home/chose/Documenti")
        self.cycle_counter = 0
        self.cycle_running = False
        self.run_thread = None
        self.api = MeasurementAPI("192.168.0.250")
        self.ensure_api_connection()

        self.GPIO_PIN_BLUE = 2
        self.GPIO_PIN_WHITE = 21

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # attribute initialization
        self.iter_time: int = 240
        self.JV_time: int = 30
        self.PL_time: int = 10
        self.EL_time: int = 10
        self.EL_voltage: float = 1.4
        self.t_recover: int = 60
        self.t_relax: int = 15
        self.exp_time: int = 1000
        self.exp_time_sc: int = 2000
        self.active_channels: int = 16
        self.cell_area: float = 0.16
        self.cell_inverted: bool = False
        self.batch_name: str = "batch"
        self.res_name: str = "Giulio Barletta"
        self.sampling_strategy: str = "decreasing"
        self.max_iter: int = 360

        if self.sampling_strategy == "decreasing":
            self.generate_schedule()

        # camera configuration
        self.USE_CAMERA = True

        # configure window
        self.title("CHOSE EL/PL GUI")
        self.geometry(f"{1400}x{600}")

        # configure grid layout (4x3)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        ################################ column 0 ################################

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # insert logo and name of application
        my_image = customtkinter.CTkImage(
            light_image=Image.open("/home/chose/Documenti/Codes/chose_logo.png"),
            dark_image=Image.open("/home/chose/Documenti/Codes/chose_logo.png"),
            size=(180, 100),
        )  # WidthxHeight
        self.logo = customtkinter.CTkLabel(self.sidebar_frame, text="", image=my_image)
        self.logo.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.logo_label_1 = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Degradation imaging bench",
            font=customtkinter.CTkFont(size=20, weight="bold"),
        )
        self.logo_label_1.grid(row=1, column=0, padx=20, pady=(20, 10), sticky="nsew")

        # important notice
        self.notice = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Carefully read\n the README\n in the textbox\n to the upper right.",
            font=customtkinter.CTkFont(size=24, weight="bold"),
            text_color="orange",
        )
        self.notice.grid(row=2, column=0, padx=20, pady=(20, 10), sticky="nsew")

        # widgets for view
        self.appearance_mode_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Appearance Mode:",
            font=customtkinter.CTkFont(size=15),
        )
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(5, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(
            self.sidebar_frame,
            values=["Light", "Dark", "System"],
            font=customtkinter.CTkFont(size=15),
            command=self.change_appearance_mode_event,
        )
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(5, 10))
        self.scaling_label = customtkinter.CTkLabel(
            self.sidebar_frame, text="UI Scaling:", font=customtkinter.CTkFont(size=15)
        )
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(5, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(
            self.sidebar_frame,
            values=["80%", "90%", "100%", "110%", "120%"],
            font=customtkinter.CTkFont(size=15),
            command=self.change_scaling_event,
        )
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(5, 20))

        # credits
        self.credits = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="GUI developed by Giulio Barletta \n<giulio.barletta@polito.it>",
            font=customtkinter.CTkFont(size=15),
        )
        self.credits.grid(row=9, column=0, padx=20, pady=(5, 0), sticky="nsew")

        ################################ column 1 ################################

        # Create the entry widget (command line input)
        self.command_line = customtkinter.CTkEntry(
            self, placeholder_text="Type a command and press Enter"
        )
        self.command_line.grid(
            row=2, column=1, columnspan=2, padx=(20, 0), pady=(20, 0), sticky="nsew"
        )

        # Bind the Enter key to the 'process_command' function
        self.command_line.bind("<Return>", self.process_command)

        # create textbox
        # Create a label for the first line with larger font size
        self.textbox = customtkinter.CTkTextbox(self, width=200)
        self.textbox.grid(
            row=0, column=1, columnspan=2, padx=(20, 0), pady=(20, 0), sticky="nsew"
        )

        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=200)
        self.tabview.grid(row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Lights")
        self.tabview.add("Camera")
        self.tabview.add("Cycle")
        self.tabview.add("Batch")
        self.tabview.tab("Lights").grid_columnconfigure(
            0, weight=1
        )  # configure grid of individual tabs
        self.tabview.tab("Camera").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Cycle").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Batch").grid_columnconfigure(0, weight=1)

        ###### LIGHTS SETTINGS ######
        self.lights_settings_label = customtkinter.CTkLabel(
            self.tabview.tab("Lights"),
            text="Lights settings",
            font=customtkinter.CTkFont(size=15),
        )
        self.lights_settings_label.grid(row=1, column=0, padx=20, pady=(5, 0))

        # Initialize the white lights' time variable
        self.white_input_button = customtkinter.CTkButton(
            self.tabview.tab("Lights"),
            text="White LED's soaking time (s)",
            command=self.open_white_dialog_event,
        )
        self.white_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        # Initialize the recovery time variable (after electrical biasing / JV)
        self.recover_input_button = customtkinter.CTkButton(
            self.tabview.tab("Lights"),
            text="Recovery time after white soak (s)",
            command=self.open_recov_dialog_event,
        )
        self.recover_input_button.grid(row=3, column=0, padx=20, pady=(10, 10))

        # Initialize the blue lights' time variable
        self.blue_input_button = customtkinter.CTkButton(
            self.tabview.tab("Lights"),
            text="Blue LEDs' soaking time (s)",
            command=self.open_blue_dialog_event,
        )
        self.blue_input_button.grid(row=4, column=0, padx=20, pady=(10, 10))

        # Initialize the relaxing time variable (after blue light soaking)
        self.relax_input_button = customtkinter.CTkButton(
            self.tabview.tab("Lights"),
            text="Recovery time after blue soak (s)",
            command=self.open_relax_dialog_event,
        )
        self.relax_input_button.grid(row=5, column=0, padx=20, pady=(10, 10))

        # Initialize the lights' off time variable
        self.off_input_button = customtkinter.CTkButton(
            self.tabview.tab("Lights"),
            text="LEDs' off time for EL (s)",
            command=self.open_off_dialog_event,
        )
        self.off_input_button.grid(row=6, column=0, padx=20, pady=(10, 10))

        ###### CAMERA SETTINGS ######
        self.camera_settings_label = customtkinter.CTkLabel(
            self.tabview.tab("Camera"),
            text="Camera settings",
            font=customtkinter.CTkFont(size=15),
        )
        self.camera_settings_label.grid(row=1, column=0, padx=20, pady=(5, 0))

        # Initialize the exposure time variable
        self.exposure_input_button = customtkinter.CTkButton(
            self.tabview.tab("Camera"),
            text="Exposure time (ms)",
            command=self.open_exposure_dialog_event,
        )
        self.exposure_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        self.sc_exposure_input_button = customtkinter.CTkButton(
            self.tabview.tab("Camera"),
            text="SC exposure time (ms)",
            command=self.open_sc_exposure_dialog_event,
        )
        self.sc_exposure_input_button.grid(row=3, column=0, padx=20, pady=(10, 10))

        ###### CYCLE SETTINGS ######
        self.cycle_settings_label = customtkinter.CTkLabel(
            self.tabview.tab("Cycle"),
            text="Cycle settings",
            font=customtkinter.CTkFont(size=15),
        )
        self.cycle_settings_label.grid(row=1, column=0, padx=20, pady=(5, 0))

        # Initialize the iteration interval variable
        self.iter_time_input_button = customtkinter.CTkButton(
            self.tabview.tab("Cycle"),
            text="Iteration interval (s)",
            command=self.open_iter_time_dialog_event,
        )
        self.iter_time_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        # Initialize the acquisition strategy variable
        self.strategy_input_button = customtkinter.CTkButton(
            self.tabview.tab("Cycle"),
            text="Acquisition strategy",
            command=self.open_strategy_dialog_event,
        )
        self.strategy_input_button.grid(row=3, column=0, padx=20, pady=(10, 10))

        # Initialize the maximum iterarions variable
        self.max_iter_input_button = customtkinter.CTkButton(
            self.tabview.tab("Cycle"),
            text="Maximum number of iterations",
            command=self.open_max_iter_dialog_event,
        )
        self.max_iter_input_button.grid(row=4, column=0, padx=20, pady=(10, 10))

        ###### BATCH SETTINGS ######
        self.batch_settings_label = customtkinter.CTkLabel(
            self.tabview.tab("Batch"),
            text="Batch settings",
            font=customtkinter.CTkFont(size=15),
        )
        self.batch_settings_label.grid(row=1, column=0, padx=20, pady=(5, 0))

        # Initialize the channels number variable
        self.channels_input_button = customtkinter.CTkButton(
            self.tabview.tab("Batch"),
            text="Active Channels",
            command=self.open_channels_dialog_event,
        )
        self.channels_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        # Initialize the EL voltage variable
        self.VEL_input_button = customtkinter.CTkButton(
            self.tabview.tab("Batch"),
            text="EL Voltage (V)",
            command=self.open_ELvoltage_dialog_event,
        )
        self.VEL_input_button.grid(row=3, column=0, padx=20, pady=(10, 10))

        # Initialize the cell area variable
        self.area_input_button = customtkinter.CTkButton(
            self.tabview.tab("Batch"),
            text="Cell area (cm2)",
            command=self.open_area_dialog_event,
        )
        self.area_input_button.grid(row=4, column=0, padx=20, pady=(10, 10))

        # Initialize the cell inverted variable
        self.inverted_input_button = customtkinter.CTkButton(
            self.tabview.tab("Batch"),
            text="Cell inverted",
            command=self.open_inverted_dialog_event,
        )
        self.inverted_input_button.grid(row=5, column=0, padx=20, pady=(10, 10))

        # Initialize the batch name variable
        self.batch_input_button = customtkinter.CTkButton(
            self.tabview.tab("Batch"),
            text="Batch name",
            command=self.open_name_dialog_event,
        )
        self.batch_input_button.grid(row=6, column=0, padx=20, pady=(10, 10))

        # Initialize the researcher name variable
        self.res_input_button = customtkinter.CTkButton(
            self.tabview.tab("Batch"),
            text="Researcher's name",
            command=self.open_res_dialog_event,
        )
        self.res_input_button.grid(row=7, column=0, padx=20, pady=(10, 10))

        ################################ column 2 ################################

        # Terminal log display (Bottom section)
        self.log_text = Text(self, height=10, width=40, wrap="word", state="normal")
        self.log_text.grid(row=1, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")

        log_scrollbar = Scrollbar(self, command=self.log_text.yview)
        log_scrollbar.grid(row=1, column=3, sticky="ns")
        self.log_text.config(yscrollcommand=log_scrollbar.set)

        sys.stdout = RedirectText(self.log_text)

        ################################ default values ################################

        # set default values
        self.appearance_mode_optionemenu.set("Light")
        self.scaling_optionemenu.set("100%")
        self.textbox.insert("1.0", "README \n \n")

        self.textbox.insert(
            "3.0",
            "This is the user guide to this GUI, developed to have easy access to the degradation imaging bench. \n",
        )
        self.textbox.insert("4.0", "Before running the tests, make sure that: \n")
        self.textbox.insert("5.0", "- the camera USB cable is plugged in, \n")
        self.textbox.insert(
            "6.0",
            "- the power source is on at 24—26 V and the current knob is turned all the way to the max (during operation, the current should be around 1.8 A), \n",
        )
        self.textbox.insert(
            "7.0",
            "- the Arkeo PC is ON, connected to the network (IP address should be 192.168.0.216), and `Arkeo Multichannel` is running. \n\n",
        )
        self.textbox.insert(
            "9.0",
            "This app integrates PL/EL imaging and JV characterization. Before starting, go through the Tabs in the Box below, click on the buttons, and enter the desired values. \n",
        )
        self.textbox.insert(
            "10.0",
            "- LIGHTS: Enter the time intervals over which the LEDs are on/off. Make sure that the White LED stays on for at least the duration of a JV measurement. \n",
        )
        self.textbox.insert(
            "11.0",
            "- CAMERA: Enter the exposure time in milliseconds (e.g. entering 200 will result in an exposure time of 0.2 seconds). Please notice: if you enter 0 as the exposure time, the program will still run, but no image will be captured. \n",
        )
        self.textbox.insert(
            "12.0",
            "- CYCLE: Enter the desired time interval between consecutive when measurements and acquisition strategy for cyclic mode. Acquisition supports either a 'linear' or a 'decreasing' strategy: \n",
        )
        self.textbox.insert(
            "13.0",
            "   - 'linear': performs one measurement at each iteration; \n",
        )
        self.textbox.insert(
            "14.0",
            "   - 'decreasing': performs measurements more frequently in the early iterations, less in later ones. \n",
        )
        self.textbox.insert(
            "15.0",
            "- BATCH: Enter the number of active channels, the desired voltage for EL biasing, the cell area, if the cell is inverted (True means stack is n-i-p), the reference of the cell/batch, and the name of the researcher performing the measurements. \n",
        )
        self.textbox.insert(
            "16.0",
            "The acquired image(s) will be saved as '<reference>.tiff' in the folder '/home/chose/Documenti/<researcher>/<date>/<EL/PL>'. \n\n",
        )
        self.textbox.insert(
            "18.0",
            "The command line at the bottom of the GUI allows running the measurement programs. It accepts 4 commands: \n",
        )
        self.textbox.insert(
            "19.0",
            "- summary: shows the values stored by the program, always check them before running. \n",
        )
        self.textbox.insert(
            "20.0", "- run: performs one single measurement and acquires an image. \n"
        )
        self.textbox.insert(
            "21.0",
            "- cycle: runs in cyclic mode, acquiring one image per iteration, until stopped. The sampling strategy is user-defined. \n",
        )
        self.textbox.insert(
            "22.0",
            "- stop: stops the running cycles. Always use this and only this command to stop cycles. \n",
        )
        self.textbox.insert(
            "23.0",
            "IMPORTANT: counterintuitively, the iterations following the first one start with the pause. This will be fixed in the future, but it is necessary as of now. \n",
        )
        self.textbox.insert(
            "24.0",
            "This means that if you use the STOP command during the pause between iterations, you still need to wait for the JV, EL and PL measurements to take place. \n\n",
        )
        self.textbox.insert(
            "26.0",
            "If the program returns an error, and the LEDs remain ON, please CLOSE the application and read the README file on the desktop. You will find what to do in lines 41-45. \n \n",
        )
        self.textbox.insert(
            "28.0",
            "For more information, or if you have doubts, ask Giulio Barletta \n<giulio.barletta@polito.it> \nor Simon Ternes \n<ternes@ing.uniroma2.it>.",
        )
        self.textbox.configure(state="disabled")

        # Run the startup camera gate once the window is ready:
        self.after(100, self._startup_camera_gate)

        self._hb_stop = threading.Event()
        self._hb_thread = threading.Thread(
            target=_hb, args=(self._hb_stop,), daemon=True
        )
        self._hb_thread.start()

    ################################ function definitions ################################

    def _startup_camera_gate(self):
        """Run once at startup: if no camera, show a modal error window and quit on close."""
        if self.USE_CAMERA:
            # Lazy import to avoid circulars and to keep startup quick
            try:
                from camera import any_camera_connected
            except Exception:

                def any_camera_connected():
                    return False

            if not any_camera_connected():
                popup = customtkinter.CTkToplevel(self)
                popup.title("Error")
                popup.geometry("460x140")
                popup.transient(self)  # keep on top of the main window
                popup.attributes("-topmost", True)
                popup.grab_set()  # modal: block interaction with the main app

                # Message only — no extra buttons, per your request
                msg = customtkinter.CTkLabel(
                    popup,
                    text="Error: no camera detected.\nMake sure it is correctly plugged in!",
                    font=customtkinter.CTkFont(size=16, weight="bold"),
                    justify="center",
                    padx=20,
                    pady=20,
                )
                msg.pack(expand=True, fill="both")

                # When user clicks the window's ✕, close the whole app
                def _close_everything():
                    try:
                        popup.grab_release()
                    except Exception:
                        pass
                    popup.destroy()
                    self.destroy()  # shuts down the entire application

                popup.protocol("WM_DELETE_WINDOW", _close_everything)

    def _date_root(self) -> str:
        """Return the absolute path to today's date folder: <base/res_name/YYYY-MM-DD>."""
        import datetime
        import os

        today = datetime.datetime.now().date().strftime("%Y-%m-%d")
        return os.path.join(str(self.base_dir), self.res_name, today)

    # appearance functions

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    ###### lights settings functions ######

    def open_white_dialog_event(self):
        dialog_white = customtkinter.CTkInputDialog(
            text="Type in the white LED's ON time (s):", title="Lights' Settings"
        )
        # Get the input from the dialog
        white_time = dialog_white.get_input()
        if white_time is None or not str(white_time).strip():
            print(f"No value inserted, falling back to default: {self.JV_time} s")
        else:
            print(f"White LEDs ON time: {white_time} s")
            # Store the input in the instance variable
            self.JV_time = int(white_time)

    def open_recov_dialog_event(self):
        dialog_recov = customtkinter.CTkInputDialog(
            text="Type in the recovery time between white and blue light soaking (s):",
            title="Lights' Settings",
        )
        # Get the input from the dialog
        recov_time = dialog_recov.get_input()
        if recov_time is None or not str(recov_time).strip():
            print(f"No value inserted, falling back to default: {self.t_recover} s")
        else:
            print(f"Recovery time: {recov_time} s")
            # Store the input in the instance variable
            self.t_recover = int(recov_time)

    def open_blue_dialog_event(self):
        dialog_blue = customtkinter.CTkInputDialog(
            text="Type in the blue LEDs' ON time (s):", title="Lights' Settings"
        )
        # Get the input from the dialog
        blue_time = dialog_blue.get_input()
        if blue_time is None or not str(blue_time).strip():
            print(f"No value inserted, falling back to default: {self.PL_time} s")
        else:
            print(f"Blue LEDs' ON time: {blue_time} s")
            # Store the input in the instance variable
            self.PL_time = int(blue_time)

    def open_relax_dialog_event(self):
        dialog_relax = customtkinter.CTkInputDialog(
            text="Type in the recovery time between blue light soaking and electrical biasing (s):",
            title="Lights' Settings",
        )
        # Get the input from the dialog
        relax_time = dialog_relax.get_input()
        if relax_time is None or not str(relax_time).strip():
            print(f"No value inserted, falling back to default: {self.t_relax} s")
        else:
            print(f"Recovery time: {relax_time} s")
            # Store the input in the instance variable
            self.t_relax = int(relax_time)

    def open_off_dialog_event(self):
        dialog_off = customtkinter.CTkInputDialog(
            text="Type in the forward biasing time (s):", title="Lights' Settings"
        )
        # Get the input from the dialog
        off_time = dialog_off.get_input()
        if off_time is None or not str(off_time).strip():
            print(f"No value inserted, falling back to default: {self.EL_time} s")
        else:
            print(f"Forward bias time: {off_time} s")
            # Store the input in the instance variable
            self.EL_time = int(off_time)

    ###### camera settings functions ######

    def open_exposure_dialog_event(self):
        dialog_exposure = customtkinter.CTkInputDialog(
            text="Type in the exposure time for EL and PL_oc (ms):",
            title="Camera Settings",
        )
        # Get the input from the dialog
        exposure = dialog_exposure.get_input()
        if exposure is None or not str(exposure).strip():
            print(f"No value inserted, falling back to default: {self.exp_time} s")
        else:
            print(f"Exposure time: {exposure} ms")
            # Store the input in the instance variable
            self.exp_time = int(exposure)

    def open_sc_exposure_dialog_event(self):
        dialog_exposure = customtkinter.CTkInputDialog(
            text="Type in the exposure time for PL_sc (ms):", title="Camera Settings"
        )
        # Get the input from the dialog
        exposure = dialog_exposure.get_input()
        if exposure is None or not str(exposure).strip():
            print(f"No value inserted, falling back to default: {self.exp_time_sc} s")
        else:
            print(f"SC exposure time: {exposure} ms")
            # Store the input in the instance variable
            self.exp_time_sc = int(exposure)

    ###### batch settings functions ######

    def open_channels_dialog_event(self):
        dialog_channels = customtkinter.CTkInputDialog(
            text="Type in the number of active channels:",
            title="Batch Settings",
        )
        # Get the input from the dialog
        active_channels = dialog_channels.get_input()
        if active_channels is None or not str(active_channels).strip():
            print(f"No value inserted, falling back to default: {self.active_channels}")
        else:
            print(f"Active channels: {active_channels}")
            # Store the input in the instance variable
            self.active_channels = int(active_channels)

    def open_ELvoltage_dialog_event(self):
        dialog_voltage = customtkinter.CTkInputDialog(
            text="Type in the fixed voltage (V) for EL:",
            title="Batch Settings",
        )
        # Get the input from the dialog
        EL_voltage = dialog_voltage.get_input()
        if EL_voltage is None or not str(EL_voltage).strip():
            print(f"No value inserted, falling back to default: {self.EL_voltage}")
        else:
            print(f"EL Voltage: {EL_voltage}")
            # Store the input in the instance variable
            self.EL_voltage = float(EL_voltage)

    def open_area_dialog_event(self):
        dialog_area = customtkinter.CTkInputDialog(
            text="Type in the cell's active area (cm2):",
            title="Batch Settings",
        )
        # Get the input from the dialog
        cell_area = dialog_area.get_input()
        if cell_area is None or not str(cell_area).strip():
            print(f"No value inserted, falling back to default: {self.cell_area}")
        else:
            print(f"Cell Area: {cell_area} cm2")
            # Store the input in the instance variable
            self.cell_area = float(cell_area)

    def open_inverted_dialog_event(self):
        dialog_inverted = customtkinter.CTkInputDialog(
            text="Type `True` if the cell is inverted, `False` otherwise:",
            title="Batch Settings",
        )
        # Get the input from the dialog
        cell_inverted = dialog_inverted.get_input()
        print(f"Cell inverted: {cell_inverted}")
        # Store the input in the instance variable
        if cell_inverted == "True":
            self.cell_inverted = True
        elif cell_inverted == "False":
            self.cell_inverted = False
        else:
            print("Input must be one of ['True', 'False']")

    def open_name_dialog_event(self):
        dialog_name = customtkinter.CTkInputDialog(
            text="Type in the batch reference (prev. " + self.batch_name + "):",
            title="Batch Settings",
        )
        # Get the input from the dialog
        batch_name = dialog_name.get_input()
        if batch_name is None or not str(batch_name).strip():
            print(f"No value inserted, falling back to default: {self.batch_name}")
        else:
            print(f"Batch name: {batch_name}")
            # Store the input in the instance variable
            self.batch_name = batch_name

    def open_res_dialog_event(self):
        dialog_res = customtkinter.CTkInputDialog(
            text=f"Type in the researcher's name (prev. {self.res_name}):",
            title="Batch Settings",
        )
        # Get the input from the dialog
        res_name = dialog_res.get_input()
        if res_name is None or not str(res_name).strip():
            print(f"No value inserted, falling back to default: {self.res_name}")
        else:
            print("Researcher's name: " + res_name)
            # Store the input in the instance variable
            self.res_name = res_name

    ###### cycle settings functions ######

    def open_iter_time_dialog_event(self):
        dialog_iter = customtkinter.CTkInputDialog(
            text="Type in the iteration interval (s):", title="Cycle Settings"
        )
        iter_time = dialog_iter.get_input()
        if iter_time is None or not str(iter_time).strip():
            print(f"No value inserted, falling back to default: {self.iter_time} s")
        else:
            print(f"Time between iterations: {iter_time}")
            self.iter_time = int(iter_time)

    def open_strategy_dialog_event(self):
        dialog_strategy = customtkinter.CTkInputDialog(
            text=f"Type in the sampling strategy to adopt (can be one of ['linear', 'decreasing']) (prev. {self.sampling_strategy}):",
            title="Cycle Settings",
        )
        # Get the input from the dialog
        acq_strategy = dialog_strategy.get_input()
        if acq_strategy is None or not str(acq_strategy).strip():
            print(
                f"No value inserted, falling back to default: {self.sampling_strategy}"
            )
        else:
            print(f"Acquisition strategy: {acq_strategy}")
            # Store the input in the instance variable
            self.sampling_strategy = acq_strategy
        if self.sampling_strategy == "decreasing":
            self.generate_schedule()

    def open_max_iter_dialog_event(self):
        dialog_iter = customtkinter.CTkInputDialog(
            text="Type in the maximum number of iterations for one cycle:",
            title="Cycle Settings",
        )
        max_iter = dialog_iter.get_input()
        if max_iter is None or not str(max_iter).strip():
            print(f"No value inserted, falling back to default: {self.max_iter}")
        else:
            print(f"Maximum number of iterations: {max_iter}")
            self.max_iter = int(max_iter)

    # command line function

    def process_command(self, event):
        # Get the text from the entry widget
        user_input = self.command_line.get().strip().lower()

        if user_input == "summary":
            self.open_summary_window()

        elif user_input == "dark":
            self.cmd_dark()

        elif user_input == "run":
            total_time = (
                self.JV_time
                + self.t_recover
                + self.PL_time * 2
                + self.t_relax
                + self.EL_time
                + 10
            )
            if total_time > self.iter_time:
                print(
                    "The sum of individual times input cannot be larger than the iteration duration!"
                )
                print(
                    f"With these dauration, iterations should be of at least {total_time}."
                )
                return
            self.user_input = user_input
            self.run_thread = threading.Thread(target=self.process_run)
            self.run_thread.start()

        elif user_input == "cycle":
            total_time = (
                self.JV_time
                + self.t_recover
                + self.PL_time * 2
                + self.t_relax
                + self.EL_time
                + 10
            )
            if total_time > self.iter_time:
                print(
                    "The sum of individual times input cannot be larger than the iteration duration!"
                )
                print(
                    f"With these dauration, iterations should be of at least {total_time}."
                )
                return
            self.user_input = user_input
            if not self.cycle_running:
                self.cycle_counter = 0
                self.cycle_running = True
                self.cycle_thread = threading.Thread(
                    target=self.cycle_process, daemon=False
                )
                self.cycle_thread.start()
                print("\n Cycle started.")
            else:
                print("\n Cycle is already running.")

        elif user_input == "stop":
            if self.cycle_running:
                self.cycle_running = False
                print("\n Cycle stopped.")
            else:
                print("\n No cycle to stop.")

        else:
            print(
                "\n Command not valid. Enter one of 'summary', 'dark', 'run', 'cycle', or 'stop'."
            )

        # Optionally clear the entry field after processing the command
        self.command_line.delete(0, customtkinter.END)

    def update_gui(self, message):
        self.after(0, lambda: print(message))

    def stop_cycle(self):
        if self.run_thread is not None and self.run_thread.is_alive():
            self.cycle_running = False
            self.run_thread.join()
            self.update_gui("Cycle stopped")

    def open_summary_window(self):
        summary_window = customtkinter.CTkToplevel(self)
        summary_window.title("Summary")
        summary_window.geometry("450x400")

        summary_label = customtkinter.CTkLabel(
            summary_window,
            text="Settings summary. \n"
            + f"- Maximum number of iterations: {int(self.max_iter)}"
            + f"- Interval between iterations: {str(self.iter_time)} s \n"
            + f"- Number of active channels: {str(self.active_channels)} \n"
            + f"- White LED's ON time: {str(self.JV_time)} s \n"
            + f"- Recovery time between white and blue light soaking: {str(self.t_recover)} s \n"
            + f"- Blue LEDs' ON time: {str(self.PL_time)} s \n"
            + f"- Recovery time between blue light soaking and electrical biasing: {str(self.t_relax)} s \n"
            + f"- EL biasing time: {str(self.EL_time)} s \n"
            + f"- Camera exposure time (OC): {str(self.exp_time)} ms \n"
            + f"- Camera exposure time (SC): {str(self.exp_time_sc)} ms \n"
            + f"- EL Voltage: {str(self.EL_voltage)} V \n"
            + f"- Cell area: {str(self.cell_area)} cm2 \n"
            + f"- Cell inverted: {str(self.cell_inverted)} \n"
            + f"- Batch name: {self.batch_name} \n"
            + f"- Researcher's name: {self.res_name} \n"
            + f"- Acquisition strategy: {self.sampling_strategy} \n",
            justify="left",
        )
        summary_label.grid(row=0, column=0, padx=20, pady=(20, 10))

    def log_metadata(self, image_type):
        log_filename = "acquisition_log.txt"

        # Get current time
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare metadata
        if image_type in ["PL", "pl"]:
            log_entry = f"{self.current_time} - {image_type} Image: {self.res_name}, Cell {self.batch_name}, Soaking: {self.PL_time} s, Exposure (OC): {self.exp_time} ms, Exposure (SC): {self.exp_time_sc} ms \n"
        else:
            log_entry = f"{self.current_time} - {image_type} Image: {self.res_name}, Cell {self.batch_name}, Biasing: {self.EL_time} s, Exposure: {self.exp_time} ms \n"

        with open(
            self.base_dir / self.res_name / self.current_date / log_filename, "a"
        ) as log_file:
            log_file.write(log_entry)

    def process_run(self, acquire: bool = True):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO.setup(self.GPIO_PIN_BLUE, GPIO.OUT)
        GPIO.output(self.GPIO_PIN_BLUE, GPIO.LOW)
        GPIO.setup(self.GPIO_PIN_WHITE, GPIO.OUT)
        GPIO.output(self.GPIO_PIN_WHITE, GPIO.LOW)

        self.ensure_api_connection()

        if self.cycle_counter == 0:
            for channel_id in range(self.active_channels):
                print(f"[Channel {channel_id}] Setting active channel")
                self.api.set_active_channel(channel_id)

                settings_str = self.api.get_channel_settings()
                if not settings_str:
                    print(
                        f"[Channel {channel_id}] API offline; skipping initial settings."
                    )
                    continue
                try:
                    data = json.loads(settings_str)
                except Exception:
                    print(f"[Channel {channel_id}] Bad settings JSON; skipping.")
                    continue

                # Modify settings...
                data["Enable"] = True
                data["User"] = str(self.res_name)
                data["Device"] = str(self.batch_name)
                data["Channel"]["Inverted"] = self.cell_inverted
                data["Tracking"]["Algorithm"] = "Open circuit"
                # data["Tracking"]["ConstantOutput"] = float(self.EL_voltage)
                data["Tracking"]["jvInterval"] = {
                    "Value": self.iter_time,
                    "Unit": "sec",
                }
                data["Tracking"]["TestDuration"] = {
                    "Value": self.max_iter * self.iter_time,
                    "Unit": "sec",
                }
                data["Cell"]["Area (cm2)"] = float(self.cell_area)

                self.api.set_channel_settings(json.dumps(data))
                print(f"[Channel {channel_id}] Settings updated.")

        try:
            # ==== WHITE LED (JV) ====
            if int(self.JV_time) > 0:
                run_JV(
                    self.api,
                    self.active_channels,
                    self.JV_time,
                    self.cycle_counter,
                    self.GPIO_PIN_WHITE,
                )
                print(f"\n[{self.cycle_counter}] White LED OFF.")

            # ==== BLUE LED (PL) ====
            if int(self.PL_time) > 0:
                self.current_date = datetime.datetime.now().date().strftime("%Y-%m-%d")
                output_dir = (
                    f"{str(self.base_dir)}/{self.res_name}/{self.current_date}/PL"
                )
                self.PL_path = output_dir

                batch_name = (
                    f"{self.batch_name}_{self.cycle_counter}"
                    if self.cycle_running
                    else self.batch_name
                )

                try:
                    run_PL(
                        self.api,
                        self.active_channels,
                        self.t_recover,
                        self.PL_time,
                        self.exp_time,
                        self.exp_time_sc,
                        self.PL_path,
                        batch_name,
                        acquire,
                        self.USE_CAMERA,
                        self.GPIO_PIN_BLUE,
                    )
                    self.log_metadata(image_type="PL")
                except Exception as e:
                    print(f"\n[{self.cycle_counter}]: {e}")

                print(f"\n[{self.cycle_counter}] Blue LEDs OFF.")

            # ==== EL Bias (LEDs OFF) ====
            if int(self.EL_time) > 0:
                self.current_date = datetime.datetime.now().date().strftime("%Y-%m-%d")
                output_dir = (
                    f"{str(self.base_dir)}/{self.res_name}/{self.current_date}/EL"
                )
                self.EL_path = output_dir

                batch_name = (
                    f"{self.batch_name}_{self.cycle_counter}"
                    if self.cycle_running
                    else self.batch_name
                )

                try:
                    run_EL(
                        self.api,
                        self.active_channels,
                        self.t_relax,
                        self.EL_time,
                        self.exp_time,
                        self.EL_path,
                        batch_name,
                        acquire,
                        self.USE_CAMERA,
                        self.EL_voltage,
                    )
                    self.log_metadata(image_type="EL")
                except Exception as e:
                    print(f"\n[{self.cycle_counter}]: {e}")

                print(f"\n[{self.cycle_counter}] EL bias OFF.")

        finally:
            if self.user_input == "run":
                for ch in range(self.active_channels):
                    self.api.set_active_channel(ch)
                    self.api.stop_channel()
            GPIO.cleanup()
            print(f"\n[{self.cycle_counter}] GPIO cleanup complete.")

    def cycle_process(self):
        self.ensure_api_connection()

        initial_start = time.time()
        try:
            while self.cycle_running and self.cycle_counter < self.max_iter:
                scheduled_start = initial_start + self.cycle_counter * self.iter_time
                now = time.time()

                # If we’re ahead of schedule, wait until the scheduled time
                if now < scheduled_start:
                    wait_time = scheduled_start - now
                    print(f"\nStarting new iteration in {wait_time:.2f} s.")
                    time.sleep(wait_time)

                # Decide if we should acquire based on strategy
                acquire = False
                if self.sampling_strategy == "linear":
                    acquire = True
                elif self.sampling_strategy == "decreasing":
                    acquire = self.cycle_counter in self.decreasing_schedule

                print(
                    f"\nStarting iteration {self.cycle_counter} (Acquire = {acquire})..."
                )
                self.process_run(acquire=acquire)

                self.cycle_counter += 1

            if self.cycle_counter < self.max_iter:
                for ch in range(self.active_channels):
                    self.api.set_active_channel(ch)
                    self.api.stop_channel()
                self.api.disconnect()
                self.update_gui(f"Cycle stopped at iteration {self.cycle_counter}")
            else:
                for ch in range(self.active_channels):
                    self.api.set_active_channel(ch)
                    self.api.stop_channel()
                self.api.disconnect()
                self.update_gui(
                    f"Cycle stopped: reached max iterations ({self.max_iter})."
                )

        finally:
            GPIO.cleanup()
            self.cycle_running = False
            print("\nCycle process exited.")

    def generate_schedule(self, max_iter: int = None):
        if max_iter is None:
            max_iter = self.max_iter

        self.decreasing_schedule = set()
        # Acquire more often early, then gradually spread out
        step = 1
        i = 0
        while i <= max_iter:
            self.decreasing_schedule.add(i)
            if i < 100:
                step = 1
            elif i < 200:
                step = 2
            else:
                step = 5
            i += step

    def _acquire_reference_series(self, kind: str, batch: str):
        """
        Acquire a reference image series in the date folder for both PL exposures:
        - kind='dark' or 'flat'
        Saves as: <date>/<kind>_PLoc_<exp>ms.tiff and <kind>_PLsc_<exp>ms.tiffthor
        """
        date_root = self._date_root()
        os.makedirs(date_root, exist_ok=True)

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # uses self.exp_time
        if int(self.exp_time) > 0:
            name = f"{batch}_{kind}_{int(self.exp_time)}ms"
            try:
                GPIO.setup(self.GPIO_PIN_WHITE, GPIO.OUT)
                if kind == "flat":
                    GPIO.output(self.GPIO_PIN_WHITE, GPIO.HIGH)
                acquisition_PL(int(self.exp_time), name, date_root)
                print(f"[{kind}] saved: {os.path.join(date_root, name + '.tiff')}")
                if kind == "flat":
                    GPIO.cleanup()
            except Exception as e:
                print(f"[{kind}] {int(self.exp_time)} ms failed ({e})")

        # uses self.exp_time_sc
        if int(self.exp_time_sc) > 0:
            name = f"{kind}_{int(self.exp_time_sc)}ms"
            try:
                GPIO.setup(self.GPIO_PIN_WHITE, GPIO.OUT)
                acquisition_PL(int(self.exp_time_sc), name, date_root)
                if kind == "flat":
                    GPIO.output(self.GPIO_PIN_WHITE, GPIO.HIGH)
                print(f"[{kind}] saved: {os.path.join(date_root, name + '.tiff')}")
                if kind == "flat":
                    GPIO.cleanup()
            except Exception as e:
                print(f"[{kind}] {int(self.exp_time_sc)} ms failed ({e})")

    def cmd_dark(self):
        """Acquire dark images at both exposures into the date folder."""
        if not self.USE_CAMERA:
            print("[dark] Camera disabled; skipped.")
            return
        print("[dark] Acquiring dark references...")
        self._acquire_reference_series("dark", self.batch_name)

    def cmd_flat(self):
        """
        Acquire flat-field images at both exposures into the date folder.
        Make sure your uniform illuminator/diffuser is in place before running.
        """
        if not self.USE_CAMERA:
            print("[flat] Camera disabled; skipped.")
            return
        print("[flat] Acquiring flat-field references...")
        self._acquire_reference_series("flat", self.batch_name)

    # Functions to handle Arkeo API
    def ensure_api_connection(self):
        if not self.api.connection:
            try:
                self.api.connect()
                print("API connected successfully.")
            except Exception as e:
                print(f"Failed to connect to API: {e}")

    # Function to ensure workers stop
    def _on_close(self):
        # ensure worker stops
        self.cycle_running = False
        if self.run_thread and self.run_thread.is_alive():
            self.run_thread.join(timeout=3)
        # switch off LEDs you use (fill the tuple)
        gpio_safe_cleanup(pins_off=(self.GPIO_PIN_WHITE, self.GPIO_PIN_BLUE))

        try:
            self._hb_stop.set()
        except Exception:
            pass
        try:
            os.remove(HEARTBEAT)  # optional; harmless if it fails
        except Exception:
            pass
        self.destroy()


################################ main ################################


if __name__ == "__main__":
    _lock_fd = _acquire_single_instance_lock()
    app = App()
    app.mainloop()
