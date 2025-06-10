import datetime
import glob
import os
import sys
import threading
import time
import tkinter
import tkinter.messagebox
from pathlib import Path
from tkinter import END, Listbox, Scrollbar, Text

import customtkinter
import numpy as np
import tifffile
from PIL import Image, ImageTk
from RPi import GPIO
from tqdm import tqdm

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path

    configure_path()
except ImportError:
    configure_path = None

from camera import acquisition_EL, acquisition_PL

customtkinter.set_appearance_mode(
    "Light"
)  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme(
    "blue"
)  # Themes: "blue" (standard), "green", "dark-blue"


class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.text_widget.insert(END, message)
        self.text_widget.see(END)

    def flush(self):
        pass


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.base_dir = Path("/home/chose/Documenti")
        self.folder_path = Path("/home/chose/Documenti")
        self.cycle_counter = 0
        self.cycle_running = False
        self.run_thread = None

        # attribute initialization
        self.PL_time: int = 0
        self.EL_time: int = 0
        self.JV_time: int = 0
        self.exposure_time: int = 0
        self.batch_name: str = "batch"
        self.res_name: str = "giulio"
        self.cycle_time_100: int = 180
        self.cycle_time_200: int = 300
        self.cycle_time_500: int = 600
        self.max_iter: int = 500

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
            text="EL/PL/IV bench",
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
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

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

        # lights settings
        self.lights_settings_label = customtkinter.CTkLabel(
            self.tabview.tab("Lights"),
            text="Lights settings",
            font=customtkinter.CTkFont(size=15),
        )
        self.lights_settings_label.grid(row=1, column=0, padx=20, pady=(5, 0))

        # Initialize the blue lights' time variable
        self.blue_input_button = customtkinter.CTkButton(
            self.tabview.tab("Lights"),
            text="Blue LEDs' soaking time (s)",
            command=self.open_blue_dialog_event,
        )
        self.blue_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        # Initialize the lights' off time variable
        self.off_input_button = customtkinter.CTkButton(
            self.tabview.tab("Lights"),
            text="LEDs' off time (s)",
            command=self.open_off_dialog_event,
        )
        self.off_input_button.grid(row=3, column=0, padx=20, pady=(10, 10))

        # Initialize the white lights' time variable
        self.white_input_button = customtkinter.CTkButton(
            self.tabview.tab("Lights"),
            text="White LED's soaking time (s)",
            command=self.open_white_dialog_event,
        )
        self.white_input_button.grid(row=4, column=0, padx=20, pady=(10, 10))

        # camera settings
        self.camera_settings_label = customtkinter.CTkLabel(
            self.tabview.tab("Camera"),
            text="Camera settings",
            font=customtkinter.CTkFont(size=15),
        )
        self.camera_settings_label.grid(row=1, column=0, padx=20, pady=(5, 0))

        # Initialize the exposure time variable
        self.exposure_input_button = customtkinter.CTkButton(
            self.tabview.tab("Camera"),
            text="Exposure time (us)",
            command=self.open_exposure_dialog_event,
        )
        self.exposure_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        # cycle settings
        self.cycle_settings_label = customtkinter.CTkLabel(
            self.tabview.tab("Cycle"),
            text="Cycle settings",
            font=customtkinter.CTkFont(size=15),
        )
        self.cycle_settings_label.grid(row=1, column=0, padx=20, pady=(5, 0))

        # initialize the cycle time variables
        self.cycle_100_input_button = customtkinter.CTkButton(
            self.tabview.tab("Cycle"),
            text="Cycle duration (iters. 0-100) (s)",
            command=self.open_cycle_100_dialog_event,
        )
        self.cycle_100_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        self.cycle_200_input_button = customtkinter.CTkButton(
            self.tabview.tab("Cycle"),
            text="Cycle duration (iters. 101-200) (s)",
            command=self.open_cycle_200_dialog_event,
        )
        self.cycle_200_input_button.grid(row=3, column=0, padx=20, pady=(10, 10))

        self.cycle_500_input_button = customtkinter.CTkButton(
            self.tabview.tab("Cycle"),
            text="Cycle duration (iters. 201-500) (s)",
            command=self.open_cycle_500_dialog_event,
        )
        self.cycle_500_input_button.grid(row=4, column=0, padx=20, pady=(10, 10))

        # batch settings
        self.batch_settings_label = customtkinter.CTkLabel(
            self.tabview.tab("Batch"),
            text="Batch settings",
            font=customtkinter.CTkFont(size=15),
        )
        self.batch_settings_label.grid(row=1, column=0, padx=20, pady=(5, 0))

        # Initialize the batch name variable
        self.batch_input_button = customtkinter.CTkButton(
            self.tabview.tab("Batch"),
            text="Batch name",
            command=self.open_name_dialog_event,
        )
        self.batch_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        # Initialize the researcher name variable
        self.res_input_button = customtkinter.CTkButton(
            self.tabview.tab("Batch"),
            text="Researcher's name",
            command=self.open_res_dialog_event,
        )
        self.res_input_button.grid(row=3, column=0, padx=20, pady=(10, 10))

        ################################ column 2 ################################

        # Latest image display (Top section)
        self.image_label = customtkinter.CTkLabel(self, text="Latest image")
        self.image_label.grid(
            row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew"
        )

        self.update_latest_image_periodically()

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
            "This is the user guide to this GUI, developed to have easy access to the PL/EL measurement bench. \n",
        )
        self.textbox.insert(
            "4.0",
            "Before running the tests, make sure that the camera USB cable is plugged in, that the power source is on at 24 V and the current knob is turned all the way to the max.  \n \n",
        )

        self.textbox.insert(
            "6.0",
            "As of now, the system is only capable of performing PL measurements. In order to do so, go through the Tabs in the Box below, click on the buttons, and enter the desired values. \n",
        )
        self.textbox.insert(
            "7.0",
            "- LIGHTS: Since the EL and JV measurements cannot be performed, please insert 0 in the 'LED's off time' and 'White LED's soaking time'. \n",
        )
        self.textbox.insert(
            "8.0",
            "- CAMERA: Enter the exposure time in microseconds (e.g. entering 200000 will result in an exposure time of 0.2 seconds). Please notice: if you enter 0 as the exposure time, the program will still run, but no image will be captured. \n",
        )
        self.textbox.insert(
            "9.0",
            "- CYCLE: Enter the desired time interval between consecutive measurements when running in cyclic mode. Supports up to 3 different intervals for iterations (0-100), (101-200), (201-500). \n",
        )
        self.textbox.insert(
            "10.0",
            "- BATCH: Enter the reference of the cell/batch, the name of the researcher performing the measurements, and the desired time interval between consecutive when measurements when running in cyclic mode. The acquired image(s) will be saved as '<reference>.tiff' in the folder '/home/chose/Documenti/<researcher>/<date>/PL'. \n",
        )
        self.textbox.insert(
            "12.0",
            "The command line at the bottom of the GUI allows running the measurement programs. It accepts 4 commands: \n",
        )
        self.textbox.insert(
            "13.0",
            "- summary: shows the values stored by the program, always check them before running. \n",
        )
        self.textbox.insert(
            "14.0", "- run: performs one single measurement and acquires an image. \n"
        )
        self.textbox.insert(
            "15.0",
            "- cycle: runs in cyclic mode, acquiring one image per iteration, until stopped. The time between iterations is user-defined. \n",
        )
        self.textbox.insert(
            "16.0",
            "- stop: stops the running cycles. Always use this and only this command to stop cycles. \n",
        )
        self.textbox.insert(
            "18.0",
            "If the program returns an error, and the LEDs remain ON, please CLOSE the application and read the README file on the desktop. You will find what to do in lines 51-56. \n \n",
        )

        self.textbox.insert(
            "20.0",
            "For more information, or if you have doubts, ask Giulio Barletta \n<giulio.barletta@polito.it> \nor Simon Ternes \n<ternes@ing.uniroma2.it>.",
        )
        self.textbox.configure(state="disabled")

    ################################ function definitions ################################

    # appearance functions

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    # lights settings functions

    def open_blue_dialog_event(self):
        dialog_blue = customtkinter.CTkInputDialog(
            text="Type in the blue LEDs' ON time (s):", title="Lights' Settings"
        )
        # Get the input from the dialog
        blue_time = dialog_blue.get_input()
        print("Blue LEDs' ON time (s): " + blue_time)

        # Store the input in the instance variable
        self.PL_time = blue_time

    def open_off_dialog_event(self):
        dialog_off = customtkinter.CTkInputDialog(
            text="Type in the LEDs' OFF time (s):", title="Lights' Settings"
        )
        # Get the input from the dialog
        off_time = dialog_off.get_input()
        print("LEDs' OFF time (s): " + off_time)

        # Store the input in the instance variable
        self.EL_time = off_time

    def open_white_dialog_event(self):
        dialog_white = customtkinter.CTkInputDialog(
            text="Type in the white LED's ON time (s):", title="Lights' Settings"
        )
        # Get the input from the dialog
        white_time = dialog_white.get_input()
        print("White LEDs' ON time (s): " + white_time)

        # Store the input in the instance variable
        self.JV_time = white_time

    # camera settings functions

    def open_exposure_dialog_event(self):
        dialog_exposure = customtkinter.CTkInputDialog(
            text="Type in the exposure time (us):", title="Camera Settings"
        )
        # Get the input from the dialog
        exposure = dialog_exposure.get_input()
        print("Exposure time (us): " + exposure)

        # Store the input in the instance variable
        self.exposure_time = exposure

    # batch settings functions

    def open_name_dialog_event(self):
        dialog_name = customtkinter.CTkInputDialog(
            text="Type in the batch reference (prev. " + self.batch_name + "):",
            title="Batch Settings",
        )
        # Get the input from the dialog
        batch_name = dialog_name.get_input()
        print("Batch name: " + batch_name)

        # Store the input in the instance variable
        self.batch_name = batch_name

    def open_res_dialog_event(self):
        dialog_name = customtkinter.CTkInputDialog(
            text="Type in the researcher's name (prev. " + self.res_name + "):",
            title="Batch Settings",
        )
        # Get the input from the dialog
        res_name = dialog_name.get_input()
        print("Researcher's name: " + res_name)

        # Store the input in the instance variable
        self.res_name = res_name

    # cycle settings functions

    def open_cycle_100_dialog_event(self):
        dialog_name = customtkinter.CTkInputDialog(
            text="Type in the time between successive measurements (iters. 0-100) (s) (prev. "
            + str(self.cycle_time_100)
            + "):",
            title="Cycle Settings",
        )
        # Get the input from the dialog
        cycle_time = dialog_name.get_input()
        print("Time between measurements (s): " + cycle_time)

        # Store the input in the instance variable
        self.cycle_time_100 = cycle_time

    def open_cycle_200_dialog_event(self):
        dialog_name = customtkinter.CTkInputDialog(
            text="Type in the time between successive measurements (iters. 101-200) (s) (prev. "
            + str(self.cycle_time_200)
            + "):",
            title="Cycle Settings",
        )
        # Get the input from the dialog
        cycle_time = dialog_name.get_input()
        print("Time between measurements (s): " + cycle_time)

        # Store the input in the instance variable
        self.cycle_time_200 = cycle_time

    def open_cycle_500_dialog_event(self):
        dialog_name = customtkinter.CTkInputDialog(
            text="Type in the time between successive measurements (iters. 201-500) (s) (prev. "
            + str(self.cycle_time_500)
            + "):",
            title="Cycle Settings",
        )
        # Get the input from the dialog
        cycle_time = dialog_name.get_input()
        print("Time between measurements (s): " + cycle_time)

        # Store the input in the instance variable
        self.cycle_time_500 = cycle_time

    # command line function

    def process_command(self, event):
        # Get the text from the entry widget
        user_input = self.command_line.get().strip().lower()

        if user_input == "summary":
            self.open_summary_window()

        elif user_input == "run":
            self.run_thread = threading.Thread(target=self.process_run)
            self.run_thread.start()

        elif user_input == "cycle":
            if not self.cycle_running:
                self.cycle_running = True
                self.cycle_thread = threading.Thread(target=self.cycle_process)
                self.cycle_thread.start()
                print("\n Cycle started.")
            else:
                print("\n Cycle is already running.")

        elif user_input == "stop":
            if self.cycle_running:
                self.cycle_running = False
                self.cycle_counter = 0
                print("\n Cycle stopped.")
            else:
                print("\n No cycle to stop.")

        else:
            print(
                "\n Command not valid. Enter one of 'summary', 'run', 'cycle', or 'stop'."
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
        summary_window.geometry("300x200")

        summary_label = customtkinter.CTkLabel(
            summary_window,
            text="Settings summary.\n"
            + f"- Blue LEDs' ON time: {self.PL_time} s\n"
            + f"- EL biasing time: {self.EL_time} s\n"
            + f"- White LED's ON time: {self.JV_time} s\n"
            + f"- Camera exposure time: {self.exposure_time} us\n"
            + f"- Batch name: {self.batch_name} \n"
            + f"- Researcher's name: {self.res_name} \n"
            + f"- Time between measurements (0-100) (s): {self.cycle_time_100} \n"
            + f"- Time between measurements (101-200) (s): {self.cycle_time_200} \n"
            + f"- Time between measurements (201-500) (s): {self.cycle_time_500} \n",
            justify="left",
        )
        summary_label.grid(row=0, column=0, padx=20, pady=(20, 10))

    def log_metadata(self, image_type):
        log_filename = "acquisition_log.txt"

        # Get current time
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare metadata
        if image_type in ["PL", "pl"]:
            log_entry = f"{self.current_time} - {image_type} Image: {self.res_name}, Cell {self.batch_name}, Soaking: {self.PL_time} s, Exposure: {self.exposure_time} us \n"
        else:
            log_entry = f"{self.current_time} - {image_type} Image: {self.res_name}, Cell {self.batch_name}, Biasing: {self.EL_time} s, Exposure: {self.exposure_time} us \n"

        with open(
            self.base_dir / self.res_name / self.current_date / log_filename, "a"
        ) as log_file:
            log_file.write(log_entry)

    def process_run(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO_PIN_BLUE = 2
        GPIO_PIN_WHITE = 21

        GPIO.setup(GPIO_PIN_BLUE, GPIO.OUT)
        GPIO.output(GPIO_PIN_BLUE, GPIO.LOW)
        GPIO.setup(GPIO_PIN_WHITE, GPIO.OUT)
        GPIO.output(GPIO_PIN_WHITE, GPIO.LOW)

        try:
            # start PL bias
            if int(self.PL_time) > 0:
                GPIO.output(GPIO_PIN_BLUE, GPIO.HIGH)
                print("\n Blue light soaking...")
                for i in tqdm(range(int(self.PL_time)), desc="Blue light soaking..."):
                    time.sleep(1)

                self.current_date = datetime.datetime.now().date().strftime("%Y-%m-%d")
                output_dir = (
                    f"{str(self.base_dir)}/{self.res_name}/{self.current_date}/PL/"
                )
                os.makedirs(output_dir, exist_ok=True)
                self.folder_path = output_dir

                if self.cycle_running:
                    unique_batch_name = f"{self.batch_name}_{self.cycle_counter}"
                else:
                    unique_batch_name = f"{self.batch_name}"

                # PL image acquisition
                if self.USE_CAMERA is True and int(self.exposure_time) != 0:
                    try:
                        acquisition_PL(
                            int(self.exposure_time), unique_batch_name, output_dir
                        )
                        self.log_metadata(image_type="PL")

                    except Exception as e:
                        print(f"\n Error during PL acquisition: {e}")

                # stop PL bias
                GPIO.output(GPIO_PIN_BLUE, GPIO.LOW)
                print("\n Blue LEDs are OFF.")

            """
            # start EL bias
            if int(self.EL_time) > 0:
                print("\n Applying electric bias...")
                for i in tqdm(range(int(self.EL_time)), desc = "EL bias is ON."):
                    time.sleep(1)

                self.current_date = datetime.datetime.now().date().strftime("%Y-%m-%d")
                output_dir = f"{str(self.base_dir)}/{self.res_name}/{self.current_date}/EL/"
                os.makedirs(output_dir, exist_ok=True)
                self.folder_path = output_dir

                if self.cycle_running:
                    unique_batch_name = f"{self.batch_name}_{self.cycle_counter}"
                    self.cycle_counter += 1
                else:
                    unique_batch_name = f"{self.batch_name}"


                # PL image acquisition
                if self.USE_CAMERA == True and int(self.exposure_time) != 0:
                    try:
                        acquisition_PL(int(self.exposure_time), unique_batch_name, output_dir)
                        self.log_metadata(image_type="EL")

                    except Exception as e:
                        print(f"\n Error during PL acquisition: {e}")

                # stop EL bias
                print("\n EL bias is OFF.")
            """

            # White light soaking for IV curve
            if int(self.JV_time) > 0:
                GPIO.output(GPIO_PIN_WHITE, GPIO.HIGH)
                print("\n White light soaking...")
                for i in tqdm(range(int(self.JV_time)), desc="White light soaking...."):
                    time.sleep(1)

                # stop white light soaking
                GPIO.output(GPIO_PIN_WHITE, GPIO.LOW)
                print("\n White LED is OFF.")

        finally:
            GPIO.cleanup()
            print("\n GPIO cleanup complete.")

    def cycle_process(self):
        try:
            while self.cycle_running and self.cycle_counter < self.max_iter:
                start_time = time.time()  # Record the cycle start time

                print(f"\n Starting measurement {self.cycle_counter}...")
                self.process_run()

                # Calculate time elapsed during process_run()
                elapsed_time = time.time() - start_time
                wait_time = self.get_cycle_time()
                remaining_time = max(int(wait_time) - elapsed_time, 0)

                self.cycle_counter += 1
                print(
                    f"\n Cycle {self.cycle_counter} completed. Waiting {int(remaining_time)} second(s) before next run."
                )

                update_interval = 10
                total_updates = int(remaining_time / update_interval)

                for _ in tqdm(range(total_updates), desc="Time until next run"):
                    if not self.cycle_running:
                        print("\n Cycle stopped during wait period.")
                        return
                    time.sleep(update_interval)

            if self.cycle_counter >= self.max_iter:
                self.update_gui(
                    f"Cycle stopped: reached max iterations ({self.max_iter})."
                )

        finally:
            GPIO.cleanup()
            self.cycle_running = False
            print("\n Cycle process exited.")

    def get_cycle_time(self):
        """Determines the waiting time based on the iteration count"""
        if self.cycle_counter <= 100:
            return self.cycle_time_100
        elif self.cycle_counter <= 200:
            return self.cycle_time_200
        else:
            return self.cycle_time_500

    # Functions to handle image display

    def get_latest_image(self):
        folder_path = Path(self.folder_path)
        if not Path(folder_path).is_dir():
            return None

        tiff_files = [f for f in folder_path.iterdir() if f.suffix.lower() == ".tiff"]
        if not tiff_files:
            return None

        latest_file = max(tiff_files, key=lambda f: f.stat().st_ctime)
        return latest_file

    def update_latest_image(self):
        latest_image_path = self.get_latest_image()
        if latest_image_path is None:
            self.image_label.configure(text="No images available")
            return

        try:
            img_16bit = Image.open(latest_image_path)
            img_array = np.asarray(img_16bit)

            img_array = (img_array / 256).astype("uint8")
            img_8bit = Image.fromarray(img_array, mode="L")

            display_image = customtkinter.CTkImage(
                light_image=img_8bit, dark_image=img_8bit, size=(196, 108)
            )

            self.image_label.configure(image=display_image, text="")
            self.image_label.image = display_image

        except Exception as e:
            print(f"Error loading image: {e}")

    def update_latest_image_periodically(self):
        self.update_latest_image()
        self.after(20000, self.update_latest_image_periodically)


################################ main ################################


if __name__ == "__main__":
    app = App()
    app.mainloop()
