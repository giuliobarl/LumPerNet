import sys
from tkinter import END


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
