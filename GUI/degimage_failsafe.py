"""
#!/usr/bin/env python3
"""

import os
import sys
import time

from RPi import GPIO

PINS = [2, 21]  # <- your LED/relay pins
HB = "/tmp/degimage.heartbeat"
TIMEOUT = 6  # seconds without heartbeat -> force LOW

GPIO.setmode(GPIO.BCM)
for p in PINS:
    GPIO.setup(p, GPIO.OUT)


def force_low():
    for p in PINS:
        try:
            GPIO.output(p, GPIO.LOW)
        except Exception:
            pass


last = 0
while True:
    try:
        stat = os.stat(HB)
        last = stat.st_mtime
    except FileNotFoundError:
        pass
    if time.time() - last > TIMEOUT:
        force_low()
    time.sleep(1)
