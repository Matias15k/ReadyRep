#!/usr/bin/env python3
"""Live-plot raw EMG from src/collect.cpp and record EMG and temperature to a CSV in data/.

Each run writes a new file named after the time it started, such as
data/emg_temp_2026-11-13_15-30-05.csv, so old recordings are never overwritten.

Usage:
    python3 tools/emg_temp.py
    python3 tools/emg_temp.py --port /dev/cu.usbmodem1101 --window 5
"""
import argparse
import collections
import csv
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial

from plot_emg import (COLOR_LINE, COLOR_SURFACE, COLOR_TEXT_MUTED, find_port, open_port,
                      style_axes)
from plot_raw import Y_MAX_MV, Y_MIN_MV, new_csv_path

SAMPLE_RATE_HZ = 1000  # must match SAMPLE_PERIOD_TICKS in src/collect.cpp
MAX_READS_PER_FRAME = 200  # stops the plot freezing if data somehow arrives faster than it's read
SAMPLE_LINE = re.compile(r"^emg_mv:(\d+),therm_mv:(\d+),temp_f:(-?\d+(?:\.\d+)?)$")


class CollectReader:
    """Turns raw serial bytes into (emg_mv, therm_mv, temp_f) readings, keeping partial lines."""

    def __init__(self, ser):
        self.ser = ser
        self.pending = b""

    def read(self):
        # Keep reading until the port is empty. macOS only buffers about 1 KB, so one read per
        # frame can't keep up with 1000 lines a second and the board gets held back.
        for _ in range(MAX_READS_PER_FRAME):
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if not chunk:
                break
            self.pending += chunk
        *lines, self.pending = self.pending.split(b"\n")
        readings = []
        for line in lines:
            match = SAMPLE_LINE.match(line.decode("ascii", "ignore").strip())
            if match:
                readings.append((int(match.group(1)), int(match.group(2)), float(match.group(3))))
        return readings


class EmgTempPlot:
    def __init__(self, reader, window_s, writer):
        self.reader = reader
        self.writer = writer
        self.window_s = window_s
        self.count = 0
        size = window_s * SAMPLE_RATE_HZ
        self.times = collections.deque(maxlen=size)
        self.values = collections.deque(maxlen=size)

        self.fig, self.ax = plt.subplots(figsize=(12, 5), facecolor=COLOR_SURFACE)
        self.fig.canvas.manager.set_window_title("ReadyRep EMG and temperature")
        ax = self.ax
        style_axes(ax, "EMG raw signal (RAW)", "Voltage (mV)")
        ax.set_xlim(0, window_s)
        ax.set_ylim(Y_MIN_MV, Y_MAX_MV)

        self.line, = ax.plot([], [], color=COLOR_LINE, linewidth=2,
                             solid_joinstyle="round", solid_capstyle="round")
        self.status = ax.text(1, 1.02, "Waiting for data...", transform=ax.transAxes,
                              ha="right", color=COLOR_TEXT_MUTED)
        self.fig.tight_layout()

    def update(self, _frame=None):
        readings = self.reader.read()
        if not readings:
            return
        for emg_mv, therm_mv, temp_f in readings:
            t = self.count / SAMPLE_RATE_HZ
            self.times.append(t)
            self.values.append(emg_mv)
            self.writer.writerow([f"{t:.3f}", emg_mv, therm_mv, f"{temp_f:.1f}"])
            self.count += 1

        self.line.set_data(self.times, self.values)
        end = max(self.times[-1], self.window_s)
        self.ax.set_xlim(end - self.window_s, end)
        self.status.set_text(f"EMG {self.values[-1]} mV   Peak {max(self.values)} mV   "
                             f"Temperature {temp_f:.1f} F")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", help="serial port (default: auto-detect the ESP)")
    parser.add_argument("--window", type=int, default=3,
                        help="seconds of data to show (default: 3)")
    args = parser.parse_args()

    port = args.port or find_port()
    try:
        ser = open_port(port)
    except serial.SerialException as err:
        sys.exit(f"Couldn't open {port}. Close any other serial monitor first. ({err})")

    path = new_csv_path("emg_temp")
    csv_file = open(path, "x", newline="")  # "x" refuses to replace an existing file
    writer = csv.writer(csv_file)
    writer.writerow(["time_s", "emg_mv", "therm_mv", "temp_f"])

    print(f"Reading from {port}, recording to {path}. Close the plot window to stop.")
    plot = EmgTempPlot(CollectReader(ser), args.window, writer)
    anim = FuncAnimation(plot.fig, plot.update, interval=33, cache_frame_data=False)
    try:
        plt.show()
    except serial.SerialException as err:
        print(f"Lost connection to the board: {err}")
    finally:
        del anim
        ser.close()
        csv_file.close()
        print(f"Saved {plot.count} samples to {path}")


if __name__ == "__main__":
    main()
