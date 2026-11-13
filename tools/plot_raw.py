#!/usr/bin/env python3
"""Live-plot raw EMG readings from src/raw.cpp and record them to a CSV in data/.

Each run writes a new file named after the time it started, such as
data/raw_2026-11-13_15-30-05.csv, so old recordings are never overwritten.

Usage:
    python3 tools/plot_raw.py
    python3 tools/plot_raw.py --port /dev/cu.usbmodem1101 --window 5
"""
import argparse
import collections
import csv
import datetime
import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial

from plot_emg import (COLOR_LINE, COLOR_SURFACE, COLOR_TEXT_MUTED, SampleReader,
                      find_port, open_port, style_axes)

SAMPLE_RATE_HZ = 1000  # must match SAMPLE_PERIOD_US in src/raw.cpp
Y_MIN_MV = 1000
Y_MAX_MV = 2000
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def new_csv_path(prefix="raw"):
    """Returns a path in data/ that doesn't exist yet."""
    DATA_DIR.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = DATA_DIR / f"{prefix}_{stamp}.csv"
    count = 2
    while path.exists():
        path = DATA_DIR / f"{prefix}_{stamp}_{count}.csv"
        count += 1
    return path


class RawPlot:
    def __init__(self, reader, window_s, writer):
        self.reader = reader
        self.writer = writer
        self.window_s = window_s
        self.count = 0
        size = window_s * SAMPLE_RATE_HZ
        self.times = collections.deque(maxlen=size)
        self.values = collections.deque(maxlen=size)

        self.fig, self.ax = plt.subplots(figsize=(12, 5), facecolor=COLOR_SURFACE)
        self.fig.canvas.manager.set_window_title("ReadyRep raw EMG")
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
        samples, _mdfs = self.reader.read()
        if not samples:
            return
        for value in samples:
            t = self.count / SAMPLE_RATE_HZ
            self.times.append(t)
            self.values.append(value)
            self.writer.writerow([f"{t:.3f}", value])
            self.count += 1

        self.line.set_data(self.times, self.values)
        end = max(self.times[-1], self.window_s)
        self.ax.set_xlim(end - self.window_s, end)
        self.status.set_text(f"Now {self.values[-1]} mV   Peak {max(self.values)} mV")


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

    path = new_csv_path()
    csv_file = open(path, "x", newline="")  # "x" refuses to replace an existing file
    writer = csv.writer(csv_file)
    writer.writerow(["time_s", "emg_mv"])

    print(f"Reading from {port}, recording to {path}. Close the plot window to stop.")
    plot = RawPlot(SampleReader(ser), args.window, writer)
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
