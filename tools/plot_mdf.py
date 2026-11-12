#!/usr/bin/env python3
"""Live-plot median frequency from the ReadyRep firmware while the button is held.

Each button hold starts a fresh graph. When the button is let go, the hold's
average is drawn as a horizontal line.

Usage:
    python3 tools/plot_mdf.py
    python3 tools/plot_mdf.py --port /dev/cu.usbmodem1101
    python3 tools/plot_mdf.py --save holds.csv
"""
import argparse
import csv
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial

from plot_emg import (COLOR_MDF_LINE, COLOR_SURFACE, COLOR_TEXT_MUTED,
                      find_port, open_port, style_axes)

Y_MAX_HZ = 350
MIN_WINDOWS_SHOWN = 10  # the x-axis grows past this if the hold runs longer

START_LINE = "Recording median frequency..."
WINDOW_LINE = re.compile(r"^Median frequency (\d+): (\d+(?:\.\d+)?) Hz")
AVERAGE_LINE = re.compile(r"^Average median frequency: (\d+(?:\.\d+)?) Hz \((\d+) windows\)")
TOO_SHORT_LINE = "Button released too soon"
TEMP_LINE = "Temperature: "  # only from ReadyRepBLE.cpp


class LineReader:
    """Turns raw serial bytes into complete text lines."""

    def __init__(self, ser):
        self.ser = ser
        self.pending = b""

    def read(self):
        self.pending += self.ser.read(self.ser.in_waiting or 1)
        *lines, self.pending = self.pending.split(b"\n")
        return [line.decode("ascii", "ignore").strip() for line in lines]


class HoldPlot:
    def __init__(self, reader, writer=None):
        self.reader = reader
        self.writer = writer
        self.hold = 0
        self.windows = []
        self.values = []

        self.fig, self.ax = plt.subplots(figsize=(10, 5), facecolor=COLOR_SURFACE)
        self.fig.canvas.manager.set_window_title("ReadyRep median frequency")
        ax = self.ax
        style_axes(ax, "Median frequency while the button is held", "Frequency (Hz)")
        ax.set_xlabel("Window (about 1 s each)", color=COLOR_TEXT_MUTED)
        ax.set_xlim(0.5, MIN_WINDOWS_SHOWN + 0.5)
        ax.set_ylim(0, Y_MAX_HZ)

        self.line, = ax.plot([], [], color=COLOR_MDF_LINE, linewidth=2, marker="o",
                             markersize=7, markeredgecolor=COLOR_SURFACE,
                             solid_joinstyle="round", solid_capstyle="round")
        self.average_line = ax.axhline(0, color=COLOR_TEXT_MUTED, linewidth=1, visible=False)
        self.average_label = ax.text(1, 0, "", transform=ax.get_yaxis_transform(),
                                     ha="right", va="bottom", color=COLOR_TEXT_MUTED)
        self.status = ax.text(1, 1.02, "Hold the button to record", transform=ax.transAxes,
                              ha="right", color=COLOR_TEXT_MUTED)
        self.temperature = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top",
                                   color=COLOR_TEXT_MUTED)
        self.fig.tight_layout()

    def start_hold(self):
        self.hold += 1
        self.windows.clear()
        self.values.clear()
        self.line.set_data([], [])
        self.average_line.set_visible(False)
        self.average_label.set_text("")
        self.ax.set_xlim(0.5, MIN_WINDOWS_SHOWN + 0.5)
        self.status.set_text(f"Hold {self.hold}: recording...")

    def add_window(self, window, hz):
        self.windows.append(window)
        self.values.append(hz)
        self.line.set_data(self.windows, self.values)
        self.ax.set_xlim(0.5, max(MIN_WINDOWS_SHOWN, window) + 0.5)
        self.status.set_text(f"Hold {self.hold}: now {hz:.1f} Hz")
        if self.writer:
            self.writer.writerow([self.hold, window, hz])

    def show_average(self, hz, count):
        self.average_line.set_ydata([hz, hz])
        self.average_line.set_visible(True)
        self.average_label.set_y(hz)
        self.average_label.set_text(f"Average {hz:.1f} Hz ")
        self.status.set_text(f"Hold {self.hold}: average {hz:.1f} Hz over {count} windows")

    def update(self, _frame=None):
        for text in self.reader.read():
            if not text:
                continue
            print(text)
            if text.startswith(START_LINE):
                self.start_hold()
            elif text.startswith(TEMP_LINE):
                self.temperature.set_text(text)
            elif text.startswith(TOO_SHORT_LINE):
                self.status.set_text(f"Hold {self.hold}: too short, hold at least 1 s")
            elif match := WINDOW_LINE.match(text):
                self.add_window(int(match.group(1)), float(match.group(2)))
            elif match := AVERAGE_LINE.match(text):
                self.show_average(float(match.group(1)), int(match.group(2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", help="serial port (default: auto-detect the ESP)")
    parser.add_argument("--save", metavar="FILE",
                        help="also record each window's median frequency to a CSV")
    args = parser.parse_args()

    port = args.port or find_port()
    try:
        ser = open_port(port)
    except serial.SerialException as err:
        sys.exit(f"Couldn't open {port}. Close any other serial monitor first. ({err})")

    csv_file = open(args.save, "w", newline="") if args.save else None
    writer = None
    if csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["hold", "window", "mdf_hz"])

    print(f"Reading from {port}. Close the plot window to stop.")
    plot = HoldPlot(LineReader(ser), writer)
    anim = FuncAnimation(plot.fig, plot.update, interval=100, cache_frame_data=False)
    try:
        plt.show()
    finally:
        del anim
        ser.close()
        if csv_file:
            csv_file.close()
            print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
