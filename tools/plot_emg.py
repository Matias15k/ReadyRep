#!/usr/bin/env python3
"""Live-plot MyoWare EMG readings streamed over USB serial by the ESP32.

Usage:
    python3 tools/plot_emg.py
    python3 tools/plot_emg.py --port /dev/cu.usbmodem1101 --window 20
    python3 tools/plot_emg.py --save session.csv
"""
import argparse
import collections
import csv
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial
from serial.tools import list_ports

SAMPLE_RATE_HZ = 1000  # must match SAMPLE_PERIOD_US in src/main.cpp
Y_MIN_MV = 1000
Y_MAX_MV = 5000
MDF_WINDOW_S = 60  # longer than the raw graph, since fatigue is a slow trend
MDF_MAX_HZ = 450  # must match EMG_MAX_HZ in src/main.cpp
ESPRESSIF_VID = 0x303A
SAMPLE_LINE = re.compile(r"^emg_mv:(\d+)$")
MDF_LINE = re.compile(r"^mdf_hz:(\d+(?:\.\d+)?)$")

COLOR_LINE = "#2a78d6"
COLOR_MDF_LINE = "#eb6834"
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_MUTED = "#52514e"
COLOR_GRID = "#e4e3df"


def find_port():
    ports = list_ports.comports()
    matches = [p.device for p in ports if p.vid == ESPRESSIF_VID]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        sys.exit(f"Several ESP boards found, choose one with --port: {matches}")
    found = [p.device for p in ports] or "none"
    sys.exit(f"No ESP board found. Is it plugged in? Serial ports: {found}")


def open_port(port):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = 115200
    ser.timeout = 0
    # Keep these low so opening the port doesn't reset the board.
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


class SampleReader:
    """Turns raw serial bytes into samples and median frequencies, keeping partial lines."""

    def __init__(self, ser):
        self.ser = ser
        self.pending = b""

    def read(self):
        """Returns (samples, mdfs). Each mdf is (samples read before it, Hz)."""
        self.pending += self.ser.read(self.ser.in_waiting or 1)
        *lines, self.pending = self.pending.split(b"\n")
        samples = []
        mdfs = []
        for line in lines:
            text = line.decode("ascii", "ignore").strip()
            match = SAMPLE_LINE.match(text)
            if match:
                samples.append(int(match.group(1)))
                continue
            match = MDF_LINE.match(text)
            if match:
                print(text)
                mdfs.append((len(samples), float(match.group(1))))
        return samples, mdfs


def style_axes(ax, title, ylabel):
    ax.set_facecolor(COLOR_SURFACE)
    ax.set_title(title, loc="left", color=COLOR_TEXT, fontsize=13)
    ax.set_xlabel("Time (s)", color=COLOR_TEXT_MUTED)
    ax.set_ylabel(ylabel, color=COLOR_TEXT_MUTED)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLOR_TEXT_MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_GRID)


class LivePlot:
    def __init__(self, reader, window_s, writer=None):
        self.reader = reader
        self.writer = writer
        self.window_s = window_s
        self.count = 0
        size = window_s * SAMPLE_RATE_HZ
        self.times = collections.deque(maxlen=size)
        self.values = collections.deque(maxlen=size)
        self.mdf_times = collections.deque(maxlen=MDF_WINDOW_S)
        self.mdf_values = collections.deque(maxlen=MDF_WINDOW_S)

        self.fig, (self.ax, self.mdf_ax) = plt.subplots(
            1, 2, figsize=(14, 5), width_ratios=[2, 1], facecolor=COLOR_SURFACE
        )
        self.fig.canvas.manager.set_window_title("ReadyRep EMG")
        ax = self.ax
        style_axes(ax, "EMG raw signal (RAW)", "Voltage (mV)")
        ax.set_xlim(0, window_s)
        ax.set_ylim(Y_MIN_MV, Y_MAX_MV)

        self.line, = ax.plot([], [], color=COLOR_LINE, linewidth=2,
                             solid_joinstyle="round", solid_capstyle="round")
        self.status = ax.text(1, 1.02, "Waiting for data...", transform=ax.transAxes,
                              ha="right", color=COLOR_TEXT_MUTED)

        mdf_ax = self.mdf_ax
        style_axes(mdf_ax, "Median frequency", "Frequency (Hz)")
        mdf_ax.set_xlim(0, MDF_WINDOW_S)
        mdf_ax.set_ylim(0, MDF_MAX_HZ)

        self.mdf_line, = mdf_ax.plot([], [], color=COLOR_MDF_LINE, linewidth=2, marker="o",
                                     markersize=6, markeredgecolor=COLOR_SURFACE,
                                     solid_joinstyle="round", solid_capstyle="round")
        self.mdf_status = mdf_ax.text(1, 1.02, "Waiting for data...",
                                      transform=mdf_ax.transAxes, ha="right",
                                      color=COLOR_TEXT_MUTED)
        self.fig.tight_layout()

    def update(self, _frame=None):
        samples, mdfs = self.reader.read()
        for offset, hz in mdfs:
            self.mdf_times.append((self.count + offset) / SAMPLE_RATE_HZ)
            self.mdf_values.append(hz)
        if mdfs:
            self.mdf_line.set_data(self.mdf_times, self.mdf_values)
            end = max(self.mdf_times[-1], MDF_WINDOW_S)
            self.mdf_ax.set_xlim(end - MDF_WINDOW_S, end)
            self.mdf_status.set_text(f"Now {self.mdf_values[-1]:.1f} Hz")
        if not samples:
            return
        for value in samples:
            t = self.count / SAMPLE_RATE_HZ
            self.times.append(t)
            self.values.append(value)
            self.count += 1
            if self.writer:
                self.writer.writerow([f"{t:.3f}", value])

        self.line.set_data(self.times, self.values)
        end = max(self.times[-1], self.window_s)
        self.ax.set_xlim(end - self.window_s, end)
        self.status.set_text(
            f"Now {self.values[-1]} mV   Peak {max(self.values)} mV"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", help="serial port (default: auto-detect the ESP)")
    parser.add_argument("--window", type=int, default=3,
                        help="seconds of data to show (default: 10)")
    parser.add_argument("--save", metavar="FILE", help="also record samples to a CSV")
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
        writer.writerow(["time_s", "emg_mv"])

    print(f"Reading from {port}. Close the plot window to stop.")
    plot = LivePlot(SampleReader(ser), args.window, writer)
    anim = FuncAnimation(plot.fig, plot.update, interval=33, cache_frame_data=False)
    try:
        plt.show()
    except serial.SerialException as err:
        print(f"Lost connection to the board: {err}")
    finally:
        del anim
        ser.close()
        if csv_file:
            csv_file.close()
            print(f"Saved {plot.count} samples to {args.save}")


if __name__ == "__main__":
    main()
