# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ReadyRep is PlatformIO/Arduino firmware for a Seeed XIAO ESP32-S3 that measures muscle fatigue from a MyoWare EMG sensor. Fatigue shows up as a falling **median frequency (MDF)** of the EMG power spectrum, so the firmware samples EMG at 1 kHz, runs a 1024-point FFT per ~1 s window, and reports the MDF over USB serial. Python tools in `tools/` live-plot that serial stream.

Note: the enclosing git repository is rooted at the user's home directory, not this folder. Scope any git commands to this path (e.g. `git status -- .`, `git add <specific files>`).

## Commands

Each `src/*.cpp` file defines its own `setup()`/`loop()`, so each is built as a separate PlatformIO environment via `build_src_filter` in `platformio.ini`:

| Env | Source | Purpose |
|---|---|---|
| `seeed_xiao_esp32s3` (default) | `src/main.cpp` | Continuous EMG stream + MDF, for `tools/plot_emg.py` |
| `readyrep` | `src/ReadyRep.cpp` | Main app: button-gated MDF recording, for `tools/plot_mdf.py` |
| `readyrep_ble` | `src/ReadyRepBLE.cpp` | Main app: `readyrep` + thermistor on A1 + BLE notifications (nRF Connect) |
| `adc_test` | `src/adc_test.cpp` | Standalone thermistor test (waits for a 1.5 °C rise) |

```sh
pio run -e readyrep                     # build
pio run -e readyrep -t upload -t monitor  # flash and open serial monitor (115200 baud)
pio run                                 # builds the default env (main.cpp)

python3 tools/plot_mdf.py [--port PORT] [--save holds.csv]   # pairs with env readyrep
python3 tools/plot_emg.py [--port PORT] [--window S] [--save session.csv]  # pairs with default env
```

When adding a new `.cpp` with its own `setup()`/`loop()`, add a new env and exclude the file from the default env's `build_src_filter`, or the default build will fail with duplicate symbols. There are no unit tests (`test/` is the PlatformIO placeholder). The Python tools need `matplotlib` and `pyserial`; only one program can hold the serial port, so close `pio device monitor` before running a plot script.

## Architecture

- **Dual-core split:** `loop()` (core 1) does timed sampling with `micros()` at `SAMPLE_PERIOD_US`; full windows are handed through a FreeRTOS queue to `analysisTask` pinned to core 0, which runs the FFT so it never delays sampling. Windows are dropped (`xQueueSend(..., 0)`) if analysis falls behind.
- **`include/median_frequency.h`:** header-only radix-2 FFT and `medianFrequency()` (mean removal, Hann window, band-limited 20–450 Hz power split). It overwrites its input buffers.
- **`include/thermistor.h`:** Beta-equation conversion for a 10k NTC in a divider (3V3 → 10k → ADC → thermistor → GND). Returns `NAN` outside 200–2900 mV (sensor unplugged or shorted), so callers must check `isnan`. Thermistor support in `ReadyRep.cpp` is commented out; `ReadyRepBLE.cpp` has the working version.
- **EMG + thermistor timing in `ReadyRepBLE.cpp`:** each 1 ms tick samples EMG first. Every 100th tick also reads the thermistor, discarding one read first because the ADC sample cap still holds the EMG voltage. Ten reads are averaged into one value per second, which goes through a separate small `temperatureQueue` (the 4 KB EMG window queue stays EMG-only). If the loop falls more than 3 ms behind, the partial window is dropped and timing resyncs, instead of bunching samples together.
- **`ReadyRep.cpp` specifics:** a debounced button on D3 (`INPUT_PULLDOWN`, high while held) gates recording. Each window message carries its *measured* sample rate (the loop can run slower than 1 kHz), and releasing the button enqueues an `endOfRecording` marker so the analysis task prints a per-hold average. ADC calibration (`esp_adc_cal_characterize`) is done once in `setup()` because `analogReadMilliVolts()` recomputes it every call and is too slow.

## Serial protocol coupling

The Python plotters parse firmware output with regexes, so changing a `Serial.printf` format breaks them:

- `plot_emg.py` expects `emg_mv:<int>` and `mdf_hz:<float>` lines (from `main.cpp`), and hardcodes `SAMPLE_RATE_HZ` and `MDF_MAX_HZ` that must match the firmware.
- `plot_mdf.py` matches the human-readable lines from `ReadyRep.cpp`: `Recording median frequency...`, `Median frequency N: X Hz`, `Average median frequency: X Hz (N windows)`, and `Button released too soon`. It imports shared helpers (`find_port`, `open_port`, `style_axes`, colors) from `plot_emg.py`.
- `ReadyRepBLE.cpp` prints the same serial lines as `ReadyRep.cpp` (all output comes from `analysisTask`, including the start event, which `loop()` queues as a `START` message). It also prints `Temperature: …` once a second, which `plot_mdf.py` shows in the corner of the plot. It sends short UTF-8 notifications on two characteristics (device name `ReadyRep`, custom UUIDs in the file):
  - MDF: `rec start`, `mdf N: X Hz`, `avg XHz n=N`, `too short`
  - Temperature: `33.4C +0.3`, with ` warm` appended after the rise target is reached Keep BLE text ≤ 20 bytes, since the default MTU truncates anything longer. It uses the core's `BLEDevice` library and re-advertises in `onDisconnect`. The platform is unpinned and currently resolves to arduino-esp32 3.3.x, where the S3 BLE backend is NimBLE: the notification descriptor (0x2902) is added automatically, so `BLE2902` is only added under `CONFIG_BLUEDROID_ENABLED`. Use `ADC_ATTEN_DB_12`, not the deprecated `_DB_11`.
- `open_port()` keeps DTR/RTS low so opening the port doesn't reset the board; `find_port()` auto-detects by Espressif USB VID `0x303A`.
