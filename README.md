# ReadyRep

Muscle fatigue and warm-up tracker for the Seeed XIAO ESP32-S3.

- **Fatigue:** a MyoWare EMG sensor is sampled at 1 kHz. While the button is held, the board computes the EMG's median frequency about once per second. Median frequency drops as a muscle tires.
- **Warm-up:** a thermistor tracks skin temperature and flags when it has risen 1.5 °C.

Results go to a phone over Bluetooth (nRF Connect) and to a computer over USB.

## Wiring

| Part | Connection |
|---|---|
| MyoWare RAW out | A0 (power the MyoWare from 3V3, so its output stays within the ADC's 0–3.1 V range) |
| Thermistor (10k NTC, B = 3950) | 3V3 → 10k resistor → A1 → thermistor → GND |
| Record button | 3V3 → button → D3 |

If your thermistor has a different B value, change `THERMISTOR_BETA` in `include/thermistor.h`.

## Build and flash

Install [PlatformIO](https://platformio.org/install), then:

```sh
pio run -e readyrep_ble -t upload -t monitor
```

| Environment | Firmware |
|---|---|
| `readyrep_ble` | **Main app:** EMG + thermistor + button, Bluetooth + serial |
| `readyrep` | EMG + button, serial only |
| `seeed_xiao_esp32s3` (default) | Streams raw EMG continuously, for `plot_emg.py` |
| `adc_test` | Thermistor-only test |

## Using it with nRF Connect

1. Open nRF Connect, scan, and connect to **ReadyRep**.
2. Open the unknown service. Tap the notify (triple-arrow) button on both characteristics.
3. Long-press each value and pick **UTF-8** to show it as text.

| Characteristic | Messages |
|---|---|
| EMG median frequency | `rec start`, `mdf 3: 92.4 Hz` (one per second), `avg 90.1Hz n=8` on release, `too short` |
| Temperature | `33.4C +0.3` every second, `34.9C +1.5 warm` once warmed up, `no sensor` if unplugged |

Hold the button for at least 1 s to get a reading. The temperature rise is measured from the first reading after power-on, so attach the thermistor before switching on.

## Plotting on a computer

```sh
pip install pyserial matplotlib
python3 tools/plot_mdf.py               # with readyrep or readyrep_ble
python3 tools/plot_emg.py               # with the default environment
```

Add `--save file.csv` to record data. Close `pio device monitor` first, because only one program can use the serial port at a time.
