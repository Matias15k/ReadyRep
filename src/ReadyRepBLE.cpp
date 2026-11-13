// ReadyRep with Bluetooth LE: EMG on A0, thermistor on A1, record button on D3.
// Sends median frequency and temperature as short UTF-8 notifications that
// nRF Connect can show, and prints the same serial lines as ReadyRep.cpp so
// tools/plot_mdf.py still works over USB.
// Build with: pio run -e readyrep_ble
#include <Arduino.h>
#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <esp_adc_cal.h>

#include "adc_range.h"
#include "median_frequency.h"
#include "thermistor.h"

const uint8_t EMG_PIN = A0;  // MyoWare RAW output
const uint8_t THERMISTOR_PIN = A1;
const uint8_t BUTTON_PIN = D3;  // 3V3 -> button -> D3, reads high while held

// One sample per FreeRTOS tick (1 ms), so 1 kHz.
const TickType_t SAMPLE_PERIOD_TICKS = pdMS_TO_TICKS(1);
const float SAMPLE_RATE_HZ = 1000;
static_assert(CONFIG_FREERTOS_HZ == 1000, "sampling assumes a 1 ms FreeRTOS tick");
// If sampling falls this far behind, the samples since would be bunched up,
// so the partial window is thrown away and timing restarts from now.
const TickType_t MAX_LATE_TICKS = 3;

const size_t FFT_SIZE = 1024;  // about 1 s of data, must be a power of two
const float EMG_MIN_HZ = 20;  // below this is mostly motion artifact
const float EMG_MAX_HZ = 450;

// Temperature changes over seconds, so 10 reads a second averaged to one
// value a second is plenty, and keeps the extra ADC work off most EMG ticks.
const int TEMP_TICK_INTERVAL = 100;  // read the thermistor every 100th sample (10 Hz)
const int TEMP_SAMPLES = 10;  // averaged into one temperature a second
const float RISE_C = 1.5;  // warm-up target above the first reading

const unsigned long STARTUP_DELAY_MS = 2000;  // gives the serial monitor time to connect
const unsigned long DEBOUNCE_MS = 30;

const char *BLE_DEVICE_NAME = "ReadyRep";
const char *SERVICE_UUID = "6b1a0001-3c1e-4f5a-9d2b-8e7f0c4a5d10";
const char *MDF_CHAR_UUID = "6b1a0002-3c1e-4f5a-9d2b-8e7f0c4a5d10";
const char *TEMP_CHAR_UUID = "6b1a0003-3c1e-4f5a-9d2b-8e7f0c4a5d10";
// The default BLE MTU leaves 20 bytes per notification, and longer text gets cut off.
const size_t BLE_TEXT_MAX = 20;

// A full window of EMG samples, or a marker that the button was pressed or released.
struct AnalysisMessage {
  enum Kind { WINDOW, START, END } kind;
  float sampleRateHz;  // measured, since a slow loop samples below SAMPLE_RATE_HZ
  float samples[FFT_SIZE];
};

static QueueHandle_t analysisQueue;
static QueueHandle_t temperatureQueue;  // averaged thermistor mV, one a second
// analogReadMilliVolts() recomputes this on every call, which is slow, so it's done once in setup().
static esp_adc_cal_characteristics_t adcCalibration;

static BLECharacteristic *mdfCharacteristic;
static BLECharacteristic *tempCharacteristic;
static volatile bool bleConnected = false;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *) override { bleConnected = true; }

  void onDisconnect(BLEServer *) override {
    bleConnected = false;
    BLEDevice::startAdvertising();  // the stack stops advertising once a phone connects
  }
};

BLECharacteristic *addTextCharacteristic(BLEService *service, const char *uuid,
                                         const char *label) {
  BLECharacteristic *characteristic = service->createCharacteristic(
      uuid, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
#if defined(CONFIG_BLUEDROID_ENABLED)
  // Lets the phone turn notifications on. NimBLE (the ESP32-S3 default) adds this itself.
  characteristic->addDescriptor(new BLE2902());
#endif
  BLEDescriptor *description = new BLEDescriptor(BLEUUID((uint16_t)0x2901));
  description->setValue(label);  // the name nRF Connect shows
  characteristic->addDescriptor(description);
  characteristic->setValue("waiting");
  return characteristic;
}

void setupBle() {
  BLEDevice::init(BLE_DEVICE_NAME);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *service = server->createService(SERVICE_UUID);
  mdfCharacteristic = addTextCharacteristic(service, MDF_CHAR_UUID, "EMG median frequency");
  tempCharacteristic = addTextCharacteristic(service, TEMP_CHAR_UUID, "Temperature");
  service->start();

  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  BLEDevice::startAdvertising();
}

// Prints serialLine over USB, and sends bleText to a connected phone.
void report(BLECharacteristic *characteristic, const char *serialLine, const char *bleText) {
  Serial.println(serialLine);
  // Cut to what fits in one notification; also what a phone's read returns.
  characteristic->setValue((uint8_t *)bleText, strnlen(bleText, BLE_TEXT_MAX));
  if (bleConnected) {
    characteristic->notify();
  }
}

void reportTemperature(float mv) {
  static float baselineC = NAN;
  static bool riseReached = false;
  char serialLine[96];
  char bleText[32];  // report() sends at most BLE_TEXT_MAX of it

  float celsius = thermistorCelsius(mv);
  if (isnan(celsius)) {
    snprintf(serialLine, sizeof(serialLine), "Temperature: no sensor (%.0f mV on A1)", mv);
    report(tempCharacteristic, serialLine, "no sensor");
    return;
  }
  if (isnan(baselineC)) {
    baselineC = celsius;
  }

  float rise = celsius - baselineC;
  if (!riseReached && rise >= RISE_C) {
    riseReached = true;
    Serial.printf("Rose %.1f C: now %.1f F (started at %.1f F)\n", rise, toFahrenheit(celsius),
                  toFahrenheit(baselineC));
  }
  snprintf(serialLine, sizeof(serialLine), "Temperature: %.1f C (%.1f F), rise %+.1f C",
           celsius, toFahrenheit(celsius), rise);
  snprintf(bleText, sizeof(bleText), "%.1fC %+.1f%s", celsius, rise, riseReached ? " warm" : "");
  report(tempCharacteristic, serialLine, bleText);
}

// Lower priority than sampleTask, so on the single-core C3 sampling interrupts the FFT,
// serial and BLE work instead of waiting for it.
void analysisTask(void *) {
  static AnalysisMessage message;
  static float im[FFT_SIZE];
  char serialLine[96];
  char bleText[32];  // report() sends at most BLE_TEXT_MAX of it
  float mdfTotal = 0;
  int windows = 0;

  for (;;) {
    // Times out so temperatures still go out when no EMG is being recorded.
    if (xQueueReceive(analysisQueue, &message, pdMS_TO_TICKS(100)) == pdTRUE) {
      if (message.kind == AnalysisMessage::START) {
        mdfTotal = 0;
        windows = 0;
        report(mdfCharacteristic, "Recording median frequency...", "rec start");
      } else if (message.kind == AnalysisMessage::WINDOW) {
        float mdf = medianFrequency(message.samples, im, FFT_SIZE, message.sampleRateHz,
                                    EMG_MIN_HZ, EMG_MAX_HZ);
        mdfTotal += mdf;
        windows++;
        snprintf(serialLine, sizeof(serialLine),
                 "Median frequency %d: %.1f Hz (sampled at %.0f Hz)", windows, mdf,
                 message.sampleRateHz);
        snprintf(bleText, sizeof(bleText), "mdf %d: %.1f Hz", windows, mdf);
        report(mdfCharacteristic, serialLine, bleText);
      } else if (windows == 0) {
        snprintf(serialLine, sizeof(serialLine),
                 "Button released too soon: hold it at least %.1f s to measure",
                 FFT_SIZE / SAMPLE_RATE_HZ);
        report(mdfCharacteristic, serialLine, "too short");
      } else {
        float average = mdfTotal / windows;
        snprintf(serialLine, sizeof(serialLine), "Average median frequency: %.1f Hz (%d windows)",
                 average, windows);
        snprintf(bleText, sizeof(bleText), "avg %.1fHz n=%d", average, windows);
        report(mdfCharacteristic, serialLine, bleText);
      }
    }

    float mv;
    while (xQueueReceive(temperatureQueue, &mv, 0) == pdTRUE) {
      reportTemperature(mv);
    }
  }
}

// Returns the button state once it has stayed the same for DEBOUNCE_MS.
bool buttonHeld() {
  static bool stable = false;
  static bool last = false;
  static unsigned long changedAt = 0;

  bool raw = digitalRead(BUTTON_PIN) == HIGH;
  if (raw != last) {
    last = raw;
    changedAt = millis();
  }
  if (millis() - changedAt >= DEBOUNCE_MS) {
    stable = raw;
  }
  return stable;
}

uint32_t readMilliVolts(uint8_t pin) {
  return esp_adc_cal_raw_to_voltage(analogRead(pin), &adcCalibration);
}

// Spreads thermistor reads across ticks so they never delay an EMG sample.
void sampleTemperature() {
  static int tick = 0;
  static uint32_t total = 0;
  static int count = 0;

  if (++tick < TEMP_TICK_INTERVAL) {
    return;
  }
  tick = 0;
  // The ADC's sampling capacitor still holds the EMG voltage, and the divider's
  // 5k source can't recharge it in one conversion, so the first read is discarded.
  analogRead(THERMISTOR_PIN);
  total += readMilliVolts(THERMISTOR_PIN);
  if (++count == TEMP_SAMPLES) {
    float mv = (float)total / TEMP_SAMPLES;
    xQueueSend(temperatureQueue, &mv, 0);
    total = 0;
    count = 0;
  }
}

// A task instead of loop(), because on the single-core C3 Arduino pauses loop() for 5 ms
// every 2 s, which would throw away a window each time. It sleeps between samples so
// analysisTask can run.
void sampleTask(void *) {
  static AnalysisMessage window;
  size_t filled = 0;
  unsigned long windowStartUs = 0;
  bool recording = false;

  TickType_t lastWake = xTaskGetTickCount();
  for (;;) {
    vTaskDelayUntil(&lastWake, SAMPLE_PERIOD_TICKS);
    TickType_t now = xTaskGetTickCount();
    if (now - lastWake > MAX_LATE_TICKS) {
      lastWake = now;
      filled = 0;
    }

    // Sampled first, so the button and thermistor work can't shift its timing.
    unsigned long sampledAt = micros();
    uint32_t mv = readMilliVolts(EMG_PIN);

    bool held = buttonHeld();
    if (held != recording) {
      // Queued in order after any windows still being analyzed, so they all count.
      recording = held;
      filled = 0;
      window.kind = held ? AnalysisMessage::START : AnalysisMessage::END;
      xQueueSend(analysisQueue, &window, portMAX_DELAY);
    }

    if (recording) {
      if (filled == 0) {
        windowStartUs = sampledAt;
      }
      window.samples[filled++] = mv;
      if (filled == FFT_SIZE) {
        window.kind = AnalysisMessage::WINDOW;
        window.sampleRateHz = (FFT_SIZE - 1) * 1000000.0f / (sampledAt - windowStartUs);
        xQueueSend(analysisQueue, &window, 0);  // skips this window if the queue is full
        filled = 0;
      }
    }

    sampleTemperature();
  }
}

void setup() {
  Serial.begin(115200);
#if ARDUINO_USB_MODE && ARDUINO_USB_CDC_ON_BOOT
  // Without a computer attached, USB serial writes would otherwise wait to time out.
  Serial.setTxTimeoutMs(0);
#endif
  pinMode(BUTTON_PIN, INPUT_PULLDOWN);
  analogReadResolution(12);
  analogSetPinAttenuation(EMG_PIN, (adc_attenuation_t)ADC_FULL_RANGE);
  analogSetPinAttenuation(THERMISTOR_PIN, (adc_attenuation_t)ADC_FULL_RANGE);
  esp_adc_cal_characterize(ADC_UNIT_1, ADC_FULL_RANGE, ADC_WIDTH_BIT_12, 1100, &adcCalibration);
  setupBle();
  delay(STARTUP_DELAY_MS);

  Serial.printf("Ready: hold the button to record median frequency (BLE name \"%s\")\n",
                BLE_DEVICE_NAME);

  analysisQueue = xQueueCreate(3, sizeof(AnalysisMessage));
  temperatureQueue = xQueueCreate(4, sizeof(float));
  // Larger stack than ReadyRep.cpp, for float formatting and BLE calls.
  xTaskCreatePinnedToCore(analysisTask, "analysis", 8192, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(sampleTask, "sample", 4096, NULL, 5, NULL, 0);
}

void loop() {
  vTaskDelay(portMAX_DELAY);  // everything happens in sampleTask and analysisTask
}
