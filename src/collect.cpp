// Prints every raw MyoWare EMG reading (RAW pin on A0) and thermistor reading (A1) at 1 kHz as
// "emg_mv:<mV>,therm_mv:<mV>,temp_f:<F>".
// Graph and record it with tools/emg_temp.py.
// Build with: pio run -e collect -t upload
#include <Arduino.h>
#include <esp_adc_cal.h>

#include "thermistor.h"

const uint8_t EMG_PIN = A0;  // MyoWare RAW output
const uint8_t THERMISTOR_PIN = A1;  // 3V3 -> 10k resistor -> A1 -> thermistor -> GND
// One sample per FreeRTOS tick (1 ms), must match SAMPLE_RATE_HZ in tools/emg_temp.py
const TickType_t SAMPLE_PERIOD_TICKS = pdMS_TO_TICKS(1);
static_assert(CONFIG_FREERTOS_HZ == 1000, "sampling assumes a 1 ms FreeRTOS tick");
// Temperature is worked out from the average of the last 100 thermistor readings (100 ms),
// since single ADC readings are noisy.
const int TEMP_AVERAGE_SAMPLES = 100;

// analogReadMilliVolts() recomputes this on every call, which is slow, so it's done once in setup().
static esp_adc_cal_characteristics_t adcCalibration;

uint32_t readMilliVolts(uint8_t pin) {
  return esp_adc_cal_raw_to_voltage(analogRead(pin), &adcCalibration);
}

// A task instead of loop(), because on the single-core C3 Arduino pauses loop() for 5 ms
// every 2 s, which would skip samples.
void sampleTask(void *) {
  static uint32_t recent[TEMP_AVERAGE_SAMPLES];
  uint32_t first = readMilliVolts(THERMISTOR_PIN);
  for (int i = 0; i < TEMP_AVERAGE_SAMPLES; i++) {
    recent[i] = first;
  }
  uint32_t recentTotal = first * TEMP_AVERAGE_SAMPLES;
  int oldest = 0;

  TickType_t lastWake = xTaskGetTickCount();
  for (;;) {
    vTaskDelayUntil(&lastWake, SAMPLE_PERIOD_TICKS);
    uint32_t emgMv = readMilliVolts(EMG_PIN);
    uint32_t thermMv = readMilliVolts(THERMISTOR_PIN);

    recentTotal += thermMv - recent[oldest];
    recent[oldest] = thermMv;
    oldest = (oldest + 1) % TEMP_AVERAGE_SAMPLES;
    float tempF = toFahrenheit(thermistorCelsius((float)recentTotal / TEMP_AVERAGE_SAMPLES));

    Serial.printf("emg_mv:%lu,therm_mv:%lu,temp_f:%.1f\n", (unsigned long)emgMv,
                  (unsigned long)thermMv, tempF);
  }
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(EMG_PIN, ADC_11db);  // about 0-2.5 V on the C3
  analogSetPinAttenuation(THERMISTOR_PIN, ADC_11db);
  esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1100, &adcCalibration);
  xTaskCreatePinnedToCore(sampleTask, "sample", 4096, NULL, 5, NULL, ARDUINO_RUNNING_CORE);
}

void loop() {
  vTaskDelay(portMAX_DELAY);  // everything happens in sampleTask
}
