// Prints every raw MyoWare EMG reading (RAW pin on A0) as "emg_mv:<value>" at 1 kHz.
// Graph it with tools/plot_raw.py.
// Build with: pio run -e raw -t upload
#include <Arduino.h>
#include <esp_adc_cal.h>

const uint8_t EMG_PIN = A0;  // MyoWare RAW output
// One sample per FreeRTOS tick (1 ms), must match SAMPLE_RATE_HZ in tools/plot_raw.py
const TickType_t SAMPLE_PERIOD_TICKS = pdMS_TO_TICKS(1);
static_assert(CONFIG_FREERTOS_HZ == 1000, "sampling assumes a 1 ms FreeRTOS tick");

// analogReadMilliVolts() recomputes this on every call, which is slow, so it's done once in setup().
static esp_adc_cal_characteristics_t adcCalibration;

// A task instead of loop(), because on the single-core C3 Arduino pauses loop() for 5 ms
// every 2 s, which would skip samples.
void sampleTask(void *) {
  TickType_t lastWake = xTaskGetTickCount();
  for (;;) {
    vTaskDelayUntil(&lastWake, SAMPLE_PERIOD_TICKS);
    uint32_t mv = esp_adc_cal_raw_to_voltage(analogRead(EMG_PIN), &adcCalibration);
    Serial.printf("emg_mv:%lu\n", (unsigned long)mv);
  }
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(EMG_PIN, ADC_11db);  // about 0-2.5 V on the C3
  esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1100, &adcCalibration);
  xTaskCreatePinnedToCore(sampleTask, "sample", 4096, NULL, 5, NULL, ARDUINO_RUNNING_CORE);
}

void loop() {
  vTaskDelay(portMAX_DELAY);  // everything happens in sampleTask
}
