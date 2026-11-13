#include <Arduino.h>

#include "median_frequency.h"

const uint8_t EMG_PIN = A0;  // MyoWare RAW output
// One sample per FreeRTOS tick (1 ms), so 1 kHz.
const TickType_t SAMPLE_PERIOD_TICKS = pdMS_TO_TICKS(1);
const float SAMPLE_RATE_HZ = 1000;
static_assert(CONFIG_FREERTOS_HZ == 1000, "sampling assumes a 1 ms FreeRTOS tick");

const size_t FFT_SIZE = 1024;  // about 1 s of data, must be a power of two
const float EMG_MIN_HZ = 20;  // below this is mostly motion artifact
const float EMG_MAX_HZ = 450;

static QueueHandle_t windowQueue;

// Lower priority than sampleTask, so on the single-core C3 sampling interrupts the FFT
// instead of waiting for it.
void analysisTask(void *) {
  static float re[FFT_SIZE];
  static float im[FFT_SIZE];

  for (;;) {
    xQueueReceive(windowQueue, re, portMAX_DELAY);
    float mdf = medianFrequency(re, im, FFT_SIZE, SAMPLE_RATE_HZ, EMG_MIN_HZ, EMG_MAX_HZ);
    Serial.printf("mdf_hz:%.1f\n", mdf);
  }
}

// A task instead of loop(), because on the single-core C3 Arduino pauses loop() for 5 ms
// every 2 s, which would skip samples. It sleeps between samples so analysisTask can run.
void sampleTask(void *) {
  static float window[FFT_SIZE];
  size_t filled = 0;

  TickType_t lastWake = xTaskGetTickCount();
  for (;;) {
    vTaskDelayUntil(&lastWake, SAMPLE_PERIOD_TICKS);
    uint32_t mv = analogReadMilliVolts(EMG_PIN);
    Serial.printf("emg_mv:%lu\n", (unsigned long)mv);

    window[filled++] = mv;
    if (filled == FFT_SIZE) {
      xQueueSend(windowQueue, window, 0);  // skips this window if still busy
      filled = 0;
    }
  }
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(EMG_PIN, ADC_11db);  // about 0-2.5 V on the C3

  windowQueue = xQueueCreate(1, sizeof(float) * FFT_SIZE);
  xTaskCreatePinnedToCore(analysisTask, "analysis", 4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(sampleTask, "sample", 4096, NULL, 5, NULL, ARDUINO_RUNNING_CORE);
}

void loop() {
  vTaskDelay(portMAX_DELAY);  // everything happens in sampleTask and analysisTask
}
