#include <Arduino.h>

#include "median_frequency.h"

const uint8_t EMG_PIN = A0;  // MyoWare RAW output
const unsigned long SAMPLE_PERIOD_US = 1000;  // 1 kHz
const float SAMPLE_RATE_HZ = 1000000.0f / SAMPLE_PERIOD_US;

const size_t FFT_SIZE = 1024;  // about 1 s of data, must be a power of two
const float EMG_MIN_HZ = 20;  // below this is mostly motion artifact
const float EMG_MAX_HZ = 450;

static QueueHandle_t windowQueue;

// Runs on the other core so the FFT can't delay sampling in loop().
void analysisTask(void *) {
  static float re[FFT_SIZE];
  static float im[FFT_SIZE];

  for (;;) {
    xQueueReceive(windowQueue, re, portMAX_DELAY);
    float mdf = medianFrequency(re, im, FFT_SIZE, SAMPLE_RATE_HZ, EMG_MIN_HZ, EMG_MAX_HZ);
    Serial.printf("mdf_hz:%.1f\n", mdf);
  }
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(EMG_PIN, ADC_11db);  // full 0-3.1 V input range

  windowQueue = xQueueCreate(1, sizeof(float) * FFT_SIZE);
  xTaskCreatePinnedToCore(analysisTask, "analysis", 4096, NULL, 1, NULL, 0);
}

void loop() {
  static unsigned long lastSample = micros();
  static float window[FFT_SIZE];
  static size_t filled = 0;

  unsigned long now = micros();
  if (now - lastSample >= SAMPLE_PERIOD_US) {
    lastSample += SAMPLE_PERIOD_US;
    uint32_t mv = analogReadMilliVolts(EMG_PIN);
    Serial.printf("emg_mv:%lu\n", (unsigned long)mv);

    window[filled++] = mv;
    if (filled == FFT_SIZE) {
      xQueueSend(windowQueue, window, 0);  // skips this window if still busy
      filled = 0;
    }
  }
}
