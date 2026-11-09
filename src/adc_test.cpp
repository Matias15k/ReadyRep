// Records a starting temperature on A0, then reports once it has risen 1.5 C.
// Build with: pio run -e adc_test
#include <Arduino.h>

#include "thermistor.h"

const uint8_t ADC_PIN = A1;
const int SAMPLES_PER_READING = 16;  // averaging smooths ADC noise

const float RISE_C = 1.5;
const unsigned long STARTUP_DELAY_MS = 2000;  // gives the serial monitor time to connect

static float baselineC;

float readCelsius() {
  uint32_t total = 0;
  for (int i = 0; i < SAMPLES_PER_READING; i++) {
    total += analogReadMilliVolts(ADC_PIN);
  }
  return thermistorCelsius((float)total / SAMPLES_PER_READING);
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(ADC_PIN, ADC_11db);  // full 0-3.1 V input range
  delay(STARTUP_DELAY_MS);

  baselineC = readCelsius();
  Serial.printf("Starting temperature: %.1f F. Waiting for a %.1f C rise...\n",
                toFahrenheit(baselineC), RISE_C);
}

void loop() {
  static bool reached = false;
  if (reached) {
    return;
  }

  float celsius = readCelsius();
  if (celsius - baselineC >= RISE_C) {
    reached = true;
    Serial.printf("Rose %.1f C: now %.1f F (started at %.1f F)\n", celsius - baselineC,
                  toFahrenheit(celsius), toFahrenheit(baselineC));
  }
  delay(100);
}
