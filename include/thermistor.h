#pragma once

#include <math.h>

// Divider: 3V3 -> 10k resistor -> ADC pin -> thermistor -> GND
const float SUPPLY_MV = 3300;
const float SERIES_OHMS = 10000;
const float THERMISTOR_OHMS_AT_25C = 10000;
const float THERMISTOR_BETA = 3950;  // check the "B" value on your part's listing

// 0-60 C reads about 2500-660 mV. Outside this band the sensor is unplugged
// (the pin floats up to the ADC's ~3.1 V ceiling) or shorted (near 0 V).
const float THERMISTOR_MIN_MV = 200;
const float THERMISTOR_MAX_MV = 2900;

// Returns NAN if the reading is outside the band a connected sensor can produce.
inline float thermistorCelsius(float mv) {
  if (mv < THERMISTOR_MIN_MV || mv > THERMISTOR_MAX_MV) {
    return NAN;
  }
  float ohms = SERIES_OHMS * mv / (SUPPLY_MV - mv);
  float kelvin = 1.0f / (1.0f / 298.15f + logf(ohms / THERMISTOR_OHMS_AT_25C) / THERMISTOR_BETA);
  return kelvin - 273.15f;
}

inline float toFahrenheit(float celsius) {
  return celsius * 9.0f / 5.0f + 32.0f;
}
