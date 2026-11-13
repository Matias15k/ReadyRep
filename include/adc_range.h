#pragma once

#include <Arduino.h>
#include <esp_adc_cal.h>

// The ADC's full-range setting was renamed in arduino-esp32 3.x, so this picks
// whichever name the installed core has. Both mean about 0-2.5 V on the C3.
#if ESP_ARDUINO_VERSION_MAJOR >= 3
const adc_atten_t ADC_FULL_RANGE = ADC_ATTEN_DB_12;
#else
const adc_atten_t ADC_FULL_RANGE = ADC_ATTEN_DB_11;
#endif
