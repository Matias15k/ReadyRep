// EMG on A0, record button on D3. (Thermistor on A1 is commented out for now.)
// Prints each window's median frequency and the average per hold, for tools/plot_mdf.py.
// Build with: pio run -e readyrep
#include <Arduino.h>
#include <esp_adc_cal.h>

#include "median_frequency.h"
// #include "thermistor.h"

const uint8_t EMG_PIN = A0;  // MyoWare RAW output
// const uint8_t THERMISTOR_PIN = A1;
const uint8_t BUTTON_PIN = D3;  // 3V3 -> button -> D3, reads high while held

// One sample per FreeRTOS tick (1 ms), so 1 kHz.
const TickType_t SAMPLE_PERIOD_TICKS = pdMS_TO_TICKS(1);
const float SAMPLE_RATE_HZ = 1000;
static_assert(CONFIG_FREERTOS_HZ == 1000, "sampling assumes a 1 ms FreeRTOS tick");

const size_t FFT_SIZE = 1024;  // about 1 s of data, must be a power of two
const float EMG_MIN_HZ = 20;  // below this is mostly motion artifact
const float EMG_MAX_HZ = 450;

// const float RISE_C = 1.5;
// const int TEMP_TICK_INTERVAL = 10;  // ADC reads are slow, so only every 10th tick also reads temperature
// const int TEMP_SAMPLES = 10;  // averaged, so a new temperature every 100 ms
// const int BASELINE_SAMPLES = 100;  // the starting temperature gets extra averaging
const unsigned long STARTUP_DELAY_MS = 2000;  // gives the serial monitor time to connect
const unsigned long DEBOUNCE_MS = 30;

// Either a full window of EMG samples, or a marker that the button was released.
struct AnalysisMessage {
  bool endOfRecording;
  float sampleRateHz;  // measured, since a slow loop samples below SAMPLE_RATE_HZ
  float samples[FFT_SIZE];
};

static QueueHandle_t analysisQueue;
// analogReadMilliVolts() recomputes this on every call, which is slow, so it's done once in setup().
static esp_adc_cal_characteristics_t adcCalibration;
// static float baselineC;

// Lower priority than sampleTask, so on the single-core C3 sampling interrupts the FFT
// instead of waiting for it.
void analysisTask(void *) {
  static AnalysisMessage message;
  static float im[FFT_SIZE];
  float mdfTotal = 0;
  int windows = 0;

  for (;;) {
    xQueueReceive(analysisQueue, &message, portMAX_DELAY);
    if (!message.endOfRecording) {
      float mdf = medianFrequency(message.samples, im, FFT_SIZE, message.sampleRateHz,
                                  EMG_MIN_HZ, EMG_MAX_HZ);
      mdfTotal += mdf;
      windows++;
      Serial.printf("Median frequency %d: %.1f Hz (sampled at %.0f Hz)\n", windows, mdf,
                    message.sampleRateHz);
      continue;
    }

    if (windows == 0) {
      Serial.printf("Button released too soon: hold it at least %.1f s to measure\n",
                    FFT_SIZE / SAMPLE_RATE_HZ);
    } else {
      Serial.printf("Average median frequency: %.1f Hz (%d windows)\n",
                    mdfTotal / windows, windows);
    }
    mdfTotal = 0;
    windows = 0;
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

// float readCelsius() {
//   uint32_t total = 0;
//   for (int i = 0; i < BASELINE_SAMPLES; i++) {
//     total += analogReadMilliVolts(THERMISTOR_PIN);
//   }
//   return thermistorCelsius((float)total / BASELINE_SAMPLES);
// }

// A task instead of loop(), because on the single-core C3 Arduino pauses loop() for 5 ms
// every 2 s, which would skip samples. It sleeps between samples so analysisTask can run.
void sampleTask(void *) {
  static AnalysisMessage window;
  size_t filled = 0;
  unsigned long windowStartUs = 0;
  bool recording = false;
  // int tick = 0;
  // uint32_t tempTotal = 0;
  // int tempCount = 0;
  // bool riseReached = false;

  TickType_t lastWake = xTaskGetTickCount();
  for (;;) {
    vTaskDelayUntil(&lastWake, SAMPLE_PERIOD_TICKS);
    bool held = buttonHeld();
    if (held && !recording) {
      recording = true;
      filled = 0;
      Serial.println("Recording median frequency...");
    } else if (!held && recording) {
      // Queued after any windows still being analyzed, so they all count.
      recording = false;
      window.endOfRecording = true;
      xQueueSend(analysisQueue, &window, portMAX_DELAY);
    }

    unsigned long sampledAt = micros();
    uint32_t mv = esp_adc_cal_raw_to_voltage(analogRead(EMG_PIN), &adcCalibration);
    // Serial.printf("emg_mv:%lu\n", (unsigned long)mv);  // streamed for tools/plot_mdf.py

    if (recording) {
      if (filled == 0) {
        windowStartUs = sampledAt;
      }
      window.samples[filled++] = mv;
      if (filled == FFT_SIZE) {
        window.endOfRecording = false;
        window.sampleRateHz = (FFT_SIZE - 1) * 1000000.0f / (sampledAt - windowStartUs);
        xQueueSend(analysisQueue, &window, 0);  // skips this window if the queue is full
        filled = 0;
      }
    }

    // // Spread the temperature reading across ticks so it never delays EMG sampling.
    // if (++tick < TEMP_TICK_INTERVAL) {
    //   continue;
    // }
    // tick = 0;
    // tempTotal += analogReadMilliVolts(THERMISTOR_PIN);
    // if (++tempCount == TEMP_SAMPLES) {
    //   float celsius = thermistorCelsius((float)tempTotal / TEMP_SAMPLES);
    //   tempTotal = 0;
    //   tempCount = 0;
    //   if (!riseReached && celsius - baselineC >= RISE_C) {
    //     riseReached = true;
    //     Serial.printf("Rose %.1f C: now %.1f F (started at %.1f F)\n", celsius - baselineC,
    //                   toFahrenheit(celsius), toFahrenheit(baselineC));
    //   }
    // }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLDOWN);
  analogReadResolution(12);
  analogSetPinAttenuation(EMG_PIN, ADC_11db);  // about 0-2.5 V on the C3
  esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1100, &adcCalibration);
  // analogSetPinAttenuation(THERMISTOR_PIN, ADC_11db);
  delay(STARTUP_DELAY_MS);

  // baselineC = readCelsius();
  // Serial.printf("Starting temperature: %.1f F. Waiting for a %.1f C rise...\n",
  //               toFahrenheit(baselineC), RISE_C);
  Serial.println("Ready: hold the button to record median frequency");

  analysisQueue = xQueueCreate(3, sizeof(AnalysisMessage));
  xTaskCreatePinnedToCore(analysisTask, "analysis", 4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(sampleTask, "sample", 4096, NULL, 5, NULL, ARDUINO_RUNNING_CORE);
}

void loop() {
  vTaskDelay(portMAX_DELAY);  // everything happens in sampleTask and analysisTask
}
