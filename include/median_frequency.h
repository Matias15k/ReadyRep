#pragma once

#include <math.h>
#include <stddef.h>

// In-place radix-2 FFT. n must be a power of two.
inline void fft(float *re, float *im, size_t n) {
  for (size_t i = 1, j = 0; i < n; i++) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      float tmp = re[i];
      re[i] = re[j];
      re[j] = tmp;
      tmp = im[i];
      im[i] = im[j];
      im[j] = tmp;
    }
  }

  for (size_t len = 2; len <= n; len <<= 1) {
    for (size_t k = 0; k < len / 2; k++) {
      float angle = -2.0f * (float)M_PI * k / len;
      float wr = cosf(angle);
      float wi = sinf(angle);
      for (size_t a = k; a < n; a += len) {
        size_t b = a + len / 2;
        float tr = re[b] * wr - im[b] * wi;
        float ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

// Returns the frequency that splits the signal's power in [minHz, maxHz] in
// half, or 0 if that band has no power. Overwrites re and im.
inline float medianFrequency(float *re, float *im, size_t n, float sampleRateHz,
                             float minHz, float maxHz) {
  float mean = 0;
  for (size_t i = 0; i < n; i++) {
    mean += re[i] / n;
  }
  for (size_t i = 0; i < n; i++) {
    float hann = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (n - 1));
    re[i] = (re[i] - mean) * hann;
    im[i] = 0;
  }

  fft(re, im, n);

  float binHz = sampleRateHz / n;
  size_t first = (size_t)ceilf(minHz / binHz);
  size_t last = (size_t)floorf(maxHz / binHz);
  if (last > n / 2) {
    last = n / 2;
  }

  float total = 0;
  for (size_t k = first; k <= last; k++) {
    total += re[k] * re[k] + im[k] * im[k];
  }
  if (total <= 0) {
    return 0;
  }

  float cumulative = 0;
  for (size_t k = first; k <= last; k++) {
    cumulative += re[k] * re[k] + im[k] * im[k];
    if (cumulative >= total / 2) {
      return k * binHz;
    }
  }
  return last * binHz;
}
