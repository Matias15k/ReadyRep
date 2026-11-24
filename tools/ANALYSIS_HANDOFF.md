# ReadyRep EMG and temperature analysis: corrected plan and handoff

**Status (23 Nov 2026).** The pipeline runs end to end on all 82 usable recordings from 16 participants. It is on branch `analysis/corrected-pipeline` (commit `e9b82a5`), which is based on `main` at `655be04`. `main` still holds an earlier, separately written version of `tools/analyze_readyrep.py` that lacks the corrections below. A pull request is needed so the team can choose one.

This document covers:
1. what the study tests
2. what the pipeline does now and why
3. every correction made along the way
4. approaches that were tried and rejected, so nobody repeats them
5. current results
6. what is left to do.

---

## 1. What is being tested

**H1, the readiness rule.** The biceps is "ready" when two things hold together, and both have levelled off:
- EMG median frequency (MDF) has risen at least **10%**.
- Skin temperature over the muscle has risen at least **1.5 °C** from a resting baseline taken with the strap on.

**Protocol.** One-arm dumbbell **preacher curls** on the dominant arm. MyoWare 2.0 EMG electrodes and a 10 kΩ NTC thermistor are taped over the biceps. Sets run in this order:

| File | Set |
|---|---|
| `rep_1` | heavy set, cold |
| `wu_1`, `wu_2`, `wu_3` | light warm-up sets |
| `rep_2` | heavy set, warm |

Emilie and Miki did a heavy set after every warm-up set (`rep_1` to `rep_4`). For them "warm" means the last heavy set.

**Data.** Files are in `data/` and named `<participant>_<rep|wu>_<n>[_tag]_<YYYY-MM-DD_HH-MM-SS>.csv`. Each has columns `time_s, emg_mv, therm_mv, temp_f`, sampled at 1 kHz.
- Each file includes 2–5 s of idle before and after the set.
- `time_s` is **made up by the logger** (`sample_count / 1000`, see `tools/emg_temp.py`), so the true sampling rate cannot be checked. Any error in it scales every MDF value equally.

## 2. How to run

```bash
python3 tools/analyze_readyrep.py --self-test                       # known-answer checks, about 30 s
python3 tools/analyze_readyrep.py data/*.csv --out analysis_out/all # full run, about 25 s
python3 tools/analyze_readyrep.py data/leo_rep_1_*.csv data/leo_rep_2_*.csv --out analysis_out/leo
# options: --overrides overrides.csv  --ratings ratings.csv  --expected-reps N  --no-figures
```

Requirements: numpy, scipy, pandas (version 3 works), matplotlib, statsmodels.

The formats for the overrides and ratings files are described in the script's docstring. Every tuning constant is a named constant at the top of the script and is also printed at the end of `stats.txt`.

**Outputs** (`--out` folder):

| File | One row per | Key columns |
|---|---|---|
| `reps.csv` | rep | `onset_s`, `offset_s`, `snr`, `mdf_hz` (primary, whole rep), `mdf_lift_hz`, `mdf_highact_hz`, `mdf_denoised_hz`, `mdf_multitaper_hz`, `mdf_firmware_hz`, `spike_samples`, `rail_samples`, `valid`, `invalid_reason`, `valid_sensitivity` |
| `sets.csv` | file | set medians of each MDF variant, `n_valid_reps`, `invalid_reasons`, `noise_rms_mv`, `temp_pre_c`, `temp_pre_source`, `delta_t_c`, `mdf_pct_vs_ref`, `mdc_pct`, rule flags, `flags` |
| `participants.csv` | participant | cold and warm MDF, `mdf_pct`, `mdc_pct`, sensitivity % changes, interval and noise changes, `delta_t_warm_c`, `baseline_source`, `temp_flag`, `rule_ready_set` |
| `paired_outcomes.csv` | outcome | n, k up and k down, Hodges–Lehmann estimate, exact CI, Wilcoxon p, Holm p |
| `qc.csv` | file | flags, including files skipped and why |
| `stats.txt` | (text) | summary, **participant inclusion list**, paired results, mixed models, threshold sweep, notes, constants |
| `figures/qc/*.png` | file | filtered EMG, envelopes, thresholds, reps, lifting windows, temperature. **Look at these before trusting any number.** |
| `figures/participants/*.png`, `figures/group_*.png`, `figures/threshold_sweep.png` | | within-person and group plots |

Figures are not committed to git; running the script regenerates them.

## 3. Pipeline as it stands now

### 3.1 Files
- The filename regex gives participant, set type, index, tag and start time. Sets are ordered by timestamp, and time is measured as **minutes since the cold heavy set started**.
- Files with a tag or no timestamp are skipped by default and can be put back with an override. These are `carolyn_wu_2_BAD_CONNECTION…`, `kaitlyn_rep_1_more_weight…` and `miki_rep_2_too much`.
- A `use_as` override lets a file stand in for another set.

### 3.2 EMG cleaning
1. Subtract the file median (the MyoWare output sits near 1580 mV).
2. Band-pass **20–450 Hz**, 4th-order Butterworth, zero-phase (`sosfiltfilt`). This is the same band as the firmware.
3. Notch 60/120/180 Hz **only if** a mains peak is more than 5× its neighbourhood. None of the data has one.
4. **Spikes**: any sample above **8× the local robust SD** (200 ms window) is a spike. The spike sample and 2 samples either side are set to zero before any spectrum is computed. A rep with **5 or more** spike samples is marked `artifact`. For comparison, 95% of reps have zero such samples, while the artifact reps in rae_rep_2 and emilie_rep_1 have 5–38.
5. **ADC rails**: a sample at or below 10 mV or at or above 2930 mV is saturated. The calibrated ADC reading tops out at 2940 mV, which shows as a flat top in rae_rep_2. Any rail sample inside a rep marks it `clipped`.

### 3.3 Envelopes and noise
- **Envelope**: a moving median of |EMG| × 1.4826 over 100 ms, which is a robust SD. Two low-passed versions are used:
  - **5 Hz** for rep boundaries and the lifting phase
  - **1 Hz** for counting reps.
- **Idle**: stretches of at least 0.5 s within the first or last 5 s where the envelope is below 1.5× the file's 10th percentile. At least 1 s of idle is required, or the file falls back to its quietest samples and gets a flag.
- **Noise level**: robust SD of the idle samples. **Noise spectrum**: Welch over the idle stretches, used only for a sensitivity check.

### 3.4 Rep segmentation
1. Find candidate peaks on the **1 Hz envelope**: at least 1.5 s apart, height at least 1.5× noise, prominence at least 0.3× noise.
2. A peak must also have **prominence of at least 25% of its own height**. Lower "shoulders" are the lowering phase riding on the lift, and they merge into the rep before them.
3. **A** = mean of the top half of candidate peak heights. Peaks below **40% of A** are dropped; these are pick-up and put-down bursts.
4. **Edge rule**: the first or last burst is also dropped as a handling burst if it is more than 2 s from the next rep **and** weaker than 75% of the median of the other reps. It is flagged `handling_burst_dropped`.
5. **Boundaries** are taken from the 5 Hz envelope:
   - The rep peak is searched within ±0.4 s of the 1 Hz peak, so a tall spike elsewhere cannot capture it.
   - Onset and offset are where the envelope stays below **noise + 3 × noise SD** for at least **100 ms**.
   - Quiet stretches are found over the whole signal, so a gap that crosses the split point still counts for both reps.
   - Reps shorter than 0.4 s are dropped.
6. **Flags**:
   - `reps_not_separated` when the median valley between reps is above 2× noise, meaning activity never comes back down between reps.
   - `--expected-reps N` checks the rep count.
   - `overrides.csv` with `set_rep`, `exclude_rep`, `exclude`, `include` or `use_as` corrects problems by hand.

### 3.5 Spectral features per rep
- **Primary MDF is computed over the whole rep** (onset to offset) on the spike-repaired signal, using Welch with `nperseg=250`, `noverlap=125`, a Hann window and no zero-padding.
- MDF is the point at half the cumulative power in 20–450 Hz. Each bin's power counts at its **upper edge**, then it is interpolated linearly.
- MNF is also computed.
- **Validity**: a rep is excluded as `clipped`, `artifact`, or `low_snr` (whole-rep RMS / noise below **3**). A set needs at least 2 valid reps.
- **Sensitivity variants**, all stored and tested:

| Variant | Column |
|---|---|
| 500 ms window centred on the first major sub-peak (the lifting phase) | `mdf_lift_hz` |
| Samples at or above 50% of the rep's peak envelope | `mdf_highact_hz` |
| Noise spectrum subtracted, with a looser SNR gate of 2 | `mdf_denoised_hz`, `valid_sensitivity` |
| Multitaper estimate (NW=3, K=5) | `mdf_multitaper_hz` |
| The firmware's own method, a 1024-sample Hann window (`include/median_frequency.h`) | `mdf_firmware_hz` |

- **Set summary**: median of the per-rep MDF values, with IQR, and a fatigue slope across reps.

### 3.6 Rep speed (proxy)
The time between the peaks of consecutive reps (`interval_s`) is the main proxy. Burst duration and onset-to-peak time are also stored. All of these are **EMG-timing proxies, not bar velocity**.

### 3.7 Temperature
1. Samples with therm_mv below 500 mV are dropouts and are removed.
2. Each 1 s block gets a **10% trimmed mean** of mV, converted to °C with the firmware Beta equation (B=3950, 10 kΩ divider, 3.3 V).
3. A block is valid when all three hold:
   - at least half its samples are connected
   - it is within **28–38 °C**
   - it is within **1 °C of the file's median valid block**. Partial contact gives in-between readings, such as 730–1006 mV in carolyn_wu_1.
4. A file is usable if it has at least 3 valid blocks. It is flagged `temp_partial` if under 80% of its blocks are valid.
5. **Pre-set temperature** = median of at least 2 valid idle blocks before the first rep. If those don't exist, it uses the first valid blocks within the first 10 s of the file, at least 2 of them. That case is recorded in `temp_pre_source = early_set`; across 70 files where both are available, it differs from the idle value by a median of +0.02 °C.
6. **ΔT** = pre-set temperature minus the pre-set temperature of the cold heavy set. Only changes are interpreted, because absolute values depend on the assumed supply voltage and ADC calibration.
7. The early dip is reported and never used as a baseline. There is no lag adjustment.
8. **Probe settling flag**: the total rise is at least 0.3 °C and at least 60% of it is already there by the first set after the cold set.

### 3.8 Readiness rule, noise and plateau
- **Primary comparison**: warm heavy set against cold heavy set, at the same load. H1 as written (each warm-up set against wu_1) is reported but not relied on, because many light reps fail the SNR gate.
- **MDC95**: 1.96 × √(SE_a² + SE_b²), where SE = 1.253 × SD / √n.
  - SD is the **pooled, detrended rep-to-rep SD across all participants' heavy sets**: currently 3.53 Hz, giving a median MDC of **9.3%**.
  - Temperature MDC comes from the SD of the 1 s blocks, which is optimistic because neighbouring blocks are correlated.
- **A change "counts"** if it is at least max(10%, MDC) and at least max(1.5 °C, MDC_T). The thresholds were fixed in advance and are **not tuned**.
- **Threshold sweep** (MDF 0–20%, ΔT 0–2 °C): exploratory only.
- **Plateau**: descriptive only. A set has plateaued if its change from the previous set **of the same load** is smaller than the MDC.
- **Rule-ready set**: the first set that meets both thresholds. With `--ratings`, it is compared with the rated set in an agreement table (exact, ±1, signed difference; no kappa).

### 3.9 Statistics
- **Paired, across participants**: Hodges–Lehmann estimate, **exact Wilcoxon (Walsh-average) 95% CI**, exact signed-rank p, and k-of-n up or down.
- **Primary outcome**: MDF % change (whole rep). All other outcomes, including the sensitivity variants, get a Holm-corrected p.
- **Mixed model**: `log(MDF) ~ condition * rep_index + (1|participant)`, using the same participants as the paired test. A confound version adds `interval_s`, `rms_norm` and `noise_rms_mv`. Wald p-values with 13 clusters are optimistic, so read the CIs.

## 4. Corrections made since the first plan, with evidence

| # | What the first plan or code did | Problem found | Correction |
|---|---|---|---|
| 1 | Threshold-crossing rep detection | Leo rep_2 merged two reps | Count reps on the 1 Hz envelope peaks |
| 2 | Peaks on the 5 Hz envelope | Each rep is a lift peak followed by a lower lowering peak more than 1 s later. adam_rep_1 gave 10 "reps", and sanjana 14–16 | 1 Hz counting, plus the prominence rule (≥ 25% of the peak's own height) |
| 3 | A = median of peak heights | Collapsed when small peaks were many: emilie_rep_2 had A/noise = 1.9, so every peak was kept | A = mean of the top half |
| 4 | 40% of A was the only handling-burst rule | Pick-up bursts at 0.55·A survived (andrew_rep_2), and also removed the pre-set temperature | Edge rule: more than 2 s isolated and below 75% of the median rep |
| 5 | Welch 256 / 50% with nfft=1024 on 500 ms | Only 2 segments, the last 116 ms unused, and zero-padding adds no resolution | `nperseg=250`, `noverlap=125`, no padding |
| 6 | MDF interpolated with cumulative power at bin centres | Biased low by half a bin, about 1.4 Hz | Cumulative power at bin upper edges; a flat-spectrum test gives 234 Hz against a true 235 Hz |
| 7 | Temperature as a 1 s median of integer mV | Snapped to steps of about 0.045 °C | 10% trimmed mean |
| 8 | Clipping counted at 2450 mV or above, and at 50 mV or below | The ADC actually tops out at **2940 mV**, so valid samples were called "clipped" (rae_rep_2) | Rails at 10 and 2930 mV, checked per rep |
| 9 | RMS envelope | Electrode or cable **spikes** of more than 1000 mV in rae_rep_2 and emilie_rep_1 pulled rep boundaries and lifting windows onto single spikes | Robust median envelope, anchored peak search, spike repair, and an `artifact` gate |
| 10 | Thresholds crossed at any single sample | With the robust envelope, a brief dip between the lift and lowering phases ended a rep early, by 1.41 s in a synthetic test | 100 ms quiet dwell |
| 11 | **Primary MDF from a 500 ms lifting window** | MDF varies by a **median of 29 Hz** within one rep (IQR 20–39 Hz) depending on window position. A small change in window placement flipped participants, e.g. andrew −6.9% → +12.1%. Rep-to-rep SD was 8.6%, giving an MDC of about 15% | **Whole-rep MDF**: SD 5.6%, MDC about 9.7%. Lift and high-activation windows kept as sensitivity analyses |
| 12 | 80% of the **whole file** valid, otherwise no temperature | Discarded clean pre-set readings in files with later dropouts: jeff_rep_1, matias_rep_2, among others | Validity judged per block, plus skin-range and outlier checks |
| 13 | Per-sample mV range gate | Cut off near-room-temperature readings and hid the probe-off file (leo_wu_2, about 24 °C) | Range checked on blocks; leo_wu_2 is flagged |
| 14 | Equilibration judged from the slope within one file | A 20 s slope is noise (Leo −0.26 to +0.32 °C/min) | Judged across files (settling flag) |
| 15 | Per-person MDC from a 4-rep SD | Very unstable: the 95% CI of an SD from 4 reps spans ×0.57 to ×3.7 | SD pooled across all heavy sets, detrended |
| 16 | Plateau rule | Off-by-one, mixed light and heavy loads, and "1/3 of rise" undefined when there is no rise | Descriptive only, same load, needs k ≥ 2 |
| 17 | Bootstrap CI of the median, uncorrected secondaries | Percentile bootstrap under-covers at n ≈ 13–15 | Exact Hodges–Lehmann / Wilcoxon CI; Holm correction |
| 18 | Code review bugs | Quiet gaps split between reps (51 boundaries); `set_rep` overrides sorted as text; mixed model using different participants from the paired test; fallback pre-set temperature using 1 block or blocks during the set; CSVs rounded to 10 ms | All fixed and verified on the reviewer's test cases |
| 19 | Participants silently missing from tests | It looked like data had been lost (16 participants, but n = 13 / 11) | `stats.txt` now lists every exclusion and its reason |
| 20 | `r.flags` on a pandas row | Returns pandas' own `Flags` object, not the column | Use `r["flags"]` (watch for this in new code too) |

## 5. Tried and did not work (don't repeat)

**Segmentation**
- A simple amplitude threshold (merged reps).
- 5 Hz peak picking (double counts).
- "Merge neighbouring peaks whose valley stays above 1.3× noise": valleys between real reps are also 1.4–5× noise, so it merges whole sets.
- Using only the 40% amplitude rule to drop handling bursts.

**MDF window**
- **"First 500 ms after onset"**: rank correlation with the other methods was only 0.25–0.32, and it was the noisiest (paired CI −5% to +19%).
- A 500 ms window at the envelope maximum, or at the first sub-peak: valid, but too sensitive to placement to serve as the primary measure.
- Noise-spectrum subtraction as the primary method: it changes heavy-set MDF by 0.4 Hz or less, and clipping at zero biases low-SNR reps upward. Kept as a sensitivity check only.

**Artifact detection**
- Kurtosis of the 500 ms window (> 10) separated artifacts well in short windows, but over a whole rep the rise and fall of activity raises kurtosis naturally (median 4.8, 99th percentile 13). Replaced by counting spike samples.

**Temperature**
- Using `temp_f` from the firmware: its moving average smears dropouts into values from −459 to 2845 °F.
- The rules rejected in corrections 12–14 above.

**Self-test**
- A tolerance of ±2 Hz per rep on a single synthetic recording: a 500 ms window genuinely scatters by about ±6 Hz. The test now judges the mean bias over 5 seeds (currently −0.2%).

**Earlier results that are withdrawn**
- "MDF fell in 10 of 13, −4.6%". It came from window-placement noise plus Emilie's artifacts.
- "Light-set MDF +5.5%". Same cause.

## 6. Current results (full run, `analysis_out/all/stats.txt`)

### Data volume
- 82 files and 605 reps.
- Heavy sets: 142 of 161 reps valid.
- Light sets: 275 of 444 reps valid.

### Who is included

| Comparison | n | Left out, and why |
|---|---|---|
| MDF cold vs warm (primary) | 13/16 | Carolyn: rep_1 weak (1/2 usable). Emilie: rep_1 low SNR plus ADC saturation (1/5). Rae: rep_2 spike artifacts reaching the ADC limits (1/5) |
| MDF sensitivity (SNR ≥ 2) | 15/16 | Rae |
| Skin ΔT at the warm set | 14/16 | Adam: rep_1 90% thermistor dropout. Pam: rep_2 93% dropout |
| Light-set MDF (H1 as written) | 12/16 | Carolyn, Justin, Leo, Matias: warm-up EMG below 3× noise |

### Paired results, cold vs warm heavy set

| Outcome | n | k up | Hodges–Lehmann | 95% CI |
|---|---|---|---|---|
| **MDF % (primary, whole rep)** | 13 | 9 | **+2.5%** | +0.3 to +4.1 (p = 0.02) |
| MDF %, SNR ≥ 2 with noise subtracted | 15 | 11 | +1.3% | −2.1 to +3.3 |
| MDF %, 500 ms lifting window | 13 | 6 | +1.4% | −3.2 to +5.7 |
| MDF %, high-activation samples | 13 | 9 | +1.7% | −0.7 to +4.4 |
| Mixed model (log MDF) | 13 | – | +2.4% | −1.7 to +6.8 |
| Inter-rep interval (s) | 16 | 4 | −0.19 s | −0.73 to +0.05 |
| **Skin ΔT (°C)** | 14 | 14 | **+0.99** | +0.68 to +1.21 (p < 0.001) |
| Skin ΔT, idle baselines only | 12 | 12 | +1.00 | +0.72 to +1.34 |
| Light-set MDF %, last wu vs wu_1 | 12 | 8 | +2.4% | −0.7 to +5.5 |

### What these say
- MDF rose by **about 2%**, and every estimate's upper bound is **at most about 7%**. That is below the 10% threshold at group level, and the result agrees across all MDF variants.
- Skin warmed by **about 1 °C**, also below 1.5 °C at the warm set. It reached at least 1.5 °C at only 3 sets: carolyn wu_3, justin wu_2 and justin rep_2.
- Only one set met the MDF part of the rule: yaerim wu_3, at +12.0% against an MDC of 11.9%.
- **The readiness rule was met by 0 of 16 participants.**
- Probe settling is flagged for andrew, emmanuel, pam and yaerim.
- The confound model (tempo, effort and noise floor) leaves the MDF effect unchanged: +2.2%, 95% CI −2.2 to +6.8.

## 7. Open items, in priority order

1. **Visual QC of segmentation.** Review `figures/qc/*.png`, starting with:
   - `reps_not_separated`: kaitlyn_rep_1, kaitlyn_rep_2, matias_rep_1, miki_rep_1, pam_rep_1, pam_wu_3, pam_rep_2
   - `handling_burst_dropped`: adam_rep_2, andrew_rep_2, carolyn_rep_1, carolyn_rep_2, and several warm-up sets
   - known borderline cases: yaerim_rep_1 (the 6th burst is probably the put-down) and carolyn_rep_1 (weak, spiky).

   Fix any problems with `overrides.csv`, re-run, and keep the overrides file in git as an audit trail.
2. **Decide on the skipped files.** Which Kaitlyn rep_1 was at the same load as her rep_2? Was carolyn_wu_2 really unusable? Was miki_rep_2 "too much" a repeat? Use `include` or `use_as` overrides.
3. **Reconcile with `main`.** Open a PR from `analysis/corrected-pipeline`; the version on `main` lacks every correction above.
4. **Ratings and buzz data.** None were found in the repo. Add `ratings.csv` (participant, rated_set) to produce the agreement table.
5. **Report wording**:
   - Describe the study as exploratory.
   - Call the rep-speed measure an EMG-timing proxy.
   - State that the sampling rate can't be verified.
   - State that absolute temperatures are uncalibrated.
   - Explain probe settling.
   - Present the plateau results as descriptive.
   - Report the result as "no evidence of a 10% MDF rise; about a 2% rise at most about 7%".
6. **Nice to have**:
   - Per-participant figures for Emilie and Miki, who have 4 heavy sets each and so the only same-load MDF trajectories.
   - A Bland–Altman comparison of offline and firmware MDF, if the device's real-time readout matters for the report.

### Recommendations for future sessions
- Keep the probe taped on for at least 10 minutes before the baseline reading, with an insulating cover.
- Run a no-warm-up control session.
- Add a standard 5 s isometric hold at 90° elbow with a fixed load before the warm-up and after each set. It gives a steady, high-SNR MDF.
- Use a metronome for tempo.
- Do at least 6 reps per heavy set to bring the per-person MDC below 10%.
- Record the load and a felt-readiness rating per set.
- Firmware:
  - log real `micros()` timestamps
  - fix the thermistor connector, which dropped out in about 20 files
  - secure the EMG cable, which caused spike artifacts
  - consider 2 kHz sampling.

## 8. Short answers to the team's seven planning questions

1. **Feature design**: keep the per-set median of per-rep MDF, but compute MDF over the **whole rep**, not a 500 ms or lifting-only window, because MDF moves by about 29 Hz within one rep. Gate reps on SNR at least 3 and on artifacts.
2. **Thresholds**: keep 10% and 1.5 °C fixed in advance, and count a change only if it also exceeds the MDC (pooled noise). With 4–5 reps the MDC is about 9%, borderline for judging one person at 10%. The threshold sweep is exploratory only.
3. **Plateau**: with at most 5 noisy points, report it descriptively (step smaller than the MDC, same load only). It can't be properly tested with this protocol.
4. **Skin temperature**: use the pre-set level change from the cold-set baseline. Don't use rate of change or a lag adjustment. Report the early dip but never use it as a baseline, and watch for probe settling.
5. **Tests**: within-person plots; a paired exact Wilcoxon test with Hodges–Lehmann CI; a mixed model over all reps; an agreement table once ratings exist.
6. **Validity**: claim group-level changes and their CIs only. Don't claim that the rule detects readiness, that the warm-up caused the change, that skin temperature reflects muscle temperature, or that the device is validated.
7. **Missing pieces**: see the recommendations at the end of section 7.
