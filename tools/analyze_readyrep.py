#!/usr/bin/env python3
"""Analyse ReadyRep EMG and skin-temperature recordings (biceps, one-arm preacher curls).

Input: CSVs written by tools/emg_temp.py (time_s, emg_mv, therm_mv, temp_f at 1 kHz), named
    <participant>_<rep|wu>_<n>[_tag]_<YYYY-MM-DD_HH-MM-SS>.csv
where rep = heavy set and wu = light warm-up set.

For every file the script:
  1. cleans the EMG: removes the MyoWare offset, band-passes 20-450 Hz (zero-phase) and notches
     mains only if a mains peak is actually present,
  2. finds the idle stretches before and after the set and measures the noise floor,
  3. segments reps (a lifting burst and its lowering burst count as one rep) and drops the
     pick-up and put-down bursts,
  4. measures median frequency (MDF) on a fixed 500 ms window centred on the lifting phase of
     every rep, and marks reps below the SNR gate,
  5. recomputes skin temperature from the raw thermistor voltage, ignoring sensor dropouts.

It then groups files into one session per participant, compares the first (cold) and last
(warm) heavy sets, applies the H1 readiness rule and writes tables, statistics and figures.

Usage:
    python3 tools/analyze_readyrep.py data/leo_rep_1_*.csv data/leo_rep_2_*.csv --out analysis_out
    python3 tools/analyze_readyrep.py data/*.csv --out analysis_out
    python3 tools/analyze_readyrep.py data/*.csv --overrides overrides.csv --ratings ratings.csv
    python3 tools/analyze_readyrep.py --self-test

overrides.csv (optional; one row per instruction; blank cells are fine):
    file,action,rep,start_s,end_s,role,note
    carolyn_wu_2_BAD_CONNECTION2026-11-17_18-17-51.csv,include,,,,,checked QC figure
    leo_wu_2_2026-11-17_20-18-46.csv,exclude,,,,,
    leo_rep_2_2026-11-17_20-24-04.csv,exclude_rep,3,,,,motion artifact
    adam_rep_1_2026-11-18_14-47-46.csv,set_rep,1,4.2,8.6,,manual rep window
    kaitlyn_rep_1_more_weight_2026-11-15_19-41-51.csv,use_as,,,,rep_1,same load as rep_2
  include      use a file excluded by default (it has a tag or no timestamp)
  exclude      drop a whole file, or one rep when `rep` is given (exclude_rep does the same)
  set_rep      replace automatic segmentation for that file with the manual windows given
  use_as       treat the file as another set (role like rep_1); the file it replaces is dropped

ratings.csv (optional): participant,rated_set   e.g.  leo,wu_3
    The set the participant said felt ready (or where the device buzzed). It is compared with
    the set the rule picks.

Every tuning constant is a module-level constant below and is copied into stats.txt.
"""
import argparse
import math
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal, stats  # noqa: E402
from scipy.ndimage import median_filter, uniform_filter1d  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Tuning constants (all written to stats.txt)
# ---------------------------------------------------------------------------------------------
FS_HZ = 1000  # src/collect.cpp samples once per 1 ms FreeRTOS tick

# EMG cleaning
EMG_BAND_HZ = (20.0, 450.0)  # same band as the firmware (src/ReadyRep.cpp)
BANDPASS_ORDER = 4
MAINS_HZ = (60.0, 120.0, 180.0)
MAINS_PEAK_RATIO = 5.0  # notch only if a mains bin is this many times the nearby median
NOTCH_Q = 30.0
# ADC rails: the calibrated ESP32-C3 reading saturates at 2940 mV (seen as a flat top in
# rae_rep_2) and at 0 mV. Values between 2450 and 2940 mV are real signal, not clipping.
CLIP_LOW_MV, CLIP_HIGH_MV = 10, 2930
SPIKE_Z = 8.0  # sample > 8 x local robust SD = spike (electrode pop / cable movement)
SPIKE_LOCAL_S = 0.200  # window for the local robust SD
SPIKE_PAD = 2  # samples zeroed either side of a spike before spectra are computed
SPIKE_MAX_SAMPLES = 5  # a rep with this many spike samples or more is marked as an artifact

# Envelopes
RMS_WINDOW_S = 0.100  # envelope window; the envelope is a moving robust amplitude
MAD_TO_SD = 1.4826  # median(|x|) x 1.4826 = SD for Gaussian noise, unaffected by sparse spikes
ENV_BOUNDARY_LP_HZ = 5.0  # for rep boundaries and the lifting phase
ENV_COUNT_LP_HZ = 1.0  # for counting reps (merges lifting and lowering bursts)

# Idle / noise
IDLE_SEARCH_S = 5.0  # look for idle in the first and last 5 s
IDLE_ENV_FACTOR = 1.5  # idle = envelope below 1.5 x the file's 10th percentile
IDLE_MIN_RUN_S = 0.5
IDLE_MIN_TOTAL_S = 1.0

# Rep segmentation
REP_MIN_SPACING_S = 1.5
REP_CANDIDATE_MIN_HEIGHT = 1.5  # x noise RMS (1 Hz envelope)
REP_CANDIDATE_MIN_PROMINENCE = 0.3  # x noise RMS
REP_MIN_RELATIVE_PROMINENCE = 0.25  # a rep peak must stand out by 25% of its own height;
#   lower shoulders are lowering phases and get merged into the rep before them
REP_KEEP_FRACTION_OF_A = 0.40  # A = mean of the top half of candidate peaks
EDGE_BURST_GAP_S = 2.0  # first/last burst this far from the set and ...
EDGE_BURST_FRACTION = 0.75  # ... weaker than 75% of the other reps' median = handling burst
NOT_SEPARATED_VALLEY = 2.0  # median valley between reps above 2 x noise = reps not separated
BOUNDARY_NOISE_SD = 3.0  # onset/offset where the 5 Hz envelope crosses noise RMS + 3 SD
REP_MIN_DURATION_S = 0.4
QUIET_DWELL_S = 0.10  # envelope must stay below the threshold this long to end/start a rep
ANCHOR_HALF_WIDTH = 400  # samples: rep peak is searched within +/-0.4 s of the 1 Hz peak
LIFT_SUBPEAK_FRACTION = 0.5  # lifting phase = first sub-peak >= 50% of the rep maximum

# Spectral features. Primary MDF = whole rep (onset to offset): MDF moves by a median 29 Hz
# across a rep, so a short window is placement-sensitive; the whole rep had the lowest
# rep-to-rep SD (5.6% of MDF vs 8-9% for 500 ms windows). The 500 ms lifting window and the
# high-activation (>= 50% of rep peak) samples are kept as sensitivity analyses.
HIGH_ACTIVATION_FRACTION = 0.5
MDF_WINDOW_S = 0.5
WELCH_NPERSEG = 250
WELCH_NOVERLAP = 125
MULTITAPER_NW, MULTITAPER_K = 3.0, 5
FIRMWARE_FFT_SIZE = 1024  # include/median_frequency.h + src/ReadyRep.cpp
SNR_GATE = 3.0  # rep RMS / noise RMS
SNR_GATE_SENSITIVITY = 2.0  # sensitivity analysis: looser gate with noise-PSD-subtracted MDF
MIN_VALID_REPS_PER_SET = 2

# Temperature
THERM_DROPOUT_MV = 500  # below this the thermistor wire is disconnected
TEMP_BLOCK_OUTLIER_C = 1.0  # a block this far from the file's median is a partial-contact reading
TEMP_MIN_VALID_BLOCKS = 3  # fewer valid 1 s blocks than this = no usable temperature for the file
TEMP_PRE_MIN_BLOCKS = 2  # pre-set temperature needs at least this many valid idle blocks
TEMP_BLOCK_S = 1.0
TEMP_BLOCK_MIN_FRACTION = 0.5  # a block needs half its samples free of dropouts
TEMP_BLOCK_TRIM = 0.1  # trimmed-mean fraction cut from each end of a 1 s block
TEMP_FILE_MIN_VALID = 0.8  # below this the file is used but flagged temp_partial
TEMP_SKIN_RANGE_C = (28.0, 38.0)
EARLY_DIP_WINDOW_S = 30.0
PRE_TEMP_FALLBACK_S = 10.0  # if no idle block precedes the set, use valid blocks in the first 10 s
SUPPLY_MV, SERIES_OHMS, THERM_R25_OHMS, THERM_BETA = 3300.0, 10000.0, 10000.0, 3950.0

# Readiness rule (H1, pre-specified; not tuned on these data)
MDF_RISE_THRESHOLD_PCT = 10.0
TEMP_RISE_THRESHOLD_C = 1.5
SETTLING_MIN_RISE_C = 0.3
SETTLING_FIRST_STEP_FRACTION = 0.6
SWEEP_MDF_PCT = np.arange(0.0, 20.01, 2.5)
SWEEP_TEMP_C = np.arange(0.0, 2.01, 0.25)
ALPHA = 0.05

# Figure style (dataviz reference palette; blue/orange validated as a pair)
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_2 = "#52514e"
COLOR_GRID = "#e4e3de"
COLOR_NEUTRAL = "#a3a29b"
COLOR_TRACE = "#8a8983"
COLOR_1 = "#2a78d6"  # heavy / cold / primary
COLOR_2 = "#eb6834"  # light / warm / secondary
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FILENAME_RE = re.compile(
    r"^(?P<participant>[A-Za-z]+)_(?P<kind>rep|wu)_(?P<index>\d+)(?P<tag>.*?)_?"
    r"(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})?\.csv$")


# ---------------------------------------------------------------------------------------------
# File parsing and overrides
# ---------------------------------------------------------------------------------------------
def parse_filename(path):
    name = Path(path).name
    m = FILENAME_RE.match(name)
    if not m:
        return None
    tag = m.group("tag").strip("_ ")
    stamp = m.group("stamp")
    start = datetime.strptime(stamp, "%Y-%m-%d_%H-%M-%S") if stamp else None
    return {
        "file": name,
        "path": str(path),
        "participant": m.group("participant").lower(),
        "kind": m.group("kind"),
        "index": int(m.group("index")),
        "tag": tag,
        "start": start,
    }


def load_overrides(path):
    if not path:
        return pd.DataFrame(columns=["file", "action", "rep", "start_s", "end_s", "role", "note"])
    ov = pd.read_csv(path, dtype=str).fillna("")
    for col in ["file", "action", "rep", "start_s", "end_s", "role", "note"]:
        if col not in ov.columns:
            ov[col] = ""
    ov["action"] = ov["action"].str.strip().str.lower()
    return ov


def overrides_for(ov, filename):
    stem = Path(filename).stem
    sel = ov["file"].str.strip().isin([filename, stem])
    return ov[sel]


# ---------------------------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------------------------
def lowpass(x, fc, order=2):
    sos = signal.butter(order, fc, btype="low", fs=FS_HZ, output="sos")
    return signal.sosfiltfilt(sos, x)


def bandpass(x):
    sos = signal.butter(BANDPASS_ORDER, EMG_BAND_HZ, btype="bandpass", fs=FS_HZ, output="sos")
    return signal.sosfiltfilt(sos, x)


def mains_present(x):
    """True if any mains harmonic stands out from its neighbourhood in a 1 Hz-resolution PSD."""
    f, p = signal.welch(x, FS_HZ, nperseg=min(len(x), FS_HZ))
    for f0 in MAINS_HZ:
        k = np.argmin(np.abs(f - f0))
        near = (np.abs(f - f0) >= 3) & (np.abs(f - f0) <= 10)
        if near.any() and p[k] > MAINS_PEAK_RATIO * np.median(p[near]):
            return True
    return False


def notch_mains(x):
    for f0 in MAINS_HZ:
        b, a = signal.iirnotch(f0, NOTCH_Q, FS_HZ)
        x = signal.filtfilt(b, a, x)
    return x


def moving_rms(x, window_s=RMS_WINDOW_S):
    return np.sqrt(np.maximum(uniform_filter1d(x ** 2, max(1, int(window_s * FS_HZ))), 0))


def robust_envelope(x, window_s=RMS_WINDOW_S):
    """Moving median of |x| scaled to an SD. Unlike RMS, a few spike-artifact samples
    (electrode or cable movement) cannot dominate it, so they cannot pull rep boundaries."""
    return MAD_TO_SD * median_filter(np.abs(x), size=max(1, int(window_s * FS_HZ)), mode="nearest")


def spike_mask(x):
    """(core, padded) boolean masks of spike samples: |x| > SPIKE_Z x local robust SD."""
    local = MAD_TO_SD * median_filter(np.abs(x), size=int(SPIKE_LOCAL_S * FS_HZ), mode="nearest")
    core = np.abs(x) > SPIKE_Z * np.maximum(local, 1e-9)
    padded = np.convolve(core.astype(float), np.ones(2 * SPIKE_PAD + 1), "same") > 0
    return core, padded


def high_activation_mdf(xr, env5, on, off):
    """MDF over the samples where the rep envelope is >= HIGH_ACTIVATION_FRACTION of its peak,
    power-weighted across contiguous runs of at least one Welch segment."""
    e = env5[on:off + 1]
    runs = [(on + a, on + b) for a, b in runs_of(e >= HIGH_ACTIVATION_FRACTION * e.max())
            if b - a >= WELCH_NPERSEG]
    if not runs:
        return np.nan
    psds = [welch_psd(xr[a:b]) for a, b in runs]
    p = np.average([pp for _, pp in psds], axis=0, weights=[b - a for a, b in runs])
    return mdf_mnf(psds[0][0], p)[0]


def robust_sd(x):
    return float(MAD_TO_SD * np.median(np.abs(x))) if len(x) else np.nan


def runs_of(mask):
    """(start, stop) index pairs of True runs; stop is exclusive."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def mdf_mnf(f, p, band=EMG_BAND_HZ):
    """Median and mean frequency. Each bin's power is spread evenly over its width, so the
    cumulative sum sits at bin upper edges (placing it at bin centres biases MDF low)."""
    df = f[1] - f[0]
    m = (f >= band[0]) & (f <= band[1])
    fb, pb = f[m], np.clip(p[m], 0, None)
    total = pb.sum()
    if total <= 0 or len(fb) < 2:
        return np.nan, np.nan
    edges = np.r_[fb[0] - df / 2, fb + df / 2]
    cum = np.r_[0.0, np.cumsum(pb)]
    return float(np.interp(total / 2, cum, edges)), float((fb * pb).sum() / total)


def welch_psd(seg):
    nper = min(WELCH_NPERSEG, len(seg))
    nover = min(WELCH_NOVERLAP, nper // 2)
    return signal.welch(seg, FS_HZ, window="hann", nperseg=nper, noverlap=nover,
                        detrend="constant", scaling="density")


def multitaper_psd(seg):
    n = len(seg)
    tapers = signal.windows.dpss(n, MULTITAPER_NW, MULTITAPER_K)
    spec = np.fft.rfft(tapers * (seg - seg.mean()), axis=1)
    return np.fft.rfftfreq(n, 1 / FS_HZ), (np.abs(spec) ** 2).mean(axis=0) / FS_HZ


def firmware_mdf(raw_seg):
    """Re-implementation of medianFrequency() in include/median_frequency.h."""
    n = len(raw_seg)
    x = (raw_seg - raw_seg.mean()) * (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    spec = np.fft.fft(x)
    bin_hz = FS_HZ / n
    first = int(math.ceil(EMG_BAND_HZ[0] / bin_hz))
    last = min(int(math.floor(EMG_BAND_HZ[1] / bin_hz)), n // 2)
    power = np.abs(spec[first:last + 1]) ** 2
    if power.sum() <= 0:
        return np.nan
    k = np.searchsorted(np.cumsum(power), power.sum() / 2)
    return (first + k) * bin_hz


def therm_mv_to_c(mv):
    mv = np.asarray(mv, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ohms = SERIES_OHMS * mv / (SUPPLY_MV - mv)
        kelvin = 1.0 / (1.0 / 298.15 + np.log(ohms / THERM_R25_OHMS) / THERM_BETA)
    return kelvin - 273.15


def theil_sen_slope(t, y):
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(stats.theilslopes(y[ok], t[ok])[0])


# ---------------------------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------------------------
def load_csv(path):
    df = pd.read_csv(path, on_bad_lines="skip")
    missing = {"emg_mv", "therm_mv"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["emg_mv", "therm_mv"])
    return df.reset_index(drop=True)


def find_idle(env5, t):
    """Idle samples: low-envelope stretches of >= IDLE_MIN_RUN_S in the first/last IDLE_SEARCH_S."""
    flags = []
    thr = IDLE_ENV_FACTOR * np.percentile(env5, 10)
    cand = (t < IDLE_SEARCH_S) | (t > t[-1] - IDLE_SEARCH_S)
    mask = np.zeros(len(env5), bool)
    for s, e in runs_of(cand & (env5 < thr)):
        if (e - s) / FS_HZ >= IDLE_MIN_RUN_S:
            mask[s:e] = True
    if mask.sum() / FS_HZ < IDLE_MIN_TOTAL_S:
        flags.append("short_idle_fallback")
        mask = env5 <= np.percentile(env5, 20)
    return mask, flags


def segment_reps(env5, env1, noise_rms, noise_sd):
    """Count reps on the 1 Hz envelope, then refine boundaries on the 5 Hz envelope.

    Returns (reps, candidates): reps is a list of dicts; candidates holds every 1 Hz peak and
    whether it was kept, so the QC figure can show what was dropped and why.
    """
    n = len(env5)
    cand, props = signal.find_peaks(env1, distance=int(REP_MIN_SPACING_S * FS_HZ),
                                    height=REP_CANDIDATE_MIN_HEIGHT * noise_rms,
                                    prominence=REP_CANDIDATE_MIN_PROMINENCE * noise_rms)
    # A shoulder that barely stands out (the lowering phase riding on the lift) is not a rep.
    cand = cand[props["prominences"] >= REP_MIN_RELATIVE_PROMINENCE * env1[cand]]
    thr = noise_rms + BOUNDARY_NOISE_SD * noise_sd
    info = {"idx": cand, "kept": np.zeros(len(cand), bool), "A": np.nan, "boundary_thr": thr,
            "flags": []}
    if len(cand) == 0:
        return [], info
    heights = env1[cand]
    top = np.sort(heights)[::-1][:int(math.ceil(len(heights) / 2))]
    amp_ref = float(top.mean())
    kept = heights >= REP_KEEP_FRACTION_OF_A * amp_ref

    def bounds(i):
        p = cand[i]
        # Search limits: the deepest valley towards each neighbouring candidate (kept or not),
        # so a dropped put-down burst is never absorbed into the last rep.
        lo = 0 if i == 0 else cand[i - 1] + int(np.argmin(env5[cand[i - 1]:p]))
        hi = n - 1 if i == len(cand) - 1 else p + int(np.argmin(env5[p:cand[i + 1]]))
        return boundaries_in(env5, lo, hi, thr, anchor=p, quiet=quiet)

    # Quiet runs over the whole signal, so a gap that straddles the split point between two reps
    # still counts as one quiet run for both of them.
    dwell = int(QUIET_DWELL_S * FS_HZ)
    quiet = [(a, b) for a, b in runs_of(env5 < thr) if b - a >= dwell]
    reps = {i: bounds(i) for i in np.flatnonzero(kept)}
    for i in list(reps):
        if (reps[i]["offset"] - reps[i]["onset"]) / FS_HZ < REP_MIN_DURATION_S:
            del reps[i]
            kept[i] = False
    # Isolated weak burst at either end of the set = picking up or putting down the dumbbell.
    for _ in range(2):
        order = sorted(reps)
        if len(order) < 3:
            break
        for end, nb in [(order[0], order[1]), (order[-1], order[-2])]:
            gap = (reps[nb]["onset"] - reps[end]["offset"]) if end < nb else \
                  (reps[end]["onset"] - reps[nb]["offset"])
            others = np.median([env1[cand[j]] for j in order if j != end])
            if gap / FS_HZ > EDGE_BURST_GAP_S and env1[cand[end]] < EDGE_BURST_FRACTION * others:
                del reps[end]
                kept[end] = False
                info["flags"].append("handling_burst_dropped")
                break
    order = sorted(reps)
    valleys = [env5[cand[a]:cand[b]].min() / noise_rms for a, b in zip(order, order[1:])]
    if len(valleys) >= 2 and np.median(valleys) > NOT_SEPARATED_VALLEY:
        info["flags"].append("reps_not_separated(check QC figure)")
    info.update(kept=kept, A=amp_ref)
    return [reps[i] for i in order], info


def boundaries_in(env5, lo, hi, thr, anchor=None, quiet=None):
    """Onset/offset where the 5 Hz envelope crosses thr, searched outward from the rep peak.
    The peak is taken near the 1 Hz rep-count peak (anchor), not anywhere in [lo, hi], so a
    single tall artifact elsewhere in the range cannot move the rep."""
    if anchor is not None:
        a = max(lo, anchor - ANCHOR_HALF_WIDTH)
        b = min(hi, anchor + ANCHOR_HALF_WIDTH)
        peak = a + int(np.argmax(env5[a:b + 1]))
    else:
        peak = lo + int(np.argmax(env5[lo:hi + 1]))
    # The rep ends where the envelope stays below thr for QUIET_DWELL_S (a brief dip between the
    # lifting and lowering phases is not the end of the rep); the same rule applies to the onset.
    if quiet is None:
        dwell = int(QUIET_DWELL_S * FS_HZ)
        quiet = [(a, b) for a, b in runs_of(env5 < thr) if b - a >= dwell]
    before = [b for a, b in quiet if b <= peak]
    after = [a for a, b in quiet if a >= peak]
    onset = min(max(before[-1], lo), peak) if before else lo
    offset = max(min(after[0] - 1, hi), peak) if after else hi
    return {"onset": onset, "offset": max(offset, onset + 1), "manual": False}


def lifting_peak(env5, onset, offset, noise_rms):
    """First sub-peak of the rep that reaches LIFT_SUBPEAK_FRACTION of the rep maximum."""
    seg = env5[onset:offset + 1]
    peaks, _ = signal.find_peaks(seg, prominence=0.5 * noise_rms)
    if len(peaks):
        big = peaks[seg[peaks] >= LIFT_SUBPEAK_FRACTION * seg.max()]
        if len(big):
            return onset + int(big[0])
    return onset + int(np.argmax(seg))


def analysis_window(center, onset, offset, n):
    """A MDF_WINDOW_S window centred on the lifting peak, kept inside the rep when it fits."""
    w = int(MDF_WINDOW_S * FS_HZ)
    start = center - w // 2
    if offset - onset + 1 >= w:
        start = min(max(start, onset), offset + 1 - w)
    start = min(max(start, 0), n - w)
    return start, start + w


def analyse_temperature(therm_mv, t, first_onset, last_offset):
    out = {"flags": []}
    mv = therm_mv.astype(float).copy()
    mv[mv < THERM_DROPOUT_MV] = np.nan
    block = int(TEMP_BLOCK_S * FS_HZ)
    nblocks = len(mv) // block
    if nblocks == 0:
        out["flags"].append("temp_too_short")
        return out, pd.DataFrame(columns=["t_s", "mv", "temp_c", "valid"])
    blocks = mv[:nblocks * block].reshape(nblocks, block)
    ok_frac = np.isfinite(blocks).mean(axis=1)
    # 10% trimmed mean per block: robust to spikes, and not stuck on whole-mV steps like a median.
    med_mv = np.array([stats.trim_mean(b[np.isfinite(b)], TEMP_BLOCK_TRIM) if np.isfinite(b).any()
                       else np.nan for b in blocks])
    temp_c = therm_mv_to_c(med_mv)
    # A block is valid if most samples are connected, it is in the skin range, and it agrees
    # with the rest of the file (partial contact gives in-between readings, e.g. carolyn_wu_1).
    valid = (ok_frac >= TEMP_BLOCK_MIN_FRACTION) & (temp_c >= TEMP_SKIN_RANGE_C[0]) & \
            (temp_c <= TEMP_SKIN_RANGE_C[1])
    if valid.any():
        valid &= np.abs(temp_c - np.median(temp_c[valid])) <= TEMP_BLOCK_OUTLIER_C
    series = pd.DataFrame({"t_s": (np.arange(nblocks) + 0.5) * TEMP_BLOCK_S, "mv": med_mv,
                           "temp_c": temp_c, "valid": valid})
    out["temp_valid_frac"] = float(valid.mean())
    out["temp_dropout_frac"] = float(1 - np.isfinite(mv).mean())
    good = series[series.valid]
    med_all = float(np.nanmedian(therm_mv_to_c(med_mv[np.isfinite(med_mv)]))) \
        if np.isfinite(med_mv).any() else np.nan
    out["temp_median_any_c"] = med_all
    file_ok = len(good) >= TEMP_MIN_VALID_BLOCKS
    if out["temp_dropout_frac"] > 0.2:
        out["flags"].append(f"therm_dropout({out['temp_dropout_frac']:.0%})")
    if np.isfinite(med_all) and not (TEMP_SKIN_RANGE_C[0] <= med_all <= TEMP_SKIN_RANGE_C[1]) \
            and len(good) < TEMP_MIN_VALID_BLOCKS:
        out["flags"].append(f"temp_off_skin({med_all:.1f}C)")
    if file_ok and valid.mean() < TEMP_FILE_MIN_VALID:
        out["flags"].append(f"temp_partial({valid.mean():.0%}_valid)")
    if not file_ok:
        out["flags"].append(f"temp_unusable({len(good)}_valid_blocks)")
    out["temp_valid"] = bool(file_ok)
    if not file_ok:
        return out, series
    out["temp_median_c"] = float(good.temp_c.median())
    pre = good[good.t_s + TEMP_BLOCK_S / 2 <= first_onset / FS_HZ] if first_onset is not None \
        else good.iloc[:0]
    out["temp_pre_source"] = "idle"
    if len(pre) < TEMP_PRE_MIN_BLOCKS:
        pre = pre.iloc[:0]
        out["temp_pre_source"] = ""
    if not len(pre) and first_onset is not None:
        # No clean idle before the set: use the first valid blocks if they are in the first
        # 10 s. Across 70 files with both, this differs from the idle value by a median
        # +0.02 °C (IQR -0.02 to +0.05), so it is allowed but recorded as early_set.
        early = good[good.t_s <= PRE_TEMP_FALLBACK_S].iloc[:3]
        if len(early) >= TEMP_PRE_MIN_BLOCKS:
            pre = early
            out["temp_pre_source"] = "early_set"
            out["flags"].append("pre_set_temp_from_early_set")
    out["temp_pre_c"] = float(pre.temp_c.median()) if len(pre) else np.nan
    out["temp_pre_n_blocks"] = int(len(pre))
    out["temp_pre_sd_c"] = float(detrended_sd(pre.t_s.values, pre.temp_c.values))
    if first_onset is not None:
        during = good[(good.t_s >= first_onset / FS_HZ) & (good.t_s <= last_offset / FS_HZ)]
        post = good[good.t_s - TEMP_BLOCK_S / 2 >= last_offset / FS_HZ]
        early = good[(good.t_s >= first_onset / FS_HZ) &
                     (good.t_s <= first_onset / FS_HZ + EARLY_DIP_WINDOW_S)]
        out["temp_during_c"] = float(during.temp_c.mean()) if len(during) else np.nan
        out["temp_post_c"] = float(post.temp_c.median()) if len(post) else np.nan
        out["temp_dip_c"] = float(out["temp_pre_c"] - early.temp_c.min()) \
            if len(early) and np.isfinite(out["temp_pre_c"]) else np.nan
    out["temp_slope_c_per_min"] = theil_sen_slope(good.t_s.values / 60, good.temp_c.values)
    if not np.isfinite(out["temp_pre_c"]):
        out["flags"].append("no_pre_set_temp")
    return out, series


def detrended_sd(t, y):
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    coef = np.polyfit(t[ok], y[ok], 1)
    return float(np.std(y[ok] - np.polyval(coef, t[ok]), ddof=2))


def analyse_file(meta, df, ov_rows):
    """Everything that can be computed from one CSV on its own."""
    res = {"meta": meta, "flags": []}
    raw = df["emg_mv"].to_numpy(float)
    n = len(raw)
    t = np.arange(n) / FS_HZ
    res["duration_s"] = n / FS_HZ
    clip = (raw <= CLIP_LOW_MV) | (raw >= CLIP_HIGH_MV)
    res["clip_frac"] = float(clip.mean())

    x = bandpass(raw - np.median(raw))
    res["mains_notched"] = mains_present(x)
    if res["mains_notched"]:
        x = notch_mains(x)
    spike_core, spike_pad = spike_mask(x)
    xr = np.where(spike_pad, 0.0, x)  # spike-repaired signal, used for every spectrum
    res["spike_samples"] = int(spike_core.sum())
    amp = robust_envelope(x)
    env5 = np.maximum(lowpass(amp, ENV_BOUNDARY_LP_HZ), 0)
    env1 = np.maximum(lowpass(amp, ENV_COUNT_LP_HZ), 0)

    idle, idle_flags = find_idle(env5, t)
    res["flags"] += idle_flags
    noise_rms = robust_sd(x[idle])  # robust SD = RMS for clean noise, ignores idle spikes
    noise_sd = float(np.std(env5[idle]))
    res.update(noise_rms_mv=noise_rms, noise_env_sd_mv=noise_sd, idle_s=idle.sum() / FS_HZ)
    f_noise, p_noise = welch_psd_concat(xr, idle)

    manual = ov_rows[ov_rows.action == "set_rep"]
    if len(manual):
        reps = []
        manual = manual.assign(_k=pd.to_numeric(manual.start_s, errors="coerce"))
        for _, r in manual.sort_values("_k").iterrows():
            on = int(float(r.start_s) * FS_HZ)
            off = min(int(float(r.end_s) * FS_HZ), n - 1)
            reps.append({"onset": on, "offset": off, "manual": True})
        seg_info = {"idx": np.array([], int), "kept": np.array([], bool), "A": np.nan,
                    "boundary_thr": noise_rms + BOUNDARY_NOISE_SD * noise_sd}
        res["flags"].append("manual_reps")
    else:
        reps, seg_info = segment_reps(env5, env1, noise_rms, noise_sd)
        res["flags"] += sorted(set(seg_info["flags"]))
    res["n_candidates"] = int(len(seg_info["idx"]))
    res["amp_ref_over_noise"] = seg_info["A"] / noise_rms if noise_rms > 0 else np.nan

    excluded_reps = set()
    for _, r in ov_rows[ov_rows.action.isin(["exclude_rep"]) |
                        ((ov_rows.action == "exclude") & (ov_rows.rep.str.strip() != ""))].iterrows():
        excluded_reps.add(int(float(r.rep)))

    rep_rows = []
    prev_peak = None
    for k, rep in enumerate(reps, start=1):
        on, off = rep["onset"], rep["offset"]
        lift = lifting_peak(env5, on, off, noise_rms)
        w0, w1 = analysis_window(lift, on, off, n)
        whole = xr[on:off + 1]
        f, p = welch_psd(whole)
        mdf, mnf = mdf_mnf(f, p)  # primary: whole rep
        mdf_dn = np.nan
        if f_noise is not None and len(f_noise) == len(f):
            mdf_dn, _ = mdf_mnf(f, p - p_noise)
        fm, pm = multitaper_psd(whole)
        mdf_mt, _ = mdf_mnf(fm, pm)
        mdf_lift, _ = mdf_mnf(*welch_psd(xr[w0:w1]))
        mdf_hi = high_activation_mdf(xr, env5, on, off)
        h = FIRMWARE_FFT_SIZE // 2
        a, b = max(0, lift - h), min(n, lift - h + FIRMWARE_FFT_SIZE)
        mdf_fw = firmware_mdf(raw[a:b]) if b - a == FIRMWARE_FFT_SIZE else np.nan
        rep_rms = float(np.sqrt(np.mean(whole ** 2)))
        snr = rep_rms / noise_rms if noise_rms > 0 else np.nan
        rail = int(clip[on:off + 1].sum())  # samples at the ADC rails inside the rep
        spikes = int(spike_core[on:off + 1].sum())
        kurt = float(stats.kurtosis(x[w0:w1], fisher=False))
        excluded = k in excluded_reps
        reason = ("excluded_by_override" if excluded else
                  "clipped" if rail > 0 else
                  "artifact" if spikes >= SPIKE_MAX_SAMPLES else
                  "low_snr" if snr < SNR_GATE else "")
        valid = reason == ""
        # Sensitivity: looser SNR gate, same artifact rules.
        valid_sens = (not excluded) and rail == 0 and spikes < SPIKE_MAX_SAMPLES and \
            snr >= SNR_GATE_SENSITIVITY
        rep_rows.append({
            "file": meta["file"], "participant": meta["participant"], "rep": k,
            "onset_s": on / FS_HZ, "offset_s": off / FS_HZ, "duration_s": (off - on) / FS_HZ,
            "lift_peak_s": lift / FS_HZ, "onset_to_peak_s": (lift - on) / FS_HZ,
            "interval_s": (lift - prev_peak) / FS_HZ if prev_peak is not None else np.nan,
            "window_start_s": w0 / FS_HZ, "window_end_s": w1 / FS_HZ,
            "rms_mv": rep_rms, "peak_env_mv": float(env5[on:off + 1].max()), "snr": snr,
            "mdf_hz": mdf, "mnf_hz": mnf, "mdf_denoised_hz": mdf_dn, "mdf_multitaper_hz": mdf_mt,
            "mdf_lift_hz": mdf_lift, "mdf_highact_hz": mdf_hi, "mdf_firmware_hz": mdf_fw,
            "rail_samples": rail, "spike_samples": spikes, "lift_window_kurtosis": kurt, "manual": rep["manual"], "valid": bool(valid),
            "invalid_reason": reason, "valid_sensitivity": bool(valid_sens),
        })
        prev_peak = lift
    res["reps"] = rep_rows

    if clip.any():
        res["flags"].append(f"adc_rail_samples({int(clip.sum())})")
    if res["spike_samples"]:
        res["flags"].append(f"spike_samples({res['spike_samples']})")
    if not reps:
        res["flags"].append("no_reps_found")
    first_on = reps[0]["onset"] if reps else None
    last_off = reps[-1]["offset"] if reps else None
    temp, series = analyse_temperature(df["therm_mv"].to_numpy(float), t, first_on, last_off)
    res["flags"] += temp.pop("flags")
    res["temp"] = temp
    res["plot"] = {"t": t, "x": x, "env5": env5, "env1": env1, "idle": idle, "seg": seg_info,
                   "temp_series": series}
    return res


def welch_psd_concat(x, idle):
    """Noise PSD averaged over idle runs that are long enough for one Welch segment."""
    psds, weights, f = [], [], None
    for s, e in runs_of(idle):
        if e - s >= WELCH_NPERSEG:
            f, p = welch_psd(x[s:e])
            psds.append(p)
            weights.append(e - s)
    if not psds:
        return None, None
    return f, np.average(np.array(psds), axis=0, weights=weights)


# ---------------------------------------------------------------------------------------------
# Session assembly, readiness rule and statistics
# ---------------------------------------------------------------------------------------------
def set_summary(res):
    meta, temp = res["meta"], res["temp"]
    reps = pd.DataFrame(res["reps"])
    valid = reps[reps.valid] if len(reps) else reps
    row = {
        "participant": meta["participant"], "file": meta["file"], "kind": meta["kind"],
        "index": meta["index"], "set_label": f"{meta['kind']}_{meta['index']}",
        "tag": meta["tag"], "start": meta["start"], "duration_s": res["duration_s"],
        "idle_s": res["idle_s"], "noise_rms_mv": res["noise_rms_mv"],
        "mains_notched": res["mains_notched"], "clip_frac": res["clip_frac"],
        "n_candidates": res["n_candidates"], "amp_ref_over_noise": res["amp_ref_over_noise"],
        "n_reps": len(reps), "n_valid_reps": len(valid),
    }
    for col, name in [("mdf_hz", "mdf"), ("mnf_hz", "mnf"), ("mdf_denoised_hz", "mdf_denoised"),
                      ("mdf_multitaper_hz", "mdf_multitaper"), ("mdf_lift_hz", "mdf_lift"),
                      ("mdf_highact_hz", "mdf_highact"), ("mdf_firmware_hz", "mdf_firmware")]:
        ok = len(valid) >= MIN_VALID_REPS_PER_SET
        row[f"{name}_median_hz"] = float(valid[col].median()) if ok else np.nan
    if len(valid) >= MIN_VALID_REPS_PER_SET:
        q = valid.mdf_hz.quantile([0.25, 0.75])
        row["mdf_iqr_hz"] = float(q.iloc[1] - q.iloc[0])
        row["mdf_slope_hz_per_rep"] = float(np.polyfit(valid.rep, valid.mdf_hz, 1)[0]) \
            if len(valid) >= 3 else np.nan
        row["rms_median_mv"] = float(valid.rms_mv.median())
    else:
        row.update(mdf_iqr_hz=np.nan, mdf_slope_hz_per_rep=np.nan, rms_median_mv=np.nan)
    sens = reps[reps.valid_sensitivity] if len(reps) else reps
    row["n_valid_sensitivity"] = len(sens)
    row["mdf_sensitivity_median_hz"] = float(sens.mdf_denoised_hz.median()) \
        if len(sens) >= MIN_VALID_REPS_PER_SET else np.nan
    row["invalid_reasons"] = ";".join(f"{k}x{v}" for k, v in
                                      reps.invalid_reason[reps.invalid_reason != ""]
                                      .value_counts().items()) if len(reps) else ""
    row["snr_median"] = float(reps.snr.median()) if len(reps) else np.nan
    row["interval_median_s"] = float(reps.interval_s.median()) if len(reps) > 1 else np.nan
    row["duration_median_s"] = float(reps.duration_s.median()) if len(reps) else np.nan
    row["onset_to_peak_median_s"] = float(reps.onset_to_peak_s.median()) if len(reps) else np.nan
    for k in ["temp_valid", "temp_valid_frac", "temp_dropout_frac", "temp_median_c", "temp_pre_c",
              "temp_pre_source",
              "temp_pre_n_blocks", "temp_pre_sd_c", "temp_during_c", "temp_post_c", "temp_dip_c",
              "temp_slope_c_per_min"]:
        row[k] = temp.get(k, np.nan)
    row["flags"] = ";".join(res["flags"])
    return row


def pooled_residual_sd(reps, set_ids):
    """Rep-to-rep SD of MDF after removing each set's mean and linear fatigue trend, pooled."""
    ss, dof = 0.0, 0
    for _, g in reps[reps.file.isin(set_ids) & reps.valid].groupby("file"):
        y = g.mdf_hz.to_numpy()
        if len(y) >= 3:
            coef = np.polyfit(g.rep.to_numpy(), y, 1)
            ss += np.sum((y - np.polyval(coef, g.rep.to_numpy())) ** 2)
            dof += len(y) - 2
        elif len(y) == 2:
            ss += np.sum((y - y.mean()) ** 2)
            dof += 1
    return (math.sqrt(ss / dof), dof) if dof > 0 else (np.nan, 0)


def mdc95_of_change(sd, n_a, n_b):
    """95% minimal detectable change between two set medians (median SE ~ 1.253 SD / sqrt n)."""
    if not (np.isfinite(sd) and n_a > 0 and n_b > 0):
        return np.nan
    return 1.96 * math.sqrt((1.253 * sd) ** 2 / n_a + (1.253 * sd) ** 2 / n_b)


def build_sessions(sets, reps, ratings):
    """Adds session order, references, % changes, rule flags; returns (sets, participants, info)."""
    sets = sets.sort_values(["participant", "start"], na_position="last").copy()
    sets["set_order"] = sets.groupby("participant").cumcount() + 1

    heavy_ids = sets[sets.kind == "rep"].file
    light_ids = sets[sets.kind == "wu"].file
    sd_heavy, dof_heavy = pooled_residual_sd(reps, heavy_ids)
    sd_light, dof_light = pooled_residual_sd(reps, light_ids)
    pre = sets[sets.temp_valid.fillna(False).astype(bool) & sets.temp_pre_sd_c.notna()]
    sd_temp = float(np.sqrt(np.mean(pre.temp_pre_sd_c ** 2))) if len(pre) else np.nan
    info = {"sd_mdf_heavy_hz": sd_heavy, "dof_heavy": dof_heavy,
            "sd_mdf_light_hz": sd_light, "dof_light": dof_light, "sd_temp_block_c": sd_temp}

    cols = ["minutes_since_cold", "ref_label", "mdf_pct_vs_ref", "mdc_pct", "mdf_met",
            "delta_t_c", "mdc_temp_c", "temp_met", "both_met", "mdf_plateau", "temp_plateau"]
    for c in cols:
        sets[c] = np.nan
    for c in ["ref_label", "mdf_met", "temp_met", "both_met", "mdf_plateau", "temp_plateau"]:
        sets[c] = pd.Series([None] * len(sets), index=sets.index, dtype=object)
    part_rows = []
    for pid, g in sets.groupby("participant", sort=True):
        g = g.sort_values("set_order")
        heavy = g[g.kind == "rep"].sort_values("index")
        light = g[g.kind == "wu"].sort_values("index")
        cold = heavy.iloc[0] if len(heavy) else None
        warm = heavy.iloc[-1] if len(heavy) > 1 else None
        t0 = cold.start if cold is not None and pd.notna(cold.start) else g.start.dropna().min()
        if pd.notna(t0):
            sets.loc[g.index, "minutes_since_cold"] = [
                (s - t0).total_seconds() / 60 if pd.notna(s) else np.nan for s in g.start]

        # Temperature baseline: pre-set temperature of the cold heavy set.
        base_src = "cold_set"
        baseline = cold.temp_pre_c if cold is not None else np.nan
        base_n = cold.temp_pre_n_blocks if cold is not None else 0
        if not np.isfinite(baseline):
            base_src = "none"
            later = g[g.temp_pre_c.notna()]
            if len(later):
                baseline, base_n = later.iloc[0].temp_pre_c, later.iloc[0].temp_pre_n_blocks
                base_src = f"fallback_{later.iloc[0].set_label}"

        # References for MDF: cold heavy set for heavy sets, wu_1 for light sets (H1 as written).
        refs = {"rep": cold, "wu": light.iloc[0] if len(light) else None}
        prev = {"rep": None, "wu": None}
        prev_temp = None
        for idx, s in g.iterrows():
            ref = refs[s.kind]
            sd = sd_heavy if s.kind == "rep" else sd_light
            if ref is not None and s.file != ref.file and np.isfinite(ref.mdf_median_hz) \
                    and np.isfinite(s.mdf_median_hz):
                pct = 100 * (s.mdf_median_hz / ref.mdf_median_hz - 1)
                mdc = 100 * mdc95_of_change(sd, ref.n_valid_reps, s.n_valid_reps) / ref.mdf_median_hz
                sets.loc[idx, ["ref_label", "mdf_pct_vs_ref", "mdc_pct"]] = [ref.set_label, pct, mdc]
                sets.loc[idx, "mdf_met"] = pct >= max(MDF_RISE_THRESHOLD_PCT,
                                                      mdc if np.isfinite(mdc) else 0)
                p = prev[s.kind]
                if p is not None and np.isfinite(p) and np.isfinite(mdc):
                    sets.loc[idx, "mdf_plateau"] = abs(pct - p) < mdc
                prev[s.kind] = pct
            elif ref is not None and s.file == ref.file:
                prev[s.kind] = 0.0 if np.isfinite(s.mdf_median_hz) else None
            if np.isfinite(baseline) and np.isfinite(s.temp_pre_c) and base_src != "none":
                dt = s.temp_pre_c - baseline
                se_b = sd_temp / math.sqrt(max(base_n, 1))
                se_s = sd_temp / math.sqrt(max(s.temp_pre_n_blocks, 1))
                mdc_t = 1.96 * math.sqrt(se_b ** 2 + se_s ** 2) if np.isfinite(sd_temp) else np.nan
                sets.loc[idx, ["delta_t_c", "mdc_temp_c"]] = [dt, mdc_t]
                sets.loc[idx, "temp_met"] = dt >= max(TEMP_RISE_THRESHOLD_C,
                                                      mdc_t if np.isfinite(mdc_t) else 0)
                if prev_temp is not None and np.isfinite(mdc_t):
                    sets.loc[idx, "temp_plateau"] = abs(dt - prev_temp) < mdc_t
                prev_temp = dt
        gg = sets.loc[g.index]
        both = gg.mdf_met.eq(True) & gg.temp_met.eq(True)
        sets.loc[g.index, "both_met"] = both
        rule_set = gg[both].sort_values("set_order").set_label.iloc[0] if both.any() else ""

        # Probe settling: most of the total rise already there at the first set after cold.
        settling = ""
        dts = gg[gg.delta_t_c.notna() & (gg.file != (cold.file if cold is not None else ""))]
        if len(dts):
            total = dts.sort_values("set_order").delta_t_c.iloc[-1]
            first = dts.sort_values("set_order").delta_t_c.iloc[0]
            if total >= SETTLING_MIN_RISE_C and first >= SETTLING_FIRST_STEP_FRACTION * total \
                    and len(dts) > 1:
                settling = "possible_probe_settling"

        row = {"participant": pid, "n_files_used": len(g), "baseline_temp_c": baseline,
               "baseline_source": base_src, "temp_flag": settling, "rule_ready_set": rule_set}
        if cold is not None:
            row.update(cold_set=cold.set_label, cold_file=cold.file,
                       mdf_cold_hz=cold.mdf_median_hz, n_valid_cold=cold.n_valid_reps,
                       mnf_cold_hz=cold.mnf_median_hz, interval_cold_s=cold.interval_median_s,
                       noise_cold_mv=cold.noise_rms_mv, rms_cold_mv=cold.rms_median_mv)
        if warm is not None:
            w = sets.loc[warm.name]
            row.update(warm_set=warm.set_label, warm_file=warm.file,
                       mdf_warm_hz=warm.mdf_median_hz, n_valid_warm=warm.n_valid_reps,
                       mnf_warm_hz=warm.mnf_median_hz, interval_warm_s=warm.interval_median_s,
                       noise_warm_mv=warm.noise_rms_mv, rms_warm_mv=warm.rms_median_mv,
                       mdf_pct=w.mdf_pct_vs_ref, mdc_pct=w.mdc_pct,
                       mdf_change_exceeds_mdc=(abs(w.mdf_pct_vs_ref) > w.mdc_pct)
                       if np.isfinite(w.mdf_pct_vs_ref) and np.isfinite(w.mdc_pct) else np.nan,
                       delta_t_warm_c=w.delta_t_c if base_src == "cold_set" else np.nan,
                       delta_t_warm_idle_only_c=w.delta_t_c if base_src == "cold_set" and
                       cold.temp_pre_source == "idle" and warm.temp_pre_source == "idle" else np.nan)
            if cold is not None:
                row["mnf_pct"] = pct_change(cold.mnf_median_hz, warm.mnf_median_hz)
                row["mdf_pct_sensitivity"] = pct_change(cold.mdf_sensitivity_median_hz,
                                                        warm.mdf_sensitivity_median_hz)
                row["mdf_pct_lift"] = pct_change(cold.mdf_lift_median_hz, warm.mdf_lift_median_hz)
                row["mdf_pct_highact"] = pct_change(cold.mdf_highact_median_hz,
                                                    warm.mdf_highact_median_hz)
                row["rms_pct"] = pct_change(cold.rms_median_mv, warm.rms_median_mv)
                row["interval_change_s"] = warm.interval_median_s - cold.interval_median_s
                row["noise_pct"] = pct_change(cold.noise_rms_mv, warm.noise_rms_mv)
        if len(light) > 1:
            last = sets.loc[light.iloc[-1].name]
            row.update(light_first_set=light.iloc[0].set_label, light_last_set=last.set_label,
                       light_mdf_pct=last.mdf_pct_vs_ref, light_mdc_pct=last.mdc_pct)
        if ratings is not None:
            rated = ratings.get(pid, "")
            row["rated_set"] = rated
            order = list(g.set_label)
            if rated and rule_set and rated in order and rule_set in order:
                diff = order.index(rule_set) - order.index(rated)
                row.update(rule_minus_rated_sets=diff, agree_exact=diff == 0,
                           agree_within_1=abs(diff) <= 1)
        part_rows.append(row)
    return sets, pd.DataFrame(part_rows), info


def pct_change(a, b):
    return 100 * (b / a - 1) if np.isfinite(a) and np.isfinite(b) and a != 0 else np.nan


def signed_rank_cdf(n):
    """Exact null CDF of the Wilcoxon signed-rank statistic W+ for n untied, non-zero pairs."""
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1
    for k in range(1, n + 1):
        new = counts.copy()
        new[k:] += counts[:-k]
        counts = new
    return np.cumsum(counts) / 2 ** n


def hodges_lehmann_ci(d, alpha=ALPHA):
    """Hodges-Lehmann estimate and the exact Wilcoxon (Walsh-average) confidence interval."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    i, j = np.triu_indices(n)
    walsh = np.sort((d[i] + d[j]) / 2)
    est = float(np.median(walsh))
    cdf = signed_rank_cdf(n)
    ok = np.flatnonzero(cdf <= alpha / 2)
    if len(ok) == 0:
        return est, np.nan, np.nan, np.nan
    c = int(ok[-1]) + 1  # CI = [W_(c), W_(M+1-c)] in 1-based order statistics
    m = len(walsh)
    return est, float(walsh[c - 1]), float(walsh[m - c]), float(1 - 2 * cdf[c - 1])


def wilcoxon_p(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.all(d == 0):
        return np.nan
    try:
        return float(stats.wilcoxon(d, method="exact" if len(d) <= 25 and
                                    len(np.unique(np.abs(d))) == len(d) and np.all(d != 0)
                                    else "auto").pvalue)
    except ValueError:
        return np.nan


def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[k]))
        out[k] = running
    return out


def paired_outcomes(parts):
    rows = []
    specs = [
        ("MDF % change (primary: whole rep)", "mdf_pct", "%"),
        ("MDF % change, sensitivity: SNR>=2, noise-subtracted", "mdf_pct_sensitivity", "%"),
        ("MDF % change, sensitivity: 500 ms lifting window", "mdf_pct_lift", "%"),
        ("MDF % change, sensitivity: high-activation samples", "mdf_pct_highact", "%"),
        ("MNF % change", "mnf_pct", "%"),
        ("RMS % change (effort)", "rms_pct", "%"),
        ("Inter-rep interval change", "interval_change_s", "s"),
        ("Noise RMS % change", "noise_pct", "%"),
        ("Skin dT at warm set", "delta_t_warm_c", "C"),
        ("Skin dT at warm set, sensitivity: idle baselines only", "delta_t_warm_idle_only_c", "C"),
        ("Light-set MDF % change, last wu vs wu_1 (H1 as written; low SNR)", "light_mdf_pct", "%"),
    ]
    for label, col, unit in specs:
        d = parts[col].to_numpy(float) if col in parts else np.array([])
        d = d[np.isfinite(d)]
        est, lo, hi, cov = hodges_lehmann_ci(d)
        rows.append({"outcome": label, "column": col, "unit": unit, "n": len(d),
                     "n_increase": int((d > 0).sum()), "n_decrease": int((d < 0).sum()),
                     "median": float(np.median(d)) if len(d) else np.nan,
                     "hodges_lehmann": est, "ci_low": lo, "ci_high": hi, "ci_coverage": cov,
                     "wilcoxon_p": wilcoxon_p(d)})
    out = pd.DataFrame(rows)
    out["holm_p_secondary"] = np.nan
    sec = out.index[1:]
    out.loc[sec, "holm_p_secondary"] = holm(out.loc[sec, "wilcoxon_p"].to_numpy())
    return out


def mixed_models(reps, sets, parts):
    """log(MDF) ~ condition * rep_index (+ confounds), random intercept per participant."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return "statsmodels not installed; mixed models skipped.\n"
    cold = dict(zip(parts.participant, parts.get("cold_file", pd.Series(dtype=str))))
    warm = dict(zip(parts.participant, parts.get("warm_file", pd.Series(dtype=str))))
    d = reps[reps.valid].copy()
    d["condition"] = np.where(d.file.map(lambda f: f in set(cold.values())), "cold",
                              np.where(d.file.map(lambda f: f in set(warm.values())), "warm", ""))
    d = d[d.condition != ""].copy()
    d = d[d.participant.isin(parts.participant[parts.warm_file.notna()])] \
        if "warm_file" in parts else d.iloc[:0]
    if d.participant.nunique() < 3:
        return "Fewer than 3 participants with valid cold and warm reps; mixed models skipped.\n"
    d["log_mdf"] = np.log(d.mdf_hz)
    d["rep_index"] = d.rep - 1
    d["interval_s"] = d.groupby("file").interval_s.transform(lambda s: s.fillna(s.median()))
    d["interval_s"] = d.interval_s.fillna(d.interval_s.median())
    cold_rms = d[d.condition == "cold"].groupby("participant").rms_mv.median()
    d["rms_norm"] = d.rms_mv / d.participant.map(cold_rms)
    d["noise_rms_mv"] = d.file.map(dict(zip(sets.file, sets.noise_rms_mv)))
    # Only participants with valid reps in both conditions, as in the paired analysis.
    d = d[d.participant.isin(parts.participant[parts.mdf_pct.notna()])]
    out = []
    formulas = [
        ("Primary", "log_mdf ~ C(condition, Treatment('cold')) * rep_index"),
        ("Confound check", "log_mdf ~ C(condition, Treatment('cold')) * rep_index + interval_s"
                           " + rms_norm + noise_rms_mv"),
    ]
    for name, formula in formulas:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cols = re.findall(r"[a-z_]+(?=\s*(?:\+|\*|$))", formula.split("~")[1]) + ["log_mdf"]
                dd = d.dropna(subset=[c for c in cols if c in d.columns])
                fit = smf.mixedlm(formula, dd, groups=dd["participant"]).fit(reml=True)
            term = "C(condition, Treatment('cold'))[T.warm]"
            b, se = fit.params[term], fit.bse[term]
            with pd.option_context("display.width", 200, "display.max_columns", 20):
                table = fit.summary().tables[1].to_string()
            out.append(f"{name}: {formula}  (n reps={len(dd)}, participants={dd.participant.nunique()})\n"
                       f"  warm vs cold at first rep: {100 * (math.exp(b) - 1):+.1f}% "
                       f"(95% CI {100 * (math.exp(b - 1.96 * se) - 1):+.1f} to "
                       f"{100 * (math.exp(b + 1.96 * se) - 1):+.1f}%), Wald p={fit.pvalues[term]:.3f}\n"
                       + "\n".join("    " + line for line in table.splitlines())
                       + "\n")
        except Exception as err:  # noqa: BLE001 - report, don't crash the whole analysis
            out.append(f"{name}: failed to fit ({err})\n")
    out.append("Wald p-values with this few participants are optimistic; read the CIs and the "
               "paired Wilcoxon results first.\n")
    return "\n".join(out)


def threshold_sweep(parts):
    rows = []
    if "mdf_pct" not in parts or "delta_t_warm_c" not in parts:
        return pd.DataFrame()
    p = parts[parts.mdf_pct.notna() & parts.delta_t_warm_c.notna()]
    for m in SWEEP_MDF_PCT:
        for tc in SWEEP_TEMP_C:
            ready = (p.mdf_pct >= m) & (p.delta_t_warm_c >= tc)
            rows.append({"mdf_threshold_pct": m, "temp_threshold_c": tc, "n": len(p),
                         "n_ready": int(ready.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def style(ax, title=None, ylabel=None, xlabel=None):
    ax.set_facecolor(COLOR_SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT_2, labelsize=8)
    ax.grid(True, color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=COLOR_TEXT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=COLOR_TEXT_2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=COLOR_TEXT_2, fontsize=9)


def legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=COLOR_TEXT_2, **kw)
    return leg


def qc_figure(res, path):
    P = res["plot"]
    t, x = P["t"], P["x"]
    noise = res["noise_rms_mv"]
    reps = res["reps"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, facecolor=COLOR_SURFACE,
                             gridspec_kw={"height_ratios": [1.2, 1.2, 0.8]})
    ax = axes[0]
    ax.plot(t, x, color=COLOR_TRACE, linewidth=0.4)
    for s, e in runs_of(P["idle"]):
        ax.axvspan(s / FS_HZ, e / FS_HZ, color=COLOR_NEUTRAL, alpha=0.15, linewidth=0)
    for r in reps:
        c = COLOR_1 if r["valid"] else COLOR_NEUTRAL
        ax.axvspan(r["onset_s"], r["offset_s"], color=c, alpha=0.10, linewidth=0)
        ax.axvspan(r["window_start_s"], r["window_end_s"], color=c, alpha=0.28, linewidth=0)
    meta = res["meta"]
    style(ax, f"{meta['file']}   noise {noise:.0f} mV   "
              f"{sum(r['valid'] for r in reps)}/{len(reps)} valid reps"
              + (f"   flags: {';'.join(res['flags'])}" if res["flags"] else ""),
          "Filtered EMG (mV)")

    ax = axes[1]
    ax.plot(t, P["env5"] / noise, color=COLOR_1, linewidth=1.0, label="Envelope, 5 Hz (boundaries)")
    ax.plot(t, P["env1"] / noise, color=COLOR_2, linewidth=1.5, label="Envelope, 1 Hz (rep count)")
    seg = P["seg"]
    if "boundary_thr" in seg:
        ax.axhline(seg["boundary_thr"] / noise, color=COLOR_TEXT_2, linewidth=0.8, linestyle="--",
                   label="Onset/offset threshold")
    if np.isfinite(seg["A"]):
        ax.axhline(REP_KEEP_FRACTION_OF_A * seg["A"] / noise, color=COLOR_TEXT_2, linewidth=0.8,
                   linestyle=":", label=f"Keep threshold ({REP_KEEP_FRACTION_OF_A:.0%} of A)")
    idx, kept = seg["idx"], seg["kept"]
    if len(idx):
        ax.plot(t[idx[kept]], P["env1"][idx[kept]] / noise, "o", color=COLOR_2, markersize=6,
                markeredgecolor=COLOR_SURFACE, label="Rep peak (kept)")
        if (~kept).any():
            ax.plot(t[idx[~kept]], P["env1"][idx[~kept]] / noise, "o", markersize=6,
                    markerfacecolor="none", markeredgecolor=COLOR_TEXT_2, label="Dropped burst")
    top = ax.get_ylim()[1]
    for r in reps:
        ax.text(r["lift_peak_s"], top * 0.97, f"{r['rep']}\n{r['mdf_hz']:.0f} Hz\nSNR {r['snr']:.1f}",
                ha="center", va="top", fontsize=7,
                color=COLOR_TEXT if r["valid"] else COLOR_NEUTRAL)
    style(ax, None, "Envelope (x noise RMS)")
    legend(ax, loc="center right")

    ax = axes[2]
    ts = P["temp_series"]
    if len(ts):
        good = ts[ts.valid]
        ax.plot(good.t_s, good.temp_c, "o-", color=COLOR_1, markersize=3, linewidth=1.2)
        for _, b in ts[~ts.valid].iterrows():
            ax.axvspan(b.t_s - TEMP_BLOCK_S / 2, b.t_s + TEMP_BLOCK_S / 2, color=COLOR_NEUTRAL,
                       alpha=0.3, linewidth=0)
    temp = res["temp"]
    note = "valid" if temp.get("temp_valid") else "NOT USED"
    style(ax, f"Skin temperature (1 s trimmed means of therm_mv; grey = rejected blocks): {note}",
          "Temp (°C)", "Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=COLOR_SURFACE)
    plt.close(fig)


def participant_figure(pid, sets, reps, plots, path):
    g = sets[sets.participant == pid].sort_values("set_order")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), facecolor=COLOR_SURFACE)
    ax = axes[0]
    labels = []
    shown = set()
    for pos, (_, s) in enumerate(g.iterrows()):
        labels.append(s.set_label)
        r = reps[reps.file == s.file]
        color = COLOR_1 if s.kind == "rep" else COLOR_2
        jitter = np.linspace(-0.15, 0.15, max(len(r), 1))[:len(r)]
        v = r.valid.to_numpy()
        lab = "Heavy set rep" if s.kind == "rep" else "Light set rep"
        if v.any():
            ax.plot(pos + jitter[v], r.mdf_hz[v], "o", color=color, markersize=6,
                    markeredgecolor=COLOR_SURFACE, label=None if lab in shown else lab)
            shown.add(lab)
        if (~v).any():
            ax.plot(pos + jitter[~v], r.mdf_hz[~v], "o", markersize=6, markerfacecolor="none",
                    markeredgecolor=COLOR_NEUTRAL,
                    label=None if "gated" in shown else f"Rep below SNR {SNR_GATE:g} (not used)")
            shown.add("gated")
        if np.isfinite(s.mdf_median_hz):
            ax.plot([pos - 0.25, pos + 0.25], [s.mdf_median_hz] * 2, color=COLOR_TEXT, linewidth=2,
                    label=None if "med" in shown else "Set median (valid reps)")
            shown.add("med")
            if np.isfinite(s.mdf_pct_vs_ref):
                ax.text(pos + 0.28, s.mdf_median_hz, f"{s.mdf_pct_vs_ref:+.0f}%", fontsize=8,
                        va="center", color=COLOR_TEXT_2)
    ax.set_xticks(range(len(labels)), labels)
    style(ax, f"{pid}: median frequency per rep (lifting-phase window)", "MDF (Hz)")
    legend(ax, loc="best")

    ax = axes[1]
    base = None
    pp = g[g.delta_t_c.notna()]
    if len(pp):
        base = (pp.temp_pre_c - pp.delta_t_c).iloc[0]
    for _, s in g.iterrows():
        ts = plots.get(s.file)
        if ts is None or base is None or not s.temp_valid or not np.isfinite(s.minutes_since_cold):
            continue
        good = ts[ts.valid]
        ax.plot(s.minutes_since_cold + good.t_s / 60, good.temp_c - base, color=COLOR_NEUTRAL,
                linewidth=1)
    if len(pp):
        tx = pp.minutes_since_cold + pp.temp_pre_n_blocks / 120.0
        ax.plot(tx, pp.delta_t_c, "o-", color=COLOR_1, markersize=7, linewidth=2,
                markeredgecolor=COLOR_SURFACE, label="Pre-set temperature")
        for xx, yy, lab in zip(tx, pp.delta_t_c, pp.set_label):
            ax.annotate(lab, (xx, yy), textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=8, color=COLOR_TEXT_2)
        ax.plot([], [], color=COLOR_NEUTRAL, linewidth=1, label="1 s trace during each file")
    ax.axhline(TEMP_RISE_THRESHOLD_C, color=COLOR_TEXT_2, linestyle="--", linewidth=0.8,
               label=f"H1 threshold +{TEMP_RISE_THRESHOLD_C} °C")
    ax.axhline(0, color=COLOR_GRID, linewidth=0.8)
    style(ax, f"{pid}: skin temperature change from cold-set baseline", "ΔT (°C)",
          "Minutes since cold heavy set started")
    legend(ax, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=COLOR_SURFACE)
    plt.close(fig)


def dodge(y, min_gap_frac=0.035):
    """Nudge label positions apart so names at the line ends do not overlap."""
    y = np.asarray(y, float)
    if len(y) < 2:
        return y
    gap = min_gap_frac * (np.nanmax(y) - np.nanmin(y) or 1.0)
    order = np.argsort(y)
    out = y.copy()
    for k in range(1, len(order)):
        out[order[k]] = max(out[order[k]], out[order[k - 1]] + gap)
    shift = (out[order].mean() - y[order].mean())
    return out - shift


def group_figures(sets, parts, sweep, outdir, primary):
    if "warm_file" not in parts:
        return
    p = parts[parts.warm_file.notna()]
    if len(p) == 0:
        return
    specs = [("MDF (Hz)", "mdf_cold_hz", "mdf_warm_hz"),
             ("Inter-rep interval (s)", "interval_cold_s", "interval_warm_s"),
             ("Noise RMS (mV)", "noise_cold_mv", "noise_warm_mv")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor=COLOR_SURFACE)
    for ax, (lab, a, b) in zip(axes, specs):
        q = p[p[a].notna() & p[b].notna()]
        for _, r in q.iterrows():
            ax.plot([0, 1], [r[a], r[b]], color=COLOR_NEUTRAL, linewidth=1, marker="o",
                    markersize=5)
        for y, name in zip(dodge(q[b].to_numpy(float)), q.participant):
            ax.text(1.04, y, name, fontsize=7, va="center", color=COLOR_TEXT_2)
        if len(q):
            ax.plot([0, 1], [q[a].median(), q[b].median()], color=COLOR_1, linewidth=2.5,
                    marker="o", markersize=8, label="Median")
            legend(ax, loc="upper left")
        ax.set_xticks([0, 1], ["Cold heavy set", "Warm heavy set"])
        ax.set_xlim(-0.2, 1.35)
        style(ax, f"{lab}  (n={len(q)})", lab)
    hl = primary.iloc[0]
    fig.suptitle(f"Cold vs warm heavy sets. MDF change: Hodges–Lehmann {hl.hodges_lehmann:+.1f}% "
                 f"(95% CI {hl.ci_low:+.1f} to {hl.ci_high:+.1f}), {hl.n_increase}/{hl.n} increased",
                 x=0.01, ha="left", fontsize=10, color=COLOR_TEXT)
    fig.tight_layout()
    fig.savefig(outdir / "group_paired_cold_vs_warm.png", dpi=110, facecolor=COLOR_SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=COLOR_SURFACE)
    ax = axes[0]
    heavy = sets[(sets.kind == "rep")].copy()
    for pid, g in heavy.groupby("participant"):
        g = g.sort_values("index")
        y = [0.0 if i == 0 else v for i, v in enumerate(g.mdf_pct_vs_ref)]
        if np.isfinite(g.mdf_median_hz.iloc[0]):
            ax.plot(range(1, len(g) + 1), y, color=COLOR_NEUTRAL, linewidth=1, marker="o",
                    markersize=4)
    ax.axhline(MDF_RISE_THRESHOLD_PCT, color=COLOR_TEXT_2, linestyle="--", linewidth=0.8,
               label=f"H1 threshold +{MDF_RISE_THRESHOLD_PCT:g}%")
    ax.axhline(0, color=COLOR_GRID)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    style(ax, "MDF change across heavy sets (one line per participant)", "MDF vs cold set (%)",
          "Heavy set number")
    legend(ax, loc="best")
    ax = axes[1]
    for pid, g in sets[sets.delta_t_c.notna()].groupby("participant"):
        g = g.sort_values("set_order")
        ax.plot(g.minutes_since_cold, g.delta_t_c, color=COLOR_NEUTRAL, linewidth=1, marker="o",
                markersize=4)
    ax.axhline(TEMP_RISE_THRESHOLD_C, color=COLOR_TEXT_2, linestyle="--", linewidth=0.8,
               label=f"H1 threshold +{TEMP_RISE_THRESHOLD_C} °C")
    ax.axhline(0, color=COLOR_GRID)
    style(ax, "Pre-set skin temperature change (one line per participant)", "ΔT (°C)",
          "Minutes since cold heavy set started")
    legend(ax, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "group_trajectories.png", dpi=110, facecolor=COLOR_SURFACE)
    plt.close(fig)

    if len(sweep) and sweep.n.iloc[0] > 0:
        grid = sweep.pivot(index="temp_threshold_c", columns="mdf_threshold_pct", values="n_ready")
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=COLOR_SURFACE)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
        im = ax.imshow(grid.values, origin="lower", aspect="auto", cmap=cmap,
                       vmin=0, vmax=sweep.n.iloc[0])
        ax.set_xticks(range(grid.shape[1]), [f"{v:g}" for v in grid.columns])
        ax.set_yticks(range(grid.shape[0]), [f"{v:g}" for v in grid.index])
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid.values[i, j]
                ax.text(j, i, int(v), ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 * sweep.n.iloc[0] else COLOR_TEXT)
        style(ax, f"Participants classed ready at the warm set (n={sweep.n.iloc[0]}), exploratory",
              "Skin ΔT threshold (°C)", "MDF rise threshold (%)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="Participants ready")
        fig.tight_layout()
        fig.savefig(outdir / "threshold_sweep.png", dpi=110, facecolor=COLOR_SURFACE)
        plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def choose_files(paths, ov):
    """Parse names, apply include/exclude/use_as, return (metas, skipped rows)."""
    metas, skipped = [], []
    for p in paths:
        meta = parse_filename(p)
        if meta is None:
            skipped.append({"file": Path(p).name, "reason": "name does not match pattern"})
            continue
        rows = overrides_for(ov, meta["file"])
        actions = set(rows.action)
        whole_excl = rows[(rows.action == "exclude") & (rows.rep.str.strip() == "")]
        if len(whole_excl):
            skipped.append({"file": meta["file"], "reason": "excluded by override"})
            continue
        use_as = rows[rows.action == "use_as"]
        if len(use_as):
            m = re.match(r"^(rep|wu)_(\d+)$", use_as.role.iloc[0].strip())
            if not m:
                skipped.append({"file": meta["file"], "reason": "bad use_as role"})
                continue
            meta["kind"], meta["index"] = m.group(1), int(m.group(2))
            meta["use_as"] = True
        elif (meta["tag"] or meta["start"] is None) and "include" not in actions:
            why = f"tag '{meta['tag']}'" if meta["tag"] else "no timestamp"
            skipped.append({"file": meta["file"], "reason": f"excluded by default ({why}); "
                                                             "add an include override to use it"})
            continue
        metas.append(meta)
    # A use_as file replaces the file that had that role.
    roles = {}
    for m in metas:
        key = (m["participant"], m["kind"], m["index"])
        if key in roles:
            old = roles[key]
            loser = old if m.get("use_as") else m
            winner = m if loser is old else old
            roles[key] = winner
            skipped.append({"file": loser["file"],
                            "reason": f"replaced by {winner['file']} for {key[1]}_{key[2]}"})
        else:
            roles[key] = m
    return list(roles.values()), skipped


def inclusion_report(sets, parts):
    """Which participants each outcome uses, and the reason every other one is missing."""
    by_file = sets.set_index("file")

    def set_reason(file, label):
        if not isinstance(file, str) or file not in by_file.index:
            return f"no {label} file"
        r = by_file.loc[file]
        why = r.invalid_reasons if isinstance(r.invalid_reasons, str) and r.invalid_reasons \
            else "too few reps found"
        return f"{r.set_label}: {int(r.n_valid_reps)}/{int(r.n_reps)} usable reps ({why})"

    def temp_reason(file, label):
        if not isinstance(file, str) or file not in by_file.index:
            return f"no {label} file"
        r = by_file.loc[file]
        fl = [f for f in str(r["flags"]).split(";") if f.startswith(("temp", "therm", "no_pre"))]
        return f"{r.set_label} temperature: {', '.join(fl) or 'no clean idle before the set'}"

    def light_reason(r):
        g = sets[(sets.participant == r.participant) & (sets.kind == "wu")].sort_values("index")
        if len(g) < 2:
            return "fewer than two light sets"
        bad = [set_reason(f, "light") for f, n in [(g.file.iloc[0], g.n_valid_reps.iloc[0]),
                                                    (g.file.iloc[-1], g.n_valid_reps.iloc[-1])]
               if n < MIN_VALID_REPS_PER_SET]
        return "; ".join(bad)

    lines = [f"Participant inclusion (participants with data: {len(parts)})"]
    checks = [
        ("MDF cold vs warm (primary)", "mdf_pct",
         lambda r: "; ".join(x for x in [
             set_reason(r.get("cold_file"), "cold") if not (r.get("n_valid_cold", 0) >= MIN_VALID_REPS_PER_SET) else "",
             set_reason(r.get("warm_file"), "warm") if not (r.get("n_valid_warm", 0) >= MIN_VALID_REPS_PER_SET) else ""]
             if x)),
        ("MDF sensitivity (SNR>=2)", "mdf_pct_sensitivity", lambda r: "fewer than 2 reps pass even the looser gate"),
        ("Skin dT at warm set", "delta_t_warm_c",
         lambda r: temp_reason(r.get("cold_file"), "cold") if r.get("baseline_source") != "cold_set"
         else temp_reason(r.get("warm_file"), "warm")),
        ("Light-set MDF (H1 as written)", "light_mdf_pct", lambda r: light_reason(r)),
    ]
    for label, col, why in checks:
        if col not in parts:
            continue
        ok = parts[col].notna()
        lines.append(f"  {label}: n = {int(ok.sum())} of {len(parts)}")
        for _, r in parts[~ok].iterrows():
            lines.append(f"      - {r.participant}: {why(r)}")
    return lines


def write_stats(path, info, primary, mm_text, sets, parts, skipped, sweep):
    const = {k: v for k, v in globals().items() if k.isupper() and not k.startswith("COLOR")
             and k not in {"FILENAME_RE", "SEQ_BLUE"}}
    lines = ["ReadyRep analysis", "=" * 60, ""]
    lines.append(f"Files analysed: {len(sets)}   participants: {sets.participant.nunique()}   "
                 f"skipped: {len(skipped)}")
    lines.append(f"Reps found: {int(sets.n_reps.sum())}   valid (SNR >= {SNR_GATE:g}): "
                 f"{int(sets.n_valid_reps.sum())}")
    by_kind = sets.groupby("kind").agg(files=("file", "count"), reps=("n_reps", "sum"),
                                       valid=("n_valid_reps", "sum"), snr=("snr_median", "median"))
    lines += ["", "By set type (snr = median of per-set median SNR):", by_kind.to_string(), ""]
    lines.append("Measurement noise")
    lines.append(f"  pooled rep-to-rep MDF SD, heavy sets (detrended): {info['sd_mdf_heavy_hz']:.2f} Hz "
                 f"(dof {info['dof_heavy']})")
    lines.append(f"  pooled rep-to-rep MDF SD, light sets (detrended): {info['sd_mdf_light_hz']:.2f} Hz "
                 f"(dof {info['dof_light']})")
    lines.append(f"  pooled SD of 1 s temperature blocks before the set (detrended): {info['sd_temp_block_c']:.3f} °C "
                 "(blocks are autocorrelated, so the temperature MDC is optimistic)")
    if "mdc_pct" in parts and parts.mdc_pct.notna().any():
        lines.append(f"  MDC95 for cold->warm MDF change: median {parts.mdc_pct.median():.1f}% "
                     f"(range {parts.mdc_pct.min():.1f}-{parts.mdc_pct.max():.1f}%). "
                     f"A single person's change below this cannot be told apart from noise.")
    lines += [""] + inclusion_report(sets, parts)
    lines += ["", "Paired cold vs warm heavy sets (participants with both sets valid)",
              "  Hodges-Lehmann estimate with exact Wilcoxon CI; Holm correction on secondary outcomes",
              primary.to_string(index=False, float_format=lambda v: f"{v:.3g}"), ""]
    lines += ["Mixed models (valid heavy-set reps, cold vs warm)", mm_text, ""]
    if "rule_ready_set" in parts:
        n_ready = (parts.rule_ready_set != "").sum()
        lines.append(f"Readiness rule (MDF >= max({MDF_RISE_THRESHOLD_PCT:g}%, MDC) and "
                     f"dT >= max({TEMP_RISE_THRESHOLD_C:g} °C, MDC)): met in {n_ready} of "
                     f"{len(parts)} participants at some set.")
        if "temp_flag" in parts:
            lines.append(f"Possible probe settling (most of the rise by the first set after cold): "
                         f"{(parts.temp_flag != '').sum()} participants.")
    if "agree_exact" in parts:
        a = parts[parts.agree_exact.notna()]
        lines.append(f"Agreement with rated set: exact {int(a.agree_exact.sum())}/{len(a)}, "
                     f"within one set {int(a.agree_within_1.sum())}/{len(a)}.")
    if len(sweep):
        lines += ["", "Threshold sweep (exploratory; n ready at the warm set):",
                  sweep.pivot(index="temp_threshold_c", columns="mdf_threshold_pct",
                              values="n_ready").to_string()]
    lines += ["", "Skipped files:"] + [f"  {s['file']}: {s['reason']}" for s in skipped]
    lines += ["", "Notes",
              "  * Rep speed values are EMG-timing proxies (burst timing), not bar velocity.",
              "  * Sampling times are reconstructed as sample_count/1000 by tools/emg_temp.py; any "
              "error in the true rate scales every MDF value.",
              "  * Absolute temperatures assume a 3.3 V supply and default ADC calibration; only "
              "changes are interpreted.",
              "  * Many warm-up (light) reps fail the SNR gate, so H1's wu_k vs wu_1 comparison is "
              "reported but not relied on.",
              "  * Primary MDF is computed over the whole rep. MDF moves by a median 29 Hz across one "
              "rep, so a single 500 ms window depends heavily on where it is placed; the lifting-"
              "window and high-activation results are listed above as sensitivity analyses.",
              "  * Isolated spikes (> 8 x local robust SD) are zeroed before spectra; reps with "
              f">= {SPIKE_MAX_SAMPLES} spike samples or any ADC-rail sample are excluded.",
              "  * A pre-set temperature taken from the first seconds of the set (early_set) differs "
              "from an idle one by a median +0.02 °C; the idle-only sensitivity row shows its effect.",
              "", "Constants"] + [f"  {k} = {v}" for k, v in const.items()]
    Path(path).write_text("\n".join(lines) + "\n")


def run(paths, outdir, overrides_path=None, ratings_path=None, expected_reps=None, figures=True):
    outdir = Path(outdir)
    (outdir / "figures" / "qc").mkdir(parents=True, exist_ok=True)
    (outdir / "figures" / "participants").mkdir(parents=True, exist_ok=True)
    ov = load_overrides(overrides_path)
    metas, skipped = choose_files(paths, ov)
    ratings = None
    if ratings_path:
        r = pd.read_csv(ratings_path, dtype=str).fillna("")
        ratings = dict(zip(r.participant.str.lower().str.strip(), r.rated_set.str.strip()))

    set_rows, rep_rows, qc_rows, temp_plots = [], [], [], {}
    for meta in sorted(metas, key=lambda m: (m["participant"], m["start"] or datetime.max)):
        try:
            df = load_csv(meta["path"])
            res = analyse_file(meta, df, overrides_for(ov, meta["file"]))
        except Exception as err:  # noqa: BLE001
            skipped.append({"file": meta["file"], "reason": f"failed to analyse: {err}"})
            continue
        if expected_reps and len(res["reps"]) != expected_reps and meta["kind"] == "rep":
            res["flags"].append(f"rep_count_{len(res['reps'])}_expected_{expected_reps}")
        set_rows.append(set_summary(res))
        rep_rows += res["reps"]
        temp_plots[meta["file"]] = res["plot"]["temp_series"]
        qc_rows.append({"file": meta["file"], "participant": meta["participant"],
                        "set": f"{meta['kind']}_{meta['index']}", "n_reps": len(res["reps"]),
                        "n_valid_reps": sum(r["valid"] for r in res["reps"]),
                        "temp_valid": res["temp"].get("temp_valid"),
                        "flags": ";".join(res["flags"])})
        if figures:
            qc_figure(res, outdir / "figures" / "qc" / f"{Path(meta['file']).stem}.png")
        print(f"  {meta['file']}: {len(res['reps'])} reps "
              f"({sum(r['valid'] for r in res['reps'])} valid)"
              + (f"  [{';'.join(res['flags'])}]" if res["flags"] else ""))

    if not set_rows:
        sys.exit("No files could be analysed.")
    sets = pd.DataFrame(set_rows)
    reps = pd.DataFrame(rep_rows)
    sets, parts, info = build_sessions(sets, reps, ratings)
    primary = paired_outcomes(parts)
    mm_text = mixed_models(reps, sets, parts)
    sweep = threshold_sweep(parts)

    reps.to_csv(outdir / "reps.csv", index=False, float_format="%.6g")
    sets.drop(columns=[]).to_csv(outdir / "sets.csv", index=False, float_format="%.6g")
    parts.to_csv(outdir / "participants.csv", index=False, float_format="%.6g")
    qc = pd.DataFrame(qc_rows + [{"file": s["file"], "flags": "SKIPPED: " + s["reason"]}
                                 for s in skipped])
    qc.to_csv(outdir / "qc.csv", index=False)
    primary.to_csv(outdir / "paired_outcomes.csv", index=False, float_format="%.6g")
    if len(sweep):
        sweep.to_csv(outdir / "threshold_sweep.csv", index=False)
    write_stats(outdir / "stats.txt", info, primary, mm_text, sets, parts, skipped, sweep)

    if figures:
        for pid in sets.participant.unique():
            participant_figure(pid, sets, reps, temp_plots,
                               outdir / "figures" / "participants" / f"{pid}.png")
        group_figures(sets, parts, sweep, outdir / "figures", primary)
    print(f"\nWrote results to {outdir.resolve()}")
    return sets, reps, parts


# ---------------------------------------------------------------------------------------------
# Self-test on synthetic data with known answers
# ---------------------------------------------------------------------------------------------
def synthetic_recording(seed, n_reps=8, dur=45.0):
    """Idle noise plus n_reps two-hump reps (lift, then a weaker lowering) with a known MDF."""
    rng = np.random.default_rng(seed)
    n = int(dur * FS_HZ)
    t = np.arange(n) / FS_HZ
    noise = bandpass(rng.normal(0, 1, n))
    noise *= 30 / noise.std()
    sos = signal.butter(2, [40, 110], btype="bandpass", fs=FS_HZ, output="sos")
    carrier = signal.sosfiltfilt(sos, rng.normal(0, 1, n))
    carrier /= carrier.std()
    fgrid = np.arange(0, 500.25, 0.25)
    _, h = signal.sosfreqz(sos, worN=fgrid, fs=FS_HZ)
    true_mdf, _ = mdf_mnf(fgrid, np.abs(h) ** 4)  # filtfilt squares the magnitude response
    amp = np.zeros(n)
    truth = []
    for k in range(n_reps):
        start = 6.0 + 4.0 * k
        lift = (t >= start) & (t < start + 1.0)
        amp[lift] = 300 * np.sin(np.pi * (t[lift] - start) / 1.0)
        low = (t >= start + 0.8) & (t < start + 2.4)
        amp[low] = np.maximum(amp[low], 120 * np.sin(np.pi * (t[low] - start - 0.8) / 1.6))
        truth.append((start, start + 2.4))
    emg = 1580 + noise + amp * carrier
    therm = np.full(n, 1367.0) + rng.normal(0, 35, n)
    therm[10000:11000] = 0  # one second of thermistor dropout
    spikes = rng.choice(np.arange(2000, 4000), 6, replace=False)  # electrode-pop spikes in idle
    emg[spikes] += 900
    df = pd.DataFrame({"time_s": t, "emg_mv": np.round(emg), "therm_mv": np.round(therm),
                       "temp_f": 91.4})
    return df, truth, true_mdf


def self_test(seeds=(1, 2, 3, 4, 5)):
    """Known-answer checks. MDF accuracy is judged over several random recordings, because a
    500 ms window has real estimation scatter (about +/-6 Hz per rep)."""
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok &= bool(cond)

    meta = {"file": "synthetic_rep_1_2026-01-01_00-00-00.csv", "path": "", "participant": "synthetic",
            "kind": "rep", "index": 1, "tag": "", "start": datetime(2026, 1, 1)}
    medians, mt_medians, worst_rep, counts, on_err, off_err = [], [], 0.0, [], 0.0, 0.0
    true_mdf, temp = np.nan, {}
    for seed in seeds:
        df, truth, true_mdf = synthetic_recording(seed)
        res = analyse_file(meta, df, load_overrides(None))
        reps = pd.DataFrame(res["reps"])
        counts.append(len(reps))
        if len(reps) == len(truth):
            on_err = max(on_err, np.abs(reps.onset_s.to_numpy() - [a for a, _ in truth]).max())
            off_err = max(off_err, np.abs(reps.offset_s.to_numpy() - [b for _, b in truth]).max())
        medians.append(reps.mdf_hz.median())
        mt_medians.append(reps.mdf_multitaper_hz.median())
        worst_rep = max(worst_rep, (reps.mdf_hz - true_mdf).abs().max())
        if seed == seeds[0]:
            temp = res["temp"]
            check(reps.valid.all(), "all reps valid (SNR gate passed, no artifact/clipping flags)")
    check(all(c == 8 for c in counts), f"rep count 8 in every recording ({counts})")
    check(on_err <= 0.15, f"onsets within 150 ms (worst {on_err * 1000:.0f} ms)")
    check(off_err <= 0.25, f"offsets within 250 ms (worst {off_err * 1000:.0f} ms)")
    bias = 100 * (np.mean(medians) / true_mdf - 1)
    check(abs(bias) <= 3, f"mean set-median MDF {np.mean(medians):.1f} Hz vs truth {true_mdf:.1f} Hz "
                          f"(bias {bias:+.1f}%, limit ±3%)")
    check(max(abs(m / true_mdf - 1) for m in medians) <= 0.06,
          f"every set median within ±6% ({', '.join(f'{m:.1f}' for m in medians)} Hz)")
    check(worst_rep <= 0.25 * true_mdf, f"every rep within ±25% (worst {worst_rep:.1f} Hz)")
    check(abs(np.mean(mt_medians) / true_mdf - 1) <= 0.03,
          f"multitaper mean set median {np.mean(mt_medians):.1f} Hz (±3%)")
    check(temp.get("temp_valid"), f"temperature valid ({temp.get('temp_valid_frac', 0):.0%} of blocks)")
    check(abs(temp.get("temp_median_c", 0) - therm_mv_to_c(1367)) < 0.1,
          f"temperature {temp.get('temp_median_c', np.nan):.2f} °C vs {float(therm_mv_to_c(1367)):.2f} °C")
    f = np.arange(0, 501, 4.0)
    check(abs(mdf_mnf(f, np.ones_like(f))[0] - 235) < 2.5,
          f"flat-spectrum MDF {mdf_mnf(f, np.ones_like(f))[0]:.1f} Hz vs 235 Hz")
    est, lo, hi, cov = hodges_lehmann_ci(np.arange(1, 11, dtype=float))
    check(abs(est - 5.5) < 1e-9 and lo < 5.5 < hi and cov >= 0.95, f"HL {est} CI ({lo}, {hi}) cov {cov:.3f}")
    # Exact signed-rank CDF against brute force for n = 8.
    n = 8
    sums = [sum(i + 1 for i in range(n) if m >> i & 1) for m in range(2 ** n)]
    brute = np.cumsum(np.bincount(sums, minlength=n * (n + 1) // 2 + 1)) / 2 ** n
    check(np.allclose(brute, signed_rank_cdf(n)), "signed-rank null distribution matches brute force")
    print("Self-test", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*", help="recording CSVs (rep and wu files)")
    ap.add_argument("--out", default="analysis_out", help="output folder (default: analysis_out)")
    ap.add_argument("--overrides", help="CSV of manual include/exclude/rep-window corrections")
    ap.add_argument("--ratings", help="CSV of participant,rated_set for the agreement table")
    ap.add_argument("--expected-reps", type=int, help="flag heavy sets with a different rep count")
    ap.add_argument("--no-figures", action="store_true", help="skip PNG figures (faster)")
    ap.add_argument("--self-test", action="store_true", help="run the synthetic-data checks")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not args.csv:
        ap.error("give at least one CSV file, or --self-test")
    run(args.csv, args.out, args.overrides, args.ratings, args.expected_reps, not args.no_figures)


if __name__ == "__main__":
    main()
