#!/usr/bin/env python3
"""Analyze ReadyRep recordings. Times are nominal at 1 kHz.

Overrides use file basenames and optionally rep,start_s,end_s,exclude,reviewed.
A file-only row opts a flagged file in. Bounds replace the automatic rep
with that number, including its lowering phase. Ratings use participant,ready_set
(or buzz_set), with set labels such as wu_2.
"""
import argparse
from itertools import product
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats

FS = 1000
BAND = (20, 450)
MIN_SNR = 3
PEAK_FRACTION = 0.40
MDF_TARGET, TEMP_TARGET = 10, 1.5
FILENAME = re.compile(r"(?P<participant>.+?)_(?P<kind>rep|wu)_(?P<index>\d+)(?P<suffix>.*)")
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
PROTOCOL = ["rep_1", "wu_1", "wu_2", "wu_3", "rep_2"]
NAN = float("nan")


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[::2], edges[1::2]))


def welch(values):
    return signal.welch(values, fs=FS, window="hann", nperseg=256,
                        noverlap=128, nfft=1024)


def preprocess(raw):
    emg = signal.sosfiltfilt(signal.butter(4, BAND, btype="bandpass", fs=FS,
                                         output="sos"), raw - np.median(raw))
    f, power = welch(emg)
    notches = []
    for hz in (60, 120, 180):
        distance = abs(f - hz)
        background = np.median(power[(distance >= 3) & (distance <= 12)])
        # Require a narrow local maximum at the line frequency.
        peak = np.argmax(np.where(distance <= 1, power, -1))
        if background > 0 and power[peak] > 5 * background and power[peak] == max(power[peak-2:peak+3]):
            emg = signal.filtfilt(*signal.iirnotch(hz, 30, FS), emg)
            notches.append(hz)
    rms = np.sqrt(np.maximum(ndimage.uniform_filter1d(emg**2, 100), 0))
    envelope = signal.sosfiltfilt(signal.butter(2, 5, fs=FS, output="sos"), rms)
    samples = np.arange(len(raw))
    idle = [(a, b) for a, b in runs(
        ((samples < 5*FS) | (samples >= len(raw)-5*FS))
        & (envelope <= 1.5*np.percentile(envelope, 10))) if b-a >= FS]
    idle_envelope = np.concatenate([envelope[a:b] for a, b in idle]) if idle else envelope[
        envelope <= np.percentile(envelope, 10)]
    noise = np.median(idle_envelope)
    return emg, envelope, idle, noise, np.std(idle_envelope), notches


def segment(envelope, noise, noise_sd):
    peaks, _ = signal.find_peaks(envelope, prominence=max(3*noise_sd, 1e-9), distance=FS)
    if not len(peaks):
        return [], NAN, NAN, 0, 0
    heights = envelope[peaks]
    # Estimate the reference after removing near-idle candidates, which can
    # otherwise dominate the median in a short heavy set.
    preliminary = heights >= max(0.40*np.percentile(heights, 75), noise+3*noise_sd)
    if not preliminary.any():
        return [], NAN, NAN, len(peaks), 0
    amplitude = np.median(heights[preliminary])
    on = noise + 0.25*(amplitude-noise)
    off = noise + 0.15*(amplitude-noise)
    kept = peaks[heights >= max(PEAK_FRACTION*amplitude, on, noise+3*noise_sd)]
    active = envelope >= off
    for a, b in runs(~active):
        if a and b < len(active) and b-a < 250:
            active[a:b] = True

    bouts = [(a, b, kept[(kept >= a) & (kept < b)]) for a, b in runs(active)]
    bouts = [(a, b, p) for a, b, p in bouts if len(p)]
    intervals = np.diff(kept)
    # The upper half avoids using short lift-to-lower spacing as the cycle.
    rhythm = np.median(intervals[intervals >= np.median(intervals)]) if len(intervals) >= 3 else NAN
    phase_gap = min(1.65*FS, 0.62*rhythm) if np.isfinite(rhythm) else 1.25*FS
    candidates = []
    for a, b, group in bouts:
        edges = [a]
        for left, right in zip(group, group[1:]):
            valley = left + np.argmin(envelope[left:right])
            if right-left >= phase_gap and envelope[valley] < on:
                edges.append(valley)
        edges.append(b)
        candidates.extend(zip(edges, edges[1:]))

    bouts = []
    joined = 0
    for a, b in candidates:
        if bouts:
            previous_a, previous_b = bouts[-1]
            first_peak = kept[(kept >= previous_a) & (kept < previous_b)][0]
            next_peak = kept[(kept >= a) & (kept < b)][0]
            if next_peak-first_peak < phase_gap:
                bouts[-1] = (previous_a, b)
                joined += 1
                continue
        bouts.append((a, b))
    joined += len(kept) - len(candidates)
    # A recurring shorter lift-to-lower gap and weaker second peak can
    # distinguish paused curls even when both phases return to idle.
    if len(bouts) >= 6 and len(bouts) % 2 == 0:
        anchors = np.array([a+np.argmax(envelope[a:b]) for a, b in bouts])
        gaps = np.diff(anchors)
        within, between = gaps[::2], gaps[1::2]
        lifts, lowers = envelope[anchors[::2]]-noise, envelope[anchors[1::2]]-noise
        if (np.max(within) < 0.88*np.min(between) and
                np.median(lowers/lifts) < 0.92 and np.mean(lowers < lifts) >= 0.75):
            joined += len(bouts)//2
            bouts = [(first[0], second[1]) for first, second in zip(bouts[::2], bouts[1::2])]
    return [(a, b) for a, b in bouts if b-a >= 400], on, off, len(peaks)-len(kept), joined


def spectral(values, noise_psd=None, firmware=False):
    if len(values) < (1024 if firmware else 256):
        return NAN, NAN
    if firmware:
        f = np.fft.rfftfreq(1024, 1/FS)
        power = abs(np.fft.rfft((values-np.mean(values))*np.hanning(1024)))**2
    else:
        f, power = welch(values)
        if noise_psd is not None:
            power = np.maximum(power - noise_psd, 0)
    band = (f >= BAND[0]) & (f <= BAND[1])
    f, power = f[band], power[band]
    if power.sum() <= 0:
        return NAN, NAN
    cumulative = np.cumsum(power)
    half = cumulative[-1]/2
    mdf = (f[np.searchsorted(cumulative, half)] if firmware else
           np.interp(half, np.r_[0, cumulative], np.r_[f[0]-(f[1]-f[0]), f]))
    return mdf, np.average(f, weights=power)


def temperature(frame, bounds, idle, flag):
    mv = pd.to_numeric(frame.therm_mv, errors="coerce").to_numpy(dtype=float)
    celsius = np.full(len(mv), NAN)
    plausible = (mv >= 1000) & (mv <= 1650)
    ohms_ratio = mv[plausible] / (3300-mv[plausible])
    celsius[plausible] = 1/(1/298.15 + np.log(ohms_ratio)/3950)-273.15
    celsius[(celsius < 25) | (celsius > 45)] = NAN
    if (~np.isfinite(celsius)).any():
        flag("thermistor_dropout", f"{(~np.isfinite(celsius)).sum()} invalid samples")
    starts = np.arange(0, len(mv)-FS+1, FS)
    t = (starts + FS/2)/FS
    temp = np.array([np.median(celsius[a:a+FS]) for a in starts])
    valid = np.isfinite(temp)
    result = dict(temp_valid_fraction=valid.mean() if len(valid) else 0,
                  temp_pre_c=NAN, temp_active_c=NAN, temp_post_c=NAN,
                  temp_slope_c_min=NAN, temp_pre_slope_c_min=NAN,
                  temp_early_dip_c=NAN, temp_mdc_c=NAN)
    median = np.nanmedian(temp) if valid.any() else NAN
    usable = result["temp_valid_fraction"] >= 0.8 and 28 <= median <= 38
    if result["temp_valid_fraction"] < 0.8:
        flag("temperature_low_coverage", "Fewer than 80% of one-second blocks are valid")
    if np.isfinite(median) and not 28 <= median <= 38:
        flag("probe_off_skin", f"Median {median:.2f} °C")
    # A detached probe can lie above the raw voltage gate throughout.
    if np.nanmedian(mv) > 1650:
        flag("probe_off_skin", "Median raw thermistor voltage exceeds 1650 mV")
    if usable:
        if valid.sum() >= 2:
            result["temp_slope_c_min"] = stats.theilslopes(temp[valid], t[valid]/60).slope
        if bounds:
            first, last = bounds[0][0], bounds[-1][1]
            pre = np.zeros(len(t), dtype=bool)
            post = pre.copy()
            for a, b in idle:
                inside = (starts >= a) & (starts+FS <= b)
                pre |= inside & (starts+FS <= first)
                post |= inside & (starts >= last)
            pre &= valid
            post &= valid
            active = valid & (t >= first/FS) & (t < last/FS)
            early = active & (t <= first/FS+30)
            if pre.any():
                result["temp_pre_c"] = np.median(temp[pre])
            if pre.sum() >= 2:
                result["temp_pre_slope_c_min"] = stats.theilslopes(temp[pre], t[pre]/60).slope
                result["temp_mdc_c"] = mdc(temp[pre])
            if active.any():
                result["temp_active_c"] = np.mean(temp[active])
            if post.any():
                result["temp_post_c"] = np.median(temp[post])
            if early.any():
                result["temp_early_dip_c"] = result["temp_pre_c"] - np.min(temp[early])
    return result, t, temp, valid & usable


def mdc(values):
    return 1.96*np.sqrt(2)*1.253*np.std(values, ddof=1)/np.sqrt(len(values)) if len(values) >= 2 else NAN


def truthy(value):
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def apply_overrides(bounds, edits, length):
    by_rep = dict(enumerate(bounds, 1))
    seen = set()
    for row in edits:
        if pd.isna(row.get("rep", NAN)):
            continue
        rep = int(row["rep"])
        if rep < 1 or rep != float(row["rep"]) or rep in seen:
            raise ValueError("Override rep numbers must be unique positive integers")
        seen.add(rep)
        if truthy(row.get("exclude")):
            by_rep.pop(rep, None)
        else:
            a, b = (round(float(row[key])*FS) for key in ("start_s", "end_s"))
            if not 0 <= a < b <= length or b-a < 400:
                raise ValueError("Override bounds must be inside the recording and at least 400 ms long")
            by_rep[rep] = (a, b)
    ordered = [by_rep[k] for k in sorted(by_rep)]
    if any(b > c for (_, b), (c, _) in zip(ordered, ordered[1:])):
        raise ValueError("Override reps overlap or are out of time order")
    return ordered


def analyze_file(path, edits, expected, figures, qc):
    def flag(code, reason):
        qc.append(dict(file=path.name, flag=code, reason=reason))

    meta = dict(file=path.name, participant="", kind="", index=NAN, set_label="",
                timestamp=pd.NaT, tag="", included=False, reviewed=False, rep_count=0)
    match = FILENAME.fullmatch(path.stem)
    if match:
        participant, kind, index, suffix = match.groups()
        timestamp = TIMESTAMP.search(suffix)
        tag = (suffix[:timestamp.start()] if timestamp else suffix).strip("_- ")
        when = pd.to_datetime(timestamp.group() if timestamp else None,
                              format="%Y-%m-%d_%H-%M-%S", errors="coerce")
        meta.update(participant=participant, kind=kind, index=int(index),
                    set_label=f"{kind}_{int(index)}", timestamp=when, tag=tag)
    if meta["tag"]:
        flag("tagged_file", meta["tag"])
    if pd.isna(meta["timestamp"]):
        flag("missing_timestamp", "Session time is unavailable")
    file_edits = [row for row in edits if pd.isna(row.get("rep", NAN))]
    excluded = any(truthy(row.get("exclude")) for row in file_edits)
    if excluded:
        flag("manual_file_exclusion", "File excluded by override")
    meta["included"] = bool(match and not excluded and
                            (edits or (not meta["tag"] and pd.notna(meta["timestamp"]))))
    meta["reviewed"] = any(truthy(row.get("reviewed")) for row in file_edits)
    try:
        frame = pd.read_csv(path)
        raw = pd.to_numeric(frame.emg_mv, errors="coerce").to_numpy(dtype=float)
        if len(raw) < 500 or not np.isfinite(raw).all() or "therm_mv" not in frame:
            raise ValueError("Recording needs 500 finite EMG samples and a therm_mv column")
    except (AttributeError, OSError, ValueError) as exc:
        flag("invalid_recording", str(exc))
        meta["included"] = False
        figure, axis = plt.subplots(figsize=(9, 3))
        axis.text(0.5, 0.5, str(exc), ha="center", va="center", transform=axis.transAxes)
        axis.set_title(path.name)
        save(figure, figures / f"qc_{path.stem}.png")
        return meta, [], None

    emg, envelope, idle, noise, noise_sd, notches = preprocess(raw)
    if not idle:
        flag("insufficient_idle", "No qualifying one-second rest stretch")
    if notches:
        flag("notch_applied", ", ".join(map(str, notches)) + " Hz")
    bounds, on, off, dropped, joined = segment(envelope, noise, noise_sd)
    if dropped:
        flag("peaks_rejected", f"{dropped} peak candidates below the amplitude gate")
    if joined:
        flag("within_cycle_peaks", f"{joined} extra peaks grouped into curl cycles; review phase assignment")
    bounds = apply_overrides(bounds, edits, len(raw))
    if any(pd.notna(row.get("rep", NAN)) for row in edits):
        flag("manual_rep_override", "Rep boundaries or exclusions supplied")
    if not meta["reviewed"]:
        flag("segmentation_review_pending", "Automatic boundaries are provisional")
    if expected is not None and len(bounds) != expected:
        flag("rep_count_mismatch", f"Expected {expected}, found {len(bounds)}")
    if not bounds:
        flag("no_reps", "No qualifying curl cycles")
    spectra = [welch(emg[a:b])[1] for a, b in idle]
    noise_psd = np.average(spectra, axis=0, weights=[b-a for a, b in idle]) if idle else None
    clipped = (raw <= 50) | (raw >= 2450)
    records = []
    previous_peak = None
    for rep, (a, b) in enumerate(bounds, 1):
        peak = a+np.argmax(envelope[a:a+max(1, round(0.6*(b-a)))])
        start = min(max(a, peak-250), max(a, b-500))
        window = emg[start:start+500] if b-a >= 500 else np.array([])
        burst = emg[a:b]
        rms = np.sqrt(np.mean(burst**2))
        snr = rms/noise if noise > 0 else NAN
        mdf, mnf = spectral(window, noise_psd)
        whole_mdf, whole_mnf = spectral(burst, noise_psd)
        fw_mdf = firmware_reference = NAN
        if len(burst) >= 1024:
            fw_start = min(max(a, peak-512), b-1024)
            fw_slice = slice(fw_start, fw_start+1024)
            fw_mdf, _ = spectral(raw[fw_slice], firmware=True)
            firmware_reference, _ = spectral(emg[fw_slice], noise_psd)
        clipping = clipped[a:b].mean()
        low_snr = not np.isfinite(snr) or snr < MIN_SNR
        if low_snr:
            flag("low_snr", f"Rep {rep}, SNR {snr:.2f}×")
        if clipping > 0.01:
            flag("rep_clipping", f"Rep {rep}, {clipping:.1%} clipped")
        if len(window) == 0:
            flag("short_spectral_window", f"Rep {rep} has fewer than 500 active samples")
        interval = (peak-previous_peak)/FS if previous_peak is not None else NAN
        previous_peak = peak
        records.append(dict(file=path.name, participant=meta["participant"], set_label=meta["set_label"],
                            rep=rep, start_s=a/FS, end_s=b/FS, lift_proxy_peak_s=peak/FS,
                            lift_window_start_s=start/FS if len(window) else NAN,
                            mdf_hz=mdf, mnf_hz=mnf, rms_mv=rms, whole_mdf_hz=whole_mdf,
                            whole_mnf_hz=whole_mnf, firmware_mdf_hz=fw_mdf,
                            firmware_delta_hz=firmware_reference-fw_mdf, snr=snr,
                            low_snr=low_snr, clip_fraction=clipping,
                            valid_mdf=bool(idle and not low_snr and clipping <= 0.01 and np.isfinite(mdf)),
                            emg_burst_duration_s=(b-a)/FS, emg_onset_to_peak_s=(peak-a)/FS,
                            emg_peak_interval_s=interval, emg_cadence_per_min=60/interval if interval > 0 else NAN))
    temp, t, trace, valid = temperature(frame, bounds, idle, flag)
    if meta["set_label"] == "rep_1" and temp["temp_pre_slope_c_min"] > 0.1:
        flag("probe_not_equilibrated", "Baseline pre-set slope exceeds +0.1 °C/min")
    good = [r for r in records if r["valid_mdf"]]
    mdfs = np.array([r["mdf_hz"] for r in good])
    meta.update(temp, rep_count=len(bounds), valid_reps=len(good), noise_rms_mv=noise,
                noise_sd_mv=noise_sd, clip_fraction=clipped.mean(), notch_hz=",".join(map(str, notches)),
                mdf_hz=np.median(mdfs) if len(mdfs) else NAN,
                mdf_iqr_hz=np.subtract(*np.percentile(mdfs, [75, 25])) if len(mdfs) else NAN,
                mdf_mdc_hz=mdc(mdfs),
                mdf_slope_hz_rep=stats.linregress([r["rep"] for r in good], mdfs).slope if len(mdfs)>1 else NAN)
    for key in ("mnf_hz", "whole_mdf_hz", "rms_mv", "emg_burst_duration_s", "emg_peak_interval_s"):
        source = good if key in ("mnf_hz", "whole_mdf_hz") else records
        values = [r[key] for r in source if np.isfinite(r[key])]
        meta[key] = np.median(values) if values else NAN
    qc_plot(path, figures, emg, envelope, on, off, bounds, records, t, trace, valid)
    return meta, records if meta["included"] else [], (t, trace, valid)


def save(figure, path):
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def plot_temperature(axis, t, trace, valid, offset=0, scale=1):
    axis.plot(offset+t/scale, np.where(valid, trace, NAN), ".-", color="#ae4a19", ms=3)
    for a, b in runs(~valid):
        axis.axvspan(offset+(t[a]-0.5)/scale, offset+(t[b-1]+0.5)/scale,
                     color="0.8", alpha=0.5, lw=0)


def qc_plot(path, figures, emg, envelope, on, off, bounds, records, t, trace, valid):
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 6), layout="constrained")
    x = np.arange(len(emg))/FS
    axes[0].plot(x, emg, color="0.45", lw=0.45)
    axes[1].plot(x, envelope, color="#176b86", lw=1)
    if np.isfinite(on):
        axes[1].axhline(on, color="#964519", ls="--", lw=1, label="On")
        axes[1].axhline(off, color="#615288", ls=":", lw=1, label="Off")
        axes[1].legend(loc="upper right")
    for (a, b), rep in zip(bounds, records):
        for axis in axes[:2]:
            axis.axvspan(a/FS, b/FS, color="#176b86", alpha=0.13, lw=0)
        axes[1].axvline(rep["lift_proxy_peak_s"], color="0.4", lw=0.7)
        window = rep["lift_window_start_s"]
        if np.isfinite(window):
            axes[0].axvspan(window, window+0.5, color="#e08d41", alpha=0.29, lw=0)
        axes[1].text(a/FS, 0.97, str(rep["rep"]), va="top",
                     transform=axes[1].get_xaxis_transform(), fontsize=8)
    plot_temperature(axes[2], t, trace, valid)
    for axis, label in zip(axes, ("Filtered EMG (mV)", "Envelope (mV)", "Skin temp. (°C)")):
        axis.set_ylabel(label)
    axes[2].set_xlabel("Recording time (s)")
    axes[0].set_title(path.name, loc="left")
    save(figure, figures / f"qc_{path.stem}.png")


def plateau(values, margin):
    values = np.asarray(values)
    if not len(values) or not np.isfinite(values[-1]) or not np.isfinite(margin):
        return "insufficient data"
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return "insufficient data"
    rise = values[-1]-values[0]
    step = abs(values[-1]-values[-2])
    return "yes" if rise > 0 and step < margin and step < rise/3 else "no"


def unique_sets(group):
    # Use the ordinary recording first if a tagged repeat is opted in.
    return group.assign(tagged=group.tag.ne("")).sort_values(
        ["tagged", "timestamp"], na_position="last").drop_duplicates("set_label")


def paired_values(cold, warm):
    result = {}
    for key in ("mnf_hz", "rms_mv", "emg_burst_duration_s", "emg_peak_interval_s", "temp_pre_c"):
        result[key+"_cold"] = cold.get(key, NAN)
        result[key+"_warm"] = warm.get(key, NAN)
        result[key+"_change"] = result[key+"_warm"]-result[key+"_cold"]
    for key, name in (("mdf_hz", "mdf_change_pct"), ("whole_mdf_hz", "whole_mdf_change_pct")):
        reference = cold.get(key, NAN)
        result[name] = 100*(warm.get(key, NAN)/reference-1) if reference > 0 else NAN
    return result


def readiness(sets, ratings, qc):
    sets = sets.copy()
    participants, thresholds = [], []
    for participant, all_sets in sets[sets.included].groupby("participant"):
        group = unique_sets(all_sets)
        by_label = group.set_index("set_label")
        cold = by_label.loc["rep_1"] if "rep_1" in by_label.index else {}
        warm = by_label.loc["rep_2"] if "rep_2" in by_label.index else {}
        first_wu = by_label.loc["wu_1"] if "wu_1" in by_label.index else {}
        baseline = cold.get("temp_pre_c", NAN)
        temp_margin = cold.get("temp_mdc_c", NAN)
        cold_mdf = cold.get("mdf_hz", NAN)
        wu_mdf = first_wu.get("mdf_hz", NAN)
        mdf_margin = 100*first_wu.get("mdf_mdc_hz", NAN)/wu_mdf if wu_mdf > 0 else NAN
        heavy_margin = 100*cold.get("mdf_mdc_hz", NAN)/cold_mdf if cold_mdf > 0 else NAN
        changes = paired_values(cold, warm)
        summary = dict(participant=participant, **changes, temp_mdc_c=temp_margin,
                       warmup_mdf_mdc_pct=mdf_margin, heavy_mdf_mdc_pct=heavy_margin,
                       heavy_effective_mdf_threshold_pct=max(MDF_TARGET, heavy_margin) if np.isfinite(heavy_margin) else NAN,
                       effective_temp_threshold_c=max(TEMP_TARGET, temp_margin) if np.isfinite(temp_margin) else NAN,
                       probe_not_equilibrated=cold.get("temp_pre_slope_c_min", NAN)>0.1,
                       rule_ready_set="", readiness_status="insufficient data",
                       heavy_mdf_plateau="insufficient data")
        if np.isfinite(heavy_margin) and np.isfinite(temp_margin):
            summary["heavy_thresholds_met"] = (
                changes["mdf_change_pct"] >= max(MDF_TARGET, heavy_margin)
                and changes["temp_pre_c_change"] >= max(TEMP_TARGET, temp_margin)
            ) if np.isfinite(changes["mdf_change_pct"]) and np.isfinite(changes["temp_pre_c_change"]) else pd.NA
        sets.loc[all_sets.index, "minutes_since_rep_1"] = (
            all_sets.timestamp-cold["timestamp"]).dt.total_seconds()/60 if "timestamp" in cold else NAN
        sets.loc[all_sets.index, "delta_temp_c"] = all_sets.temp_pre_c-baseline
        sets.loc[all_sets.index, "mdf_change_from_rep_1_pct"] = (
            100*(all_sets.mdf_hz/cold_mdf-1) if cold_mdf > 0 else NAN)
        mdf_history, temp_history = [], []
        session = group[group.set_label.isin(PROTOCOL)].sort_values("timestamp", na_position="last")
        for index, row in session.iterrows():
            dt = row.temp_pre_c-baseline
            temp_history.append(dt)
            t_flat = plateau(temp_history, temp_margin)
            if row.set_label == "rep_2":
                summary["heavy_temperature_plateau"] = t_flat
            sets.loc[index, "temperature_plateau"] = t_flat
            if row.kind != "wu":
                continue
            dm = 100*(row.mdf_hz/wu_mdf-1) if wu_mdf > 0 else NAN
            mdf_history.append(dm)
            m_flat = plateau(mdf_history, mdf_margin)
            unreliable = (row.valid_reps < row.rep_count or not row.valid_reps or
                          first_wu.get("valid_reps", 0) < first_wu.get("rep_count", 0))
            sets.loc[index, "warmup_mdf_change_pct"] = dm
            sets.loc[index, "warmup_mdf_plateau"] = m_flat
            sets.loc[index, "warmup_mdf_unreliable"] = str(unreliable).lower()
            if unreliable:
                qc.append(dict(file=row.file, flag="warmup_mdf_unreliable",
                               reason="Current or reference warm-up has missing or rejected MDF reps"))
            if "insufficient data" not in (m_flat, t_flat):
                if not summary["rule_ready_set"]:
                    summary["readiness_status"] = "criteria not met"
                if m_flat == t_flat == "yes":
                    thresholds.append((participant, dm, dt, mdf_margin, temp_margin))
                    if (not summary["rule_ready_set"] and dm >= max(MDF_TARGET, mdf_margin)
                            and dt >= max(TEMP_TARGET, temp_margin)):
                        summary.update(rule_ready_set=row.set_label, readiness_status="ready by rule")
        rating = ratings.get(participant, "")
        if rating:
            summary["rated_ready_set"] = rating
            if summary["rule_ready_set"] and rating in PROTOCOL:
                diff = PROTOCOL.index(summary["rule_ready_set"])-PROTOCOL.index(rating)
                summary.update(rating_signed_difference_sets=diff, rating_exact_match=diff == 0,
                               rating_within_one_set=abs(diff) <= 1)
        participants.append(summary)
    return sets, pd.DataFrame(participants), thresholds


def exact_signed_rank(delta):
    # Conditional sign enumeration with doubled midranks supports ties and zeros.
    nonzero = delta[delta != 0]
    if not len(nonzero):
        return 1.0
    ranks = np.rint(2*stats.rankdata(abs(nonzero))).astype(int)
    observed = sum(ranks[nonzero > 0])
    total = sum(ranks)
    counts = [1]+[0]*total
    for rank in ranks:
        for index in range(total, rank-1, -1):
            counts[index] += counts[index-rank]
    distance = abs(2*observed-total)
    return sum(count for index, count in enumerate(counts) if abs(2*index-total) >= distance)/2**len(ranks)


def session_figures(sets, reps, traces, figures):
    for participant, group in sets[sets.included].groupby("participant"):
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5), layout="constrained")
        envelope = group.sort_values("timestamp").dropna(subset=["minutes_since_rep_1"])
        for row in envelope.itertuples():
            x = row.minutes_since_rep_1
            if not reps.empty:
                subset = reps[(reps.file == row.file) & reps.valid_mdf]
                axes[0].scatter(x+subset.lift_proxy_peak_s/60, subset.mdf_hz, s=14, color="#176b86")
            if row.file in traces:
                plot_temperature(axes[1], *traces[row.file], offset=x, scale=60)
        axes[0].plot(envelope.minutes_since_rep_1, envelope.mdf_hz, "D-", color="0.2", ms=4)
        axes[0].set_ylabel("MDF (Hz)")
        axes[1].set_ylabel("Skin temperature (°C)")
        axes[1].set_xlabel("Minutes since heavy baseline")
        axes[0].set_title(participant, loc="left")
        save(figure, figures / f"session_{participant}.png")

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), layout="constrained")
    for _, group in sets[sets.included].groupby("participant"):
        ordered = unique_sets(group).set_index("set_label").reindex(PROTOCOL)
        for axis, key in zip(axes, ("mdf_change_from_rep_1_pct", "delta_temp_c")):
            if key in ordered:
                axis.plot(PROTOCOL, ordered[key], ".-", alpha=0.45, lw=1)
    axes[0].set_ylabel("MDF change vs. rep_1 (%)")
    axes[1].set_ylabel("Pre-set skin temp. change (°C)")
    if axes[1].lines:
        axes[1].axhline(TEMP_TARGET, color="0.3", ls="--", lw=1)
    save(figure, figures / "group_session.png")


def paired_figure(sets, figures):
    figure, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
    metrics = (
        ("mdf_hz", "MDF (Hz)"), ("mnf_hz", "MNF (Hz)"), ("rms_mv", "RMS (mV)"),
        ("emg_burst_duration_s", "EMG burst duration proxy (s)"),
        ("emg_peak_interval_s", "EMG peak interval proxy (s)"),
        ("temp_pre_c", "Pre-set skin temperature (°C)"))
    for _, group in sets[sets.included].groupby("participant"):
        paired = unique_sets(group).set_index("set_label").reindex(["rep_1", "rep_2"])
        for axis, (key, _) in zip(axes.flat, metrics):
            if key in paired and paired[key].notna().all():
                axis.plot([0, 1], paired[key], "o-", color="#176b86", alpha=0.52, ms=4, lw=1)
    for axis, (_, label) in zip(axes.flat, metrics):
        axis.set_xticks([0, 1], ["Cold", "Warm"])
        axis.set_ylabel(label)
    save(figure, figures / "heavy_pairs.png")


def threshold_figure(thresholds, figures):
    mdf_grid = np.arange(21)
    temp_grid = np.linspace(0, 2, 21)
    counts = np.zeros((len(temp_grid), len(mdf_grid)), dtype=int)
    for y, x in product(range(len(temp_grid)), range(len(mdf_grid))):
        counts[y, x] = len({participant for participant, dm, dt, mm, mt in thresholds
                            if dm >= max(mdf_grid[x], mm) and dt >= max(temp_grid[y], mt)})
    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    mesh = axis.pcolormesh(mdf_grid, temp_grid, counts, shading="nearest", cmap="Blues",
                           vmin=0, vmax=max(1, counts.max()))
    axis.plot(MDF_TARGET, TEMP_TARGET, marker="+", color="#ae4a19", ms=12)
    axis.set(xlabel="MDF threshold (%)", ylabel="Skin temperature threshold (°C)")
    figure.colorbar(mesh, ax=axis, label="Participants meeting criteria",
                    ticks=np.arange(max(1, counts.max())+1))
    save(figure, figures / "threshold_sweep.png")


def write_stats(path, participants, sets, reps):
    rng = np.random.default_rng(42)
    lines = [
        "ReadyRep exploratory analysis",
        "Thresholds are provisional. Automatic segmentation is provisional pending QC review.",
        f"Nominal sample rate {FS} Hz | band {BAND[0]}–{BAND[1]} Hz | "
        f"minimum SNR {MIN_SNR}× | peak fraction {PEAK_FRACTION:.0%} | "
        f"H1 thresholds +{MDF_TARGET}% MDF and +{TEMP_TARGET} °C.",
        "Time is reconstructed from the sample count. Serial gaps and actual sample rate are unknown.",
        "EMG timing measures and the lifting window are proxies. Pauses and separate phase peaks can be ambiguous.",
        "The primary comparison uses the same heavy load. Warm-up MDF has limited coverage after SNR gating.",
        "The heavy comparison has only two same-load MDF points, so its MDF plateau is indeterminate.",
        "The temperature reference is the pre-set heavy baseline. Skin temperature differs from muscle temperature.",
        "Probe settling can contribute to the observed rise. The early dip is reported separately.",
        "The protocol has no control for time, order, fatigue or potentiation. Causal effects cannot be isolated.",
        "The rule has no performance or injury endpoint. No reference EMG or thermometer validation was performed.",
        "Paired heavy comparison. Changes are warm minus cold. Bootstrap 95% CIs use 10,000 participant resamples.",
        "With five or six nonzero pairs, the smallest possible two-sided exact p is 0.0625 or 0.03125.",
    ]
    metrics = {
        "mdf_change_pct": "MDF (%)", "mnf_hz_change": "MNF (Hz)",
        "rms_mv_change": "RMS (mV)", "emg_burst_duration_s_change": "EMG burst duration proxy (s)",
        "emg_peak_interval_s_change": "EMG peak interval proxy (s)", "temp_pre_c_change": "Pre-set skin temperature (°C)",
    }
    for key, label in metrics.items():
        delta = participants.get(key, pd.Series(dtype=float)).dropna().to_numpy()
        if not len(delta):
            lines.append(f"{label} | no paired observations")
            continue
        estimate = np.median(delta)
        low, high = np.percentile(np.median(rng.choice(delta, (10000, len(delta)), replace=True), axis=1), [2.5, 97.5])
        ci = f"[{low:+.3f}, {high:+.3f}]" if len(delta) > 1 else "unavailable with one pair"
        lines.append(f"{label} | median {estimate:+.3f} | 95% CI {ci} | "
                     f"{sum(delta > 0)} of {len(delta)} increased | exact Wilcoxon p={exact_signed_rank(delta):.4g}")
    if not participants.empty:
        primary = participants.mdf_change_pct
        whole = participants.whole_mdf_change_pct
        use = primary.notna() & whole.notna()
        if use.any():
            lift, burst = np.median(primary[use]), np.median(whole[use])
            if np.sign(lift) == np.sign(burst):
                lines.append(f"Whole-burst MDF sensitivity agrees in direction on {use.sum()} matched pairs. "
                             f"Median changes were {lift:+.2f}% for the lift proxy and {burst:+.2f}% for the full burst.")
            else:
                lines.append(f"Whole-burst and lift-window MDF sensitivity did not agree in direction on {use.sum()} matched pairs.")
        lines.append(f"Readiness status counts | {participants.readiness_status.value_counts().to_dict()}")
        agreement = participants.dropna(subset=["rating_signed_difference_sets"]) if "rating_signed_difference_sets" in participants else pd.DataFrame()
        if "rated_ready_set" in participants:
            lines.append(f"Rating agreement | {len(agreement)} comparable of {participants.rated_ready_set.notna().sum()} rated")
            for row in agreement.itertuples():
                lines.append(f"{row.participant} | rule {row.rule_ready_set} | rating {row.rated_ready_set} | "
                             f"exact {row.rating_exact_match} | within one {row.rating_within_one_set} | "
                             f"signed difference {row.rating_signed_difference_sets:+.0f}")
    if not reps.empty:
        firmware = reps.loc[reps.valid_mdf, "firmware_delta_hz"].dropna().abs()
        if len(firmware):
            lines.append(f"High-SNR matched 1024-sample segments, corrected versus firmware MDF | n={len(firmware)} | "
                         f"median absolute difference {firmware.median():.2f} Hz | "
                         f"maximum {firmware.max():.2f} Hz")
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        lines.append("Secondary mixed model skipped because statsmodels is unavailable.")
    else:
        selected = [file for _, group in sets[sets.included].groupby("participant")
                    for file in unique_sets(group).file]
        model_data = reps[reps.valid_mdf & reps.file.isin(selected) & reps.set_label.isin(PROTOCOL)].copy() if not reps.empty else reps
        if (not model_data.empty and model_data.participant.nunique() >= 3 and
                model_data.set_label.nunique() >= 2 and model_data.rep.nunique() >= 2):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fit = smf.mixedlm("mdf_hz ~ C(set_label, Treatment('rep_1')) + rep",
                                      model_data, groups=model_data.participant).fit()
                lines.append(f"Secondary mixed model | MDF by protocol condition and rep index | "
                             f"participant random intercept | converged={fit.converged}")
                lines.append(fit.summary().as_text())
                lines.extend(str(warning.message) for warning in caught)
            except (ValueError, np.linalg.LinAlgError) as exc:
                lines.append(f"Secondary mixed model failed: {exc}")
        else:
            lines.append("Secondary mixed model has insufficient valid participants, conditions or rep indices.")
    lines.extend([
        "Future sessions. Allow at least 10 minutes for probe equilibration and insulate the probe.",
        "Add a timed no-warm-up control and a standardized fixed-load contraction at each measurement.",
        "Use a metronome and record load, effort or readiness ratings, room temperature and electrode placement.",
        "Log real sample timestamps, repair the thermistor connector and evaluate 2 kHz sampling.",
    ])
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def self_test():
    rng = np.random.default_rng(755)
    n = 26*FS
    time = np.arange(n)/FS
    source = signal.sosfiltfilt(signal.butter(6, [50, 87], btype="bandpass", fs=FS,
                                              output="sos"), rng.normal(size=n))
    source /= np.std(source)
    raw = 1580 + rng.normal(0, 2, n)
    truth = [(4, 6), (9, 11), (14, 16), (19, 21)]
    for index, (a, b) in enumerate(truth):
        phase = time[a*FS:b*FS]-a
        # Separate lift and lowering peaks with varying top pauses.
        pause = [0.18, 0.36, 0.08, 0.48][index]
        lift = np.clip(np.minimum(phase/0.10, (0.93-phase)/0.10), 0, 1)
        lower = np.clip(np.minimum((phase-0.93-pause)/0.10, (2-phase)/0.10), 0, 1)
        raw[a*FS:b*FS] += (90*lift+79*lower)*source[a*FS:b*FS]
    emg, envelope, idle, noise, sd, _ = preprocess(raw)
    bounds, _, _, _, joined = segment(envelope, noise, sd)
    assert len(bounds) == len(truth), bounds
    assert joined > 0, "Fixture must exercise merging retained lift and lowering peaks"
    assert np.max(abs(np.array(bounds)/FS-np.array(truth))) <= 0.1, bounds
    idle_psd = np.mean([welch(emg[a:b])[1] for a, b in idle], axis=0)
    measured, expected = [], []
    for a, _ in truth:
        window = slice(a*FS+200, a*FS+700)
        measured.append(spectral(emg[window], idle_psd)[0])
        expected.append(spectral(90*source[window])[0])
    assert np.max(abs(np.subtract(measured, expected))) <= 2, (measured, expected)
    assert 60 <= np.mean(expected) <= 80, expected
    for delay in (0.6, 1.1, 1.5, 1.8, 2.2):
        trace = np.full(n, 10.)
        for start, _ in truth:
            trace += 100*np.exp(-((time-start)/0.24)**2)
            trace += 85*np.exp(-((time-start-delay)/0.24)**2)
        cycles, *_ = segment(trace, 10, 0.25)
        assert len(cycles) == len(truth), (delay, cycles)
    trace = np.full(n, 10.)
    for index, peak in enumerate(np.arange(4, 24, 2.5)):
        amplitude = 100 if index % 2 == 0 else 85
        trace += amplitude*np.exp(-((time-peak)/0.24)**2)
    cycles, *_ = segment(trace, 10, 0.25)
    assert len(cycles) == 8, cycles
    print(f"Self-test passed: four lift–lower cycles, MDF {np.mean(measured):.2f} Hz "
          f"(clean reference {np.mean(expected):.2f} Hz).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=Path("analysis_out"))
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--expected-reps", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if not args.files:
        if args.self_test:
            return
        parser.error("provide one or more CSV files or --self-test")
    if args.expected_reps is not None and args.expected_reps < 0:
        parser.error("--expected-reps must be nonnegative")
    names = [path.name for path in args.files]
    if len(set(names)) != len(names):
        parser.error("input CSV basenames must be unique")
    reserved = [args.out / name for name in ("reps.csv", "sets.csv", "participants.csv", "qc.csv", "stats.txt")]
    if {path.resolve() for path in args.files} & {path.resolve() for path in reserved}:
        parser.error("output would replace an input recording")
    edits = {}
    if args.overrides:
        table = pd.read_csv(args.overrides)
        for name, group in table.groupby("file"):
            edits[name] = group.to_dict("records")
    ratings = {}
    if args.ratings:
        table = pd.read_csv(args.ratings, dtype=str).set_index("participant")
        for key in ("buzz_set", "ready_set"):
            if key in table:
                ratings.update(table[key].dropna().to_dict())
        if not set(table.columns) & {"ready_set", "buzz_set"}:
            parser.error("ratings must contain ready_set or buzz_set")
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    set_records, rep_records, qc, traces = [], [], [], {}
    for path in args.files:
        try:
            row, reps, traces[path.name] = analyze_file(path, edits.get(path.name, []),
                                                        args.expected_reps, figures, qc)
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(f"{path.name}: {exc}")
        set_records.append(row)
        rep_records.extend(reps)
    sets = pd.DataFrame(set_records).sort_values(["participant", "timestamp", "file"])
    for column in ("temp_pre_c", "mdf_hz", "minutes_since_rep_1"):
        if column not in sets:
            sets[column] = NAN
    reps = pd.DataFrame(rep_records) if rep_records else pd.DataFrame(
        columns=["file", "participant", "set_label", "rep", "start_s", "end_s", "mdf_hz",
                 "valid_mdf", "emg_burst_duration_s", "emg_peak_interval_s"])
    sets, participants, thresholds = readiness(sets, ratings, qc)
    reps.to_csv(args.out / "reps.csv", index=False)
    sets.to_csv(args.out / "sets.csv", index=False)
    if participants.empty:
        participants = pd.DataFrame(columns=["participant", "mdf_change_pct", "rule_ready_set", "readiness_status"])
    participants.to_csv(args.out / "participants.csv", index=False)
    pd.DataFrame(qc, columns=["file", "flag", "reason"]).to_csv(args.out / "qc.csv", index=False)
    write_stats(args.out / "stats.txt", participants, sets, reps)
    session_figures(sets, reps, traces, figures)
    paired_figure(sets, figures)
    threshold_figure(thresholds, figures)
    print(f"Wrote {len(reps)} reps and {len(sets)} file summaries to {args.out}")


if __name__ == "__main__":
    main()
