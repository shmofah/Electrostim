#This grid is for viewing H reflex recruitment curves when you use the aux channel on the TMSI
import os, sys, time, json, argparse
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# --- Poly5 reader import (same pattern you used) ---
sys.path.insert(0, r"C:\\Users\\Dell\\Documents\\RESEARCH\\tmsi-python-interface-main\\TMSiFileFormats\\file_readers")
from poly5reader import Poly5Reader


# -------------------------
# IO
# -------------------------
def read_poly(fname):
    path = fname if fname.endswith(".poly5") else fname + ".poly5"
    r = Poly5Reader(path)
    if hasattr(r, "samples"):
        return r.samples
    for m in ("read_data", "read", "get_data"):
        if hasattr(r, m):
            out = getattr(r, m)()
            if out is not None:
                return out
    for a in ("data", "signal_data", "sample_data"):
        if hasattr(r, a):
            return getattr(r, a)
    candidates = [x for x in dir(r) if ("read" in x.lower() or "data" in x.lower() or "sam" in x.lower())]
    raise AttributeError(f"Poly5Reader has no samples/read_data/read. Candidates: {candidates}")


# -------------------------
# Filtering
# -------------------------
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band", analog=False)
    return b, a

def butter_bandpass_filter(x, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, x)

def notch(notch_freq, samp_freq, quality_factor=30):
    b, a = iirnotch(notch_freq, quality_factor, samp_freq)
    return b, a

def notch_filter(x, notch_fs, fs, q=30):
    b, a = notch(notch_fs, fs, q)
    return filtfilt(b, a, x)

def filt_emg_1ch(x, fs=2000, lowcut=20, highcut=500, order=3, notch_fs=50, notch_q=30):
    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=order)
    y = notch_filter(y, notch_fs, fs, notch_q)
    return y


# -------------------------
# Trigger segmentation
# -------------------------
def segment_trigs(trigs, threshold=0):
    stim_idx = {}
    d = np.diff(trigs)
    events = np.where(d < -threshold)[0]
    if events.size < 2:
        raise RuntimeError(f"Not enough trigger edges found (found {events.size}). Try adjusting --trig_threshold.")
    stim_idx["start"] = int(events[0])
    stim_idx["stims"] = events[1:].astype(int).tolist()
    return stim_idx

def subset_events_by_time(event_dict, fs, t0_s, t1_s):
    t0 = int(t0_s * fs)
    t1 = int(t1_s * fs)
    stims = np.asarray(event_dict["stims"], dtype=int)
    keep = (stims >= t0) & (stims <= t1)
    out = dict(event_dict)
    out["stims"] = stims[keep].tolist()
    return out


# -------------------------
# Recruitment extraction
# -------------------------
def compute_recruitment(
    emg_f, iso_sig, stims, fs,
    vis_win_L_ms, vis_win_U_ms,
    hreflex_win_ms, mwave_win_ms,
    debug=False
):
    s_ms = fs / 1000.0
    win_pre = vis_win_L_ms
    win_post = vis_win_U_ms
    win_samp = int((win_pre + win_post) * s_ms)

    h0 = int((win_pre + hreflex_win_ms[0]) * s_ms)
    h1 = int((win_pre + hreflex_win_ms[1]) * s_ms)
    m0 = int((win_pre + mwave_win_ms[0]) * s_ms)
    m1 = int((win_pre + mwave_win_ms[1]) * s_ms)

    stim_amp = []
    Hpp = []
    Mpp = []

    n_total = len(stims)
    n_edge = 0
    n_shape = 0
    n_kept = 0

    N = len(emg_f)

    for event_idx in stims:
        start = int(event_idx - win_pre * s_ms)
        end   = int(event_idx + win_post * s_ms)

        if start < 0 or end > N:
            n_edge += 1
            continue

        seg = emg_f[start:end]
        if seg.shape[0] != win_samp:
            n_shape += 1
            continue

        stim_amp.append(np.mean(iso_sig[int(event_idx - 250):int(event_idx - 50)]))

        Hseg = seg[h0:h1]
        Mseg = seg[m0:m1]
        Hpp.append(float(np.max(Hseg) - np.min(Hseg)))
        Mpp.append(float(np.max(Mseg) - np.min(Mseg)))

        n_kept += 1

    stim_amp = np.asarray(stim_amp, dtype=float)
    Hpp = np.asarray(Hpp, dtype=float)
    Mpp = np.asarray(Mpp, dtype=float)

    if debug:
        print(f"  compute_recruitment: total={n_total} kept={n_kept} edge_skip={n_edge} shape_skip={n_shape}")
        if n_kept > 0:
            print(f"  win_samp={win_samp}, start/end example: "
                  f"{int(stims[0]-win_pre*s_ms)}..{int(stims[0]+win_post*s_ms)} (N={N})")
        elif n_total > 0:
            e0 = int(stims[0])
            s0 = int(e0 - win_pre*s_ms)
            e1 = int(e0 + win_post*s_ms)
            print(f"  first stim={e0} => slice {s0}:{e1} (N={N})")

    if stim_amp.size == 0:
        return np.array([]), np.array([]), np.array([])

    mx = np.max(stim_amp)
    if mx == 0 or not np.isfinite(mx):
        return np.array([]), np.array([]), np.array([])

    stim_norm = stim_amp / mx
    stim_bin = np.round(stim_norm, 1)

    return stim_bin, Hpp, Mpp


def plot_recruitment(out_path, flag, stim_bin, Hpp, Mpp):
    if stim_bin.size == 0 or Hpp.size == 0 or Mpp.size == 0:
        print(f"  [WARNING] No valid events to plot for {flag}. Skipping plot.")
        return

    levels = np.unique(stim_bin)
    H_means, M_means = [], []

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(flag)
    ax.set_xlabel("Normalized stim amplitude (binned)")
    ax.set_ylabel("Peak-to-peak (uV)")

    for lvl in levels:
        idx = np.where(stim_bin == lvl)[0]
        Hvals = Hpp[idx]
        Mvals = Mpp[idx]

        ax.scatter(lvl * np.ones_like(Hvals), Hvals, s=10, alpha=0.25)
        ax.scatter(lvl * np.ones_like(Mvals), Mvals, s=10, alpha=0.25)

        H_means.append(np.mean(Hvals) if Hvals.size else np.nan)
        M_means.append(np.mean(Mvals) if Mvals.size else np.nan)

    ax.plot(levels, H_means, marker="o", linewidth=1, label="H-reflex")
    ax.plot(levels, M_means, marker="o", linewidth=1, label="M-wave")

    ax.set_xlim([-0.1, 1.1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(frameon=False)

    os.makedirs(out_path, exist_ok=True)
    fig.savefig(os.path.join(out_path, f"{flag}_bipolar_recruitment.svg"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Bipolar-only recruitment curve (channels 64 & 65)")

    p.add_argument("--data_dir", default=r"C:\\Users\\Dell\\Documents\\data", type=str)
    p.add_argument("--participant_ID", default="shadow", type=str)
    p.add_argument("--exp_date", default="20260210", type=str)

    p.add_argument("--fname", default="thresholding_FLX-20260210_160511", type=str)
    p.add_argument("--fs", default=2000, type=int)

    p.add_argument("--bipolar_chans", nargs=2, type=int, default=[64, 65],
                   help="Two absolute channel indices for bipolar EMG channels (default: 64 65).")

    p.add_argument("--trigger_chan", default=-3, type=int,
                   help="Trigger channel index (python negative indexing allowed, default -3).")
    p.add_argument("--iso_chan", default=70, type=int,
                   help="Absolute channel index used to estimate stim amplitude (change to your real ISO channel).")

    p.add_argument("--trig_threshold", default=3.0, type=float)

    p.add_argument("--vis_win_L", default=20, type=float, help="ms before stim")
    p.add_argument("--vis_win_U", default=60, type=float, help="ms after stim")
    p.add_argument("--Hreflex_win", default=(20, 30), type=float, nargs=2, help="ms window")
    p.add_argument("--Mwave_win", default=(10, 15), type=float, nargs=2, help="ms window")

    p.add_argument("--segments", nargs="*", default=["36,96,seg_1", "146,204,seg_2"],
                   help='List like "t0,t1,tag" in seconds.')

    p.add_argument("--out_subdir", default="thresholding/plots_bipolar_only", type=str)

    args = p.parse_args()

    file_path = os.path.join(
        args.data_dir, args.participant_ID, args.exp_date,
        "thresholding", args.fname
    )
    print("Reading:", file_path)
    data = read_poly(file_path)  # (n_channels, n_samples)

    trigs = data[args.trigger_chan, :]
    iso = data[args.iso_chan, :]

    print(f"ISO chan {args.iso_chan}: min={np.min(iso):.6g} max={np.max(iso):.6g} std={np.std(iso):.6g}")

    event_dict = segment_trigs(trigs, threshold=args.trig_threshold)

    out_path = os.path.join(args.data_dir, args.participant_ID, args.exp_date, args.out_subdir)

    segments = []
    for s in args.segments:
        try:
            t0, t1, tag = s.split(",", 2)
            segments.append((float(t0), float(t1), tag))
        except ValueError:
            raise ValueError(f"Bad segment format: {s}. Expected t0,t1,tag")

    if len(segments) == 0:
        segments = [(None, None, "all")]

    duration_s = data.shape[1] / args.fs
    print(f"File duration: {duration_s:.2f} s  (Nsamples={data.shape[1]}, fs={args.fs})")

    for t0, t1, tag in segments:
        if t0 is None:
            seg_events = event_dict
        else:
            seg_events = subset_events_by_time(event_dict, args.fs, t0, t1)

        nst = len(seg_events["stims"])
        print(f"\nSegment {tag}: {nst} stims")

        if nst == 0:
            print(f"[WARNING] No stims in segment {tag}. Skipping.")
            continue

        print(f"  first stim idx={seg_events['stims'][0]}  last stim idx={seg_events['stims'][-1]}  N={data.shape[1]}")
        print(f"  first stim time={seg_events['stims'][0]/args.fs:.3f}s  last stim time={seg_events['stims'][-1]/args.fs:.3f}s")

        # ISO sanity check for this segment
        a = max(seg_events["stims"][0] - 500, 0)
        b = min(seg_events["stims"][-1] + 500, len(iso))
        iso_seg = iso[a:b]
        print(f"  ISO seg {tag} (chan {args.iso_chan}): min={np.min(iso_seg):.6g} max={np.max(iso_seg):.6g} std={np.std(iso_seg):.6g}")

        for ch in args.bipolar_chans:
            emg = data[ch, :]
            emg_f = filt_emg_1ch(emg, fs=args.fs)

            stim_bin, Hpp, Mpp = compute_recruitment(
                emg_f=emg_f,
                iso_sig=iso,
                stims=seg_events["stims"],
                fs=args.fs,
                vis_win_L_ms=args.vis_win_L,
                vis_win_U_ms=args.vis_win_U,
                hreflex_win_ms=args.Hreflex_win,
                mwave_win_ms=args.Mwave_win,
                debug=True
            )

            flag = f"{args.fname}_{tag}_bip_ch{ch}_iso{args.iso_chan}"
            plot_recruitment(out_path, flag, stim_bin, Hpp, Mpp)

            if stim_bin.size > 0:
                print("Saved:", os.path.join(out_path, f"{flag}_bipolar_recruitment.svg"))

    print("Done.")


if __name__ == "__main__":
    main()
