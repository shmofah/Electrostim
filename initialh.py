#This code is to viewing the initial H-reflex recruitmnent curve for a single grid channel
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
# from tmsi_dual_interface.tmsi_libraries.TMSiFileFormats.file_readers import Poly5Reader
sys.path.insert(0, r"C:\Users\Dell\Documents\RESEARCH\tmsi-python-interface-main\TMSiFileFormats\file_readers")
from poly5reader import Poly5Reader



# FILE SETUP

#FILE_PATH = r"D:\h_reflex_gui\data\dragon\20260401\thresholding\trial_1_FLX-20260401_114935.poly5"   
FILE_PATH = r"C:\Users\Dell\Documents\data\dragon\20260401\thresholding\trial_1_FLX-20260401_114935.poly5"
FS = 2000

VIS_WIN_L = 20   # ms before stim
VIS_WIN_U = 60   # ms after stim

HREFLEX_WINDOW_MS = (20, 30)
MWAVE_WINDOW_MS   = (10, 15)

EVENT_THRESHOLD = 3
ISO_AUX_INDEX = 6              # AUX channel used to amplitude
GRID_CHANNEL_TO_PLOT = 48     # choose one remapped grid channel: 

# differential mode
DIFFERENTIAL_MODE = "double_row"   # monopolar, row, column, double_col, double_row

# None = use whole file
SEGMENT_TIME_RANGE_S = None
# SEGMENT_TIME_RANGE_S = (36, 96)

#BASE_OUTPUT_PATH = r"D:\h_reflex_gui\data\dragon\20260401\thresholding"
BASE_OUTPUT_PATH = r"C:\Users\Dell\Documents\data\dragon\20260401\thresholding"
OUTDIR = os.path.join(BASE_OUTPUT_PATH, "initialhref")




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

    raise AttributeError("Could not find sample data in Poly5Reader.")




def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band", analog=False)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def notch(notch_freq, samp_freq, quality_factor=30):
    b, a = iirnotch(notch_freq, quality_factor, samp_freq)
    return b, a

def notch_filter(data, notch_fs, fs, q=30):
    b, a = notch(notch_fs, fs, q)
    return filtfilt(b, a, data)

def filt_grid(data, lowcut=20, highcut=500, fs=2000, order=3, notch_fs=50, notch_q=30):
    filt_out = np.zeros_like(data)
    for i in range(data.shape[0]):
        filt_out[i, :] = notch_filter(
            butter_bandpass_filter(data[i, :], lowcut, highcut, fs, order=order),
            notch_fs, fs, notch_q
        )
    return filt_out



# EVENTS


def segment_trigs(trigs, threshold=0):
    stim_idx = {}
    trigs = np.diff(trigs)
    events = np.where(trigs < -threshold)[0]

    if len(events) == 0:
        raise ValueError("No stimulation events found.")

    stim_idx["start"] = events[0]
    stim_idx["stims"] = events[1:]
    return stim_idx

def subset_events_by_time(event_dict, fs, t0_s, t1_s):
    t0 = int(t0_s * fs)
    t1 = int(t1_s * fs)

    stims = np.asarray(event_dict["stims"], dtype=int)
    keep = (stims >= t0) & (stims <= t1)

    out = dict(event_dict)
    out["stims"] = stims[keep].tolist()
    return out



# GRID REMAP / DIFFERENTIAL


def remap_grid(f_grid):
    grid_map = np.array([
        17, 16, 15, 14, 13, 9, 5, 1,
        22, 21, 20, 19, 18, 10, 6, 2,
        27, 26, 25, 24, 23, 11, 7, 3,
        32, 31, 30, 29, 28, 12, 8, 4,
        33, 34, 35, 36, 37, 53, 57, 61,
        38, 39, 40, 41, 42, 54, 58, 62,
        43, 44, 45, 46, 47, 55, 59, 63,
        48, 49, 50, 51, 52, 56, 60, 64
    ]) - 1
    return f_grid[grid_map, :]

def apply_differential_mode(f_grid_t, mode="double_col"):
    """
    Input:
        f_grid_t shape = (T, 64)
    Output:
        xdiff shape = (T, 64), invalid positions filled with NaN
    """
    F0 = f_grid_t.T.reshape(8, 8, -1)   # (8, 8, T)
    T = F0.shape[2]
    F = np.full((8, 8, T), np.nan, dtype=F0.dtype)

    if mode in [None, "monopolar"]:
        F[:] = F0
    elif mode == "row":
        d = np.diff(F0, axis=1)      # (8, 7, T)
        F[:, 0:7, :] = d
    elif mode == "column":
        d = np.diff(F0, axis=0)      # (7, 8, T)
        F[0:7, :, :] = d
    elif mode == "double_col":
        sd_col = np.diff(F0, axis=1)                 # (8, 7, T)
        dd_col = sd_col[:, 1:, :] - sd_col[:, :-1, :]   # (8, 6, T)
        F[:, 1:7, :] = dd_col
    elif mode == "double_row":
        sd_row = np.diff(F0, axis=0)                 # (7, 8, T)
        dd_row = sd_row[1:, :, :] - sd_row[:-1, :, :]   # (6, 8, T)
        F[1:7, :, :] = dd_row
    else:
        raise ValueError(f"Unknown differential mode: {mode}")

    xdiff = F.reshape(64, T).T
    return xdiff



# GRID RECRUITMENT


def compute_grouped_grid_recruitment(
    xdiff,
    f_aux,
    stim_events,
    fs,
    vis_win_l_ms,
    vis_win_u_ms,
    hreflex_window_ms,
    mwave_window_ms,
    iso_aux_index,
    grid_channel
):
    s_ms_factor = fs / 1000.0
    win_samp = int((vis_win_l_ms + vis_win_u_ms) * s_ms_factor)

    href0 = int((vis_win_l_ms + hreflex_window_ms[0]) * s_ms_factor)
    href1 = int((vis_win_l_ms + hreflex_window_ms[1]) * s_ms_factor)
    mw0   = int((vis_win_l_ms + mwave_window_ms[0]) * s_ms_factor)
    mw1   = int((vis_win_l_ms + mwave_window_ms[1]) * s_ms_factor)

    stim_amp_raw = np.zeros(len(stim_events), dtype=float)
    for i, event_idx in enumerate(stim_events):
        stim_amp_raw[i] = np.mean(f_aux[iso_aux_index, int(event_idx - 250):int(event_idx - 50)])

    stim_amp_norm = stim_amp_raw / np.max(stim_amp_raw)
    stim_amp_group = np.round(stim_amp_norm, 1)
    unique_levels = np.unique(stim_amp_group)

    h_means = np.full(len(unique_levels), np.nan)
    m_means = np.full(len(unique_levels), np.nan)
    raw_mean_per_level = np.full(len(unique_levels), np.nan)

    scatter_x_h = []
    scatter_y_h = []
    scatter_x_m = []
    scatter_y_m = []

    for cond_idx, stim_level in enumerate(unique_levels):
        matched_events = np.where(stim_amp_group == stim_level)[0]

        h_vals = []
        m_vals = []
        raw_vals = []

        for ev in matched_events:
            event_idx = stim_events[ev]

            sl0 = int(event_idx - vis_win_l_ms * s_ms_factor)
            sl1 = int(event_idx + vis_win_u_ms * s_ms_factor)

            if sl0 < 0 or sl1 > xdiff.shape[0]:
                continue

            epoch = xdiff[sl0:sl1, grid_channel]
            if len(epoch) != win_samp:
                continue
            if np.all(np.isnan(epoch)):
                continue

            hreflex_emg = epoch[href0:href1]
            mwave_emg   = epoch[mw0:mw1]

            if len(hreflex_emg) == 0 or len(mwave_emg) == 0:
                continue

            h_pp = np.abs(np.nanmax(hreflex_emg) - np.nanmin(hreflex_emg))
            m_pp = np.abs(np.nanmax(mwave_emg) - np.nanmin(mwave_emg))

            h_vals.append(h_pp)
            m_vals.append(m_pp)
            raw_vals.append(stim_amp_raw[ev])

            scatter_x_h.append(stim_level)
            scatter_y_h.append(h_pp)
            scatter_x_m.append(stim_level)
            scatter_y_m.append(m_pp)

        if len(h_vals) > 0:
            h_means[cond_idx] = np.mean(h_vals)
            m_means[cond_idx] = np.mean(m_vals)
            raw_mean_per_level[cond_idx] = np.mean(raw_vals)

    if np.all(np.isnan(h_means)):
        raise ValueError("No valid H-reflex values were computed for the selected grid channel.")

    hmax_idx = np.nanargmax(h_means)
    hmax = h_means[hmax_idx]
    stim_at_hmax_norm = unique_levels[hmax_idx]
    stim_at_hmax_raw = raw_mean_per_level[hmax_idx]

    return {
        "unique_levels": unique_levels,
        "h_means": h_means,
        "m_means": m_means,
        "scatter_x_h": np.asarray(scatter_x_h),
        "scatter_y_h": np.asarray(scatter_y_h),
        "scatter_x_m": np.asarray(scatter_x_m),
        "scatter_y_m": np.asarray(scatter_y_m),
        "hmax": hmax,
        "stim_at_hmax_norm": stim_at_hmax_norm,
        "stim_at_hmax_raw": stim_at_hmax_raw,
        "grid_channel": grid_channel,
    }


def plot_grouped_grid_recruitment(results, out_png):
    plt.figure(figsize=(8, 6))

    plt.scatter(
        results["scatter_x_h"], results["scatter_y_h"],
        c="b", s=8, alpha=0.25, linewidth=0, label="H individual"
    )
    plt.scatter(
        results["scatter_x_m"], results["scatter_y_m"],
        c="r", s=8, alpha=0.25, linewidth=0, label="M individual"
    )

    plt.plot(results["unique_levels"], results["h_means"], c="b", marker="o", label="H mean")
    plt.plot(results["unique_levels"], results["m_means"], c="r", marker="o", label="M mean")

    plt.scatter(
        results["stim_at_hmax_norm"], results["hmax"],
        s=140, marker="*", c="k", label="Hmax"
    )

    plt.annotate(
        f"Hmax = {results['hmax']:.2f} uV\n"
        f"Norm stim = {results['stim_at_hmax_norm']:.1f}\n"
        f"Raw stim = {results['stim_at_hmax_raw']:.4f}\n"
        f"Grid CH = {results['grid_channel']}",
        xy=(results["stim_at_hmax_norm"], results["hmax"]),
        xytext=(10, 10),
        textcoords="offset points"
    )

    plt.xlim([-0.1, 1.1])
    plt.xlabel("Normalized stimulus amplitude")
    plt.ylabel("Peak-to-peak amplitude (uV)")
    plt.title(f"Grouped recruitment curve | Grid CH {results['grid_channel']}")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()



# MAIN


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"Reading: {FILE_PATH}")
    data = read_poly(FILE_PATH)

    f_grid = data[:64, :]
    f_aux = data[64:-3, :]
    f_trigs = data[-3, :].T

    event_dict = segment_trigs(f_trigs, threshold=EVENT_THRESHOLD)

    if SEGMENT_TIME_RANGE_S is not None:
        event_dict = subset_events_by_time(
            event_dict,
            FS,
            SEGMENT_TIME_RANGE_S[0],
            SEGMENT_TIME_RANGE_S[1]
        )

    if len(event_dict["stims"]) == 0:
        raise ValueError("No stim events found in selected range.")

    f_grid = remap_grid(f_grid)
    f_grid_t = filt_grid(f_grid, fs=FS).T
    xdiff = apply_differential_mode(f_grid_t, DIFFERENTIAL_MODE)

    if GRID_CHANNEL_TO_PLOT < 0 or GRID_CHANNEL_TO_PLOT > 63:
        raise ValueError("GRID_CHANNEL_TO_PLOT must be between 0 and 63.")

    if np.all(np.isnan(xdiff[:, GRID_CHANNEL_TO_PLOT])):
        raise ValueError(
            f"Grid channel {GRID_CHANNEL_TO_PLOT} is invalid under differential mode '{DIFFERENTIAL_MODE}'."
        )

    results = compute_grouped_grid_recruitment(
        xdiff=xdiff,
        f_aux=f_aux,
        stim_events=event_dict["stims"],
        fs=FS,
        vis_win_l_ms=VIS_WIN_L,
        vis_win_u_ms=VIS_WIN_U,
        hreflex_window_ms=HREFLEX_WINDOW_MS,
        mwave_window_ms=MWAVE_WINDOW_MS,
        iso_aux_index=ISO_AUX_INDEX,
        grid_channel=GRID_CHANNEL_TO_PLOT
    )

    print("\n===== RESULTS =====")
    print(f"Grid channel: {results['grid_channel']}")
    print(f"Hmax: {results['hmax']:.2f} uV")
    print(f"Normalized stimulus amplitude at Hmax: {results['stim_at_hmax_norm']:.1f}")
    print(f"Raw stimulus amplitude at Hmax: {results['stim_at_hmax_raw']:.4f}")

    out_png = os.path.join(
        OUTDIR,
        f"grouped_grid_recruitment_CH{GRID_CHANNEL_TO_PLOT}_{DIFFERENTIAL_MODE}.png"
    )
    plot_grouped_grid_recruitment(results, out_png)
    print(f"Saved plot to: {out_png}")


if __name__ == "__main__":
    main()