#This code is for viewing specific recruitment curves of EMG channels

import argparse, sys, os, glob, time, json
from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Dell\Documents\RESEARCH\tmsi-python-interface-main\TMSiFileFormats\file_readers")
from poly5reader import Poly5Reader

print("Poly5Reader imported from:", Poly5Reader.__module__)
import inspect
print("Poly5Reader file:", inspect.getfile(Poly5Reader))


def read_poly(fname):
    path = fname if fname.endswith(".poly5") else fname + ".poly5"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Poly5 file not found: {path}")

    r = Poly5Reader(path)

    if hasattr(r, "samples") and r.samples is not None:
        return r.samples

    if hasattr(r, "readSamples"):
        samples = r.readSamples()
        if samples is not None:
            return samples

    if hasattr(r, "read_data_MNE"):
        raw = r.read_data_MNE()
        if hasattr(raw, "get_data"):
            return raw.get_data()
        return raw

    for m in ("readAll", "read_data", "read", "get_data"):
        if hasattr(r, m):
            out = getattr(r, m)()
            if out is not None:
                return out

    for a in ("data", "signal_data", "sample_data"):
        if hasattr(r, a):
            attr = getattr(r, a)
            if attr is not None:
                return attr

    candidates = [x for x in dir(r) if ("read" in x.lower() or "data" in x.lower() or "sam" in x.lower())]
    raise AttributeError(f"Poly5Reader has no readable data members. Candidates: {candidates}")


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band', analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def subset_events_by_time(event_dict, fs, t0_s, t1_s):
    t0 = int(t0_s * fs)
    t1 = int(t1_s * fs)

    stims = np.asarray(event_dict["stims"], dtype=int)
    keep = (stims >= t0) & (stims <= t1)

    out = dict(event_dict)
    out["stims"] = stims[keep].tolist()
    return out


def filt_GRID_rect(data, lowcut=20, highcut=500, fs=2000, order=2, notch_fs=50, notch_q=30, low_rect=0.1, high_rect=5):
    filt_out = np.zeros_like(data)
    for i in range(data.shape[0]):
        filt_out[i, :] = notch_filter(
            butter_bandpass_filter(data[i, :], lowcut, highcut, fs, order=3),
            notch_fs, fs, notch_q
        )
    emg_rectified = abs(filt_out) - np.min(abs(filt_out), axis=0).reshape(1, -1)
    low_pass = high_rect / (fs / 2)
    b2, a2 = butter(4, low_pass, btype='low')
    emg_envelope = np.zeros_like(data)
    for i in range(data.shape[0]):
        emg_envelope[i, :] = filtfilt(b2, a2, emg_rectified[i, :])
    return emg_envelope


def filt_GRID(data, lowcut=20, highcut=500, fs=2000, order=3, notch_fs=50, notch_q=30):
    filt_out = np.zeros_like(data)
    for i in range(data.shape[0]):
        filt_out[i, :] = notch_filter(
            butter_bandpass_filter(data[i, :], lowcut, highcut, fs, order=order),
            notch_fs, fs, notch_q
        )
    return filt_out


def notch(notch_freq, samp_freq, quality_factor=30):
    b, a = iirnotch(notch_freq, quality_factor, samp_freq)
    return b, a


def notch_filter(data, notch_fs, fs, q=30):
    b, a = notch(notch_fs, fs, q)
    y = filtfilt(b, a, data)
    return y


def segment_trigs(trigs, threshold=0):
    stim_idx = {}
    trigs = np.diff(trigs)
    events = np.where(trigs < -threshold)[0]
    stim_idx["start"] = events[0]
    stim_idx["stims"] = events[1:]
    return stim_idx


def get_selected_channels(args, n_channels=64):
    """
    Returns zero-based selected channel indices.
    User passes 1-based channel numbers, e.g. --channels 1 5 10
    If None, returns all channels.
    """
    if args.channels is None or len(args.channels) == 0:
        return list(range(n_channels))

    selected = []
    for ch in args.channels:
        if ch < 1 or ch > n_channels:
            raise ValueError(f"Channel {ch} is out of range. Valid channels are 1 to {n_channels}.")
        selected.append(ch - 1)

    return selected


def make_subplot_grid(n):
    """
    Choose a nearly square subplot layout for n plots.
    """
    if n <= 0:
        return 1, 1
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def plot_grid(args, f_grid, event_dict, out_path, flag):
    f_grid = filt_GRID(f_grid, fs=args.fs).T
    selected_channels = get_selected_channels(args, n_channels=f_grid.shape[1])

    s_ms_factor = args.fs / 1000
    plot_data_dict = np.empty((
        len(event_dict['stims']),
        int((args.vis_win_L + args.vis_win_U) * s_ms_factor),
        f_grid.shape[1]
    ))
    trial_SD = np.empty((len(event_dict['stims']), f_grid.shape[1]))

    for i, event_idx in enumerate(event_dict['stims']):
        event_data = f_grid[
            int(event_idx - args.vis_win_L * s_ms_factor):
            int(event_idx + args.vis_win_U * s_ms_factor), :
        ]
        event_data[
            int((args.vis_win_L - args.blank_win_L) * s_ms_factor):
            int((args.vis_win_L + args.blank_win_U) * s_ms_factor), :
        ] = np.zeros((
            int((args.blank_win_L + args.blank_win_U) * s_ms_factor),
            event_data.shape[1]
        ))
        plot_data_dict[i, :, :] = event_data
        trial_SD[i, :] = np.std(f_grid[int(event_idx - 1050):int(event_idx - 50), :], axis=0)

    rows, cols = make_subplot_grid(len(selected_channels))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    fig.suptitle(flag)

    max_val = np.max(np.abs(plot_data_dict))
    x_axis = np.linspace(-args.vis_win_L, args.vis_win_U, plot_data_dict.shape[1])
    axes_flat = axes.ravel()

    for plot_idx, ch in enumerate(selected_channels):
        ax = axes_flat[plot_idx]

        for k in range(plot_data_dict.shape[0]):
            ax.plot(x_axis, plot_data_dict[k, :, ch], alpha=0.5, linewidth=0.5)

        mean_sig = np.mean(plot_data_dict[:, :, ch], axis=0)
        std_sig = np.std(plot_data_dict[:, :, ch], axis=0)
        std_base = np.mean(trial_SD[:, ch], axis=0)
        yerr = mean_sig + std_sig

        ax.axvline(x=0, ymin=0.0, ymax=1.0, color='k')
        ax.axvline(x=20, ymin=0.0, ymax=1.0, color='c', alpha=0.25)
        ax.axhline(y=-std_base * 5.5, xmin=0.0, xmax=1.0, color='k', alpha=0.25)
        ax.axhline(y=std_base * 5.5, xmin=0.0, xmax=1.0, color='k', alpha=0.25)

        ax.plot(x_axis, mean_sig, c='k', alpha=0.75)
        ax.fill_between(x_axis, -yerr, yerr, color='k', alpha=0.2)
        ax.set_xlim([-args.vis_win_L, args.vis_win_U])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        max_val = max(std_base * 6, max_val)
        ax.set_ylim([-max_val, max_val])
        ax.set_xticks(np.linspace(-args.vis_win_L, args.vis_win_U, 7, dtype=int))
        ax.set_xticklabels(ax.get_xticks(), rotation=45)
        ax.set_title(f"CH {ch + 1}")

    for idx in range(len(selected_channels), len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.savefig(os.path.join(out_path, flag) + '.png', bbox_inches="tight", dpi=300)
    return


def plot_aux(args, f_grid, event_dict, out_path, incl_chan, dev_map):
    f_grid = filt_GRID(f_grid, fs=args.fs).T
    s_ms_factor = args.fs / 1000
    plot_data_dict = np.empty((
        len(event_dict['stims']),
        int((args.vis_win_L + args.vis_win_U) * s_ms_factor),
        f_grid.shape[1]
    ))
    trial_SD = np.empty((len(event_dict['stims']), f_grid.shape[1]))

    for i, event_idx in enumerate(event_dict['stims']):
        event_data = f_grid[
            int(event_idx - args.vis_win_L * s_ms_factor):
            int(event_idx + args.vis_win_U * s_ms_factor), :
        ]
        event_data[
            int((args.vis_win_L - args.blank_win_L) * s_ms_factor):
            int((args.vis_win_L + args.blank_win_U) * s_ms_factor), :
        ] = np.zeros((
            int((args.blank_win_L + args.blank_win_U) * s_ms_factor),
            event_data.shape[1]
        ))
        plot_data_dict[i, :, :] = event_data
        trial_SD[i, :] = np.std(f_grid[int(event_idx - 1050):int(event_idx - 50), :], axis=0)

    rows, cols = make_subplot_grid(len(incl_chan))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    fig.suptitle(dev_map["GRID"])

    x_axis = np.linspace(-args.vis_win_L, args.vis_win_U, plot_data_dict.shape[1])
    axes_flat = axes.ravel()

    for i, ch_name in enumerate(incl_chan):
        ax = axes_flat[i]

        for k in range(plot_data_dict.shape[0]):
            ax.plot(x_axis, plot_data_dict[k, :, i], alpha=0.5, linewidth=0.5)

        mean_sig = np.mean(plot_data_dict[:, :, i], axis=0)
        std_sig = np.std(plot_data_dict[:, :, i], axis=0)
        std_base = np.mean(trial_SD[:, i], axis=0)
        yerr = mean_sig + std_sig

        ax.set_title(dev_map[ch_name])
        ax.axvline(x=0, ymin=0.0, ymax=1.0, color='k')
        ax.axvline(x=20, ymin=0.0, ymax=1.0, color='c', alpha=0.25)
        ax.axhline(y=-std_base * 5.5, xmin=0.0, xmax=1.0, color='k', alpha=0.25)
        ax.axhline(y=std_base * 5.5, xmin=0.0, xmax=1.0, color='k', alpha=0.25)

        ax.plot(x_axis, mean_sig, c='k', alpha=0.75)
        ax.fill_between(x_axis, -yerr, yerr, color='k', alpha=0.2)
        ax.set_xlim([-args.vis_win_L, args.vis_win_U])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        max_val = max(std_base * 6, np.max(plot_data_dict[:, :, i]))
        ax.set_ylim([-max_val, max_val])
        ax.set_xticks(np.linspace(-args.vis_win_L, args.vis_win_U, 7, dtype=int))
        ax.set_xticklabels(ax.get_xticks(), rotation=45)

    for idx in range(len(incl_chan), len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.savefig(os.path.join(out_path, dev_map["GRID"]) + '_AUX.png', bbox_inches="tight", dpi=300)


def gen_MEP_vis(args):
    with open(os.path.join(args.data_dir, args.particiapnt_ID, args.exp_date, 'musclemap.json'), 'r') as j:
        muscle_map = json.loads(j.read())

    for key in muscle_map.keys():
        if args.MEP:
            file_ID, time_ID = args.fname.split('-')
            file_path = os.path.join(args.data_dir, args.particiapnt_ID, args.exp_date, 'MEPs', key + '-' + time_ID)
            out_path = os.path.join(args.data_dir, args.particiapnt_ID, args.exp_date, 'MEPs', 'plots', time_ID)
        else:
            file_ID, time_ID = args.fname.split('-')
            file_path = os.path.join(args.data_dir, args.particiapnt_ID, args.exp_date, file_ID[:-3] + key + '-' + time_ID)
            out_path = os.path.join(args.data_dir, args.particiapnt_ID, args.exp_date, 'plots', time_ID)

        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        print(file_path)
        incl_chan = []
        for chan in muscle_map[key].keys():
            if muscle_map[key][chan] != 'N/A':
                incl_chan.append(chan)

        data = read_poly(file_path)
        f_trigs = data[-3, :].T
        event_dict = segment_trigs(f_trigs)

        if len(data) > 67 and args.AUX:
            f_aux = data[64:-3, :]
            plot_aux(args, f_aux, event_dict, out_path, incl_chan[1:], muscle_map[key])

        if args.GRID:
            f_grid = data[:64, :]
            plot_grid(args, f_grid, event_dict, out_path, muscle_map[key]["GRID"])


def gen_rcrt_th_vis(args):
    with open(os.path.join(args.data_dir, args.particiapnt_ID, args.exp_date, 'musclemap.json'), 'r') as j:
        muscle_map = json.loads(j.read())

    for key in ("FLX",):
        file_path = os.path.join(
            args.data_dir,
            args.particiapnt_ID,
            args.exp_date,
            'thresholding',
            args.fname
        )
        out_path = os.path.join(
            args.data_dir,
            args.particiapnt_ID,
            args.exp_date,
            'thresholding',
            'selected_plots',
            args.fname
        )

        print("Looking for:", file_path if file_path.endswith(".poly5") else file_path + ".poly5")

        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        data = read_poly(file_path)
        f_trigs = data[-3, :].T
        event_dict = segment_trigs(f_trigs, threshold=3)
        f_aux = data[64:-3, :]
        f_grid = data[:64, :]

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
        f_grid = f_grid[grid_map, :]

        segments = [
            (36, 96, 'seg_1'),
            (146, 204, 'seg_2'),
        ]

        for t0, t1, tag in segments:
            seg_events = subset_events_by_time(event_dict, args.fs, t0, t1)

            if len(seg_events['stims']) == 0:
                print(f"[WARNING] No stim events found in time segment {t0}-{t1}s for {key}. Skipping.")
                continue

            plot_flag = f"{muscle_map[key]['GRID']}_{tag}"
            plot_grid_MEPs(args, f_grid, f_aux, seg_events, out_path, plot_flag)
            plot_grid_recruitment(args, f_grid, f_aux, seg_events, out_path, plot_flag)


def plot_grid_MEPs(args, f_grid, f_aux, event_dict, out_path, flag):
    f_grid = filt_GRID(f_grid, fs=args.fs).T
    selected_channels = get_selected_channels(args, n_channels=f_grid.shape[1])

    s_ms_factor = args.fs / 1000
    iso = args.iso_aux_index
    stim_amp_all = np.zeros((len(event_dict['stims'])))

    for i, event_idx in enumerate(event_dict['stims']):
        stim_amp_all[i] = np.mean(f_aux[iso, int(event_idx - 250):int(event_idx - 50)])
    stim_amp_all = np.round(stim_amp_all / np.max(stim_amp_all), 1)

    rows, cols = make_subplot_grid(len(selected_channels))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    fig.suptitle(flag)

    c_list = matplotlib.cm.rainbow(np.linspace(0, 1, len(np.unique(stim_amp_all))))
    axes_flat = axes.ravel()

    for cond_idx, stim_amp in enumerate(reversed(np.unique(stim_amp_all))):
        matched_events = np.where(stim_amp_all == stim_amp)[0]
        plot_data_dict = np.zeros((
            len(matched_events),
            int((args.vis_win_L + args.vis_win_U) * s_ms_factor),
            f_grid.shape[1]
        ))

        for i, event in enumerate(matched_events):
            event_idx = event_dict['stims'][event]
            event_data = f_grid[
                int(event_idx - args.vis_win_L * s_ms_factor):
                int(event_idx + args.vis_win_U * s_ms_factor), :
            ]
            event_data[
                int((args.vis_win_L - args.blank_win_L) * s_ms_factor):
                int((args.vis_win_L + args.blank_win_U) * s_ms_factor), :
            ] = np.zeros((
                int((args.blank_win_L + args.blank_win_U) * s_ms_factor),
                event_data.shape[1]
            ))
            plot_data_dict[i, :, :] = event_data

        x_axis = np.linspace(-args.vis_win_L, args.vis_win_U, plot_data_dict.shape[1])

        for plot_idx, ch in enumerate(selected_channels):
            ax = axes_flat[plot_idx]

            for k in range(plot_data_dict.shape[0]):
                ax.plot(x_axis, plot_data_dict[k, :, ch], alpha=0.5, linewidth=0.5, color=c_list[cond_idx])

            mean_sig = np.mean(plot_data_dict[:, :, ch], axis=0)
            ax.axvline(x=0, ymin=0.0, ymax=1.0, color='k')
            ax.axvline(x=20, ymin=0.0, ymax=1.0, color='c', alpha=0.25)
            ax.set_ylim([-args.max_val, args.max_val])
            ax.set_title(f"CH {ch + 1}")

            ax.plot(x_axis, mean_sig, color=c_list[cond_idx], alpha=0.75)
            ax.set_xlim([-args.vis_win_L, args.vis_win_U])
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            ax.set_xticks(np.linspace(-args.vis_win_L, args.vis_win_U, 7, dtype=int))
            ax.set_xticklabels(ax.get_xticks(), rotation=45)

    for idx in range(len(selected_channels), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.savefig(os.path.join(out_path, flag) + '_MEPs.svg', bbox_inches="tight", dpi=50)


def compute_window_metrics(window_emg, window_start_ms, fs):
    if window_emg.size == 0:
        return np.array([]), np.array([])

    p2p = np.nanmax(window_emg, axis=1) - np.nanmin(window_emg, axis=1)
    peak_idx = np.nanargmax(np.abs(window_emg), axis=1)
    ttp_ms = window_start_ms + (peak_idx / fs) * 1000.0

    return p2p, ttp_ms


def plot_grid_recruitment(args, f_grid, f_aux, event_dict, out_path, flag):
    f_grid = filt_GRID(f_grid, fs=args.fs).T

    F0 = f_grid.T.reshape(8, 8, -1)
    T = F0.shape[2]

    F = np.full((8, 8, T), np.nan, dtype=F0.dtype)
    mode = args.differentialaxis

    if mode in [None, "monopolar"]:
        F[:] = F0
    elif mode == "row":
        d = np.diff(F0, axis=1)
        F[:, 0:7, :] = d
    elif mode == "column":
        d = np.diff(F0, axis=0)
        F[0:7, :, :] = d
    elif mode == "double_col":
        sd_col = np.diff(F0, axis=1)
        dd_col = sd_col[:, 1:, :] - sd_col[:, :-1, :]
        F[:] = np.nan
        F[:, 1:7, :] = dd_col
    elif mode == "double_row":
        sd_row = np.diff(F0, axis=0)
        dd_row = sd_row[1:, :, :] - sd_row[:-1, :, :]
        F[:] = np.nan
        F[1:7, :, :] = dd_row
    else:
        raise ValueError(f"Unknown differentialaxis: {mode}")

    rows_all, cols_all = 8, 8
    xdiff = F.reshape(rows_all * cols_all, T).T

    selected_channels = get_selected_channels(args, n_channels=xdiff.shape[1])

    s_ms_factor = args.fs / 1000.0
    iso = args.iso_aux_index

    stim_amp_all = np.zeros(len(event_dict['stims']))
    for i, event_idx in enumerate(event_dict['stims']):
        raw_val = np.mean(f_aux[iso, int(event_idx - 250):int(event_idx - 50)])
        stim_amp_all[i] = raw_val

    stim_amp_all = np.round(stim_amp_all / np.max(stim_amp_all), 1)
    uniq_stim = np.unique(stim_amp_all)

    rows, cols = make_subplot_grid(len(selected_channels))
    fig_rcrt, axes_rcrt = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    fig_rcrt.suptitle(flag)
    axes_flat = axes_rcrt.ravel()

    Hreflex_list = np.full((len(uniq_stim), xdiff.shape[1]), np.nan)
    Mwave_list = np.full((len(uniq_stim), xdiff.shape[1]), np.nan)
    H_ttp_list = np.full((len(uniq_stim), xdiff.shape[1]), np.nan)
    M_ttp_list = np.full((len(uniq_stim), xdiff.shape[1]), np.nan)

    h0 = int((args.vis_win_L + args.Hreflex_win[0]) * s_ms_factor)
    h1 = int((args.vis_win_L + args.Hreflex_win[1]) * s_ms_factor)
    m0 = int((args.vis_win_L + args.Mwave_win[0]) * s_ms_factor)
    m1 = int((args.vis_win_L + args.Mwave_win[1]) * s_ms_factor)

    for cond_idx, stim_amp in enumerate(uniq_stim):
        matched_events = np.where(stim_amp_all == stim_amp)[0]
        n_ch = xdiff.shape[1]
        win_samp = int((args.vis_win_L + args.vis_win_U) * s_ms_factor)
        plot_data_dict = np.full((len(matched_events), win_samp, n_ch), np.nan)

        for ii, event in enumerate(matched_events):
            event_idx = event_dict['stims'][event]
            event_data = xdiff[
                int(event_idx - args.vis_win_L * s_ms_factor):
                int(event_idx + args.vis_win_U * s_ms_factor), :
            ]
            plot_data_dict[ii, :, :] = event_data

        for plot_idx, ch in enumerate(selected_channels):
            ax = axes_flat[plot_idx]
            ch_data = plot_data_dict[:, :, ch]

            if np.all(np.isnan(ch_data)):
                ax.axis("off")
                continue

            Hreflex_emg = ch_data[:, h0:h1]
            Mwave_emg = ch_data[:, m0:m1]

            Hreflex, H_ttp = compute_window_metrics(Hreflex_emg, args.Hreflex_win[0], args.fs)
            Mwave, M_ttp = compute_window_metrics(Mwave_emg, args.Mwave_win[0], args.fs)

            mean_hreflex = np.nanmean(Hreflex)
            mean_mwave = np.nanmean(Mwave)
            mean_h_ttp = np.nanmean(H_ttp)
            mean_m_ttp = np.nanmean(M_ttp)

            Hreflex_list[cond_idx, ch] = mean_hreflex
            Mwave_list[cond_idx, ch] = mean_mwave
            H_ttp_list[cond_idx, ch] = mean_h_ttp
            M_ttp_list[cond_idx, ch] = mean_m_ttp

            ax.scatter(stim_amp * np.ones_like(Hreflex), Hreflex, c='b', s=5, alpha=0.25, linewidth=0)
            ax.scatter(stim_amp * np.ones_like(Mwave), Mwave, c='r', s=5, alpha=0.25, linewidth=0)

            ax.scatter(stim_amp, mean_hreflex, c='b', s=30, linewidth=0)
            ax.scatter(stim_amp, mean_mwave, c='r', s=30, linewidth=0)

            ax.set_xlim([-0.1, 1.1])
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

    for plot_idx, ch in enumerate(selected_channels):
        ax = axes_flat[plot_idx]
        if not ax.axison:
            continue

        ax.plot(uniq_stim, Hreflex_list[:, ch], c='b')
        ax.plot(uniq_stim, Mwave_list[:, ch], c='r')
        ax.set_title(f"CH {ch + 1}", fontsize=8)

        if np.any(~np.isnan(Hreflex_list[:, ch])):
            hmax_idx = np.nanargmax(Hreflex_list[:, ch])
            hmax_y = Hreflex_list[hmax_idx, ch]
            hmax_x = uniq_stim[hmax_idx]
            hmax_ttp = H_ttp_list[hmax_idx, ch]
            m_at_hmax = Mwave_list[hmax_idx, ch]
            m_ttp_at_hmax = M_ttp_list[hmax_idx, ch]

            ax.scatter(hmax_x, hmax_y, c='k', s=60, marker='*', zorder=5)

            txt = (
                f"Hmax={hmax_y:.1f} uV\n"
                f"H ttp={hmax_ttp:.2f} ms\n"
                f"M p2p={m_at_hmax:.1f} uV\n"
                f"M ttp={m_ttp_at_hmax:.2f} ms"
            )

            ax.text(
                0.03, 0.97, txt,
                transform=ax.transAxes,
                va='top', ha='left',
                fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
            )

    for idx in range(len(selected_channels), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig_rcrt.savefig(
        os.path.join(out_path, flag) + '_recruitment.svg',
        bbox_inches="tight", dpi=80
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description='Poly5_2_pkl')

    parser.add_argument('--data_dir', default=r"C:\Users\Dell\Documents\data", type=str, help="Data directory")
    parser.add_argument('--fs', default=2000, type=int, help="sampling freq of data")
    parser.add_argument('--vis_win_L', default=20, type=int, help="time in ms before stim")
    parser.add_argument('--vis_win_U', default=60, type=int, help="time in ms after stim")
    parser.add_argument('--blank_win_L', default=1.5, type=int, help="time in ms before stim to blank")
    parser.add_argument('--blank_win_U', default=1.5, type=int, help="time in ms after stim to blank")
    parser.add_argument('--max_val', default=750, type=int, help="Yaxis max value for MEPs (uV)")
    parser.add_argument('--Hreflex_win', default=[20, 30], type=list, help="expected window for H reflex peak")
    parser.add_argument('--Mwave_win', default=[10, 15], type=list, help="expected window for M wave peak")
    parser.add_argument('--GRID', default=True, type=bool, help="Plot grid")
    parser.add_argument('--AUX', default=True, type=bool, help="Plot other chans")

    today = time.strftime("%Y%m%d")
    parser.add_argument('--exp_date', default='20260401', type=str, help="Data directory")
    parser.add_argument('--MEP', default=False, type=bool, help="Is the file an MEP scan")
    parser.add_argument('--thresholding', default=True, type=bool, help="Is the file a recruitment curve")
    parser.add_argument('--particiapnt_ID', default='fairy', type=str, help="Data directory")
    parser.add_argument('--differentialaxis', default='row', type=str, help="Data directory")
    parser.add_argument('--fname', default="thresholding_5FLX-20260401_124307", type=str, help="File name of the trial")

    parser.add_argument(
        '--diff', '--differential', dest='differentialaxis',
        type=str.lower,
        choices=['monopolar', 'row', 'column', 'double_col', 'double_row'],
        default='double_col',
        help="Channel referencing: 'monopolar' (no diff), 'row' (8x7), 'column' (7x8), 'double_col' (8x6), or 'double_row' (6x8)."
    )

    parser.add_argument(
        '--iso_aux_index',
        default=6,
        type=int,
        help="0-based row index within f_aux to use as ISO amplitude channel."
    )

    parser.add_argument(
        '--channels',
        nargs='+',
        type=int,
        default=[17,38],
        help="1-based channel numbers to plot. Example: --channels 1 2 3 10. Leave empty to plot all channels."
    )

    args = parser.parse_args(sys.argv[1:])

    if args.MEP:
        gen_MEP_vis(args)
    if args.thresholding:
        gen_rcrt_th_vis(args)