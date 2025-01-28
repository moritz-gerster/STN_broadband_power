"""Look through all rawdata, annotate bad segments, and save.

This script requires pre-processed data for better visual inspection. Can be
used to 1) annotate recording for the first time or to 2) review annotations.

1) to annotate for the first time:
    - select (descriptions='uncleaned')
    - will be saved to derivatives/annotations
2) review annotations:
    - select (descriptions='cleaned')
    - if annotations are modified, changes will be saved to
      derivatives/annotations, even if the original comes from
      artifact_annotation_main (Thomas annotations)

Once all files have been annotated, the script bidsify_sourcedata should be
run again to add all annotations and to remove all uncleaned files.

This script does not load concatenated data because proper concatenation only
works if all files are annotated already.

If data are preprocessed, the channel names have changed. The script
bidsify_sourcedata however loads bad channels and expects the original channel
names. Therefore it must be differentiated between annotating for the first
time and reviewing annotations. Bad channels are saved twice using
descriptions="NewNames", and "OriginalNames" respectively."""
from json import dump, load
from os import makedirs
from os.path import isfile
from tkinter.messagebox import askyesno, showinfo

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mne import channel_indices_by_type, Annotations
from mne.io import read_raw
from mne.preprocessing import annotate_amplitude, annotate_nan
from mne_bids import find_matching_paths

import scripts.config as cfg
from scripts.utils import _delete_dirty_files
from scripts._03_make_dataframe import _info_from_ch_names

# M1 chip requires tkinter to start before the script for some reason...
showinfo(message="Starting annotation.")


def annotate_bad_segments(root="derivatives/preprocessed",
                          subjects: list[str] = None,
                          processings: list[str] = None,
                          sessions: list[str] = None,
                          only_uncleaned: bool = False,
                          auto_annotate: bool = True,
                          recordings=cfg.RECORDINGS,
                          acquisitions=None,
                          runs=None,
                          tasks=None,
                          annotate_dbs=False,
                          overwrite: bool = True):
    """Load data, inspect interactively, and save."""

    # decide whether to review present annotations or to add only missing ones
    descriptions = "uncleaned" if only_uncleaned else None
    bids_paths = find_matching_paths(root=root,
                                     processings=processings,
                                     subjects=subjects,
                                     descriptions=descriptions,
                                     sessions=sessions,
                                     recordings=recordings,
                                     acquisitions=acquisitions,
                                     runs=runs,
                                     extensions=".fif",
                                     tasks=tasks,
                                     check=False)

    matplotlib.use('TkAgg')

    for bids_path in bids_paths:

        if only_uncleaned:
            cleaned = find_matching_paths(root=cfg.ANNOTATIONS,
                                          processings=bids_path.processing,
                                          subjects=bids_path.subject,
                                          descriptions='NewNames',
                                          sessions=bids_path.session,
                                          recordings=bids_path.recording,
                                          acquisitions=bids_path.acquisition,
                                          tasks=tasks,
                                          check=False)
            if cleaned:
                continue

        # Load
        raw = read_raw(bids_path.fpath, verbose=0)

        # For inspection delete tremor annotations (Hirschmann) and add later
        # again
        if bids_path.recording == "Hirschmann" and only_uncleaned:
            anno_remove = _remove_tremor_annotations(raw)

        # rename channels for plotting
        new_names, old_names = _rename_chs_plot(raw.ch_names)
        raw.rename_channels(new_names)

        if auto_annotate and bids_path.description == "uncleaned":
            _annotate_noisy_flat_channels(raw)
            if bids_path.recording == "Tan" and annotate_dbs:
                _annotate_dbs(raw, picks=None, dbs_bad=True, threshold=.6,
                              plot_threshold=True)

        # Plot PSD before cleaning
        title = bids_path.basename + " (pre/post cleaning)"
        plot_psd_cleaning(raw, title=title)

        ch_order = _extract_types(raw.info)

        annotation_done = False
        while not annotation_done:
            raw.plot(title=bids_path.basename,
                     duration=10,
                     n_channels=27,
                     order=ch_order,
                     scalings=dict(ecog=100e-6, dbs=40e-6, emg=200e-6),
                     remove_dc=True,
                     highpass=None, block=True)

            # Plot PSD after cleaning
            plot_psd_cleaning(raw, title=title)
            annotation_done = askyesno(message="Are you done annotating?")

        # rename back to original names
        raw.rename_channels(old_names)

        if bids_path.recording == "Hirschmann" and only_uncleaned:
            # add deleted subset of annotations back to data
            annotations_new = raw.annotations
            raw.set_annotations(annotations_new + anno_remove)

        # Very short annotations happen by accident & can't be removed manually
        _remove_short_annotations(raw, thresh_ms=50)

        # Export annotations in separate folder which is tracked by github
        _export_annotations(raw.annotations, bids_path, overwrite=overwrite)
        _export_bad_chs(raw.info["bads"], bids_path, overwrite=overwrite)

        # Save changes
        # Don't overwrite raw files. Things get dirty. Clearly separate the
        # task of each script. This script modifies annotations,
        # bidsify_sourcedata applies them to the raw data. Also, concatenate
        # script needs to be run again anyways.
        bids_path.update(description="cleaned")
        raw.load_data()  # call before deletion of dirty files
        if overwrite:
            _delete_dirty_files(bids_path)
        raw.save(bids_path.fpath, overwrite=overwrite)

        keep_annotating = askyesno(message="Continue with next file?")
        if not keep_annotating:
            break


def _extract_types(info, show_distant=True):
    type_dic = channel_indices_by_type(info)
    ecog = type_dic['ecog']

    ch_nms = info.ch_names
    matched_names = _info_from_ch_names(ch_nms)
    (_, _, _, _, _, distant_bip_chs) = matched_names
    dbs_bip = [idx for idx in type_dic['dbs'] if "-" in ch_nms[idx]
               and ch_nms[idx] not in distant_bip_chs]
    distant_bip_chs = [idx for idx in type_dic['dbs'] if ch_nms[idx]
                       in distant_bip_chs]
    dbs_mon = [idx for idx in type_dic['dbs'] if "-" not in ch_nms[idx]]
    dbs_lar = [idx for idx in type_dic['dbs']
               if "_LSTN" in ch_nms[idx] or "_RSTN" in ch_nms[idx]]
    dbs_mon = [idx for idx in dbs_mon if idx not in dbs_lar]
    dbs_bip_dir = [idx for idx in dbs_bip if "a" in ch_nms[idx]
                   or "b" in ch_nms[idx] or "c" in ch_nms[idx]]
    dbs_bip_nondir = [idx for idx in dbs_bip if idx not in dbs_bip_dir]
    ch_order = dbs_bip_nondir + dbs_lar + dbs_mon + dbs_bip_dir + ecog
    if show_distant:
        ch_order += distant_bip_chs
    return ch_order


def _rename_chs_plot(ch_names):
    new_names = {ch: ch.replace("_STN_MT", "") for ch in ch_names}
    new_names = {key: val.replace("_STN_BS", "")
                 for key, val in new_names.items()}
    new_names = {key: val.replace("_STN_SJ", "")
                 for key, val in new_names.items()}
    new_names = {key: val.replace("_SMC_AT", "")
                 for key, val in new_names.items()}
    new_names = {key: val.replace("_AO", "")
                 for key, val in new_names.items()}
    new_names = {key: val.replace("_TM", "")
                 for key, val in new_names.items()}
    old_names = {v: k for k, v in new_names.items()}
    return new_names, old_names


def _export_annotations(annotations, bids_path, suffix=".csv",
                        overwrite=False):
    """Export annotations to a separate folder which is tracked by GitHub."""
    # Export annotations to derivatives folder which is tracked by github
    save_dic = dict(suffix="events", extension=".tsv", description=None,
                    root=cfg.ANNOTATIONS)
    bids_path_annotations = bids_path.copy().update(**save_dic)
    makedirs(bids_path_annotations.directory, exist_ok=True)
    # Manually change ".tsv" extension to ".csv" because write_raw_bids
    # only accepts ".tsv" while mne.annotations.save() only accepts ".csv".
    save_path = bids_path_annotations.fpath.with_suffix(suffix)
    annotations.save(save_path, overwrite=overwrite)


def _export_bad_chs(bad_channels, bids_path, overwrite=False):
    """Export annotations to a separate folder which is tracked by GitHub.

    Important: Files are first preprocessed and channels renamed. However,
    here the bad channels need to be annotated, named to their original names
    and then to be preprocessed again."""
    save_dic = dict(suffix="channels", extension=".json", root=cfg.ANNOTATIONS,
                    description="NewNames")
    save_path = bids_path.copy().update(**save_dic)
    makedirs(save_path.directory, exist_ok=True)
    if isfile(save_path.fpath) and not overwrite:
        raise FileExistsError(f"File already exists: {save_path.fpath}")

    with open(save_path.fpath, 'w') as file:
        dump(bad_channels, file, indent=2)
    if bids_path.recording != "Neumann":
        return None
    # rename to original channel names before preprocessing
    original_nms = open(cfg.NEW_CH_NAMES)
    original_nms = load(original_nms)
    original_nms = original_nms[bids_path.subject]
    original_nms = {v: k for k, v in original_nms.items()}
    new_bad_chs = []
    for ch in bad_channels:
        if ch in original_nms:
            new_bad_chs.append(original_nms[ch])
    if bids_path.subject == "EL005" and bids_path.session == "EcogLfpMedOff02":
        # very annoying: sub 5 has different channel names in ON and OFF
        # condition because of varying number of channels
        new_bad_chs.extend(["LFP_L_06_STN_MT", "LFP_L_07_STN_MT"])
    save_path.update(description="OriginalNames")
    with open(save_path.fpath, 'w') as file:
        dump(new_bad_chs, file, indent=2)


def _remove_tremor_annotations(raw):
    """For inspection delete tremor annotations (and add later again)."""
    # get set of annotation descriptions and select which to keep for plotting
    annos_orig = raw.annotations
    anno_names = set(annos_orig.description)
    plot_annos = [anno for anno in anno_names
                  if "trem" not in anno and 'undef' not in anno]
    anno_plot_list = anno_names.intersection(plot_annos)

    # separate annotations to keep for plotting and those to remove
    df_annotations = annos_orig.to_data_frame()
    mask_plot = df_annotations.description.isin(anno_plot_list)

    # convert dataframe back to annotations
    anno_plot = Annotations(annos_orig.onset[mask_plot],
                            annos_orig.duration[mask_plot],
                            annos_orig.description[mask_plot],
                            orig_time=annos_orig.orig_time)

    anno_rm = Annotations(annos_orig.onset[~mask_plot],
                          annos_orig.duration[~mask_plot],
                          annos_orig.description[~mask_plot],
                          orig_time=annos_orig.orig_time)

    # set annotations for plotting
    raw.set_annotations(anno_plot)
    return anno_rm


def _remove_short_annotations(raw, thresh_ms=30):
    """Remove very brief annotations which happen by accident."""
    thresh_s = thresh_ms / 1000
    short_annotations = raw.annotations.duration < thresh_s
    raw.set_annotations(raw.annotations[~short_annotations])


def _annotate_noisy_flat_channels(raw):
    # Check that there are no nan segments
    annot_nan = annotate_nan(raw)
    assert len(annot_nan) == 0, "Nan Channels in recording!"

    # Find flat channels and segments
    flat_segments, flat_channels = annotate_amplitude(raw, flat=1e-15,
                                                      bad_percent=99.9)
    raw.info['bads'] += flat_channels
    raw.set_annotations(raw.annotations + flat_segments)


def plot_psd_cleaning(raw, title=None, fmax=None, sharey=True,
                      show_bandwidth=False):
    """Plot PSDs of raw data before and after cleaning."""
    # Find number of ch_types
    all_chs = raw.copy().pick("data", exclude=[])
    ch_types = all_chs.get_channel_types(picks="data", unique=True,
                                         only_data_chs=True)
    plot_cols = len(ch_types)

    # PSD kwargs
    fmax = fmax if fmax else raw.info["lowpass"]
    psd_kwargs = dict(n_fft=int(raw.info["sfreq"]), fmax=fmax)
    psd_kwargs_good = dict(reject_by_annotation=True, picks=None, **psd_kwargs)
    psd_kwargs_bad = dict(reject_by_annotation=False, picks=all_chs.ch_names,
                          **psd_kwargs)

    # Plot kwargs
    plot_kwargs = dict(xscale="log", dB=False, show=False)
    plot_kwargs_bad = dict(exclude=[], picks=all_chs.ch_names, **plot_kwargs)
    plot_kwargs_good = dict(exclude="bads", **plot_kwargs)

    # Figure
    fig, axes = plt.subplots(2, plot_cols, figsize=(15, 9), sharex=True,
                             sharey=sharey, num=title)

    raw.compute_psd(**psd_kwargs_bad).plot(axes=axes[0], **plot_kwargs_bad)
    try:
        raw.compute_psd(**psd_kwargs_good).plot(axes=axes[1],
                                                **plot_kwargs_good)
        if plot_cols > 1:
            [ax.set_xlabel("Cleaned") for ax in axes[1, :].flatten()]
        else:
            axes[1].set_xlabel("Cleaned")
    except ValueError:
        # error if ch_type all bad because of axis mismatch. Solution: plot
        # all channels, even the bad ones.
        plot_kwargs_good["exclude"] = []
        plot_kwargs_good["picks"] = all_chs.ch_names
        psd_kwargs_good["picks"] = all_chs.ch_names
        raw.compute_psd(**psd_kwargs_good).plot(axes=axes[1],
                                                **plot_kwargs_good)
        if plot_cols > 1:
            [ax.set_xlabel("Cleaned") for ax in axes[1, :].flatten()]
        else:
            axes[1].set_xlabel("Cleaned")
    flat_axes = axes.flatten() if plot_cols > 1 else axes
    [ax.set_yscale("log") for ax in flat_axes]
    if plot_cols > 1:
        [ax.set_xlabel("Dirty") for ax in axes[0, :].flatten()]
    else:
        axes[0].set_xlabel("Dirty")
    if raw.info['line_freq'] and show_bandwidth:
        for ax in axes.flatten():
            bandwidth_line = ax.get_lines()[2]
            label = f"Analogue Bandwidth 0-{raw.info['line_freq']:.0f} Hz"
            ax.legend([bandwidth_line], [label])
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def _annotate_dbs(raw, picks=None, dbs_bad=False, threshold=.6, ch_type=None,
                  mark_transitions=True, plot_threshold=False):
    dbs_segments, scores = _get_dbs_segments(raw, picks,
                                             threshold=threshold,
                                             ch_type=ch_type,
                                             min_length_good=2,
                                             filter_freq=(800, 999),
                                             dbs_bad=dbs_bad)
    if plot_threshold:
        _plot_scores(scores, threshold)

    if mark_transitions:
        trans_start, trans_end = _get_transition_segments(dbs_segments)
        new_annotations = dbs_segments + trans_start + trans_end
    else:
        new_annotations = dbs_segments
    raw.set_annotations(raw.annotations + new_annotations)


def _get_transition_segments(dbs_segments, transition=1,
                             exclude_post_dbs=20):
    """Mark transitions of segments as bad to avoid borderline segregation
    of DBS vs no-DBS."""
    transitions_start = dbs_segments.copy()
    transitions_end = dbs_segments.copy()
    if not len(dbs_segments):
        return transitions_start, transitions_end

    # add 1 second buffer to always catch start and end of dbs
    transitions_start.onset -= transition / 2
    transitions_start.duration = transition

    dbs_ends = dbs_segments.onset + dbs_segments.duration
    transitions_end.onset = dbs_ends - transition
    transitions_end.duration = transition + exclude_post_dbs

    annotation = np.unique(dbs_segments.description)[0]
    trans_dic = {annotation: 'BAD_transition'}
    post_dic = {annotation: 'BAD_postDBS'}
    transitions_start.rename(trans_dic)
    transitions_end.rename(post_dic)
    return transitions_start, transitions_end


def _get_dbs_segments(raw, picks=None, threshold=10, ch_type=None,
                      min_length_good=0.1, filter_freq=(110, 140),
                      dbs_bad=False):
    """MODIFIED from mne.preprocessing.annotate_muscle_zscore.

    Detects data segments containing activity in the frequency range given by
    ``filter_freq`` whose envelope magnitude exceeds the specified
    threshold, when summed across channels and divided by ``sqrt(n_channels)``.
    False-positive transient peaks are prevented by low-pass filtering the
    resulting z-score time series at 4 Hz. Only operates on a single channel
    type, if ``ch_type`` is ``None`` it will select the first type in the list
    ``mag``, ``grad``, ``eeg``.

    Parameters
    ----------
    raw : instance of Raw
        Data to estimate segments with muscle artifacts.
    picks : list of str | None
        Channels to include. If None, all channels are used.
    threshold : float
        The threshold in uV for marking segments as containing DBS
        activity artifacts.
    ch_type : 'mag' | 'grad' | 'eeg' | None
        The type of sensors to use. If ``None`` it will take the first type in
        ``mag``, ``grad``, ``eeg``.
    min_length_good : float | None
        The shortest allowed duration of "good data" (in seconds) between
        adjacent annotations; shorter segments will be incorporated into the
        surrounding annotations. ``None`` is equivalent to ``0``.
        Default is ``0.1``.
    filter_freq : array-like, shape (2,)
        The lower and upper frequencies of the band-pass filter.
        Default is ``(110, 140)``.
    dbs_bad : bool
        Whether to annotate DBS as bad (to exclude it) or as neutral (to
        analyze it).

    Returns
    -------
    annot : mne.Annotations
        Periods with muscle artifacts annotated as BAD_muscle.
    scores_dbs : array
        Z-score values averaged across channels for each sample.
    """
    from mne.annotations import _adjust_onset_meas_date

    raw_copy = raw.copy().load_data()

    if ch_type:
        ch_type = {ch_type: True}
        raw_copy.pick_types(**ch_type)

    if picks:
        # don't pick bad channels
        picks = list(set(picks) - set(raw.info['bads']))
        raw_copy.pick(picks)

    raw_copy.filter(
        filter_freq[0],
        filter_freq[1],
        fir_design="firwin",
        pad="reflect_limited",
    )
    raw_copy.apply_hilbert(envelope=True)

    data = raw_copy.get_data(reject_by_annotation="NaN")
    finite_mask = ~np.isnan(data[0])
    sfreq = raw_copy.info["sfreq"]

    art_scores = data[:, finite_mask]
    art_scores = art_scores.sum(axis=0) / art_scores.shape[0]
    art_scores = np.abs(art_scores)

    scores_dbs = np.zeros(data.shape[1])
    scores_dbs[finite_mask] = art_scores

    dbs_mask = scores_dbs > threshold * 1e-6  # convert to uV
    # return muscle scores with NaNs
    scores_dbs[~finite_mask] = np.nan

    # remove artifact free periods shorter than min_length_good
    min_length_good = 0 if min_length_good is None else min_length_good
    min_samps = min_length_good * sfreq

    min_samps = int(min_samps)
    last_samp = raw_copy._last_samps[0] + 1
    assert len(dbs_mask) == last_samp
    for step in range(0, last_samp, min_samps):
        if dbs_mask[step:step + min_samps].any():
            dbs_mask[step:step + min_samps] = True

    annotation_description = 'BAD_DBS' if dbs_bad else 'DBS'
    annot = _annotations_from_mask(raw_copy.times, dbs_mask,
                                   annotation_description,
                                   orig_time=raw.info["meas_date"])
    _adjust_onset_meas_date(annot, raw)
    return annot, scores_dbs


def _annotations_from_mask(times, mask, annot_name, orig_time=None):
    """Construct annotations from boolean mask of the data."""
    from scipy.ndimage import distance_transform_edt
    from scipy.signal import find_peaks

    mask_tf = distance_transform_edt(mask)
    # Overcome the shortcoming of find_peaks
    # in finding a marginal peak, by
    # inserting 0s at the front and the
    # rear, then subtracting in index
    ins_mask_tf = np.concatenate((np.zeros(1), mask_tf, np.zeros(1)))
    left_midpt_index = find_peaks(ins_mask_tf)[0] - 1
    right_midpt_index = (
        np.flip(len(ins_mask_tf) - 1 - find_peaks(ins_mask_tf[::-1])[0]) - 1
    )
    onsets_index = left_midpt_index - mask_tf[left_midpt_index].astype(int) + 1
    ends_index = right_midpt_index + mask_tf[right_midpt_index].astype(int)
    # Ensure onsets_index >= 0,
    # otherwise the duration starts from the beginning
    onsets_index[onsets_index < 0] = 0
    # Ensure ends_index < len(times),
    # otherwise the duration is to the end of times
    if len(times) == len(mask):
        ends_index[ends_index >= len(times)] = len(times) - 1
    # To be consistent with the original code,
    # possibly a bug in tests code
    else:
        ends_index[ends_index >= len(mask)] = len(mask)
    onsets = times[onsets_index]
    ends = times[ends_index]
    durations = ends - onsets
    desc = [annot_name] * len(durations)
    return Annotations(onsets, durations, desc, orig_time=orig_time)


def _plot_scores(scores, threshold):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2)
    ax1 = axes[0]
    ax2 = axes[1]
    ax1.hist(scores, bins=100, range=(0, threshold*1e-6*2))
    # convert xtick labels to uV by multiplying with 1e6
    xticks = ax1.get_xticks()
    xticks = [f"{xtick*1e6:.0f}" for xtick in xticks]
    ax1.set_xticklabels(xticks)
    # indicate threshold
    ylim = ax1.get_ylim()
    ax1.vlines(threshold*1e-6, *ylim, color='r')
    ax2.hist(scores, bins=100)
    # convert xtick labels to uV by multiplying with 1e6
    xticks = ax2.get_xticks()
    xticks = [f"{xtick*1e6:.0f}" for xtick in xticks]
    ax2.set_xticklabels(xticks)
    # indicate threshold
    ylim = ax2.get_ylim()
    ax2.vlines(threshold*1e-6, *ylim, color='r')
    plt.show()


if __name__ == "__main__":
    annotate_bad_segments()
