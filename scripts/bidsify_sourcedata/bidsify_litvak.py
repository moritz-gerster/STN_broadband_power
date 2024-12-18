"""Read sourcedata, bidsify, and save in rawdata."""
from json import load
from os import listdir
from os.path import basename, join
from warnings import warn

import numpy as np
from mne import read_annotations
from mne.io import read_raw
from mne_bids import BIDSPath, find_matching_paths, make_dataset_description
from tqdm import tqdm

import scripts.config as cfg
from scripts.bidsify_sourcedata import (_add_info, _correct_units, _move_files,
                                        loadmat)
from scripts.utils import _copy_files_and_dirs, _delete_dirty_files, _save_bids


def bidsify_sourcedata_litvak(only_cleaned=True) -> None:
    """Read sourcedata, bidsify, and save in rawdata."""
    raw_root = cfg.RAWDATA
    source_root = join(cfg.SOURCEDATA, "BIDS_Litvak_MEG_LFP")
    session_map = cfg.LITVAK_SESSION_MAP
    subject_map = cfg.LITVAK_SUBJECT_MAP
    raw_dir = "sourcedata_annotated"
    task = cfg.TASKS[0]
    # obtain filenames of all files to convert
    for subj_old, subj_new in tqdm(subject_map.items(), desc="Bidsify Litvak"):
        for cond, session in session_map.items():
            path_subj = join(source_root, raw_dir, task, subj_old, cond)
            files = listdir(path_subj)
            # choose the file that starts with "subj"
            fname = [file for file in files if file.startswith("subj")][0]
            fname = join(path_subj, fname)

            raw = read_raw(fname)
            # print(f'Sample rate: {raw.info["sfreq"]}\n'
            #       f'highpass: {raw.info["highpass"]}\n'
            #       f'lowpass: {raw.info["lowpass"]}')
            _set_litvak_chs(raw)
            _add_missing_chs(raw)
            _correct_units(raw)
            _add_mni_coords(raw, subj_old)
            # make bids path
            bids_path = BIDSPath(subject=subj_new,
                                 session=session,
                                 task=task.title(),
                                 run=1,
                                 suffix="ieeg",
                                 acquisition="StimOff",
                                 description="uncleaned",
                                 datatype="ieeg",
                                 recording="Litvak",
                                 extension=".vhdr",
                                 root=raw_root)

            _add_bad_segments(raw, bids_path, modify_bidspath=True,
                              only_cleaned=only_cleaned)
            _add_bad_channels(raw, bids_path, only_cleaned=only_cleaned)

            # Add infos to raw.info
            _add_info(raw, bids_path)
            raw.pick_types(meg=False, dbs=True, exclude=[])
            _save_bids(raw, bids_path)
            _delete_dirty_files(bids_path)

    # add dataset description in the end
    make_dataset_description(path=raw_root, **cfg.LITVAK_META, overwrite=True)
    _copy_files_and_dirs(source_root, join(raw_root, "meta_infos_Litvak"),
                         ["meta_infos"])
    # move participants.tsv, participants.json, and dataset_description.json
    # to meta_infos directory
    _move_files(raw_root, join(raw_root, "meta_infos_Litvak"))
    print(f"{basename(__file__).strip('.py')} done.")


def _add_missing_chs(raw):
    needed_chs = {'LFP_R_1-2_STN_MT', 'LFP_R_2-3_STN_MT', 'LFP_R_3-4_STN_MT',
                  'LFP_L_1-2_STN_MT', 'LFP_L_2-3_STN_MT', 'LFP_L_3-4_STN_MT'}
    missing_chs = needed_chs - set(raw.ch_names)
    if not missing_chs:
        return
    # add dummy channels
    dummy_ch = raw.copy().pick_types(dbs=True).pick(0).load_data()
    dummy_ch._data.fill(np.nan)
    raw.load_data()
    for ch in missing_chs:
        fill_ch = dummy_ch.copy()
        fill_ch.rename_channels({dummy_ch.ch_names[0]: ch})
        raw.add_channels([fill_ch])
        raw.info['bads'].append(ch)


def _set_litvak_chs(raw):
    ch_map = cfg.CH_NME_MAP_LITVAK
    ch_type_map = cfg.LITVAK_CHTYPE_MAP
    try:
        raw.rename_channels(ch_map)
    # if channels are missing:
    except ValueError:
        for key, val in ch_map.items():
            try:
                raw.rename_channels({key: val})
            except ValueError:
                continue
    try:
        raw.set_channel_types(ch_type_map)  # change channel types
    # if channels are missing:
    except ValueError:
        for key, val in ch_type_map.items():
            try:
                raw.set_channel_types({key: val})
            except ValueError:
                continue


def _add_bad_segments(raw, bids_path: BIDSPath, modify_bidspath=True,
                      only_cleaned=True):
    anno_path = _get_annotation_path(bids_path, "events")
    if anno_path is None:
        description = 'uncleaned'
    elif anno_path.startswith('derivatives/annotations'):
        description = 'cleaned'
    else:
        raise ValueError(f"No annotations found: {anno_path}")

    try:
        annotations = read_annotations(anno_path)
    except (FileNotFoundError, TypeError):
        msg = f"\n\n{bids_path.basename} has not been annotated yet!\n\n"
        if only_cleaned:
            raise FileNotFoundError(msg)
        else:
            warn(msg)
            if modify_bidspath:
                bids_path.update(description="uncleaned")
            return None
    except IndexError:
        # this just means that there are no annotations, this is fine
        annotations = None
    try:
        raw.set_annotations(annotations)
    except RuntimeError:
        raw.set_meas_date(0)  # set recording date to 1970-01-01
        raw.set_annotations(annotations)
    if modify_bidspath:
        bids_path.update(description=description, suffix="ieeg")
    return annotations


def _add_bad_channels(raw, bids_path, only_cleaned=True):
    anno_path = _get_annotation_path(bids_path, 'channels')
    try:
        with open(anno_path) as bad_chs:
            bad_chs = load(bad_chs)
    except (FileNotFoundError, TypeError):
        if only_cleaned:
            msg = f"\n\n{bids_path.basename} has not been annotated yet!\n\n"
            raise FileNotFoundError(msg)
        else:
            return None
    ch_mismatch = set(bad_chs) - set(raw.ch_names)
    if ch_mismatch:
        # bad channels could be bipolar channels such as STN_R_12 while the
        # channels are monopolar: STN_R_1 and STN_R_2
        msg = "Channel names don't match!"
        bipolar_stn_chs = {f"STN_{hemi}_{i}{i+1}" for i in range(1, 4)
                           for hemi in ["L", "R"]}
        assert set(ch_mismatch).issubset(bipolar_stn_chs), msg
        # remove bipolar channels from bad_chs
        bad_chs = set(bad_chs) - ch_mismatch
        # remove last character from bipolar bad channels to make monopolar:
        # STN_R_12 -> STN_R_1
        bad_mono_channels1 = {ch[:-1] for ch in ch_mismatch}
        # remove second last character from bipolar bad channels to make
        # monopolar: STN_R_12 -> STN_R_2
        bad_mono_channels2 = {ch[:-2] + ch[-1] for ch in ch_mismatch}
        bad_mono_channels = bad_mono_channels1.union(bad_mono_channels2)
        # apply mapping to convert to current channel names
        bad_mono_channels = {ch.replace('STN', 'LFP') + '_STN_MT'
                             for ch in bad_mono_channels}
        bad_chs = bad_chs.union(bad_mono_channels)
        ch_mismatch = set(bad_chs) - set(raw.ch_names)
        assert not ch_mismatch, msg
    raw.info["bads"] += list(bad_chs)


def _get_annotation_path(bids_path, suffix):
    anno_root = join('derivatives', 'annotations')
    bids_path = bids_path.copy().update(run=None)
    session = bids_path.session.strip('01')  # this changed since annotation
    bids_info = dict(subjects=bids_path.subject, sessions=session,
                     tasks=bids_path.task, runs=bids_path.run, root=anno_root,
                     suffixes=suffix, descriptions=None)
    anno_path = find_matching_paths(**bids_info)
    if anno_path:
        msg = f"More than one annotation file found for {bids_path}"
        assert len(anno_path) < 2, msg
        return str(anno_path[0].fpath)
    else:
        return None


def _add_mni_coords(raw, subj_old):
    lead_path = "sourcedata/BIDS_Litvak_MEG_LFP/meta_infos/lead_reconstruction"
    sub_dir = subj_old.replace("subj", "S")
    file_path = join(lead_path, sub_dir, "ea_reconstruction.mat")
    mat_file = loadmat(file_path)["reco"]

    try:
        mni_coords = mat_file["mni"]["coords_mm"]
    except TypeError:
        mni_coords = mat_file["mni"]  # mat73 file
    assert len(mni_coords) == 2, "More or less than two hemispheres found"
    mni_right = mni_coords[0]
    mni_left = mni_coords[1]
    if isinstance(mni_right, dict):
        mni_right = mni_right["coords_mm"]
        mni_left = mni_left["coords_mm"]
    assert isinstance(mni_right, np.ndarray), "Wrong type of mni coords"

    # mne requires SI units mm -> m
    mni_right *= 1e-3
    mni_left *= 1e-3

    x_ax, _, z_ax = 0, 1, 2
    right_hemi_positive = np.all(mni_right[:, x_ax] > 0)
    left_hemi_negative = np.all(mni_left[:, x_ax] < 0)
    assert right_hemi_positive and left_hemi_negative, "Wrong hemispheres"

    # Make sure coords are ordered from inferior to superior, L_0 -> L_4
    assert np.all(np.diff(mni_right[:, z_ax]) > 0), "Wrong order of z-coords"
    assert np.all(np.diff(mni_left[:, z_ax]) > 0), "Wrong order of z-coords"

    LFP_R_1 = mni_right[0]
    LFP_R_2 = mni_right[1]
    LFP_R_3 = mni_right[2]
    LFP_R_4 = mni_right[3]

    LFP_L_1 = mni_left[0]
    LFP_L_2 = mni_left[1]
    LFP_L_3 = mni_left[2]
    LFP_L_4 = mni_left[3]

    LFP_R_12 = (LFP_R_1 + LFP_R_2) / 2
    LFP_R_23 = (LFP_R_2 + LFP_R_3) / 2
    LFP_R_34 = (LFP_R_3 + LFP_R_4) / 2

    LFP_L_12 = (LFP_L_1 + LFP_L_2) / 2
    LFP_L_23 = (LFP_L_2 + LFP_L_3) / 2
    LFP_L_34 = (LFP_L_3 + LFP_L_4) / 2

    ch_names = ["LFP_R_1-2_STN_MT", "LFP_R_2-3_STN_MT", "LFP_R_3-4_STN_MT",
                "LFP_L_1-2_STN_MT", "LFP_L_2-3_STN_MT", "LFP_L_3-4_STN_MT"]
    coords = [LFP_R_12, LFP_R_23, LFP_R_34,
              LFP_L_12, LFP_L_23, LFP_L_34]
    for ch_name, coords in zip(ch_names, coords):
        try:
            raw.info["chs"][raw.ch_names.index(ch_name)]["loc"][:3] = coords
        except ValueError:
            # warn(f"{subj_old} misses channel {ch_name}.")
            continue


if __name__ == "__main__":
    bidsify_sourcedata_litvak()