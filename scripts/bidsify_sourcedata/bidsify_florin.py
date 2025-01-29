from json import load
from os import listdir
from os.path import basename, isdir, join
from pathlib import Path
from warnings import warn

import numpy as np
from mne import create_info, read_annotations
from mne.io import RawArray, read_raw
from mne_bids import BIDSPath, find_matching_paths, make_dataset_description
from tqdm import tqdm

import scripts.config as cfg
from scripts.bidsify_sourcedata import _add_info, _move_files, loadmat
from scripts.utils import _delete_dirty_files, _save_bids


def bidsify_sourcedata_florin(only_cleaned=True) -> None:
    """Read sourcedata, bidsify, and save in rawdata."""
    bids_path = BIDSPath(suffix="ieeg",
                         extension=".vhdr",  # fif does not work
                         description="uncleaned",
                         datatype="ieeg",
                         task="Rest",
                         recording="Florin",
                         root=cfg.RAWDATA)
    _bidsify_noise()
    source_path = join(cfg.SOURCEDATA_FLORIN, 'Share_LFP')
    subjects = sorted(listdir(source_path))
    # ignore files like '.DS_Store' and 'Kanalbelegung.txt'
    subjects = [sub for sub in subjects if isdir(join(source_path, sub))]
    sub_map = cfg.FLORIN_SUBJECT_MAP
    for subject in tqdm(subjects, desc="Bidsify Florin"):
        for cond in ['off', 'on']:
            dir_path = join(source_path, subject)
            raw, run = _raw_from_mat(dir_path, cond)
            # print(f'Sample rate: {raw.info["sfreq"]}\n'
            #         f'highpass: {raw.info["highpass"]}\n'
            #         f'lowpass: {raw.info["lowpass"]}')
            if raw is None:
                continue

            bids_path.update(subject=sub_map[subject],
                             run=run, acquisition='StimOff',
                             session=f'LfpMed{cond.capitalize()}01')

            _add_bad_channels(raw, bids_path, only_cleaned=only_cleaned)
            _add_bad_segments(raw, bids_path, only_cleaned=only_cleaned)

            # from scripts.utils_plot import plot_psd_units
            # plot_psd_units(raw, title='Florin')

            _add_info(raw, bids_path)
            _save_bids(raw, bids_path)
            _delete_dirty_files(bids_path)

    # add dataset description in the end
    meta_data = cfg.FLORIN_META
    raw_root = cfg.RAWDATA
    meta_data["path"] = raw_root
    make_dataset_description(**meta_data, overwrite=True)
    # move participants.tsv, participants.json, and dataset_description.json
    # to meta_infos directory
    meta_path = join(raw_root, "meta_infos_Florin")
    Path(meta_path).mkdir(parents=True, exist_ok=True)
    _move_files(raw_root, meta_path)
    print(f"{basename(__file__).strip('.py')} done.")


def _bidsify_noise():
    """Create single recording from 16 noise recordings and save."""
    amp_root = join(cfg.SOURCEDATA, "BIDS_Florin_NoiseFloorRecordings")

    raws = []
    for fname in listdir(amp_root):
        if not fname.endswith('.fif'):
            continue
        fpath = join(amp_root, fname)
        ch = fname.split('_')[1].strip('.fif')
        raw = read_raw(fpath, preload=True)
        raw.pick([f'EEG{ch}'])
        raws.append(raw)

    # concat to shortest time
    duration_min = min(raw.times[-1] for raw in raws)
    raws = [raw.crop(tmax=duration_min) for raw in raws]

    raw = raws[0].add_channels(raws[1:])  # concat recordings as single file
    raw.rename_channels(cfg.FLORIN_CHNAME_MAP_EMPTY)
    raw.reorder_channels(sorted(raw.ch_names))  # sort channels
    raw.set_channel_types({ch: 'dbs' for ch in raw.ch_names})

    # save in BIDS format in rawdata
    bids_path = BIDSPath(suffix="ieeg",
                         extension=".vhdr",
                         description="cleaned",
                         datatype="ieeg",
                         task="noise",
                         run=1,
                         subject='FloEmptyroom',
                         session='ElektaNeuromag20240208',
                         recording="Florin",
                         root=cfg.RAWDATA)
    _add_info(raw, bids_path)
    _save_bids(raw, bids_path)

    # Also save for Hirschmann
    bids_path.update(subject='HirEmptyroom', recording="Hirschmann")
    _add_info(raw, bids_path)
    _save_bids(raw, bids_path)


def _add_mni_coords(raw, dir_path):
    file_path = join(dir_path, "ea_reconstruction.mat")
    try:
        mat_file = loadmat(file_path)["reco"]
    except FileNotFoundError:
        assert dir_path.split('/')[-1] == 'S002'
        return

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

    x_ax = 0
    right_hemi_positive = np.all(mni_right[:, x_ax] > 0)
    left_hemi_negative = np.all(mni_left[:, x_ax] < 0)
    assert right_hemi_positive and left_hemi_negative, "Wrong hemispheres"

    # combine right and left mnicoords
    coords = np.vstack((mni_right, mni_left))
    for ch_name, coords in zip(raw.ch_names, coords):
        raw.info["chs"][raw.ch_names.index(ch_name)]["loc"][:3] = coords


def _raw_from_mat(directory_path, cond):
    """Each subject has two conditions and 3 runs from the same day."""
    subject = basename(directory_path)
    good_run = False
    run = 1
    while not good_run:
        fname = f'matrix_{subject}_M{cond.upper()}_run{run}.mat'
        try:
            run1 = loadmat(join(directory_path, fname))
        except FileNotFoundError:
            warn(f"Subject {subject} has no {cond} condition.")
            return None, None
        times = run1["Time"]
        if times[-1] > 300:
            good_run = True
        else:
            run += 1
    sample_rate = int(run1["Fs"])
    ch_names = list(run1["label"])
    data = run1["rawdata"]

    info = create_info(ch_names, sample_rate, ch_types='dbs')
    raw = RawArray(data, info)

    _add_mni_coords(raw, directory_path)

    if subject == 'S020':
        rename_dic = cfg.FLORIN_CHNAME_MAP_BOSTON
    else:
        rename_dic = cfg.FLORIN_CHNAME_MAP_STJUDE

    raw.rename_channels(rename_dic)
    raw.reorder_channels(sorted(raw.ch_names))
    assert raw.times[-1] > 300
    return raw, run


def _add_bad_channels(raw, bids_path, only_cleaned=True):
    """Add bad channels and bad segments."""
    anno_path = _get_annotation_path(bids_path, extension='.json')
    try:
        with open(anno_path) as bad_chs:
            bad_chs = load(bad_chs)
    except (FileNotFoundError, TypeError):
        if only_cleaned:
            raise FileNotFoundError(f"No annotations found: {anno_path}")
        return None
    bad_chs = set(bad_chs).intersection(set(raw.ch_names))
    bad_chs = list(bad_chs)
    raw.info["bads"] = bad_chs


def _add_bad_segments(raw, bids_path, modify_bidspath=True, only_cleaned=True):
    anno_path = _get_annotation_path(bids_path, extension='.csv')
    if anno_path is None:
        warn(f"\n\n{bids_path.basename} has not been annotated yet!\n\n")
        description = 'uncleaned'
    elif anno_path.startswith(cfg.ANNOTATIONS):
        description = 'cleaned'
    else:
        raise ValueError(f"No annotations found: {anno_path}")

    try:
        annotations = read_annotations(anno_path)
    except (FileNotFoundError, TypeError):
        msg = f"\n\n{bids_path.basename} has not been annotated yet!\n\n"
        if only_cleaned:
            raise FileNotFoundError(msg)
        warn(msg)
        if modify_bidspath:
            bids_path.update(description="uncleaned")
        return None
    except IndexError:
        # this just means that there are no annotations, this is fine
        annotations = None
    raw.set_meas_date(0)  # set recording date to 1970-01-01
    raw.set_annotations(annotations)
    if modify_bidspath:
        bids_path.update(description=description, suffix="ieeg")


def _get_annotation_path(bids_path, extension=None):
    anno_root = cfg.ANNOTATIONS
    bids_info = dict(subjects=bids_path.subject, sessions=bids_path.session,
                     tasks=bids_path.task,
                    #  runs=bids_path.run,
                     root=anno_root,
                     extensions=extension, descriptions=None,
                     acquisitions=bids_path.acquisition)
    anno_path = find_matching_paths(**bids_info)
    msg = f"More than one annotation file found for {bids_path}"
    if anno_path:
        assert len(anno_path) < 2, msg
        return str(anno_path[0].fpath)
    # Don't load old annotations. Very messy. Annotate again.
    else:
        return None


if __name__ == "__main__":
    bidsify_sourcedata_florin()
