"""Module for reading annotations and creating montages."""
from json import load
from os import listdir
from os.path import basename, isfile, join
from shutil import copy
from warnings import warn

import numpy as np
import pandas as pd
from mne import read_annotations
from mne.channels import make_dig_montage, make_standard_montage
from mne.io import Raw
from mne_bids import BIDSPath, find_matching_paths, read_raw_bids
from tqdm import tqdm

import scripts.config as cfg
from scripts.bidsify_sourcedata import _add_info
from scripts.load_tmsi_data import Poly5Reader
from scripts.utils import _copy_files_and_dirs, _delete_dirty_files, _save_bids


def bidsify_sourcedata_neumann(subjects=None, only_cleaned=True) -> None:
    """Read sourcedata, bidsify, and save in rawdata.

    Files are saved in fif format which does NOT follow the BIDS standard.
    However, the function mne_bids.write_raw_bids is very buggy at the moment.
    """
    raw_root = cfg.RAWDATA
    source_root = cfg.SOURCEDATA_NEU
    dict_of_good_files = load(open(cfg.GOOD_FILES_JSON, "r"))
    if subjects is None:
        subjects = [file_dic["sub"] for file_dic in dict_of_good_files]
    bids_paths = []
    for file_dic in dict_of_good_files:
        subject = file_dic["sub"]
        bids_path = BIDSPath(root=source_root, subject=subject,
                             session=file_dic["ses"], task=file_dic["task"],
                             acquisition=file_dic["acq"], run=file_dic["run"],
                             extension=".vhdr", datatype="ieeg")
        if subject in subjects:
            bids_paths.append(bids_path)

    # Add single channel noise floor recording
    _bidsify_noise()  # breaks in debug mode, works when ran as a file...
    # Add various impedances noise floor recording
    amp_root = join(cfg.SOURCEDATA, "BIDS_Neumann_NoiseFloorRecordings")
    amp_dic = dict(root=amp_root, extensions=".vhdr",
                   sessions="TMSiSAGA20220916")
    amp_path_TMSi = find_matching_paths(**amp_dic)
    bids_paths += amp_path_TMSi
    for bids_path in tqdm(bids_paths, desc="BIDSify Neumann: "):
        bids_path_old = bids_path.copy()

        _clean_electrodes_tsv(bids_path)

        # ValueError for read_raw: "New channel names are not unique,
        # renaming failed" -> Bug in mne_bids
        raw = read_raw_bids(bids_path, verbose=False)
        # raw = read_raw(bids_path.fpath)
        # print(f'Sample rate: {raw.info["sfreq"]}\n'
        #       f'highpass: {raw.info["highpass"]}\n'
        #       f'lowpass: {raw.info["lowpass"]}')

        # Check that all channels are in the set "expected_ch_names"
        _check_channel_names(raw, bids_path)

        # Correct montage. Otherwise MNE BIDS bug ValueError (CapTrak)
        # The electrode.tsv files aren't read and written correclty.
        _correct_coord_frame(raw)

        # Add infos to raw.info
        bids_path.recording = "Neumann"

        # Save path
        _add_bad_segments(raw, bids_path, remove_eeg_annotations=True,
                          only_cleaned=only_cleaned)
        _add_bad_channels(raw, bids_path, descriptions="OriginalNames",
                          drop_ctx=True, only_cleaned=only_cleaned)
        sub_new = f"{cfg.SUB_PREFIX_NEU}{bids_path.subject}"
        sub_new = 'NeuEmptyroom' if sub_new == 'Neuemptyroom' else sub_new
        bids_path.update(root=raw_root, suffix="ieeg", extension=".fif",
                         subject=sub_new)
        _add_info(raw, bids_path, bids_path_old=bids_path_old)
        _remove_bad_end(raw)
        if bids_path.subject != "NeuEmptyroom":
            # EL029 has wrong channel types
            if bids_path.subject == 'NeuEL029':
                dbs_chs = [ch for ch in raw.ch_names if ch.startswith('LFP_')]
                dbs_types = ['dbs'] * len(dbs_chs)
                raw.set_channel_types(dict(zip(dbs_chs, dbs_types)))

            # don't use try: raw.pick_types, will modify raw amplifier
            raw.pick_types(ecog=False, eeg=False, dbs=True, exclude=[])
        _save_bids(raw, bids_path)
        # Path(bids_path.directory).mkdir(parents=True, exist_ok=True)
        # write_raw_bids does not save raw.info correctly
        # raw.save(bids_path.fpath, split_naming="bids", overwrite=True)
        _copy_meta_files(bids_path_old, bids_path)
        _delete_dirty_files(bids_path)

    # Copy meta info as is
    _copy_files_and_dirs(source_root, join(raw_root, "meta_infos_Neumann"),
                         cfg.BIDS_FILES_TO_COPY)
    print(f"{basename(__file__).strip('.py')} done.")


def _bidsify_noise():
    """Bidsify noise recordings."""
    amp_root = join(cfg.SOURCEDATA, "BIDS_Neumann_NoiseFloorRecordings",
                    'sub-emptyroom', 'ses-TMSiSAGA20240122')

    # list all files in path
    sessions = listdir(amp_root)
    skip_sessions = ['.DS_Store',
                     'sub-emptyroom_ses-TMSiSAGA20220916_scans.tsv',
                     'SAGA20240122_Ch2+3+4 - 20240122T144144']
    raws = []
    for session in sessions:
        if session in skip_sessions:
            continue
        ch = session.split(' - ')[0].split('_')[1].strip('Ch')
        if int(ch) > 16:
            continue
        fname = listdir(join(amp_root, session))
        fname.remove('Record.xses')
        fname = fname[0]
        filename = join(amp_root, session, fname)
        raw = Poly5Reader(filename=filename).read_data_MNE()
        raw.pick_channels([f'Ch{ch}'])
        raws.append(raw)

    # concat to shortest time
    duration_min = 44.5  # artifact at end of recording
    raws = [raw.crop(tmax=duration_min) for raw in raws]

    raw = raws[0].add_channels(raws[1:])  # concat recordings as single file
    raw.rename_channels(cfg.NEUMANN_CHNAME_MAP_EMPTY)
    raw.reorder_channels(sorted(raw.ch_names))  # sort channels
    raw.set_channel_types({ch: 'dbs' for ch in raw.ch_names})

    # save in BIDS format in rawdata
    bids_path = BIDSPath(suffix="ieeg",
                         extension=".vhdr",
                         description="cleaned",
                         datatype="ieeg",
                         task="noise",
                         run=1,
                         subject='NeuEmptyroom',
                         session='TMSiSAGA20240122',
                         recording="Neumann",
                         root=cfg.RAWDATA)
    _add_info(raw, bids_path)
    _save_bids(raw, bids_path)


def _remove_bad_end(raw):
    """Files have bad recording ends which cause huge filter artifacts.

    In theory, 6 seconds is too conservative. The edge artifact is probably
    always in the last second of the recording. Therefore, 1 second should be
    sufficient. But we have long enough data anyways, so no problem."""
    crop_end = 6  # seconds
    tmax = raw.times[-1] - crop_end
    raw.crop(tmax=tmax)


def _clean_electrodes_tsv(bids_path):
    """
    Remove nan values from electrodes tsv files becauses it causes errors.

    Subject 7 has fused LFP coords LFP_020304_STN_MT. Therefore, mni
    coordinates are not correctly read from electrodes.tsv file. Correct file.
    """
    if bids_path.subject == "emptyroom":
        return
    update = dict(
        task=None,
        acquisition=None,
        run=None,
        suffix="electrodes",
        space="MNI152NLin2009bAsym",
        extension=".tsv",
    )
    electrodes_path = bids_path.copy().update(**update)
    df = pd.read_csv(electrodes_path.fpath, sep="\t")

    # keep copy of original file
    orig_path = str(electrodes_path.fpath).replace(".tsv", "_orig.tsv")
    df.to_csv(orig_path, sep="\t", index=False)

    # save after excluding nan values
    df.dropna(subset="x", inplace=True)

    if (bids_path.subject == "EL007" and "On" in bids_path.session):
        assert bids_path.basename in cfg.DIRECTIONAL_LEADS_AVERAGED
        _average_dir_mni(df, leading_zeros=True)
    elif (bids_path.subject == "L012" and "On" in bids_path.session):
        assert bids_path.basename in cfg.DIRECTIONAL_LEADS_AVERAGED
        _average_dir_mni(df, leading_zeros=False)

    df.to_csv(electrodes_path.fpath, sep="\t", index=False)


def _average_dir_mni(df, leading_zeros):
    """Happens to be left hemi in both subs."""
    if leading_zeros:
        L2_nme = "LFP_L_020304_STN_MT"
        L3_nme = "LFP_L_050607_STN_MT"
    elif not leading_zeros:
        L2_nme = "LFP_L_234_STN_MT"
        L3_nme = "LFP_L_567_STN_MT"

    # check if already corrected. In that case skip to avoid growing table
    if L2_nme in df.name.to_list():
        assert L3_nme in df.name.to_list()
        return df

    # add two new rows to dataframe and copy meta data
    idx_L2 = df.index[-1] + 1
    idx_L3 = df.index[-1] + 2
    df.loc[idx_L2] = df[df.name == "LFP_L_02_STN_MT"].values[0]
    df.loc[idx_L3] = df[df.name == "LFP_L_03_STN_MT"].values[0]

    # correct channel name and mni coordinates
    df.loc[idx_L2, "name"] = L2_nme
    df.loc[idx_L3, "name"] = L3_nme

    # average mni coordinates
    left_2_abc = ["LFP_L_02_STN_MT", "LFP_L_03_STN_MT", "LFP_L_04_STN_MT"]
    left_3_abc = ["LFP_L_05_STN_MT", "LFP_L_06_STN_MT", "LFP_L_07_STN_MT"]
    cols = ["x", "y", "z"]
    mni_L_2 = df.loc[df.name.isin(left_2_abc), cols].mean(0).to_numpy()
    mni_L_3 = df.loc[df.name.isin(left_3_abc), cols].mean(0).to_numpy()
    df.loc[idx_L2, cols] = mni_L_2
    df.loc[idx_L3, cols] = mni_L_3

    return df


def _check_channel_names(raw: Raw, bids_path) -> None:
    """Check if channels are in the expected format.

    Raises assertion error if unknown channels are found.
    Warns if channels are missing.
    """
    expected_ch_names = cfg.POSSIBLE_CH_NAMES
    # Raise error if unknown channel names are found
    ch_names = set(raw.ch_names)
    wrong_ch_names = sorted(list(ch_names - expected_ch_names))
    msg = f"{bids_path.basename}: Channels {wrong_ch_names} not known."
    assert not wrong_ch_names, msg


def _correct_coord_frame(raw: Raw, verbose=False) -> None:
    """Coordinate frame is set wrongly."""
    try:
        montage = raw.get_montage().get_positions()
    except (RuntimeError, AttributeError):
        subject = raw.info["subject_info"]["his_id"]
        print(f"No montage found for subject {subject}. Skipping.")
    else:
        if "eeg" in raw.get_channel_types(unique=True):
            # Add standard EEG locations (coord_frame MRI=mni_tal?)
            eeg_1020 = make_standard_montage("standard_1020").get_positions()
            Cz = eeg_1020["ch_pos"]["Cz"]
            Fz = eeg_1020["ch_pos"]["Fz"]
            eeg_nms = raw.copy().pick_types(eeg=True, exclude=[]).ch_names
            for eeg_nm in eeg_nms:
                if "CZ" in eeg_nm:
                    montage["ch_pos"][eeg_nm] = Cz
                elif "FZ" in eeg_nm:
                    montage["ch_pos"][eeg_nm] = Fz
                else:
                    raise ValueError("Neither 'Fz' nor 'Cz' in EEG channel ",
                                     eeg_nm)

        # Change coord_frame to mni_tal
        montage["coord_frame"] = "mni_tal"
        new_montage = make_dig_montage(**montage)
        raw.set_montage(new_montage, verbose=verbose)


def _add_bad_segments(raw: Raw, bids_path: BIDSPath, modify_bidspath=True,
                      remove_eeg_annotations=False, location="both",
                      only_cleaned=True) -> None:
    """Add bad segment annotations provided either by Gunnar or Thomas.

    Annotations Thomas are stored in "derivatives/artifact_annotation_main/".
    They are not agreeing with BIDS standard because there are no datatype
    "ieeg" folders. This needs to be changed manually.

    Annotations Gunnar "derivatives/annotations/". They are agreeing with BIDS
    standard because there are datatype "ieeg" folders.

    Annotations from Thomas and Gunnar almost equal but his annos dont have
    proper names such as "BAD_strict", "BAD_LFP", "BAD_EEG", etc.
    """
    if bids_path.subject == "emptyroom":
        # emptyroom doesn't need cleaning
        if modify_bidspath:
            bids_path.update(description="cleaned", suffix="ieeg")
        return None

    anno_path = _get_annotation_path(bids_path, location=location,
                                     anno_type="events")
    if anno_path.startswith(f'{cfg.BASE_DIR}/derivatives/annotations'):
        description = 'cleaned'
    elif anno_path.startswith(
        f'{cfg.BASE_DIR}/sourcedata/BIDS_Neumann_ECOG_LFP/meta_infos/'
    ):
        description = 'uncleaned'

    # read and add annotations
    try:
        annotations = read_annotations(anno_path)
    except (FileNotFoundError):
        msg = f"\n\n{bids_path.basename} has not been annotated yet!\n\n"
        if only_cleaned:
            raise FileNotFoundError(msg)
        warn(msg)
        if modify_bidspath:
            bids_path.update(description="uncleaned", suffix="ieeg")
        return None
    except IndexError:
        # this just means that there are no annotations, this is fine
        annotations = None
    if annotations and remove_eeg_annotations:
        bad_eeg_idx = np.where(annotations.description == "BAD_EEG")[0]
        for idx in reversed(list(bad_eeg_idx)):
            annotations.delete(idx)
    try:
        raw.set_annotations(annotations)
    except RuntimeError as error:
        warn(f"\n\n{bids_path.basename}: {error}\n\n")
        annotations._orig_time = None
        raw.set_annotations(annotations)
    else:
        if modify_bidspath:
            # sourcedata annotations should be checked again. Therefore,
            # considered uncleaned
            bids_path.update(description=description, suffix="ieeg")
    return annotations


def _add_bad_channels(raw: Raw, bids_path: BIDSPath, location="both",
                      descriptions="OriginalNames", drop_ctx=True,
                      only_cleaned=True) -> None:
    """Add bad segment annotations provided either by Gunnar or Thomas.

    Annotations Thomas are stored in "derivatives/artifact_annotation_main/".
    They are not agreeing with BIDS standard because there are no datatype
    "ieeg" folders. This needs to be changed manually.

    Annotations Gunnar "derivatives/annotations/". They are agreeing with BIDS
    standard because there are datatype "ieeg" folders.

    Annotations from Thomas and Gunnar almost equal but his annos dont have
    proper names such as "BAD_strict", "BAD_LFP", "BAD_EEG", etc.
    """
    if bids_path.subject == "emptyroom":
        return None
    bads_path = _get_annotation_path(bids_path, location=location,
                                     anno_type="channels",
                                     descriptions=descriptions)
    try:
        with open(bads_path) as bad_chs:
            bad_chs = load(bad_chs)
    except (FileNotFoundError, TypeError):
        if only_cleaned:
            msg = f"\n\n{bids_path.basename} has not been annotated yet!\n\n"
            raise FileNotFoundError(msg)
        return None
    if drop_ctx:
        pick = 'dbs'
        if bids_path.subject == "EL029":
            pick = ['eeg', 'dbs']
        dbs_chs = raw.copy().pick(pick).ch_names
        bad_chs = [ch for ch in bad_chs if ch in dbs_chs]
    msg = "Channel names don't match!"
    assert set(bad_chs).issubset(set(raw.ch_names)), msg
    raw.info["bads"] = bad_chs


def _get_annotation_path(bids_path, location="both", anno_type="events",
                         descriptions="OriginalNames"):
    """Get annotation path from bids path.

    Parameters
    ----------
    bids_path : _type_
        _description_
    location : str, optional
        both: preferentially return the annotations made with Gunnar but use
        Thomas' annotations if they don't exist. By default "both"
        gunnar: only return annotations made with Gunnar
        thomas: only return annotations made with Thomas

    Returns
    -------
    _type_
        _description_
    """
    if bids_path.subject == "emptyroom":
        return None

    if anno_type == "channels":
        extension = "json"
    elif anno_type == "events":
        extension = "csv"
        descriptions = None

    # extract root derivatives
    anno_root = cfg.ANNOTATIONS
    bids_info = dict(
        subjects=f'Neu{bids_path.subject}',
        sessions=bids_path.session,
        tasks=bids_path.task,
        runs=bids_path.run,
        root=anno_root,
        extensions=extension,
        descriptions=descriptions,
    )
    anno_path = find_matching_paths(**bids_info)
    msg = f"More than one annotation file found for {bids_path}"
    if anno_path:
        assert len(anno_path) < 2, msg
        return str(anno_path[0].fpath)

    # extract root Gunnar
    anno_root = join(cfg.BASE_DIR, 'sourcedata', 'BIDS_Neumann_ECOG_LFP',
                     'meta_infos', 'annotationsMoritz')
    anno_root = join(anno_root, 'annotationsMoritzGunnar')

    bids_info = dict(
        subjects=bids_path.subject,
        sessions=bids_path.session,
        tasks=bids_path.task,
        runs=bids_path.run,
        root=anno_root,
        extensions=extension,
        descriptions=descriptions,
    )
    anno_path = find_matching_paths(**bids_info)
    if not anno_path and anno_type == "channels":
        return None
    elif anno_path:
        assert len(anno_path) < 2, msg
        anno_path_gunnar = anno_path[0].fpath
        # # Extract basename and fool BIDS:
        # # "ValueError: Extension .csv is not allowed."
        # extension = ".json" if anno_type == "channels" else ".csv"
        # anno_basename = anno_path.basename.replace(".fif", extension)
        # # path without /ieeg/ datatype
        # # anno_path = str(anno_path.directory.parent)
        # anno_path = str(anno_path.directory)
        # # full path
        # anno_path_gunnar = join(anno_path, anno_basename)
        gunnar_exists = isfile(anno_path_gunnar)
    elif not anno_path and anno_type == "events":
        gunnar_exists = False
    if location == "gunnar" or (location == "both" and gunnar_exists):
        return str(anno_path_gunnar)

    if anno_type == "channels":
        # Thomas channel annotations are saved with sourcedata and always
        # automatically loaded
        return None
    else:
        # extract root Thomas: cannot use "find_matching_paths" because files
        # are not correctly saved in "ieeg" folder acoording to BIDS
        root = join(anno_root, 'artifact_annotation_main')
        bids_thomas = dict(root=root, description=None, processing=None,
                           recording=None)
        anno_path = bids_path.copy().update(**bids_thomas)
        # extract basename
        anno_basename = anno_path.basename.replace("ieeg.fif",
                                                   "annotations.csv")
        # path without /ieeg/ datatype
        anno_path = str(anno_path.directory.parent)
        # full path
        anno_path_thomas = join(anno_path, anno_basename)
        return anno_path_thomas


def _copy_meta_files(bids_path_old: BIDSPath, bids_path_new: BIDSPath):
    """Copy electrodes.tsv, ieeg.json, channels.tsv, and coordsystem.json files
    from old to new bids path to solve mne bug."""
    json_old = bids_path_old.copy().update(extension=".json")
    channels_old = bids_path_old.copy().update(suffix="channels",
                                               extension=".tsv")
    json = bids_path_new.copy().update(extension=".json")
    channels = bids_path_new.copy().update(suffix="channels", extension=".tsv")
    copy(json_old.fpath, json.fpath)
    copy(channels_old.fpath, channels.fpath)
    if bids_path_new.subject == "NeuEmptyroom":
        return None

    electrodes_old = bids_path_old.copy().update(
        suffix="electrodes",
        extension=".tsv",
        space="MNI152NLin2009bAsym",
        task=None,
        acquisition=None,
        run=None,
    )
    coords_old = bids_path_old.copy().update(
        suffix="coordsystem",
        extension=".json",
        space="MNI152NLin2009bAsym",
        task=None,
        acquisition=None,
        run=None,
    )
    electrodes = bids_path_new.copy().update(
        suffix="electrodes",
        extension=".tsv",
        space="MNI152NLin2009bAsym",
        task=None,
        acquisition=None,
        run=None,
    )
    coords = bids_path_new.copy().update(
        suffix="coordsystem",
        extension=".json",
        space="MNI152NLin2009bAsym",
        task=None,
        acquisition=None,
        run=None,
    )

    copy(electrodes_old.fpath, electrodes.fpath)
    copy(coords_old.fpath, coords.fpath)
    return None


if __name__ == "__main__":
    bidsify_sourcedata_neumann()
