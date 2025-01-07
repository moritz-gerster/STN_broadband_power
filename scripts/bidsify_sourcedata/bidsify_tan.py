from json import load
from os.path import basename, join
from pathlib import Path
from warnings import warn

import numpy as np
from mne import create_info, read_annotations
from mne.io import RawArray
from mne.preprocessing import annotate_amplitude
from mne_bids import BIDSPath, find_matching_paths, make_dataset_description
from tqdm import tqdm

import scripts.config as cfg
from scripts.bidsify_sourcedata import (_add_info, _correct_units, _move_files,
                                        loadmat)
from scripts.utils import _delete_dirty_files, _save_bids


def bidsify_sourcedata_tan(only_cleaned=True, add_dbs_subs=False) -> None:
    """Read sourcedata, bidsify, and save in rawdata."""
    raw_root = cfg.RAWDATA

    bids_path = BIDSPath(suffix="ieeg",
                         extension=".vhdr",  # fif does not work
                         description="uncleaned",
                         datatype="ieeg",
                         recording="Tan",
                         run=1,
                         root=raw_root)

    bidsify_onoff(bids_path, only_cleaned=only_cleaned)
    if add_dbs_subs:
        bidsify_dbs(bids_path, add_dbs=True)
    bidsify_noise()

    # add dataset description in the end
    meta_data = cfg.TAN_META
    meta_data["path"] = raw_root
    make_dataset_description(**cfg.TAN_META, overwrite=True)
    # move participants.tsv, participants.json, and dataset_description.json
    # to meta_infos directory
    meta_path = join(raw_root, "meta_infos_Tan")
    Path(meta_path).mkdir(parents=True, exist_ok=True)
    _move_files(raw_root, meta_path)
    print(f"{basename(__file__).strip('.py')} done.")
    return None


def bidsify_noise():
    """Bidsify noise recordings."""
    amp_root = join(cfg.SOURCEDATA, 'BIDS_TAN_NoiseFloorRecordings')

    raws_porti = _get_porti_noise(amp_root)
    raws_saga = _get_saga_noise(amp_root)  # slow due to huge file size

    amp_names = ['SAGA', 'Porti']
    ch_maps = [cfg.TAN_CHNAME_MAP_SAGA, cfg.TAN_CHNAME_MAP_PORTI]
    for amp_idx, raws in enumerate([raws_saga, raws_porti]):
        # concat to shortest time
        duration_min = min([raw.times[-1] for raw in raws])
        raws = [raw.crop(tmax=duration_min) for raw in raws]

        # concat recordings as single file
        raw = raws[0].add_channels(raws[1:])
        raw.rename_channels(ch_maps[amp_idx])
        raw.reorder_channels(sorted(raw.ch_names))  # sort channels

        # save in BIDS format in rawdata
        bids_path = BIDSPath(suffix="ieeg",
                             extension=".vhdr",
                             description="cleaned",
                             datatype="ieeg",
                             task="noise",
                             run=1,
                             subject='TanEmptyroom',
                             session=f'TMSi{amp_names[amp_idx]}20240212',
                             recording="Tan",
                             root=cfg.RAWDATA)
        _add_info(raw, bids_path)
        _save_bids(raw, bids_path)


def _get_porti_noise(amp_root):
    source_path = join(amp_root, 'NoiseTMSiPorti.mat')

    raw_mat = loadmat(source_path)["Noise"]
    sample_rate = int(raw_mat["sampling_rate"])

    # everything recorded twice, only use first recording
    ch_names = raw_mat['label'][0]
    data = raw_mat["data"][0]

    raws = []
    for ch_data, ch_nme in zip(data, ch_names):
        if int(ch_nme.strip('Ch')) > 8:
            continue
        info = create_info([ch_nme], sample_rate, ch_types='dbs')
        ch_data = ch_data.reshape(1, -1)
        raw = RawArray(ch_data, info)
        _correct_units(raw)
        raws.append(raw)
    return raws


def _get_saga_noise(amp_root):
    source_path = join(amp_root, 'Saga_Noise_Test_All_Chan.mat')
    raw_mat = loadmat(source_path)['SmrData']
    sample_rate = int(raw_mat["Fs"])

    data = raw_mat["WvData"]
    ch_names = raw_mat['WvTits']
    ch_names = [ch[0] for ch in ch_names]

    info = create_info(ch_names, sample_rate, ch_types='dbs')
    raw_all = RawArray(data, info)
    _correct_units(raw_all)  # correct units before flat segment annotation

    raws = []
    for ch_nme in ch_names:
        if ch_nme == 'CREF' or int(ch_nme.strip('UN')) > 8:
            continue
        raw = raw_all.copy().pick_channels([ch_nme])

        # Find flat channels and segments and crop out
        flat_segments, _ = annotate_amplitude(raw, flat=1e-20,
                                              min_duration=1,
                                              bad_percent=99.9)
        onset = flat_segments.onset[0]
        duration = flat_segments.duration[0]
        buffer = 3  # 3s buffer
        if len(flat_segments.onset) == 1:
            if onset == 0:
                # last segment
                tmin = duration + buffer
                tmax = None
            else:
                # first segment
                tmin = 0
                tmax = onset - buffer
        elif len(flat_segments.onset) == 2:
            tmin = onset + duration + buffer
            tmax = flat_segments.onset[1] - buffer
        else:
            raise ValueError("Wrong number of annotations")
        raw.crop(tmin=tmin, tmax=tmax)
        raws.append(raw)
    flat_segments, _ = annotate_amplitude(raw, flat=1e-20,
                                        min_duration=1,
                                        bad_percent=99.9)
    assert not len(flat_segments.onset)
    return raws



def bidsify_onoff(bids_path, pick_wiest=False, only_cleaned=True):
    all_ch_names = []
    subject_map = cfg.TAN_SUBJECT_MAP
    bids_path.update(task='RestLdopa')
    for cond in tqdm(["OFF", "ON"], desc="Bidsify Tan LDOPA", position=0):
        dir_cond = ("aperiodic exponent of subthalamic field potentials- "
                    f"Human- Meds- {cond}")
        source_path = join(cfg.SOURCEDATA, 'BIDS_TAN_EEG_LFP', dir_cond)
        for subj_old, subj_new in tqdm(subject_map.items(), position=1,
                                       desc=f"Bidsify subjects {cond}"):
            if subj_old in ['XG37', 'XG39'] and cond == "ON":
                # add ERNA name in ON condition
                subj_old += '_ERNA'
            for hemi in ["le", "ri"]:
                fname = f"{subj_old}_{hemi}STN_{cond}.mat"
                path_hemi = join(source_path, fname)
                try:
                    raw_mat = loadmat(path_hemi)["SmrData"]
                except FileNotFoundError:
                    # add dummy hemisphere to avoid missing channels
                    _update_bids_path(bids_path, cond, subj_new, hemi)
                    raw_dummy = _make_dummy_raw(hemi)
                    _add_info(raw_dummy, bids_path)
                    _save_bids(raw_dummy, bids_path)
                    _delete_dirty_files(bids_path)
                    continue

                raw = _raw_from_mat(raw_mat)
                # print(f'Sample rate: {raw.info["sfreq"]}\n'
                #     f'highpass: {raw.info["highpass"]}\n'
                #     f'lowpass: {raw.info["lowpass"]}')

                _set_tan_chs(raw)
                _pick_channels_times(raw, fname, hemi, crop_wiest=False,
                                     pick_wiest=pick_wiest, crop_gerster=True)
                _update_bids_path(bids_path, cond, subj_new, hemi)
                _add_bad_channels(raw, bids_path, pick_wiest=pick_wiest,
                                  only_cleaned=only_cleaned)
                _add_bad_segments(raw, bids_path, only_cleaned=only_cleaned)
                _rename_channels(raw, fname)
                _drop_wrong_hemisphere(raw, hemi)
                _add_missing_channels(raw, hemi)

                _add_info(raw, bids_path)
                _correct_units(raw)
                _save_bids(raw, bids_path)
                _delete_dirty_files(bids_path)
                all_ch_names.extend(raw.ch_names)
    print(f"Channels: {set(all_ch_names)}")


def _make_dummy_raw(hemisphere):
    hemi = 'L' if hemisphere == 'le' else 'R'
    dummy_chs = [f'LFP_{hemi}_1-3_STN_MT', f'LFP_{hemi}_2-4_STN_MT',
                 f'LFP_{hemi}_1-2_STN_MT', f'LFP_{hemi}_2-3_STN_MT',
                 f'LFP_{hemi}_3-4_STN_MT']
    sfreq = 2333  # choose distinct sample rate
    info = create_info(dummy_chs, sfreq, ch_types='dbs', verbose=False)
    nan_data = np.ones((len(dummy_chs), 10*sfreq))
    nan_data[:] = np.nan
    raw = RawArray(nan_data, info)
    raw.info['bads'] = dummy_chs
    return raw


def _add_missing_channels(raw, hemisphere):
    """Many subjects have only one channel which causes problems later on.

    Therefore, add missing channels with nan values and set as bad."""
    hemi = 'L' if hemisphere == 'le' else 'R'
    needed_chs = {f'LFP_{hemi}_1-2_STN_MT', f'LFP_{hemi}_2-3_STN_MT',
                  f'LFP_{hemi}_3-4_STN_MT',
                  f'LFP_{hemi}_1-3_STN_MT', f'LFP_{hemi}_2-4_STN_MT'}
    missing_chs = needed_chs - set(raw.ch_names)

    # add dummy channels
    dummy_ch = raw.copy().pick_types(dbs=True).pick(0).load_data()
    dummy_ch._data.fill(np.nan)
    raw.load_data()
    for ch in missing_chs:
        fill_ch = dummy_ch.copy()
        fill_ch.rename_channels({dummy_ch.ch_names[0]: ch})
        raw.add_channels([fill_ch])
        raw.info['bads'].append(ch)


def _drop_wrong_hemisphere(raw, hemi):
    """I have each subject twice, no need to keep both hemispheres. Often,
    the stimulated hemisphere had worse data quality even when the stimulation
    was turned off. The only two subjects missing one hemisphere (K7 and K11)
    have only channels on one hemisphere anyways."""
    right_chs = [ch for ch in raw.ch_names if ch.split('_')[1] == 'R']
    left_chs = [ch for ch in raw.ch_names if ch.split('_')[1] == 'L']
    assert set(raw.ch_names) == set(right_chs + left_chs)
    if hemi == "le":
        raw.drop_channels(right_chs)
    elif hemi == "ri":
        raw.drop_channels(left_chs)
    # only the two wiest channels (which have different names) should be left
    # as dbs channels
    assert len(raw.copy().pick_types(dbs=True).ch_names) == 2


def _add_bad_channels(raw, bids_path, pick_wiest=False, only_cleaned=True):
    """Add bad channels and bad segments."""
    anno_path = _get_annotation_path(bids_path, extension='.json')
    try:
        bad_chs = open(anno_path)
        with open(anno_path) as bad_chs:
            bad_chs = load(bad_chs)
    except (FileNotFoundError, TypeError):
        if only_cleaned:
            raise FileNotFoundError(f"No annotations found: {anno_path}")
        else:
            return None
    if pick_wiest:
        #  drop ecog (non-picked dbs channels)
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
        else:
            warn(f"\n\n{bids_path.basename} has not been annotated yet!\n\n")
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
    bids_path = bids_path.copy().update(task='Rest', run=None)
    anno_root = cfg.ANNOTATIONS
    bids_info = dict(subjects=bids_path.subject, sessions=bids_path.session,
                     tasks=bids_path.task, runs=bids_path.run, root=anno_root,
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


def _update_bids_path(bids_path, cond, subj_new, hemi):
    session = f'LfpMed{cond.capitalize()}01'
    if hemi == "le":
        hemisphere = 'Left'
    elif hemi == "ri":
        hemisphere = 'Right'
    acquisition = f"StimOn{hemisphere}"
    bids_path.update(subject=subj_new, session=session,
                     acquisition=acquisition)


def _raw_from_mat(raw_mat):
    sample_rate = int(raw_mat["Fs"])
    ch_names = _extract_chs(raw_mat)

    data = raw_mat["WvData"]
    if data.ndim == 1:
        data = data.reshape(1, -1)

    info = create_info(ch_names, sample_rate, ch_types='misc')
    raw = RawArray(data, info)

    return raw


def _extract_chs(raw_mat):
    ch_names = raw_mat["WvTits"]
    if isinstance(ch_names, str):
        ch_names = [ch_names]
    elif isinstance(ch_names, list):
        ch_names = [ch[0] for ch in ch_names]
    elif isinstance(ch_names, np.ndarray):
        ch_names = list(ch_names)
    return ch_names


def _pick_channels_times(raw, fname, hemi, crop_wiest=True, pick_wiest=True,
                         crop_gerster=False):
    dic = cfg.CH_TIME_SELECTION_WIEST
    ch_idx = dic[fname]['idx'] - 1  # matlab to python indexing
    ch_nme_raw = raw.ch_names[ch_idx]
    ch_nme_orig = dic[fname]['ch_nme']
    times = dic[fname]['time']
    msg = f"{ch_nme_raw} != {ch_nme_orig}"
    # use '.startswith' instead of '==' because the virtual channels are
    # renamed be mne to avoid duplucates: virtual -> virtual-1, virtual-2, ...
    assert raw.ch_names[ch_idx].startswith(ch_nme_orig), msg

    # set his channel choice as dbs, keep others as ecog
    raw.set_channel_types({ch_nme_raw: 'dbs'})  # change channel types
    # drop misc and eeg channels
    # also drop other dbs channels for now
    if pick_wiest:  # only keep his dbs choice
        raw.pick_types(ecog=False, dbs=True, eeg=False, misc=False)
    else:  # else, keep other channels as ecog type
        raw.pick_types(ecog=True, dbs=True, eeg=False, misc=False)
        # add Wiest pick as duplicated virtual channel such that it can be
        # chosen later even if it changes between ON and OFF condition
        wiest_pick = raw.copy().pick_channels([ch_nme_raw])
        hemi = 'L' if hemi == 'le' else 'R'
        wiest_pick.rename_channels({ch_nme_raw: f'LFP_{hemi}_WIEST_STN_MT'})
        raw.add_channels([wiest_pick])
    if crop_wiest:  # select his time window
        assert not crop_gerster
        raw.crop(*times)
    if crop_gerster:
        assert not crop_wiest
        times = cfg.TIME_SELECTION_OWN[fname]['time']
        raw.crop(*times)


def _rename_channels(raw, fname):
    """This is a huge mess. Plenty of different channel names. Sometimes, DBS
    ch index starts at 0, sometimes at 1.

    Strategy: Drop all misc and eeg channels, keep DBS channels. If one DBS
    channel starts with "0" use CH_NME_MAP_TAN_IDX0 to rename channels. If one
    channels ends with "4" use CH_NME_MAP_TAN_IDX1 to rename channels. If one
    channel starts with 'Virtual' use CH_NME_MAP_TAN_FNAME to rename channels
    which have been figured out. These files remain amibguous:
    ['G27_leSTN_ON.mat', 'G27_riSTN_ON.mat',
    'G31_leSTN_ON.mat', 'G31_riSTN_ON.mat', 'G32_riSTN_ON.mat',
    'XG39_ERNA_riSTN_ON.mat']. Here, I simply assume that the index starts at 1
    since it does not really matter for the analysis and since for the same
    subjects the index started at 1 for the OFF condition."""

    # drop monopolar dbs channels
    drop_channels = set(cfg.DROP_CHANNELS)
    to_drop = drop_channels.intersection(set(raw.ch_names))
    if to_drop:
        raw.drop_channels(to_drop)

    # rename subject via fname if available
    try:
        rename_dic = cfg.CH_NME_MAP_TAN_FNAME[fname]
    except KeyError:
        # rename subject via idx0 if correct
        rename_dic = cfg.CH_NME_MAP_TAN_IDX0
        rename_dic = {key: value for key, value in rename_dic.items()
                      if key in raw.ch_names}
        try:
            assert len(rename_dic) == len(raw.ch_names)
            raw.rename_channels(rename_dic)
            return None
        except (KeyError, AssertionError):
            rename_dic = cfg.CH_NME_MAP_TAN_IDX1
            rename_dic = {key: value for key, value in rename_dic.items()
                          if key in raw.ch_names}
            assert len(rename_dic) == len(raw.ch_names)
            raw.rename_channels(rename_dic)
            return None
    rename_dic = {key: value for key, value in rename_dic.items()
                  if key in raw.ch_names}
    assert len(rename_dic) == len(raw.ch_names)
    raw.rename_channels(rename_dic)


def _set_tan_chs(raw):
    ch_type_map = cfg.CH_TYPE_MAP_TAN
    map_chs = {ch: ch_type_map[ch] for ch in raw.ch_names}
    raw.set_channel_types(map_chs)  # change channel types


def bidsify_dbs(bids_path, add_dbs=False):
    """Some DBS subjects recordings are full of low-frequency artifacts.
    Cleaning the 60s segments not always possible because sometimes there is no
    data left, this would shrink the sample size tremendously. Therefore, keep
    dirty but do not combine with LDOPA subjects which are clean."""
    DBS_mat = join(cfg.SOURCEDATA_TAN, "DBS", "MATRIX_DBS.mat")
    DBS_mat = loadmat(DBS_mat)["MATRIX_DBS"]
    rest_data_all = DBS_mat['signal_base']
    dbs_data_all = DBS_mat['signal_dbs']
    sample_rate_all = DBS_mat['fs']
    bids_path.update(task="RestDbs", description="cleaned")
    for idx in tqdm(range(len(rest_data_all)), desc="Bidsify Tan DBS"):
        session, ch_name, subject, hemi = _get_dbs_info(idx)
        sample_rate = sample_rate_all[idx]
        info = create_info([ch_name], sample_rate, ch_types='dbs')

        rest_data = rest_data_all[idx].reshape(1, -1)
        raw = RawArray(rest_data, info)
        _correct_units(raw)
        bids_path.update(acquisition=f"StimOff{hemi}", session=session,
                         subject=subject)
        _add_info(raw, bids_path)
        _save_bids(raw, bids_path)
        _delete_dirty_files(bids_path)

        if add_dbs:
            dbs_data = dbs_data_all[idx].reshape(1, -1)
            raw = RawArray(dbs_data, info)
            _correct_units(raw)
            bids_path.update(acquisition=f"StimOn{hemi}")
            _add_info(raw, bids_path)
            _save_bids(raw, bids_path)
            _delete_dirty_files(bids_path)


def _get_dbs_info(idx):
    info = cfg.DBS_MATRIX_INFO[idx]

    # Get medication info
    if info.endswith("OFF"):
        session = "LfpMedOff01"
        info = info.strip("OFF")
    elif info.endswith('ON'):
        session = "LfpMedOn01"
        info = info.strip("ON")
    else:
        raise ValueError(f"Info {info} does not end with 'OFF' or 'ON'.")

    # Get hemisphere info
    hemi = info[-1]
    assert hemi in ["L", "R"]
    # TODO: find out which contacts were used. For now, set to 1-3
    ch_name = f'LFP_{hemi}_1-3_STN_MT'
    hemisphere = 'Left' if hemi == 'L' else 'Right'
    info = info[:-1]

    # Get subject info
    subject_old = info
    subject_new = cfg.TAN_SUBJECT_MAP[subject_old]
    return session, ch_name, subject_new, hemisphere


if __name__ == "__main__":
    bidsify_sourcedata_tan()