"""Bidsify Hirschmann and remove MEG channels."""
from json import load
from os.path import basename, join

import pandas as pd
from mne import concatenate_raws
from mne_bids import BIDSPath, find_matching_paths, read_raw_bids
from tqdm import tqdm

import scripts.config as cfg
from scripts.bidsify_sourcedata import _add_info
from scripts.utils import _copy_files_and_dirs, _delete_dirty_files, _save_bids


def bidsify_sourcedata_hirschmann() -> None:
    source_root = cfg.SOURCEDATA_HIR
    raw_root = cfg.RAWDATA

    sub_dic = cfg.SUB_MAPPING_HIR
    ses_dic = cfg.SESSION_MAPPING_HIR
    tasks = ['Rest', 'HoldL', 'HoldR', 'MoveL', 'MoveR']
    for subj_old, sub_new in tqdm(sub_dic.items(), desc="Bidsify Hirschmann"):
        for cond in ['MedOff', 'MedOn']:
            bids_paths = []
            i = 0
            while not len(bids_paths):
                # load one task for each subject and ignore others
                task = tasks[i]
                bids_paths = find_matching_paths(root=source_root,
                                                 subjects=subj_old,
                                                 acquisitions=cond,
                                                 tasks=task,
                                                 runs='1',
                                                 extensions=".fif")
                i += 1
                if i == 5:
                    assert subj_old == 'zxEhes', "Med On missing."
                    break
            if len(bids_paths) == 0:
                continue
            elif len(bids_paths) == 1:
                bids_path = bids_paths[0]
                raw = read_raw_bids(bids_path)
            else:
                # combine splits
                raws = [read_raw_bids(bids_path.update(split=f'{split:02d}'))
                        for split, bids_path in enumerate(bids_paths, 1)]
                raw = concatenate_raws(raws)
                bids_path = bids_paths[0]
            raw.pick_types(eeg=True)

            # crop out resting state
            raw, _ = get_raw_condition(raw, ['rest'])

            # rename channels
            raw = _rename_channels(raw, bids_path)
            raw.set_channel_types({ch: 'dbs' for ch in raw.ch_names})
            _add_missing_chs(raw)

            # save in rawdata
            bids_path.update(root=raw_root,
                             subject=sub_new,
                             session=ses_dic[cond],
                             task=task,
                             run=1,
                             split=None,
                             suffix="ieeg",
                             extension=".fif",
                             description="uncleaned",
                             recording="Hirschmann",
                             acquisition="StimOff",
                             datatype="ieeg")

            _add_bad_channels(raw, bids_path)
            _add_bad_segments(raw, bids_path)

            _add_info(raw, bids_path)
            _save_bids(raw, bids_path)
            _delete_dirty_files(bids_path)
    raw_root_meta = join(raw_root, "meta_infos_Hirschmann")
    _copy_files_and_dirs(source_root, raw_root_meta, cfg.BIDS_FILES_TO_COPY)
    copy_dirs = ['participants_updrs_on.tsv', 'participants_updrs_off.tsv']
    raw_root_meta = join(raw_root, "meta_infos_Hirschmann", "meta_infos")
    _copy_files_and_dirs(source_root, raw_root_meta, copy_dirs)
    print(f"{basename(__file__).strip('.py')} done.")


def _add_missing_chs(raw):
    needed_chs = {'LFP_R_1_STN_MT', 'LFP_R_2_STN_MT', 'LFP_R_3_STN_MT',
                  'LFP_R_4_STN_MT', 'LFP_L_1_STN_MT', 'LFP_L_2_STN_MT',
                  'LFP_L_3_STN_MT', 'LFP_L_4_STN_MT'}
    missing_chs = needed_chs - set(raw.ch_names)
    if not missing_chs:
        return
    # make sure suffix is correct
    assert raw.ch_names[0].split('_')[-1] == 'MT', "Suffix incorrect."

    # add dummy channels
    dummy_ch = raw.copy().pick_types(dbs=True).pick(0).load_data()
    dummy_ch._data.fill(0)
    raw.load_data()
    for ch in missing_chs:
        fill_ch = dummy_ch.copy()
        fill_ch.rename_channels({dummy_ch.ch_names[0]: ch})
        raw.add_channels([fill_ch])
        raw.info['bads'].append(ch)


def get_raw_condition(raw, conds):
    '''
    Parameters
    ----------
    raw : mne.io.Raw
        raw object
    conds : list
        list of conditions
    Returns
    -------
    tuple of rest and task segments
    if no event found in the raw.annotations --> the value will be None.
    '''
    # create a copy of the list so we can modify it without changing the
    # original list
    conditions = conds.copy()

    # available conditions in the bids dataset
    allowed_conditions = [['rest', 'HoldL'], ['rest', 'MoveL'],
                          ['rest', 'HoldR'], ['rest', 'MoveR'], ['rest']]
    msg = f'Conditions should be in {allowed_conditions}'
    assert conditions in allowed_conditions, msg

    # initialise the segments by None
    task_segments = None
    rest_segments = None

    # check that the raw actually have resting state and not only task
    # [e.g., files run-2]
    if 'rest' not in raw.annotations.description:
        conditions.remove('rest')

    for task in conditions:
        # get the onset and the duration of the event
        segments_onset = raw.annotations.onset[raw.annotations.description
                                               == task]
        segments_duration = raw.annotations.duration[
            raw.annotations.description == task]

        # substract the first_sample delay in the onset
        segments_onset = segments_onset - (raw.first_samp / raw.info['sfreq'])

        # loop trough the onset and duration to get only part of the raw that
        # are in the task
        for i, (onset, duration) in enumerate(zip(segments_onset,
                                                  segments_duration)):
            # if it is not resting state
            if task != 'rest':
                # if it is the first onset, initialise the raw object storing
                # all of the task segments
                if i == 0:
                    task_segments = raw.copy().crop(tmin=onset,
                                                    tmax=onset+duration)
                # otherwise, append the segments to the existing raw object
                else:
                    task_segments.append([
                        raw.copy().crop(tmin=onset, tmax=onset+duration)])
            # do the same for rest
            else:
                if i == 0:
                    rest_segments = raw.copy().crop(tmin=onset,
                                                    tmax=onset+duration)
                else:
                    rest_segments.append([
                        raw.copy().crop(tmin=onset, tmax=onset+duration)])
    return rest_segments, task_segments


def _rename_chs(ch):
    ch = ch.replace('-', '_')
    ch = ch.replace('right', 'R').replace('left', 'L')
    int_old = ch[-1]
    int_new = int(int_old) + 1
    ch = ch.replace(int_old, str(int_new))
    ch += '_STN_MT'
    return ch


def _rename_channels(raw, bids_path):
    # create a bids path for the montage, check=False because this is not a
    # standard bids file
    montage_path = BIDSPath(root=bids_path.root, session=bids_path.session,
                            subject=bids_path.subject, datatype='montage',
                            extension='.tsv', check=False)

    # get the montage tsv bids file
    montage = montage_path.match()[0]

    # read the file using pandas
    df = pd.read_csv(montage, sep='\t')

    # rename rows in df
    df['right_contacts_new'] = df['right_contacts_new'].apply(_rename_chs)
    df['left_contacts_new'] = df['left_contacts_new'].apply(_rename_chs)

    # create a dictionary mapping old names to new names for right
    # and left channels
    montage_mapping = {row['right_contacts_old']: row['right_contacts_new']
                       for _, row in df.iterrows()}
    montage_mapping.update({row['left_contacts_old']: row['left_contacts_new']
                            for _, row in df.iterrows()})

    # remove in the montage mapping the channels that are not in the raw
    # anymore (because they were marked as bads)
    montage_mapping = {key: value for key, value in montage_mapping.items()
                       if key in raw.ch_names}

    # rename the channels using the montage mapping scheme
    raw.rename_channels(montage_mapping)
    return raw


def _add_bad_channels(raw: Raw, bids_path: BIDSPath):
    anno_dic = dict(root=cfg.ANNOTATIONS,
                    extensions='.json',
                    suffixes='channels', descriptions=None,
                    subjects=bids_path.subject, datatypes=bids_path.datatype,
                    sessions=bids_path.session, tasks=bids_path.task,
                    acquisitions=bids_path.acquisition,
                    # runs=bids_path.run,
                    recordings=bids_path.recording)
    anno_path = find_matching_paths(**anno_dic)
    msg = f"More than one annotation file found for {bids_path}"
    assert len(anno_path) < 2, msg
    try:
        anno_path = str(anno_path[0].fpath)
        with open(anno_path) as bad_chs:
            bad_chs = load(bad_chs)
    except (FileNotFoundError, TypeError, IndexError):
        msg = f"\n\n{bids_path.basename} has not been annotated yet!\n\n"
        raise FileNotFoundError(msg)
    # only add bad channels that exist (no bipolar chs)
    bad_chs = set(bad_chs).intersection(set(raw.ch_names))
    raw.info["bads"] += list(bad_chs)


def _add_bad_segments(raw: Raw, bids_path: BIDSPath):
    anno_dic = dict(root=cfg.ANNOTATIONS,
                    extensions='.csv',
                    suffixes='events', descriptions=None,
                    subjects=bids_path.subject, datatypes=bids_path.datatype,
                    sessions=bids_path.session, tasks=bids_path.task,
                    acquisitions=bids_path.acquisition,
                    # runs=bids_path.run,
                    recordings=bids_path.recording)
    anno_path = find_matching_paths(**anno_dic)
    msg = f"More than one annotation file found for {bids_path}"
    assert len(anno_path) < 2, msg

    try:
        anno_path = str(anno_path[0].fpath)
    except (IndexError):
        raise FileNotFoundError(f"\n\n{bids_path.basename} has not been annotated yet!\n\n")
    try:
        annotations = read_annotations(anno_path)
    except IndexError:
        # no annotations present
        annotations = None
    except (FileNotFoundError, TypeError):
        raise FileNotFoundError(f"\n\n{bids_path.basename} has not been annotated yet!\n\n")
    raw.set_annotations(annotations)
    bids_path.update(description='cleaned')


# def _get_annotation_path(bids_path, anno=None, extension=None):
#     # remove run and processing and task since unambigious and was changed
#     # after annotating
#     bids_path = bids_path.copy().update(run=None, processing=None, task=None)
#     anno_root = cfg.ANNOTATIONS
#     bids_info = dict(subjects=bids_path.subject, sessions=bids_path.session,
#                      tasks=bids_path.task, runs=bids_path.run, root=anno_root,
#                      extensions=extension, descriptions=None)
#     anno_path = find_matching_paths(**bids_info)
#     msg = f"More than one annotation file found for {bids_path}"
#     if anno_path:
#         assert len(anno_path) < 2, msg
#         return str(anno_path[0].fpath)
#     # Don't load old annotations. Very messy. Annotate again.
#     elif not anno_path and anno is None:
#         return None

#     if anno is not None:
#         anno_root = join(cfg.SOURCEDATA, "BIDS_Hirschmann_MEG_LFP", anno)
#         subject = bids_path.subject.replace(cfg.SUB_PREFIX_HIR, '')
#         session = bids_path.session.replace('01', '')
#         bids_info.update(root=anno_root, subjects=subject, sessions=session)
#         anno_path = find_matching_paths(**bids_info)
#         if anno_path:
#             msg = f"More than one annotation file found for {bids_path}"
#             assert len(anno_path) < 2, msg
#             return str(anno_path[0].fpath)
#         else:
#             return None


if __name__ == "__main__":
    bidsify_sourcedata_hirschmann()