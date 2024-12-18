"""Helping functions."""
from os.path import isfile
from pathlib import Path
from shutil import move
from warnings import warn

import mat73
import numpy as np
import pandas as pd
import scipy.io as spio
from mne.io import Raw

import scripts.config as cfg


def _add_info(raw, bids_path, amp=None, bids_path_old=None):
    """Add info to raw.info."""
    # Set raw.info
    with raw.info._unlock():
        raw.info["proj_name"] = bids_path.recording

    sub = bids_path.entities["subject"]
    ses = bids_path.entities["session"]
    tsk = bids_path.entities["task"]
    run = bids_path.entities["run"]

    raw.info["line_freq"] = 50
    if bids_path.recording == "Litvak":
        description = f"Sub-{sub} Ses-{ses} Task-{tsk}"
        sub_dic = {"his_id": sub,
                   "middle_name": 3}
        raw.info["subject_info"] = sub_dic
        raw.info['device_info'] = dict(model='CTF MEG System', type=None)
    elif bids_path.recording == "Neumann":
        raw.info['device_info'] = dict(model='TMSi Saga', type=None)
        acq = bids_path.entities["acquisition"]
        ref_channel = _indicate_reference_channel(raw, bids_path_old)
        description = (f"Sub-{sub} Ses-{ses} Task-{tsk} "
                        f"Acq-{acq} Run-{run} Amp-{amp} Ref-{ref_channel}.")
        lead_num = cfg.DBS_LEAD_MAP_NEU[sub.strip("Neu")]
        sub_dic = {"his_id": sub, "middle_name": lead_num}
        raw.info["subject_info"] = sub_dic

        if bids_path.session == 'TMSiSAGA20220916':
            # only applies to old impedance measurement
            impedances = np.array([float(ch.split("_")[1].strip("kOhm"))
                                   for ch in raw.ch_names])
            mono_ch_names = [f"{imp:.2f} kOhm" for imp in impedances]
            mono_ch_names_dic = dict(zip(raw.ch_names, mono_ch_names))
            raw.rename_channels(mono_ch_names_dic, verbose=False)
            bids_path.update(datatype="ieeg", suffix="ieeg")

    elif bids_path.recording == "Tan":
        acq = bids_path.entities["acquisition"]
        if bids_path.subject == "TanEmptyroom":
            sub_dic = {"his_id": sub, "middle_name": 0}
            sample_rate = raw.info["sfreq"]
        else:
            sub_old = cfg.TAN_SUBJECT_MAP_REV[sub]
            lead_num = cfg.DBS_LEAD_MAP_TAN[sub_old]
            sub_dic = {"his_id": sub, "middle_name": lead_num}
            sample_rate = raw.info["sfreq"]

        if sample_rate == 2048:
            amp = 'TMSi Porti'
        elif sample_rate == 4096:
            amp = 'TMSi SAGA'
        else:
            amp = np.nan  # Tan subjects with missing channels
        raw.info['device_info'] = dict(model=amp, type=None)

        description = (f"Sub-{sub} Ses-{ses} Task-{tsk} "
                       f"Acq-{acq} Run-{run} Amp-{amp}")

        raw.info["subject_info"] = sub_dic
    elif bids_path.recording == "Hirschmann":
        raw.info['device_info'] = dict(model='ElektaNeuromag', type=None)
        acq = bids_path.entities["acquisition"]
        description = f"Sub-{sub} Ses-{ses} Task-{tsk}"
        if bids_path.subject == "HirEmptyroom":
            sub_dic = {"his_id": sub, "middle_name": 0}
        else:
            sub_old = cfg.SUB_MAPPING_HIR_REV[sub]
            lead_num = cfg.DBS_LEAD_MAP_HIR[sub_old]
            sub_dic = {"his_id": sub, "middle_name": lead_num}
        raw.info["subject_info"] = sub_dic
    elif bids_path.recording == "Florin":
        raw.info['device_info'] = dict(model='ElektaNeuromag', type=None)
        acq = bids_path.entities["acquisition"]
        description = (f"Sub-{sub} Ses-{ses} Task-{tsk} "
                       f"Acq-{acq} Run-{run}.")
        if sub == "FloML012":
            lead_num = 2
        elif sub == "FloEmptyroom":
            lead_num = 0
        else:
            lead_num = 5
        sub_dic = {"his_id": sub, "middle_name": lead_num}
        raw.info["subject_info"] = sub_dic

    raw.info["description"] = description


def _move_files(source, destination):
    """Move files from source to destination directory."""
    # list all files in source
    files = [f for f in Path(source).glob("*") if isfile(f)]
    for file in files:
        move(file, str(file).replace(source, destination))


def _indicate_reference_channel(raw: Raw, bids_path) -> str:
    """Set eeg reference to make reference channel explicit."""
    if bids_path is None:
        return "1 kOhm"
    elif bids_path.subject == "emptyroom":
        return "1.5 kOhm"
    elif "Litvak" in str(bids_path.root):
        return "forehead/mastoid"
    fname = bids_path.copy().update(suffix="channels", extension=".tsv").fpath
    table = pd.read_csv(fname, sep='\t')
    dbs_reference = table[table.type == "DBS"].reference.unique()
    ecog_reference = table[table.type == "ECOG"].reference.unique()
    dbs_reference = list(dbs_reference)
    ecog_reference = list(ecog_reference)
    if len(dbs_reference) > 1:
        raise ValueError("More than one reference channel for DBS: "
                         f"{dbs_reference}")
    if len(ecog_reference) > 1:
        if bids_path.subject == "EL007":
            # Subject 7 has long ecog strip with skull facing elecrodes
            # referenced to LFP_L_1.
            ecog_reference.remove("ECOG_R_12_SMC_AT")
        else:
            raise ValueError("More than one reference channel for ECOG: "
                             f"{ecog_reference}")
    if dbs_reference != ecog_reference:
        ref_channel = " + ".join(dbs_reference + ecog_reference)
    else:
        ref_channel = dbs_reference[0]
        try:
            raw.load_data().set_eeg_reference(list(dbs_reference),
                                              verbose=False)
        except ValueError:
            warn(f"Reference channel is missing for {bids_path.basename}.")
    raw.info["bads"] += [ref_channel]
    return ref_channel


def _correct_units(raw):
    muV_to_V = lambda x: x * 1e-6
    raw.load_data()
    raw.apply_function(muV_to_V, picks=["dbs", "ecog"], channel_wise=False)


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    try:
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        data = mat73.loadmat(filename)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict