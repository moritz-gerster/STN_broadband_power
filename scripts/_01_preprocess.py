"""Preprocessing script."""
import re
from itertools import combinations
from json import dump, load
from os.path import basename, join

import numpy as np
from mne import (Info, pick_channels_regexp, pick_types, rename_channels,
                 set_bipolar_reference, set_eeg_reference, set_log_level)
from mne.channels import combine_channels
from mne.io import Raw, read_raw
from mne_bids import find_matching_paths, get_entity_vals
from tqdm import tqdm

import scripts.config as cfg
from scripts.utils import (_copy_files_and_dirs, _delete_dirty_files,
                           _get_ref_from_info, _save_bids)


def preprocess(subjects=None, descriptions=None, sessions=None,
               recordings=cfg.RECORDINGS, LAR=False,
               bipolar_ref=True, bipolar_distant=True,
               bipolar_directional=False):
    """Preprocessing pipeline.

    Filter, resample, standardize DBS leads, rereference."""
    set_log_level('error')
    load_root = cfg.RAWDATA
    save_root = cfg.PREPROCESSED

    # This is important for channel name conversion of Neumann data
    process_all_subjects = False
    if subjects is None:
        subjects = get_entity_vals(load_root, "subject", ignore_subjects=None)
        if 'Neumann' in recordings:
            process_all_subjects = True

    bids_paths = find_matching_paths(load_root, extensions=".fif",
                                     recordings=recordings,
                                     descriptions=descriptions,
                                     sessions=sessions, subjects=subjects)

    for bids_path_raw in tqdm(bids_paths, desc="Preprocessing: "):
        if bids_path_raw.session == "TMSiSAGA20220916":
            # old recording
            continue

        raw = read_raw(bids_path_raw.fpath, preload=True)

        bids_path_new = bids_path_raw.copy()
        bids_path_new.update(root=save_root, processing='Highpass')

        raw.filter(cfg.HIGHPASS, cfg.LOWPASS)  # equalize all datasets

        # Apply preprocessing functions
        raw.resample(sfreq=cfg.RESAMPLE_FREQ)  # assert same srate across files
        msg = 'Recording info mismatch between raw object and BIDS path.'
        assert raw.info["proj_name"] == bids_path_new.recording, msg
        ref_kwargs = dict(raw=raw, LAR=LAR, bipolar=bipolar_ref,
                          bipolar_directional=bipolar_directional,
                          bipolar_distant=bipolar_distant)
        rereference(**ref_kwargs)

        # from scripts.utils_plot import plot_psd_units
        # plot_psd_units(raw, title=bids_path_new.subject)

        # Save raw
        _save_bids(raw, bids_path_new)
        _delete_dirty_files(bids_path_new)
    # Copy meta info as is
    # recordings = get_entity_vals(load_root, "recording")  # does not work yet
    # this would be much more elegant because it avoids a-priori specification
    # by setting recordings=None. Will be fixed in mne soon!!!!!!!!!
    recordings = [recordings] if isinstance(recordings, str) else recordings
    for recording in recordings:
        _copy_files_and_dirs(join(load_root, f"meta_infos_{recording}"),
                             join(save_root, f"meta_infos_{recording}"),
                             cfg.BIDS_FILES_TO_COPY)
    if 'Neumann' in recordings:
        _save_ch_name_conversion(process_all_subjects)
    print(f"{basename(__file__).strip('.py')} done.")


def rereference(raw: Raw, LAR=True, bipolar=True, bipolar_directional=False,
                bipolar_distant=True) -> None:
    """Reference raw data."""
    if (raw.info["proj_name"] == "Tan" and
            not raw.info["subject_info"]["his_id"].endswith('Emptyroom')):
        return None  # already bipolar
    elif raw.info["proj_name"] == "Litvak":
        if not bipolar_distant:
            return None  # already bipolar
        else:
            _distant_bip_from_adjacent_bip(raw)
            return None
    # average directional leads
    rename_dic_dbs = _standardize_dbs_leads(raw)
    _add_bad_directional_amplifier_channels(raw)
    new_channel_names = NewChannelNames()
    new_channel_names.add_failed_fit(rename_dic_dbs)  # Save new channel names
    if LAR:
        _reference_average(raw)
    if bipolar:
        _rereference_bipolar(raw, bipolar_directional=bipolar_directional,
                             bipolar_distant=bipolar_distant)


def _add_bad_directional_amplifier_channels(raw):
    """Simulate the averaging of bipolar directional leads in the case of
    bad directional channels."""
    if raw.info["subject_info"]["his_id"] not in ['FloEmptyroom',
                                                  'NeuEmptyroom']:
        return None
    # add channel 2ab and 3ab (c is bad)
    for num in [2, 3]:
        dir_ch_indices = [raw.ch_names.index(f'Amp_L_{num}a'),
                          raw.ch_names.index(f'Amp_L_{num}b')]
        average = {f'Amp_L_{num}ab': dir_ch_indices}
        comb_dic = dict(inst=raw, groups=average, method='mean', verbose=False)
        combined = combine_channels(**comb_dic)
        raw.add_channels([combined], force_update_info=True)


def _distant_bip_from_adjacent_bip(raw):
    """Get distant bipolar reference from already bipolar referenced data."""
    anodes = ['LFP_L_1-2_STN_MT', 'LFP_L_2-3_STN_MT',
              'LFP_R_1-2_STN_MT', 'LFP_R_2-3_STN_MT']
    cathodes = ['LFP_L_2-3_STN_MT', 'LFP_L_3-4_STN_MT',
                'LFP_R_2-3_STN_MT', 'LFP_R_3-4_STN_MT']
    ch_names = ['LFP_L_1-3_STN_MT', 'LFP_L_2-4_STN_MT',
                'LFP_R_1-3_STN_MT', 'LFP_R_2-4_STN_MT']

    # check if channel has data
    all_nan = lambda raw, anode: np.isnan(raw.get_data(anode)).all()

    # remove channels that do not exist or are all nan (causes silent error)
    for anode, cathode, ch_nme in list(zip(anodes, cathodes, ch_names)):
        anode_exists = anode in raw.ch_names and not all_nan(raw, anode)
        cathode_exists = cathode in raw.ch_names and not all_nan(raw, cathode)
        if not anode_exists or not cathode_exists:
            anodes.remove(anode)
            cathodes.remove(cathode)
            ch_names.remove(ch_nme)
    # apply bipolar reference only to selected channels. Single nan channel
    # destroys the whole array. Append back afterward.
    raw_bip = raw.copy().pick_channels(set(anodes + cathodes))
    bipolar_dic = dict(anode=anodes, cathode=cathodes, ch_name=ch_names)
    set_bipolar_reference(raw_bip, **bipolar_dic, copy=0, verbose=0,
                          on_bad="ignore", drop_refs=True)
    raw.add_channels([raw_bip], force_update_info=True)

    # add bipolar mni coords
    for idx, ch_nme in enumerate(ch_names):
        ch_info = raw.info["chs"]
        anode = anodes[idx]
        cathode = cathodes[idx]

        anode_idx = raw.ch_names.index(anode)
        cathode_idx = raw.ch_names.index(cathode)
        bip_idx = raw.ch_names.index(ch_nme)

        mni_anode = ch_info[anode_idx]["loc"]
        mni_cathode = ch_info[cathode_idx]["loc"]
        mni_bip = (mni_anode + mni_cathode) / 2
        assert ch_info[bip_idx]["ch_name"] == ch_nme
        ch_info[bip_idx]["loc"] = mni_bip

    # Combine channels to get bipolar electrodes 1-4
    for hemi in ["L", "R"]:
        ch_name_new = f"LFP_{hemi}_1-4_STN_MT"
        ch_names_old = [f"LFP_{hemi}_1-2_STN_MT", f"LFP_{hemi}_2-3_STN_MT",
                        f"LFP_{hemi}_3-4_STN_MT"]
        if not set(ch_names_old).issubset(set(raw.ch_names)):
            continue
        all_good = all(ch not in raw.info["bads"] for ch in ch_names_old)
        ch_idcs_old = [raw.ch_names.index(ch) for ch in ch_names_old]
        sum_chs = {ch_name_new: ch_idcs_old}

        comb_dic = dict(inst=raw, groups=sum_chs, verbose=False,
                        method=lambda data: np.sum(data, axis=0),
                        drop_bad=False)
        combined = combine_channels(**comb_dic)
        # get mean mni coordinates of directional leads
        mni_coords = np.array([raw.info["chs"][idx]["loc"] for idx
                               in ch_idcs_old]).mean(0)
        combined.info["chs"][0]["loc"] = mni_coords
        raw.add_channels([combined], force_update_info=True)
        if not all_good:
            raw.info['bads'].append(ch_name_new)


def _save_ch_name_conversion(all_subjects):
    """Save new channel names as json file if all subjects are evaluated."""
    if all_subjects:
        # delete dictionary in case of few subjects
        converted_ch_nms = {}
    else:
        converted_ch_nms = open(cfg.NEW_CH_NAMES)
        converted_ch_nms = load(converted_ch_nms)
    new_channel_names = NewChannelNames()
    for sub_dic in new_channel_names.data:
        converted_ch_nms.update(sub_dic)
    with open(cfg.NEW_CH_NAMES, 'w') as file:
        dump(converted_ch_nms, file, indent=4)


def _reference_emptyroom(raw, LAR=True, bipolar=True):
    """Needed because noise level changes with LAR and bipolar montage."""
    if raw.info["subject_info"]["his_id"].startswith('Neu'):
        # Simulate averaging of directional leads
        # randomly permute channels to eliminate increasing impedance order
        np.random.seed(seed=1)
        permute = np.random.permutation(range(len(raw.ch_names)))
        raw.reorder_channels([raw.ch_names[idx] for idx in permute])
        new_names = ['Amp_Z_1', 'Amp_Z_2a', 'Amp_Z_2b', 'Amp_Z_2c',
                     'Amp_Z_3a', 'Amp_Z_3b', 'Amp_Z_3c', 'Amp_Z_4']
        rename = dict(zip(raw.ch_names, new_names))
    elif raw.info["subject_info"]["his_id"].startswith('Flo'):
        rename = cfg.FLORIN_CHNAME_MAP_EMPTY
    raw.rename_channels(rename)
    raw.reorder_channels(sorted(raw.ch_names))

    # average
    if raw.info["subject_info"]["his_id"].startswith('Neu'):
        average = {'Amp_Z_2': [1, 2, 3], 'Amp_Z_3': [4, 5, 6]}
        comb_dic = dict(inst=raw, groups=average, method='mean', verbose=False,
                        drop_bad=False)
    elif raw.info["subject_info"]["his_id"].startswith('Flo'):
        average = {'Amp_L_2': [1, 2, 3], 'Amp_L_3': [4, 5, 6],
                   'Amp_R_2': [9, 10, 11], 'Amp_R_3': [12, 13, 14]}
        comb_dic = dict(inst=raw, groups=average, method='mean', verbose=False,
                        drop_bad=False)
    combined = combine_channels(**comb_dic)
    raw.add_channels([combined], force_update_info=True)
    raw.reorder_channels(sorted(raw.ch_names))

    if LAR:
        # average ECOG hemisphere
        raw_lar, _ = set_eeg_reference(raw, verbose=False)

        # LAR Montage
        lar_ch_names = [f"{ch_name}_LAR" for ch_name in raw_lar.ch_names]
        ch_names_lar_dic = dict(zip(raw.ch_names, lar_ch_names))
        raw_lar.rename_channels(ch_names_lar_dic)
        raw.add_channels([raw_lar], force_update_info=True)

    if bipolar:
        if raw.info["subject_info"]["his_id"].startswith('Neu'):
            anodes = ['Amp_Z_1', 'Amp_Z_2', 'Amp_Z_3', 'Amp_Z_1']
            cathodes = ['Amp_Z_2', 'Amp_Z_3', 'Amp_Z_4', 'Amp_Z_4']
            bip_ch_names = ['Amp_Z_1-2', 'Amp_Z_2-3', 'Amp_Z_3-4', 'Amp_Z_1-4']
        elif raw.info["subject_info"]["his_id"].startswith('Flo'):
            anodes = ['Amp_L_1', 'Amp_L_2', 'Amp_L_3', 'Amp_L_1',
                      'Amp_R_1', 'Amp_R_2', 'Amp_R_3', 'Amp_R_1']
            cathodes = ['Amp_L_2', 'Amp_L_3', 'Amp_L_4', 'Amp_L_4',
                        'Amp_R_2', 'Amp_R_3', 'Amp_R_4', 'Amp_R_4']
            bip_ch_names = ['Amp_L_1-2', 'Amp_L_2-3', 'Amp_L_3-4', 'Amp_L_1-4',
                            'Amp_R_1-2', 'Amp_R_2-3', 'Amp_R_3-4', 'Amp_R_1-4']
        set_bipolar_reference(raw, anodes, cathodes, bip_ch_names,
                              copy=False, drop_refs=False, on_bad="ignore",
                              verbose=False)


def _reference_average(raw: Raw) -> None:
    """Reference to local average reference (LAR)."""

    # Average reference ECOG (_ECOG)
    raw_ecog_lar = _reference_average_ctx(raw)

    # Average reference STN left (_LSTN) and right (_RSTN).
    raw_stn_lar = _reference_average_stn(raw, reference_zero=False)

    # Add
    with raw.info._unlock():
        raw.info["custom_ref_applied"] = 1
    if raw_ecog_lar is not None:
        # Important: Keep force_update_info=False! Otherwise "bads" are lost.
        raw_lar = raw_ecog_lar.add_channels(raw_stn_lar,
                                            force_update_info=False)
        raw.add_channels([raw_lar], force_update_info=False)
    else:
        raw.add_channels(raw_stn_lar)


def _reference_average_ctx(raw):
    """Get new raw object with ECOG average reference."""
    try:
        ecog_channels = raw.copy().pick_types(ecog=True, exclude=[])
    except ValueError:
        return None
    # average ECOG hemisphere
    raw_ecog_lar, _ = set_eeg_reference(ecog_channels, verbose=False)
    dic_lar = {ch: f"{ch}_ECOG" for ch in raw_ecog_lar.ch_names}
    raw_ecog_lar.rename_channels(dic_lar)

    # Exclude bipolar reference
    assert all("-" not in ch for ch in raw_ecog_lar.ch_names)
    return raw_ecog_lar


def _reference_average_stn(raw, reference_zero=True):
    """Get new raw object with STN average reference."""

    raws_stn_lar = []
    for hemi in ["L", "R"]:
        # Pick all STN channels from same hemisphere
        match_all = rf"^LFP_{hemi}_\d.*_STN_"
        match_ring = rf"^LFP_{hemi}_\d_STN_"
        stn_chs_all = list(filter(re.compile(match_all).match, raw.ch_names))
        stn_chs_ring = list(filter(re.compile(match_ring).match, stn_chs_all))

        if not len(stn_chs_all):
            # hemisphere missing
            continue

        raw_stn_all = raw.copy().pick_channels(stn_chs_all)
        raw_stn_ring = raw.copy().pick_channels(stn_chs_ring)

        # Get reference channels: all channels except reference and bads
        reference = _get_ref_from_info(raw.info)

        if reference not in raw.ch_names:
            ref_channels = set(stn_chs_ring)
        else:
            # delete this if running through
            ref_idx = raw.info["description"].find("Ref-") + 4
            reference_del = raw.info["description"][ref_idx:-1]
            assert reference_del == reference
            assert reference in raw.info["bads"]
            ref_channels = set(stn_chs_ring) - set(raw.info["bads"])

        # Average reference STN by using only the mean of the ring leads
        _, ref_mean = set_eeg_reference(raw_stn_ring,
                                        ref_channels=ref_channels,
                                        ch_type="dbs")

        # Correct ref_mean in case of bad directional leads. Bad ring leads
        # are automatically taken care of by mne.set_eeg_reference. Each bad
        # ring lead reduces the mean by 1/4*1/3 = 1/12.
        stn_chs_dir = set(stn_chs_all) - set(stn_chs_ring)
        bads_all = raw.info["bads"]
        bads_dir = set(bads_all).intersection(stn_chs_dir)
        num_bad_dir = len(bads_dir)
        if num_bad_dir > 0:
            mean_correction = 1 - (1/12 * num_bad_dir)
            ref_mean *= mean_correction

        # Apply reference to all leads of this hemisphere including bad
        # channels (to check in the PSD whether they are actually bad).
        if reference_zero:
            # But do not apply to reference channel (to keep it all zeros):
            apply_ref = list(set(stn_chs_all) - {reference})
        else:
            apply_ref = stn_chs_all
            if reference in raw_stn_all.info["bads"]:
                raw_stn_all.info["bads"].remove(reference)
        raw_stn_all.apply_function(lambda x: x - ref_mean, picks=apply_ref)

        # Rename channels
        dic_lar = {ch: f"{ch}_{hemi}STN" for ch in stn_chs_all}
        raw_stn_all.rename_channels(dic_lar)

        # 'custom_ref_applied' must be true, otherwise concatenation error
        # later when using mne.add_channels. For some reason, only subject 8
        # has some issues with this.
        if not raw_stn_all.info["custom_ref_applied"]:
            with raw_stn_all.info._unlock():
                raw_stn_all.info["custom_ref_applied"] = 1
        raws_stn_lar.append(raw_stn_all)

    return raws_stn_lar


def _rereference_bipolar(raw: Raw, drop_refs: bool = False,
                         bipolar_directional=False, bipolar_distant=False):
    """Rereference STN bipolar."""
    reference = _get_ref_from_info(raw.info)

    kwargs = dict(info=raw.info, bipolar_directional=bipolar_directional,
                  bipolar_distant=bipolar_distant)
    bipolar_dic, bip_coords = _map_ring_to_bipolar(**kwargs)

    # mark ref as good for bipolar reference and then as bad again
    if reference in raw.ch_names:
        raw.info["bads"].remove(reference)
    set_bipolar_reference(raw, **bipolar_dic, copy=0, verbose=0,
                          on_bad="ignore", drop_refs=drop_refs)
    if reference in raw.ch_names:
        raw.info["bads"] += [reference]

    # add bipolar mni coords
    bipolar_channels = bipolar_dic["ch_name"]
    for ch_dic in raw.info["chs"]:
        if ch_dic["ch_name"] in bipolar_channels:
            ch_dic["loc"] = bip_coords[ch_dic["ch_name"]]

    # reorder channels
    other_channels = set(raw.ch_names) - set(bipolar_channels)
    ch_order = sorted(list(bipolar_channels)) + sorted(list(other_channels))
    raw.reorder_channels(ch_order)


def _map_ring_to_bipolar(info: Info, bipolar_directional=False,
                         bipolar_distant=False) -> tuple[dict, list]:
    """Extract correct bipolar channel mapping.

    bids_ch_types: bids_ch_types to rereference. For example:
    bids_ch_types = ["ECOG", "LFP", "EEG"].
    Important: The BIDS channel
    type may be different from the MNE channel type. For example, the channel
    "LFP_R_1_STN_BS" corresponds to the BIDS channel type "LFP" and to the
    MNE channel type "DBS". Default is "all" which selects all available
    types.

    This function assumes that the channel names follow BIDS convention and
    does not check for it.
    """
    anodes = []
    cathodes = []
    bip_ch_names = []
    bip_coordinates = {}

    only_neighbors = True if not bipolar_distant else False
    only_rings = True if not bipolar_directional else False
    # Hemisphere in alphabetical order to match the channel names
    # which are in alphabetical order.
    for hemisphere in sorted(["L", "R"]):
        # Select current ch_type and hemisphere:
        # type_hemi = rf"^LFP_{hemisphere}_"
        type_hemi = rf"^..._{hemisphere}_"
        # to only match LFP ring leads: num = r"\d*"
        # to only match LFP directional leads: num = r"\d[abc]"
        # to also match LFP directional + ring (+ combined) leads:
        num = r"\d[abc]*"  # match 1 digit and 0 or more letters
        # match a maximum of 4 times "_" to get monopolar chs:
        # area_manufact = r"_[^_]*_[^_]*$"
        # expr = type_hemi + num + area_manufact
        expr = type_hemi + num
        idx = pick_channels_regexp(info.ch_names, expr)
        chs_type_hemi = sorted([info.ch_names[i] for i in idx
                                if not info.ch_names[i].endswith("LSTN")
                                and not info.ch_names[i].endswith("RSTN")])
        if not chs_type_hemi:
            continue

        _lfp_bipolar_names(chs_type_hemi, anodes, cathodes, bip_ch_names,
                           bip_coordinates, info,
                           only_neighbors=only_neighbors,
                           only_rings=only_rings)
        # _bipolar_mapping('LFP', chs_type_hemi, anodes, cathodes,
        #                  bip_ch_names, bip_coordinates, info)

    if info["subject_info"]["his_id"] in ['FloEmptyroom', 'NeuEmptyroom']:
        # add bipolar reference with bad channels
        # 1) 1-2ab (2c is bad)
        anodes.append('Amp_L_1')
        cathodes.append('Amp_L_2ab')
        bip_ch_names.append('Amp_L_1-2ab')
        bip_coordinates['Amp_L_1-2ab'] = np.ones(12) * np.nan

        # 2) 2ab-3abc (2c is bad)
        anodes.append('Amp_L_2ab')
        cathodes.append('Amp_L_3')
        bip_ch_names.append('Amp_L_2ab-3abc')
        bip_coordinates['Amp_L_2ab-3abc'] = np.ones(12) * np.nan

        # 3) 2ab-3ab (2c and 3c are bad)
        anodes.append('Amp_L_2ab')
        cathodes.append('Amp_L_3ab')
        bip_ch_names.append('Amp_L_2ab-3ab')
        bip_coordinates['Amp_L_2ab-3ab'] = np.ones(12) * np.nan

    bipolar_dict = dict(anode=anodes, cathode=cathodes, ch_name=bip_ch_names)
    return bipolar_dict, bip_coordinates


# def _bipolar_mapping(bids_ch_type, ch_names, anodes, cathodes, bip_ch_names,
#                      bip_coordinates, info, ):
#     """Map bipolar channels and add to mutable lists in-place."""
#     # if bids_ch_type == "EEG":
#     #     _eeg_bipolar_names(ch_names, anodes, cathodes, bip_ch_names,
#     #                        bip_coordinates, info)
#     # elif bids_ch_type == "ECOG":
#     #     _ecog_bipolar_names(ch_names,  anodes, cathodes, bip_ch_names,
#     #                         bip_coordinates, info)
#     # elif bids_ch_type == "LFP":
#     if bids_ch_type == "LFP":
#         _lfp_bipolar_names(ch_names,  anodes, cathodes, bip_ch_names,
#                            bip_coordinates, info)


def _add_coords(anode, cathode, bip_channel, bip_coordinates, info):
    """Add mni coords of bipolar channels to set bip_coordinates in-place."""
    ch_info = info["chs"]
    ch_names = info.ch_names
    mni_anode = ch_info[ch_names.index(anode)]["loc"]
    mni_cathode = ch_info[ch_names.index(cathode)]["loc"]
    bip_coordinates[bip_channel] = (mni_anode + mni_cathode) / 2
    return bip_coordinates


def _lfp_bipolar_names(ch_names, anodes, cathodes, bip_ch_names,
                       bip_coordinates, info, only_neighbors=True,
                       only_rings=True):
    """Implement all bipolar LFP combinations."""
    # Same for all ch_names
    ch_splits = ch_names[0].split("_")
    if len(ch_splits) == 5:
        bids_ch_type, hemi, _, area, manufact = ch_splits
        area_manufact = f"_{area}_{manufact}"
        assert bids_ch_type == 'LFP'
    elif len(ch_splits) == 3:
        bids_ch_type, hemi, _ = ch_splits
        assert bids_ch_type == 'Amp'
        area_manufact = ''
    type_hemi = f"{bids_ch_type}_{hemi}"
    nums = [ch.split("_")[2] for ch in ch_names]
    if only_rings:
        nums = [num for num in nums if len(num) == 1]

    permutations = list(combinations(nums, 2))
    for num1, num2 in permutations:
        int1, int2 = re.sub(r"\D", "", num1 + num2)
        if only_neighbors:
            distance = abs(np.diff([int(int1), int(int2)]))
            if distance > 1:
                continue
        letters = re.sub(r"\d", "", num1 + num2)
        if len(letters) == 1 and int1 in ["2", "3"] and int2 in ["2", "3"]:
            # we don't need e.g. LFP_R_2-3b. Only LFP_R_2-3 or LFP_R_2b-3c.
            continue
        # Check channel naming is correct
        anode = f"{type_hemi}_{num1}{area_manufact}"
        cathode = f"{type_hemi}_{num2}{area_manufact}"
        assert anode in ch_names
        assert cathode in ch_names
        bip_channel = f"{type_hemi}_{num1}-{num2}{area_manufact}"
        anodes.append(anode)
        cathodes.append(cathode)
        bip_ch_names.append(bip_channel)
        _add_coords(anode, cathode, bip_channel, bip_coordinates, info)


# def _eeg_bipolar_names(ch_names, anodes, cathodes, bip_ch_names,
#                        bip_coordinates, info):
#     assert len(ch_names) == 2, "Too many EEG electrodes found"
#     anode = ch_names.pop(0)  # avoid looping twice
#     cathode = ch_names.pop(0)
#     assert len(ch_names) == 0, "Too many EEG electrodes found"
#     bids_ch_type, area1, manufact = anode.split("_")
#     _, area2, _ = cathode.split("_")
#     bip_channel = f"{bids_ch_type}_{area1}-{area2}_{manufact}"
#     anodes.append(anode)
#     cathodes.append(cathode)
#     bip_ch_names.append(bip_channel)
#     _add_coords(anode, cathode, bip_channel, bip_coordinates, info)


# def _duplicate_ecog_cz(raw):
#     """Add ECoG channel closest to min Cz position to raw and name ECOG_CZ.

#     This will be a duplicated channel, however, it will have a uniform
#     position across subjects. The channel type will be 'SEEG' to not
#     accidentally use this duplicated channel later in the analysis."""
#     if 'ecog' not in raw.get_channel_types(unique=True):
#         return None
#     # choose ECoG closest to CZ y-coordinate
#     ecog_chs = raw.copy().pick_types(ecog=True).ch_names
#     y_axis = 1  # only use frontal-posteior axis
#     ch_info = raw.info["chs"]
#     mni_ecog_ys = np.array([ch_info[raw.ch_names.index(ch)]["loc"][y_axis]
#                             for ch in ecog_chs])
#     mni_cz_y = -0.009167
#     closest_idx = np.argmin(np.abs(mni_ecog_ys - mni_cz_y))
#     ecog_cz_ch = ecog_chs[closest_idx]

#     # give seeg channel type to not introduce a new type
#     ecog_cz = raw.copy().pick_channels([ecog_cz_ch])
#     new_name = "ECOG_CZ"  # ignore hemisphere to unify across subjects
#     ecog_cz.rename_channels({ecog_cz_ch: new_name})
#     ecog_cz.set_channel_types({new_name: "seeg"})  # change type to SEEG

#     # append new channel to raw
#     raw.add_channels([ecog_cz])


# def _ecog_bipolar_names(ch_names,  anodes, cathodes, bip_ch_names,
#                         bip_coordinates, info):
#     for ch_name in ch_names:
#         bids_ch_type, hemi, num, area, manufact = ch_name.split("_")
#         type_hemi = f"{bids_ch_type}_{hemi}"
#         area_manufact = f"{area}_{manufact}"

#         # Check channel naming is correct
#         anode = f"{type_hemi}_{num}_{area_manufact}"
#         assert anode == ch_name

#         # Get BIDS name of corresponding next neighbor cathode
#         num = int(num)
#         cathode = f"{type_hemi}_{str(num+1).zfill(2)}_{area_manufact}"
#         # Make new BIDS name indicating next neighbor: ch1-2
#         bip_channel = f"{type_hemi}_{num}-{num+1}_{area_manufact}"
#         # ... however, only if ch2 actually exists.
#         if cathode not in ch_names:
#             continue  # ch2 does not exist
#         anodes.append(anode)
#         cathodes.append(cathode)
#         bip_ch_names.append(bip_channel)
#         _add_coords(anode, cathode, bip_channel, bip_coordinates, info)


# def _lfp_bipolar_ring(ch_name, ch_names):
#    """This function might be needed later on."""
#     bids_ch_type, hemi, num, area, manufact = ch_name.split("_")
#     type_hemi = f"{bids_ch_type}_{hemi}"
#     area_manufact = f"{area}_{manufact}"

#     # Check channel naming is correct
#     anode = f"{type_hemi}_{num}_{area_manufact}"
#     assert anode == ch_name

#     # Get BIDS name of corresponding next neighbor cathode
#     # ignore directional leads containing abc
#     try:
#         # ring lead e.g. int(1)
#         num = int(num)
#     except ValueError:
#         # directional lead e.g. int(2a)
#         pass
#         # cathode, bip_channel = _bipolar_directional(anode)
#     else:
#         cathode = f"{type_hemi}_{num+1}_{area_manufact}"
#         # Some channel nums are "01", some are "1"
#         cathode2 = f"{type_hemi}_{str(num+1).zfill(2)}_{area_manufact}"
#         # Make new BIDS name indicating next neighbor: ch1-2
#         bip_channel = f"{type_hemi}_{num}-{num+1}_{area_manufact}"
#         # ... however, only if ch2 actually exists.
#     finally:
#         if cathode in ch_names:
#             cathode = cathode
#         elif cathode2 in ch_names:
#             cathode = cathode2
#         else:
#             return None  # ch2 does not exist
#         return anode, cathode, bip_channel


# def _standardize_ecog(raw: Raw, max_ecog: int = 6,
#                       areas: list[str] | None = None,
#                       manufacturers: list[str] | None = None) -> None:
#     """
#     Standardize ECOG channels. ECOG Model Ad-Tech DS12A-SP10X-000 12-contact,
#     1x6 dual sided long term monitoring strip. Simply ignore the last 6
#     channels because they face away from the brain.
#     """
#     chs_facing_upwards = []
#     try:
#         ecog_channels = raw.copy().pick_types(ecog=True, exclude=[]).ch_names
#     except ValueError:
#         ecog_channels = []
#     for ch_ecog in ecog_channels:
#         bids_ch_type, hemi, num, area, manufact = ch_ecog.split("_")
#         assert bids_ch_type == "ECOG"
#         if int(num) > max_ecog:
#             chs_facing_upwards.append(ch_ecog)
#         if hemi not in ["R", "L"]:
#             raise ValueError(f"Hemisphere must be 'R' or 'L'. Got {hemi}.")
#         if areas and area not in areas:
#             raise ValueError(f"Area must be in {areas}. Got {area}.")
#         if manufacturers and manufact not in manufacturers:
#             raise ValueError(f"Manufacturer must one of {manufacturers}. "
#                              f"Got {manufact}.")
#     raw.drop_channels(chs_facing_upwards)


def _standardize_dbs_leads(raw: Raw) -> dict[str, str] | None:
    """Standardize all DBS models to ring electrodes with different levels.

    Case 1: Non-directional DBS leads already standardized 1-...-4 (subject 3)
    Case 2: Directional extended DBS: average and rename 1-...-5 (subject 8)
    Case 3: Directional DBS 1-...-8: Average and rename 1-...-4.
            Or non-directional DBS in form 1-(234)-(567)-8. Rename 1-...-4.

    Save dbs model in raw.info["description"].

    Bad channel handling: Bad channels are used for bipolar reference and the
    new channel is marked as bad.

    and
    Directional bad channels are ignored and good channels kept for obtaining
    ring electrodes. The obtained ring electrode is marked "good".

    Return the renaming dictionary of the electrodes and save it for later use.
    """
    if raw.info["proj_name"] == "Hirschmann":
        return {}  # Leads already standardized
    elif raw.info["proj_name"] == "Hirschmann2":
        return {}  # Leads already standardized
    elif raw.info["proj_name"] == "Florin":
        _combine_dir_leads_florin(raw)
        return {}
    elif raw.info["proj_name"] == "Tan":
        return {}  # Leads already standardized
    elif raw.info["proj_name"] == "Litvak":
        raise NotImplementedError("Litvak already standardized.")
    else:
        assert raw.info["proj_name"] == "Neumann"
    try:
        subject = raw.info["subject_info"]["his_id"]
        assert subject.startswith("Neu")
        if subject == "NeuEmptyroom":
            _combine_dir_leads_florin(raw)
            return {}
    except TypeError as type_error:
        raise IndexError("Subject info not found in raw.info.") from type_error
    # Case 1. Hardcode subject 3 because DBS model cannot be inferred from
    # channel naming or raw.info.
    ref_name_old = _get_ref_from_info(raw.info)

    if subject == "NeuEL003":
        assert raw.info["subject_info"]["middle_name"] == '3'
        old_names = ['LFP_L_01_STN_MT', 'LFP_L_02_STN_MT',
                     'LFP_L_03_STN_MT', 'LFP_L_04_STN_MT',
                     'LFP_R_01_STN_MT', 'LFP_R_02_STN_MT',
                     'LFP_R_03_STN_MT', 'LFP_R_04_STN_MT']
        new_names = ['LFP_L_1_STN_MT', 'LFP_L_2_STN_MT',
                     'LFP_L_3_STN_MT', 'LFP_L_4_STN_MT',
                     'LFP_R_1_STN_MT', 'LFP_R_2_STN_MT',
                     'LFP_R_3_STN_MT', 'LFP_R_4_STN_MT']
        rename_delete = dict(zip(old_names, new_names))
        rename_dic = {ch: ch.replace("0", "") for ch  # remove leading 0
                      in raw.copy().pick(picks="dbs").ch_names}
        assert rename_dic == rename_delete
        rename_channels(raw.info, rename_dic)
        ref_name_new = rename_dic[ref_name_old]
        new_ref = raw.info["description"].replace(ref_name_old, ref_name_new)
        raw.info["description"] = new_ref
        return {subject: rename_dic}

    if len(pick_types(raw.info, dbs=True)) == 0:
        return {}

    if subject == "NeuEL008":
        # Combine directional leads:  R(1-2-3) -> R1, R(4-5-6) -> R2, ...
        old_to_new = {"0[1-3]": 1, "0[4-6]": 2, "0[7-9]": 3,
                      "1[0-2]": 4, "1[3-5]": 5, "16": 6}
    else:
        # Combine directional leads:  R(2-3-4) -> R2, R(5-6-7) -> R3
        # or rename directional leads: R-234 -> R2, R-567 -> R3
        old_to_new = {"01": 1, "0[2-4]": 2, "0[5-7]": 3, "08": 4,
                      "234": 2, "567": 3, "020304": 2, "050607": 3}

    ch_type = "LFP"
    ch_location = "STN"
    rename_dic = {}
    for hemi in ["R", "L"]:
        for ch_nums_old, ch_num_new in old_to_new.items():

            lead_info = _get_lead_info(raw, ch_nums_old, ch_num_new, hemi,
                                       ch_type, ch_location, subject)
            if lead_info is None:
                continue
            else:
                (directional_leads, old_ch_names, old_ch_indices,
                 ch_name_new, dir_ch_names_new) = lead_info

            if directional_leads:
                _combine_dir_leads(raw, ref_name_old, old_ch_names,
                                   ch_name_new, old_ch_indices,
                                   dir_ch_names_new)
                new_channel_names = dir_ch_names_new
            elif not directional_leads:
                _rename_ring_leads(raw, ref_name_old, old_ch_names,
                                   ch_name_new)
                new_channel_names = [ch_name_new]
            else:
                raise ValueError(f"Unknown channel names: {old_ch_names}.")
            update_dic = dict(zip(old_ch_names, new_channel_names))
            rename_dic.update(update_dic)
    return {subject: rename_dic}


def _rename_ring_leads(raw, reference, old_ch_names, ch_name_new):
    # Rename to ring electrode
    old_ch_name = old_ch_names[0]
    rename = {old_ch_name: ch_name_new}
    rename_channels(raw.info, rename)
    # looping over channels, therefore reference might not be present:
    if reference in old_ch_names:
        new_reference = rename[reference]
        new_ref = raw.info["description"].replace(reference, new_reference)
        raw.info["description"] = new_ref


def _combine_dir_leads(raw, ref_name_old, old_ch_names, ch_name_new,
                       old_ch_indices, dir_ch_names_new):
    """Rename dir leads, then add them to ring electrode. Ignore bad leads."""
    rename = dict(zip(old_ch_names, dir_ch_names_new))
    raw.rename_channels(rename)
    # Rename reference
    if ref_name_old in old_ch_names:
        new_reference = rename[ref_name_old]
        new_ref = raw.info["description"].replace(ref_name_old, new_reference)
        raw.info["description"] = new_ref
    all_bad = all(ch in raw.info["bads"] for ch in dir_ch_names_new)
    drop_bad = bool(not all_bad)
    # Average directional leads to ring electrode
    average = {ch_name_new: old_ch_indices}
    # Important: don't sum directional leads, average them!
    comb_dic = dict(inst=raw, groups=average, method='mean',
                    drop_bad=drop_bad, verbose=False)
    # try:
    combined = combine_channels(**comb_dic)
    # get mean mni coordinates of directional leads
    mni_coords = np.array([raw.info["chs"][idx]["loc"] for idx
                           in old_ch_indices]).mean(0)
    # assign coords to combination
    combined.info["chs"][0]["loc"] = mni_coords
    raw.add_channels([combined], force_update_info=True)
    if all_bad:
        raw.info['bads'].append(ch_name_new)


def _combine_dir_leads_florin(raw):
    """Rename dir leads, then add them to ring electrode. Ignore bad leads."""
    for hemi in ["L", "R"]:
        for dir_num in [2, 3]:
            # pick directional channel indices
            # reg_ex = f"LFP_{hemi}_{dir_num}._STN_*"
            reg_ex = f"..._{hemi}_{dir_num}.*"
            dir_ch_indices = pick_channels_regexp(raw.ch_names, reg_ex)
            dir_ch_nmes = [raw.ch_names[idx] for idx in dir_ch_indices]
            all_bad = all(ch in raw.info["bads"] for ch in dir_ch_nmes)
            dir_ch_nme = dir_ch_nmes[0]
            ch_name_new = dir_ch_nme.replace('a', ''
                                             ).replace('b', ''
                                                       ).replace('c', '')

            # Average directional leads to ring electrode
            average = {ch_name_new: dir_ch_indices}
            drop_bad = bool(not all_bad)
            # Important: don't average directional leads, sum them!
            comb_dic = dict(inst=raw, groups=average,
                            method='mean',
                            drop_bad=drop_bad, verbose=False)
            combined = combine_channels(**comb_dic)
            # get mean mni coordinates of directional leads
            mni_coords = np.array([raw.info["chs"][idx]["loc"] for idx
                                   in dir_ch_indices]).mean(0)
            # assign coords to combination
            combined.info["chs"][0]["loc"] = mni_coords
            raw.add_channels([combined], force_update_info=True)
            if all_bad:
                raw.info['bads'].append(ch_name_new)


def _get_lead_info(raw, ch_nums_old, ch_num_new, hemi, ch_type, ch_location,
                   subject):
    # Select channel indices through regular expression for different
    # manufacturers (*).
    old_ch_nms = (f"{ch_type}_{hemi}_{ch_nums_old}_"
                  f"{ch_location}_*")
    old_ch_indices = pick_channels_regexp(raw.ch_names, old_ch_nms)

    # Get electrode manufacturer by checking abbreviation for
    # exemplary electrode.
    old_ch_names = [raw.ch_names[idx] for idx in old_ch_indices]
    if not old_ch_names:
        return None
    manufacturer = old_ch_names[0].split("_")[-1]

    # New averaged channel name
    ch_name_new = (f"{ch_type}_{hemi}_{ch_num_new}_"
                   f"{ch_location}_{manufacturer}")

    if subject == "NeuEL008":
        directional_leads = len(old_ch_indices) > 1
        non_directional_leads = len(old_ch_indices) == 1
        assert raw.info["subject_info"]["middle_name"] == '4'
    else:
        directional_leads = ch_num_new in [2, 3]
        non_directional_leads = ch_num_new in [1, 4]
        assert raw.info["subject_info"]["middle_name"] in ['1', '2']
        if directional_leads:
            if len(old_ch_names) != 3:
                if ch_nums_old in ["234", "567", "020304", "050607"]:
                    rename = {old_ch_names[0]: ch_name_new}
                    raw.rename_channels(rename)
                    return None
                else:
                    msg = ("Need 3 directional leads to create ring "
                           f"lead. Found only {len(old_ch_indices)}")
                    raise ValueError(msg)
    assert directional_leads is not non_directional_leads
    if directional_leads:
        # Rename directional leads - always 3 directions!
        dir_ch_names_new = [(f"{ch_type}_{hemi}_{ch_num_new}{dir}_"
                            f"{ch_location}_{manufacturer}") for dir in "abc"]
    else:
        dir_ch_names_new = None

    lead_info = (directional_leads, old_ch_names, old_ch_indices,
                 ch_name_new, dir_ch_names_new)
    return lead_info


class NewChannelNames:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NewChannelNames, cls).__new__(cls)
            cls._instance.data = []
        return cls._instance

    def add_failed_fit(self, info):
        self.data.append(info)


if __name__ == "__main__":
    preprocess()
