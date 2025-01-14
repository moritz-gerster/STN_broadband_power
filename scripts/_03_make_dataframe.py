"""Create Pandas Dataframe with PSDs, and 1/f slopes."""
import re
import warnings
from os.path import basename, join
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne import set_log_level
from mne.time_frequency import read_spectrum
from mne_bids import find_matching_paths
from nilearn import image
from specparam import SpectralModel
from specparam.core.errors import DataError
from specparam.utils.io import load_model
from tqdm import tqdm

import scripts.config as cfg
from scripts.bidsify_sourcedata import loadmat
from scripts.utils import (FailedFits, _distant_contacts, _get_ref_from_info,
                           _ignore_warnings, _load_stn_masks)
from scripts.utils_plot import _save_fig


def psd_fooof_to_dataframe(root="derivatives/spectra",
                           recordings=cfg.RECORDINGS, subjects=None,
                           descriptions='cleaned', sessions=None,
                           verbose="error", load_fits=True,
                           normalizations=['Neumann'],
                           subtract_amplifier=False,
                           smooth_amp=False, save_plots=False) -> pd.DataFrame:
    """Load BIDS data, calculate PSDs, fit fooof, and save to dataframe."""
    set_log_level(verbose)

    # important that "SpectrumWelchEmpty" is excluded
    spec_paths = find_matching_paths(root, subjects=subjects,
                                     sessions=sessions, extensions=".fif",
                                     recordings=recordings,
                                     descriptions=descriptions)
    # Get fitted spectra as dics
    dics = []
    for spec_path in tqdm(spec_paths, desc="Fit spectra: "):
        _path_to_psd_dict(dics, spec_path, load_fits=load_fits,
                          normalizations=normalizations,
                          subtract_amplifier=subtract_amplifier,
                          smooth_amp=smooth_amp, save_plots=save_plots)

    # convert dics to dataframe
    df = pd.DataFrame(dics)
    assert df.shape[0] == len(dics)

    # save dataframe
    save_path = join(cfg.DF_PATH, cfg.DF_FOOOF_RAW)
    Path(cfg.DF_PATH).mkdir(parents=True, exist_ok=True)
    df.to_pickle(save_path)
    failed_fits = FailedFits()
    print(f'Failed fits: {failed_fits.data}')
    print(f"Fooof dataframe saved to {save_path}.")
    print(f"{basename(__file__).strip('.py')} done.")
    return df


def _path_to_psd_dict(dics, spec_path, load_fits=True,
                      normalizations=['Neumann', 'Plateau'],
                      subtract_amplifier=False,
                      smooth_amp=False, save_plots=False):
    spectrum = _read_spectrum(spec_path)
    file_dic = _get_file_dic(spec_path)
    ch_dics = _add_channels_to_dic(spectrum, file_dic,
                                   normalizations=normalizations,
                                   subtract_amplifier=subtract_amplifier,
                                   smooth_amp=smooth_amp,
                                   save_plots=save_plots)
    dic = _add_fooof_to_dic(ch_dics, spec_path, load_fits=load_fits)
    dics.extend(dic)


def _read_spectrum(spec_path):
    """Load PSDs."""
    spectrum = read_spectrum(spec_path.fpath)
    if spectrum.method == "multitaper":
        # multitaper is based on epochs and must be averaged
        # Important: Cannot average before saving spectrum object. Needs to
        # be done after loading it.
        spectrum = spectrum.average()
    return spectrum


def _cleaned_psd(spectrum, file_dic, save_plots=False, smooth_amp=False):
    """Subtract amp/empty spectrum from spectrum to clean."""
    spectrum_clean = spectrum.copy()

    # Load amplifier spectrum
    session = None
    recordings = file_dic['project']
    if file_dic['project'] == 'Tan':
        amp = spectrum.info['device_info']['model'].split(' ')[1]
        session = f'TMSi{amp}20240212'
    processing = file_dic['bids_processing']
    processing += 'Smooth' if smooth_amp else ''
    amp_path = find_matching_paths(root="derivatives/spectra",
                                   processings=processing,
                                   recordings=recordings,
                                   sessions=session,
                                   tasks='noise')
    assert len(amp_path) == 1
    amp_path = amp_path[0]
    spectrum_amp = _read_spectrum(amp_path)

    # subtract spectrum channels with matched amplifier channels
    ch_amp_dic = _match_ch_to_amp(spectrum, spectrum_clean, spectrum_amp)
    for ch_idx, ch_amp_idx in ch_amp_dic.items():
        ch_amp_data = spectrum_amp._data[ch_amp_idx]
        spectrum_clean._data[ch_idx] -= ch_amp_data

    # clip negative PSD values. Replace negative values with tiny number
    neg_idcs = spectrum_clean._data < 0
    spectrum_clean._data[neg_idcs] = 1e-18

    if save_plots:
        _plot_amp_subtraction(spectrum, spectrum_clean, spectrum_amp,
                              file_dic, ch_amp_dic, smooth_amp=smooth_amp)
    return spectrum_clean


def _match_ch_to_amp(spectrum, spectrum_clean, spectrum_amp):
    ch_names = spectrum_clean.ch_names
    matched_names = _info_from_ch_names(ch_names)
    _, _, _, _, dir_chs, _ = matched_names
    bad_dir_chs = [ch for ch in dir_chs if ch in spectrum.info['bads']]
    bad_comb_rings = [(ch.split('_')[1], ch.split('_')[2].strip('abc'))
                      for ch in bad_dir_chs]

    # get bad combined channels. Pure bad channels are fine and ignored.
    bad_comb_dir = []
    for hemi, num in bad_comb_rings:
        match = re.compile(f"LFP_{hemi}_.*{num}[^a-c].*").match
        match_list = list(filter(match, ch_names))
        bad_comb_dir += match_list
    combined_nums = set(re.search(r"\d", ch)[0] for ch in dir_chs)

    bip_chs = [ch for ch in ch_names if '-' in ch]
    msg = 'Code assumes no directional bipolar chs'
    assert not len(list(filter(re.compile(".*[a-c].*").match, bip_chs))), msg

    ch_amp_dic = {}
    for ch_idx, ch_nme in enumerate(ch_names):
        ch_splits = ch_nme.split('_')
        hemi = ch_splits[1]
        num_full = ch_splits[2]

        if len(dir_chs):
            ch_amp = f'Amp_{hemi}_{num_full}'
        else:
            if '-' in num_full:
                ch_amp = f'Amp_{hemi}_1-4'
            else:
                ch_amp = f'Amp_{hemi}_1'

        special_lead = (ch_nme in bad_comb_dir
                        or ch_amp not in spectrum_amp.ch_names
                        or combined_nums != {'2', '3'})
        if special_lead:
            bipolar = '-' in ch_nme
            monopolar = not bipolar and 'WIEST' not in ch_nme
            num = num_full.strip('abc-')
            if bipolar:
                ch1, ch2 = num.split('-')
            elif monopolar:
                ch = num
            else:
                # Wiest channel is distant bipolar
                ch_amp = f'Amp_{hemi}_1-4'

            if monopolar:
                if ch not in combined_nums:
                    ch_amp = 'Amp_L_1'
                else:
                    n_ch = 3  # always 3 dir chs

                    # for this monopolar ch, count how many dir chs are bad
                    n_bads_dir = bad_comb_rings.count((hemi, ch))

                    # subtract bad dir chs
                    num_chs = n_ch - n_bads_dir

                    if num_chs == 2:
                        ch_amp = 'Amp_L_2ab'
                    elif num_chs == 1:
                        ch_amp = 'Amp_L_2a'
                    elif num_chs == 0:
                        # all 3 chs are bad. Treat normal since excluded anyway
                        pass
                    elif num_chs == 3:
                        # all 3 chs are good but doesn't exist in standard lead
                        ch_amp = 'Amp_L_2a'
            elif bipolar:
                # count maximum number of involved chs (ring or dir)
                n_chs1 = 1 if ch1 not in combined_nums else 3
                n_chs2 = 1 if ch2 not in combined_nums else 3
                n_chs = n_chs1 + n_chs2

                # for this bipolar ch, count how many dir chs are bad
                n_bads_dir = bad_comb_rings.count((hemi, ch1))
                n_bads_dir += bad_comb_rings.count((hemi, ch2))

                # subtract bad dir chs
                num_chs = n_chs - n_bads_dir

                if num_chs == 5:
                    ch_amp = 'Amp_L_2ab-3abc'
                elif num_chs == 4:
                    if not n_bads_dir:
                        ch_amp = 'Amp_L_1-2'
                    elif n_bads_dir == 2:
                        ch_amp = 'Amp_L_2ab-3ab'
                elif num_chs == 3:
                    ch_amp = 'Amp_L_1-2ab'
                elif num_chs == 2:
                    ch_amp = 'Amp_L_1-4'
                elif num_chs == 1:
                    # all 3 chs are bad. Treat normal since excluded anyway
                    pass
                elif num_chs == 6:
                    # all dir chs good but num larger than 4 (huge dbs lead)
                    ch_amp = 'Amp_L_2-3'

        ch_amp_idx = spectrum_amp.ch_names.index(ch_amp)
        ch_amp_dic[ch_idx] = ch_amp_idx
    return ch_amp_dic


def _plot_amp_subtraction(spectrum, spectrum_clean, spectrum_amp,
                          file_dic, ch_amp_dic, smooth_amp=False):
    # Convert to ASD in nV/sqrt(Hz)
    spectrum._data = (spectrum._data**.5) * 1e9
    spectrum_clean._data = (spectrum_clean._data**.5) * 1e9
    spectrum_amp._data = (spectrum_amp._data**.5) * 1e9

    matched_names = _info_from_ch_names(spectrum.ch_names)
    _, _, _, _, dir_chs, _ = matched_names

    ch_nmes_amp = [spectrum_amp.ch_names[idx] for idx in ch_amp_dic.values()]

    mono = [ch for ch in ch_nmes_amp if '-' not in ch]
    bip = [ch for ch in ch_nmes_amp if '-' in ch]
    mono_pure = [ch for ch in mono if ch[-1] in ['1', '4']]
    mono_comb = [ch for ch in mono if ch[-1] in ['2', '3']]
    bip_pure = [ch for ch in bip if ch.split("_")[-1] in ['1-4']]
    bip_comb = [ch for ch in bip if ch.split("_")[-1] in ['2-3']]
    bip_mixed = [ch for ch in bip if ch.split("_")[-1] in ['1-2', '3-4']]

    bip_distant = _distant_contacts(spectrum.ch_names)

    all_kinds = [mono_pure, mono_comb, bip_pure, bip_comb, bip_mixed]
    all_kinds_descr = ['monopolar non-directional',
                       'monopolar fused-directional',
                       'bipolar non-directional',
                       'bipolar fused-directional',
                       'bipolar mixed']
    all_kinds_descr = [descr for kind, descr in zip(all_kinds, all_kinds_descr)
                       if len(kind)]
    all_kinds = [kind for kind in all_kinds if len(kind)]
    ncols = len(all_kinds)

    fig, axes = plt.subplots(1, ncols, figsize=(ncols*5, 9), sharey='row')
    if ncols == 1:
        axes = np.array([axes])
    for ax_idx, kind in enumerate(all_kinds):
        plot_colors = cfg.CHANNEL_PLOT_COLORS
        idx = 0
        for ch_idx, ch_amp_idx in ch_amp_dic.items():
            ch_nme_amp = spectrum_amp.ch_names[ch_amp_idx]
            if ch_nme_amp not in kind:
                continue
            ch_nme = spectrum.ch_names[ch_idx]
            ch_nme_short = '_'.join(ch_nme.split('_')[:3])
            try:
                c = plot_colors[ch_nme_short]
            except KeyError:
                if ch_nme_short == 'LFP_L_1-4':
                    c = 'orange'
                elif ch_nme_short == 'LFP_R_1-4':
                    c = 'turquoise'
                else:
                    if 'R' in ch_nme:
                        c = 'turquoise'
                    else:
                        c = 'orange'
            if ch_nme in spectrum.info['bads']:
                continue
            if not len(dir_chs):
                if ch_nme in bip_distant:
                    continue
            axes[ax_idx].loglog(spectrum.freqs, spectrum._data[ch_idx],
                                c=c, label=ch_nme)
            axes[ax_idx]. loglog(spectrum.freqs, spectrum_clean._data[ch_idx],
                                 '--', c=c)
            if idx == 0:
                axes[ax_idx].loglog(spectrum.freqs,
                                    spectrum_amp._data[ch_amp_idx],
                                    label=ch_nme_amp, c='k')
            idx += 1
        axes[ax_idx].set_title(all_kinds_descr[ax_idx])
    axes[0].set_ylabel(r'ASD [$nV/\sqrt{Hz}$]')

    for ax in axes.flatten():
        ax.set_xlabel('Frequency [Hz]')
        ax.legend()
        ax.yaxis.grid(True, which='both')
        ax.xaxis.grid(True, which='major')
    title = f'{file_dic["subject"]} {file_dic["cond"]}'
    plt.suptitle(title, y=.98)
    plt.tight_layout()
    amp_smoothed = 'smooth' if smooth_amp else 'unsmoothed'
    save_dir = join(cfg.FIG_ASD, 'subtracted', amp_smoothed)
    _save_fig(fig, title.replace(' ', '_'), save_dir)


def _get_file_dic(bids_path):
    """Extract df columns from bids_path."""
    # Add subject info
    patient_dic = _get_patient_dic(bids_path)
    updrs_dic = _get_updrs_dic(bids_path)
    basename = bids_path.basename.strip("_ieeg.fif")
    title = " ".join(basename.split("_")).replace("ses-", "")
    title = title.replace("task-", "").replace("acq-", "")
    # Add BIDS info
    file_dic = {"subject": bids_path.subject,
                "bids_session": bids_path.session,
                "bids_task": bids_path.task,
                "bids_acquisition": bids_path.acquisition,
                "bids_processing": bids_path.processing,
                "bids_run": bids_path.run,
                "bids_basename": basename,
                "title": title,
                "project": bids_path.recording,
                **patient_dic, **updrs_dic}
    return file_dic


def _get_patient_dic(bids_path):
    patient_dic = {}
    if bids_path.subject.endswith("Emptyroom"):
        pass
    elif bids_path.recording == 'Florin':
        fpath = join(cfg.SOURCEDATA_FLORIN, "Details.xlsx")
        patient_table = pd.read_excel(fpath)
        # filter for subject
        sub_old = cfg.FLORIN_SUBJECT_MAP_REV[bids_path.subject]
        mask_sub = patient_table['Subject '] == sub_old + '_PD'
        patient_table = patient_table[mask_sub]
        # drop columns
        drop = ['Subject ', 'MMSE pre', 'BDI pre ', 'UPDRS_peri',
                'UPDRS_peri.1', 'current side ', 'of impairment']
        patient_table.drop(columns=drop, inplace=True)
        # rename columns
        rename_cols = {"Age ": 'patient_age', 'Sex ': 'patient_sex',
                       'Handness ': 'patient_handedness',
                       'Dose (mg)': 'patient_LEDD'}
        patient_table.rename(columns=rename_cols, inplace=True)
        patient_dic = patient_table.iloc[0].to_dict()
        day = cfg.FLORIN_DAYS_RECORDING[sub_old]
        patient_dic['patient_days_after_implantation'] = day
    elif bids_path.recording == 'Hirschmann':
        fname = join(bids_path.root, f"meta_infos_{bids_path.recording}",
                     "participants.tsv")
        patient_table = pd.read_csv(fname, sep="\t")
        # rename subjects
        sub_dic = cfg.SUB_MAPPING_HIR
        sub_dic = {f'sub-{k}': v for k, v in sub_dic.items()}
        patient_table['participant_id'] = patient_table[
            'participant_id'].map(sub_dic)
        # filter for subject
        mask_sub = patient_table.participant_id == bids_path.subject
        patient_table = patient_table[mask_sub]
        # drop columns
        drop = ['hand', 'disease']
        patient_table.drop(columns=drop, inplace=True)
        patient_table["diseaseDuration"] = patient_table[
            "diseaseDuration"].values[0].strip('[]')
        # rename columns
        rename_cols = {"participant_id": "patient_participant_id",
                       "age": "patient_age", "sex": "patient_sex",
                       "diseaseDuration": "patient_disease_duration"}
        patient_table.rename(columns=rename_cols, inplace=True)
        patient_dic = patient_table.iloc[0].to_dict()
        patient_dic['patient_handedness'] = np.nan
        patient_dic['patient_symptom_dominant_side'] = np.nan
        patient_dic['patient_recording_site'] = np.nan
        patient_dic['patient_days_after_implantation'] = 1
    elif bids_path.recording == 'Neumann':
        fname = join(cfg.SOURCEDATA_NEU, "participants.tsv")
        patient_table = pd.read_csv(fname, sep="\t", parse_dates=True)
        # filter for subject
        sub = f"sub-{bids_path.subject.replace('Neu', '')}"
        mask_sub = patient_table.participant_id == sub
        patient_table = patient_table[mask_sub]
        # drop columns
        drop = ['DBS_target', 'DBS_hemisphere', 'DBS_contacts',
                'ECOG_target', 'ECOG_manufacturer', 'ECOG_model',
                'ECOG_location', 'ECOG_material', 'ECOG_contacts',
                'ECOG_description',
                'ECOG_hemisphere',  # ECoG hemisphere important later
                # will be added separately:
                'DBS_manufacturer', 'DBS_model', 'DBS_description',
                'DBS_directional'
                ]
        patient_table.drop(columns=drop, inplace=True)
        # rename columns
        patient_cols = ["participant_id", "sex", "handedness", "age",
                        "date_of_implantation", "disease_duration",
                        "PD_subtype", "symptom_dominant_side", "LEDD"]
        rename_cols = {col: f"patient_{col}" for col in patient_cols}
        patient_table.rename(columns=rename_cols, inplace=True)
        date_col = "patient_date_of_implantation"
        patient_table[date_col] = pd.to_datetime(patient_table[date_col])
        patient_dic = patient_table.iloc[0].to_dict()
        patient_dic[date_col] = patient_dic[date_col].to_datetime64()
    elif bids_path.recording == 'Litvak':
        # Attention: The original excel file has been modified by hand!
        fpath = join(cfg.SOURCEDATA_LIT, 'meta_infos',
                     '4Vadim_Original_PD_STN_cohort.xlsx')
        patient_table = pd.read_excel(fpath)
        # filter for subject
        sub = int(bids_path.subject.replace('LitML', ''))
        patient_table = patient_table[(patient_table.Patients == sub)]

        # rename columns
        rename_cols = {'Sex': 'sex',
                       'Age (years)': 'age',
                       'Disease Duration (years)': 'disease_duration',
                       'Predominant Symptoms': 'PD_subtype',
                       'Pre-operative Medication': 'medication_preop'}
        rename_cols = {key: f"patient_{val}" for key, val
                       in rename_cols.items()}
        patient_table.rename(columns=rename_cols, inplace=True)
        # this column contains errors
        patient_table.drop(columns='Motor UPDRS (ON/OFF)', inplace=True)
        patient_dic = patient_table.iloc[0].to_dict()
        patient_dic['patient_days_after_implantation'] = 2.5
    elif bids_path.recording == 'Tan':
        # Attention: The original excel file has been modified by hand!
        fpath = join(cfg.SOURCEDATA_TAN, 'Elife_Data4Moritz_combined.xlsx')
        patient_table = pd.read_excel(fpath)
        # filter for subject
        sub = cfg.TAN_SUBJECT_MAP_REV[bids_path.subject]
        patient_table = patient_table[(patient_table['OXF ID'] == sub)]

        # rename columns
        rename_cols = {'Gender (m/f)': 'sex',
                       'Age (yr)': 'age',
                       'Disease Duration (yr)': 'disease_duration',
                       'Predominant \nSymptoms': 'PD_subtype',
                       'Time of Recording \n(days post-OP)':
                           'days_after_implantation',
                       'Site': 'recording_site'}
        rename_cols = {key: f"patient_{val}" for key, val
                       in rename_cols.items()}
        # rename_cols['DBS number'] = 'DBS_model'  # without patient prefix
        patient_table.rename(columns=rename_cols, inplace=True)
        drop_cols = [col for col in patient_table.columns
                     if col not in rename_cols.values()]
        patient_table.drop(columns=drop_cols, inplace=True)
        patient_dic = patient_table.iloc[0].to_dict()
        n_days_str = patient_dic["patient_days_after_implantation"]
        if isinstance(n_days_str, str):
            n_days_list = [int(x) for x in n_days_str if x.isdigit()]
            n_days_float = np.array(n_days_list).mean()
        elif isinstance(n_days_str, (int, float)):
            n_days_float = n_days_str
        patient_dic["patient_days_after_implantation"] = n_days_float
    return patient_dic


def _add_channels_to_dic(spectrum, file_dic, normalizations=None,
                         subtract_amplifier=False,
                         smooth_amp=False, save_plots=False):
    """Extract channel values and return as dic for df."""
    file_dic["psd_method"] = spectrum.method
    # Important to pick all chs, otherwise bad chs are silently skipped
    psds, freqs = spectrum.get_data(picks=spectrum.ch_names, return_freqs=True)
    psd_kinds = [("standard", psds)]
    if normalizations is not None:
        for normalization in normalizations:
            psds_norm = _normalize_psd(psds, freqs, method=normalization)
            assert psds.shape == psds_norm.shape
            if normalization == 'Neumann':
                normalization_str = ''
            else:
                normalization_str = normalization
            psd_kinds.append((f"normalized{normalization_str}", psds_norm))
    if subtract_amplifier:
        if file_dic["project"] != 'Litvak':  # no amplifier recording London
            if file_dic['bids_task'] != 'noise':  # only apply to brain data
                spectrum_clean = _cleaned_psd(spectrum, file_dic,
                                              save_plots=save_plots,
                                              smooth_amp=smooth_amp)
                picks = spectrum_clean.ch_names
                psds_clean = spectrum_clean.get_data(picks=picks)
                assert psds.shape == psds_clean.shape
                psd_kinds.append(("cleaned", psds_clean))

    # This caused some bugs in the past because of distinct counting of bad
    # channels in mne.Spectrum and mne.Raw
    assert len(psds) == len(spectrum.ch_names)

    mne_types = {"eeg": "eeg", "ecog": "ecog", "lfp": "dbs", "meg": "mag",
                 "seeg": "seeg", 'amp': 'eeg'}
    info = spectrum.info
    mni_locs = _get_mni_coords(info)
    sweetspot_chs = _get_sweetspot_chs(mni_locs)
    record_ref = _get_ref_from_info(info)
    dbs_dic = _get_dbs_info(info)

    ch_names = info.ch_names
    bad_chs = info["bads"]
    wiest_pick = _indicate_wiest_picks(info)
    amplifier = info["device_info"]['model']

    # This function is needed for function _get_ref_info. Call outside of loop
    # to increase speed.
    matched_names = _info_from_ch_names(ch_names)
    _, _, _, comb_rings, dir_chs, distant_bip_chs = matched_names
    ch_dics = []
    for kind, psds_kind in psd_kinds:
        ch_dics_kind = []
        if not kind.startswith('normalized'):
            # Amplitude Spectral Density: convert V**2/Hz to nV/sqrt(Hz)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ignore_warnings()
                asds_kind = (psds_kind**.5) * 1e9
            # Power Spectral Density:
            # psds_kind = psds_kind**.5  # convert V**2/Hz to V/sqrt(Hz)
            # psds_kind *= 1e6  # convert V/sqrt(Hz) to muV/sqrt(Hz)
            # psds_kind = psds_kind**2  # convert muV/sqrt(Hz) to muV**2/Hz

            psds_kind = psds_kind * (1e6)**2  # convert V**2/Hz to muV**2/Hz

        for idx, psd_ch in enumerate(psds_kind):
            ch_name = ch_names[idx]
            ch_split = ch_name.split("_")
            ch_type = ch_split[0]
            assert ch_type in ["LFP", 'Amp']
            hemisphere = ch_split[1][-1]  # L or R or Z
            plot_column = 0 if hemisphere == "L" else 1
            ch_num = ch_split[2].lstrip("0")
            if ch_type == 'Amp':
                brain_area = np.nan
            else:
                brain_area = ch_split[3]
            ch_nme = f"{ch_type}_{hemisphere}_{ch_num}"
            ch = f"{ch_type}_{ch_num}"
            ch_dir = ch_name in dir_chs
            bad = ch_name in bad_chs
            comb_ring = ch_name in comb_rings
            bip_distant = ch_name in distant_bip_chs or ch == 'LFP_WIEST'
            mni_coords = mni_locs[ch_name]
            ref, ref_contacts = _get_ref_info(ch_name, matched_names)
            sweetspot_in_ref = sweetspot_chs.intersection(set(ref_contacts))
            # only select distant sweetspot channels. Adjacent can lead to
            # duplicates (e.g. sweet: LFP_3 -> LFP_2-3 & LFP_3-4). For adjacent
            # use arithmetic mean.
            ch_sweetspot = (len(sweetspot_in_ref) > 0) & bip_distant
            ch_sweetspot = ch_sweetspot or (ch_name in sweetspot_chs)
            inside_stn = _bipolar_in_stn(ch_name, mni_locs, ref, ref_contacts)
            record_ref_short = "_".join(record_ref.split("_")[:3])
            file_dic_copy = file_dic.copy()
            add_to_title = f" Ref-{ref} (orig. {record_ref_short})"
            file_dic_copy["title"] += add_to_title

            ch_dic = {**file_dic_copy,
                      **dbs_dic,
                      "psd_freqs": freqs,
                      "psd": psd_ch,
                      "psd_log": np.log10(psd_ch),
                      "asd": asds_kind[idx],
                      "psd_kind": kind,
                      "ch_nme": ch_nme,  # short version
                      "ch": ch,  # very short version
                      "ch_type_bids": ch_type.lower(),
                      "ch_type_mne": mne_types[ch_type.lower()],
                      "ch_directional": ch_dir,
                      "ch_combined_ring": comb_ring,
                      "ch_hemisphere": hemisphere,
                      "ch_area": brain_area,
                      "ch_bad": bad,
                      "ch_wiestpick": wiest_pick[ch_name],
                      "ch_reference": ref,
                      "ch_bip_distant": bip_distant,
                      "ch_inside_stn": inside_stn,
                      "ch_sweetspot": ch_sweetspot,
                      'ch_choice': False,
                      'ch_mean_inside_stn': False,
                      "plot_column": plot_column,
                      "ch_recording_reference": record_ref_short,
                      "ch_ref_contacts": ref_contacts,
                      "mni_coords": mni_coords,
                      "mni_x": mni_coords[0],
                      "mni_y": mni_coords[1],
                      "mni_z": mni_coords[2],
                      "amplifier": amplifier
                      }
            ch_dics_kind.append(ch_dic)
        _add_inside_stn_mean(ch_dics_kind)
        ch_dics.extend(ch_dics_kind)
    return ch_dics


def _bipolar_in_stn(ch_name, mni_locs, ref, ref_contacts):
    """Check whether bipolar channels is inside the STN."""
    if np.isnan(mni_locs[ch_name]).all():
        return np.nan
    if ref in ["monopolar", "LAR"]:
        inside_stn = _mni_inside_stn(mni_locs[ch_name])
        return inside_stn
    assert ref == "bipolar"
    ref_mnis = [mni_locs[ref_ch] for ref_ch in ref_contacts]
    refs_in_STN = [_mni_inside_stn(ref_mni) for ref_mni in ref_mnis]
    if any(refs_in_STN):
        inside_stn = True  # make inside if one channel inside STN
    elif _mni_inside_stn(mni_locs[ch_name]):
        inside_stn = True  # also make inside if middle point is inside STN
    else:
        inside_stn = False
    return inside_stn


def _add_inside_stn_mean(ch_dics):
    df = pd.DataFrame(ch_dics)

    # Filter the DataFrame for rows where inside_STN is True
    # Exclude distant bip channels since they might lie outside the STN while
    # their center may lie inside
    mask = (df.ch_inside_stn & ~df.ch_bad
            & (~df.ch_bip_distant | (df.project == 'Tan'))
            & ~df.ch_directional
            & (df.ch_reference == 'bipolar'))
    for hemi in ['L', 'R']:
        df_inside = df[mask & (df.ch_hemisphere == hemi)]

        if not len(df_inside):
            continue

        # Keep all columns which are identical for all channels
        identical_cols = _get_equal_columns(df_inside)

        # Average mni coordinates
        coords = ['mni_x', 'mni_y', 'mni_z']
        df_mean = df_inside[coords].mean(numeric_only=True)
        df_mean['mni_coords'] = (df_mean.mni_x, df_mean.mni_y, df_mean.mni_z)

        # Add all other columns which are identical for all channels
        for col in identical_cols:
            df_mean[col] = df_inside[col].values[0]

        # Manually set channels
        df_mean['ch'] = 'LFP_INSIDE'
        df_mean['ch_nme'] = f'LFP_{hemi}_INSIDE'
        assert df_inside.ch_area.nunique() == 1

        # Manually set channel information
        df_mean['ch_ref_contacts'] = np.nan
        df_mean['ch_combined_ring'] = False
        df_mean['ch_inside_stn'] = True
        df_mean['ch_mean_inside_stn'] = True
        df_mean['ch_choice'] = False
        df_mean['ch_directional'] = False
        df_mean['ch_sweetspot'] = False

        # Average numpy arrays
        assert check_all_arrays_equal(df.psd_freqs)
        df_mean['psd_freqs'] = df.psd_freqs.values[0]
        df_mean['psd'] = np.mean(np.stack(df_inside.psd), axis=0)
        df_mean['psd_log'] = np.mean(np.stack(df_inside.psd_log), axis=0)
        df_mean['asd'] = np.mean(np.stack(df_inside.asd), axis=0)

        # Add to ch_dics
        assert len(df_mean) == len(df.columns)
        ch_dic = df_mean.to_dict()
        ch_dics.append(ch_dic)


def _get_equal_columns(df):
    identical_cols = []
    for col in df.columns:
        try:
            unique = df[col].nunique()
        except TypeError:
            pass
        else:
            if unique <= 1:
                identical_cols.append(col)
    return identical_cols


def check_all_arrays_equal(series):
    first_array = series.iloc[0]
    for array in series:
        if not np.array_equal(first_array, array):
            return False
    return True


def _mni_inside_stn(mni_coords):
    """Check whether MNI coordinates are inside the STN."""
    if not np.isfinite(mni_coords).all():
        return False
    stn_mask_LR, inverse_matrix = _load_stn_masks(threshold=0.1)
    # Transform MNI coords to image coords using inverse_matrix
    idx = image.coord_transform(*mni_coords, inverse_matrix)
    mask_location = (round(idx[0]), round(idx[1]), round(idx[2]))
    # Check whether mni coords are within binary mask stn_mask_LR
    mask_value = stn_mask_LR.get_fdata()[mask_location]
    mni_inside_stn = mask_value == 1
    return mni_inside_stn


def _indicate_wiest_picks(info):
    if info['proj_name'] != 'Tan':
        return {ch_name: False for ch_name in info.ch_names}
    ch_names = info.ch_names
    ch_types = info.get_channel_types()
    wiest_pick = {}
    for idx, ch in enumerate(ch_names):
        if ch_types[idx] == 'dbs':
            wiest_pick[ch] = True
        elif ch_types[idx] == 'ecog':
            wiest_pick[ch] = False
        else:
            raise ValueError(f"Unknown channel type {ch_types[idx]}")
    return wiest_pick


def _add_fooof_to_dic(ch_dics, bids_path, load_fits=True):
    """Fit fooof and append results to dics and save plots.

    Save all plots in the same folder for fast visual inspection.
    """
    fooof_dics = []
    freqs = ch_dics[0]["psd_freqs"]
    for ch_dic in ch_dics:
        reference = ch_dic["ch_reference"]
        if reference != 'bipolar':
            continue
        project = ch_dic["project"]
        kind = ch_dic["psd_kind"]
        if kind.startswith('normalized'):
            ch_dic["fm_params"] = False
            fooof_dics.append(ch_dic)
            continue
        for fooof_setting, fooof_params in cfg.FOOOF_DICT[project].items():
            fit_range, params = fooof_params[kind]
            # Saving fit files super ugly....
            # Suggestion Thomas: Save a separate json file with all
            # fooof parameters and call it for example 'LFP-BIP'. Instead
            # of using huge file names, use 'LFP-BIP' as abbreviation.
            proc1 = f"params-{fooof_setting}_ref-{reference}_"
            proc2 = f"fitrange-{fit_range[0]}+{fit_range[1]}Hz_"
            proc3 = f"ch-{ch_dic['ch_nme']}_"
            proc4 = f"reso-{round(np.unique(np.diff(freqs))[0]*10)}_"
            proc5 = f"method-{ch_dic['psd_method']}_"
            proc6 = f"proc-{ch_dic['bids_processing']}"
            proc = proc1 + proc2 + proc3 + proc4 + proc5 + proc6
            proc = proc.replace("-", "=").replace("_", "&")

            psd = ch_dic["psd"]
            root = join(cfg.FOOOF_SAVE_JSON, kind)
            save_fit = dict(root=root, extension=".json",
                            suffix="fooof", check=False,
                            processing=proc)
            fit_path = bids_path.update(**save_fit)
            fname = fit_path.basename
            fpath = fit_path.directory
            save_params = dict(freqs=freqs, psd=psd, fname=fname,
                               fit_range=fit_range,
                               params=params, fit_path=fpath,
                               load_fits=load_fits)
            fm = _get_fooof(**save_params)
            fooof_dic = _extract_fooof_params(fm, fname)
            fit_dic = {**ch_dic, **fooof_dic, "fm_params": fooof_setting}
            fooof_dics.append(fit_dic)
    return fooof_dics


def _get_updrs_dic(bids_path, multiple_scores="average_sessions"):
    """Extract UPDRS scores from excel file.

    If one condition has multiple scores, use the latest
    (multiple_scores="latest") or average (multiple_scores="average").
    """
    updrs_dic = {}
    cond_path = _cond_from_path(bids_path)

    if bids_path.recording == 'Litvak':
        # load excel file
        fpath = join(cfg.SOURCEDATA_LIT, 'meta_infos',
                     'Details_Moritz.xlsx')
        updrs_table = pd.read_excel(fpath)
        # filter for subject
        sub = int(bids_path.subject.replace('LitML', ''))
        updrs_table = updrs_table[(updrs_table.Subject == sub)]
        # filter for condition: the first 5 columns are OFF, the last 5 are ON
        if cond_path == 'off':
            mask_cond = [True, True, True, True, True, True, False,
                         False, False, False, False, False]
        elif cond_path == 'on':
            mask_cond = [True, False, False, False, False, False, False,
                         True, True, True, True, True]
        updrs_table = updrs_table.iloc[:, mask_cond]

        # rename columns
        rename_cols = {f'Total Pre {cond_path.capitalize()}': 'III',
                       'Hemibody bradykinesia R': 'bradykinesia_right',
                       'Hemibody rigidity R': 'rigidity_right',
                       'Hemibody bradykinesia L': 'bradykinesia_left',
                       'Hemibody rigidity L ': 'rigidity_left',
                       # ON columns are renamed by pandas to avoid duplicates
                       'Hemibody bradykinesia R.1': 'bradykinesia_right',
                       'Hemibody rigidity R.1': 'rigidity_right',
                       'Hemibody bradykinesia L.1': 'bradykinesia_left',
                       'Hemibody rigidity L .1': 'rigidity_left'}
        rename_cols = {key: f"UPDRS_pre_{val}" for key, val
                       in rename_cols.items()}
        updrs_table.rename(columns=rename_cols, inplace=True)
        updrs_dic = updrs_table.iloc[0].to_dict()
        updrs_dic["cond"] = cond_path
        return updrs_dic

    elif bids_path.recording == 'Hirschmann':
        if bids_path.subject == "HirEmptyroom":
            updrs_dic["cond"] = 'Elekta Neuromag'
            return updrs_dic
        # load tsv file
        fname = join(cfg.SOURCEDATA_HIR,
                     f'participants_updrs_{cond_path}.tsv')
        updrs_table = pd.read_csv(fname, sep="\t")
        # filter for subject
        sub = cfg.SUB_MAPPING_HIR_REV[bids_path.subject]
        sub_mask = updrs_table['participant_id'] == f'sub-{sub}'
        updrs_table = updrs_table[sub_mask]

        if not len(updrs_table.dropna(axis=1)):
            # UPDRS missing for this subject
            updrs_dic["cond"] = cond_path
            return updrs_dic

        # rename columns
        rename_cols = {'SUM': 'post_III',
                       "AR right": "post_bradyrigid_right",
                       "AR left": "post_bradyrigid_left",
                       "AR sum": "post_bradyrigid_total",
                       "trem right": "post_tremor_right",
                       "trem left": "post_tremor_left",
                       "trem sum": "post_tremor_total",
                       "axial": "post_axial_total"}
        rename_cols = {key: f"UPDRS_{val}" for key, val in rename_cols.items()}
        updrs_table.rename(columns=rename_cols, inplace=True)

        # add bradykinesia and rigidity columns from UPDRS subscores
        brady_cols_right = ["3_4_a", "3_5_a", "3_6_a", "3_7_a", "3_8_a"]
        brady_cols_left = ["3_4_b", "3_5_b", "3_6_b", "3_7_b", "3_8_b"]
        rigid_cols_right = ["3_3_b", "3_3_d"]
        rigid_cols_left = ["3_3_c", "3_3_e"]
        subscores = [("UPDRS_post_bradykinesia_right", brady_cols_right),
                     ("UPDRS_post_bradykinesia_left", brady_cols_left),
                     ("UPDRS_post_rigidity_right", rigid_cols_right),
                     ("UPDRS_post_rigidity_left", rigid_cols_left)]
        for subscore, cols in subscores:
            if updrs_table[cols].isna().sum().sum() > 0:
                updrs_table[subscore] = np.nan
            else:
                updrs_table[subscore] = updrs_table[cols].sum(1).values[0]
        msg = "Subscores wrong"
        bradyrigid = (updrs_table["UPDRS_post_bradykinesia_right"]
                      + updrs_table["UPDRS_post_rigidity_right"])
        if bradyrigid.notna().all():
            assert all(bradyrigid
                       == updrs_table["UPDRS_post_bradyrigid_right"]), msg
        bradyrigid = (updrs_table["UPDRS_post_bradykinesia_left"]
                      + updrs_table["UPDRS_post_rigidity_left"])
        if bradyrigid.notna().all():
            assert all(bradyrigid
                       == updrs_table["UPDRS_post_bradyrigid_left"]), msg

        keep_cols = ["UPDRS_post_bradykinesia_left",
                     "UPDRS_post_bradykinesia_right",
                     "UPDRS_post_rigidity_left",
                     "UPDRS_post_rigidity_right"]
        keep_cols += list(rename_cols.values())
        updrs_table = updrs_table[keep_cols]

        updrs_dic = updrs_table.iloc[0].to_dict()
        updrs_dic["cond"] = cond_path
        return updrs_dic

    elif bids_path.recording == 'Tan':
        if bids_path.subject == "TanEmptyroom":
            amp = bids_path.session.replace("TMSi", "").replace("20240212", "")
            updrs_dic["cond"] = f'TMSi {amp} OX'
            return updrs_dic
        # Attention: The original excel file has been modified by hand!
        fpath = join(cfg.SOURCEDATA_TAN, 'Elife_Data4Moritz_combined.xlsx')
        patient_table = pd.read_excel(fpath)
        # filter for subject
        sub = cfg.TAN_SUBJECT_MAP_REV[bids_path.subject]
        patient_table = patient_table[(patient_table['OXF ID'] == sub)]

        # rename columns
        rename_cols = {f'UPDRS-III {cond_path.upper()} meds (pre-OP)': 'III',
                       f'UPDRS III - {cond_path.upper()} - Left':
                           'bradyrigid_left',
                       f'UPDRS III - {cond_path.upper()} - Right':
                           'bradyrigid_right'}
        rename_cols = {key: f"UPDRS_pre_{val}" for key, val
                       in rename_cols.items()}
        patient_table.rename(columns=rename_cols, inplace=True)
        drop_cols = [col for col in patient_table.columns
                     if col not in rename_cols.values()]
        patient_table.drop(columns=drop_cols, inplace=True)
        updrs_dic = patient_table.iloc[0].to_dict()
        updrs_dic["cond"] = cond_path
        return updrs_dic

    elif bids_path.recording == "Florin":
        if bids_path.subject == "FloEmptyroom":
            updrs_dic["cond"] = 'Elekta Neuromag'
            return updrs_dic
        # load excel file
        fpath = join(cfg.SOURCEDATA_FLORIN, 'updrs_data_pre_peri.xlsx')
        sub_map = cfg.FLORIN_SUBJECT_MAP_REV
        keep_cols = ['SUM', 'rigidity_right', 'rigidity_left',
                     'rigidity_total', 'bradykinesia_right',
                     'bradykinesia_left', 'bradykinesia_total',
                     'tremor_right', 'tremor_left', 'tremor_total']
        updrs_dic = {}
        for timepoint in ['pre', 'peri']:
            sheet_name = f"{timepoint}_{cond_path}"
            updrs_table = pd.read_excel(fpath, sheet_name=sheet_name)

            # filter for subject
            sub = sub_map[bids_path.subject] + '_PD'
            updrs_table = updrs_table[(updrs_table.subject_id == sub)]

            # filter for relevant columns
            updrs_table = updrs_table[keep_cols]

            # rename columns
            updrs_table.rename(columns={'SUM': 'III'}, inplace=True)
            suffix = '_pre' if timepoint == 'pre' else '_post'
            rename_cols = {col: f"UPDRS{suffix}_{col}"
                           for col in updrs_table.columns}
            updrs_table.rename(columns=rename_cols, inplace=True)

            if len(updrs_table):
                updrs_dic.update(updrs_table.iloc[0].to_dict())
        updrs_dic["cond"] = cond_path
        return updrs_dic

    elif bids_path.recording == 'Neumann':
        if bids_path.subject == "NeuEmptyroom":
            updrs_dic["cond"] = 'TMSi SAGA BER'
            return updrs_dic
        fpath = join(cfg.SOURCEDATA_NEU, 'phenotype', 'UPDRS_Berlin.xlsx')

        updrs_dic = {}
        for timepoint in ['recording', 'preop']:
            if bids_path.subject.startswith('NeuEL'):
                sheet_name = f"{timepoint}_detailed_EcogLfp"
                sheet_name2 = f"{timepoint}_detailed_Ecog"
                if timepoint == 'preop':
                    if bids_path.subject in ["NeuEL004", "NeuEL026"]:
                        continue
            elif bids_path.subject.startswith('NeuL'):
                if timepoint == 'preop':
                    # Lfp does not have preop data
                    continue
                else:
                    sheet_name = "recording_detailed_Lfp"
            else:
                raise ValueError(f"Unknown subject {bids_path.subject}.")
            parse_dates = ['Date'] if timepoint == 'recording' else False
            try:
                updrs_table = pd.read_excel(fpath, sheet_name=sheet_name,
                                            parse_dates=parse_dates)
            except ValueError:
                updrs_table = pd.read_excel(fpath, sheet_name=sheet_name2,
                                            parse_dates=parse_dates)

            sub = f"sub-{bids_path.subject.replace('Neu', '')}"
            # Filter for subject and condition.
            if timepoint == 'recording':
                if multiple_scores == 'same_session':
                    # Only use UPDRS of same session as recording.
                    sessions = [bids_path.session]
                elif multiple_scores == 'average_sessions':
                    # Average all sessions
                    sub_mask = (updrs_table.Subject == sub)
                    sessions = updrs_table[sub_mask].Session.unique()
                time_mask = ((updrs_table.Session.isin(sessions))
                             & (updrs_table["Recorded?"].isin(["Yes", "yes"])))

            elif timepoint == 'preop':
                time_mask = True
                recording_date = np.nan
            mask = ((updrs_table.Subject == sub)
                    & (updrs_table.Medication == cond_path.upper())
                    & time_mask)
            updrs_table = updrs_table[mask]

            if not len(updrs_table):
                continue

            if timepoint == 'recording':
                # Extract date of recording
                ses = bids_path.session
                mask = updrs_table.Session == ses
                recording_date = updrs_table[mask].Date
                if recording_date.empty:
                    ses2 = ses.replace('Off', '')
                    mask = updrs_table.Session == ses2
                    recording_date = updrs_table[mask].Date
                    if recording_date.empty:
                        ses3 = ses.replace('Dys', '')
                        mask = updrs_table.Session == ses3
                        recording_date = updrs_table[mask].Date
            # Indent?
            try:
                recording_date = recording_date.values[0]
            except (IndexError, AttributeError):
                recording_date = np.nan

            updrs_total = ('UPDRS_III' if timepoint == 'recording'
                           else 'UPDRS-III total')
            msg = f"UPDRS_III is missing for {sub} {cond_path} {timepoint}."
            assert np.all(updrs_table[updrs_total].notna()), msg
            assert len(updrs_table.Medication.unique()) == 1, "Multiple conds."
            cond = updrs_table.Medication.values[0].lower()
            assert cond == cond_path, f"{cond} != {cond_path}"

            # filter for relevant columns
            keep_cols = ["UPDRS_III",
                         "subscore_tremor_right",
                         "subscore_tremor_left",
                         "subscore_tremor_total",
                         "subscore_rigidity_right",
                         "subscore_rigidity_left",
                         "subscore_rigidity_total",
                         "subscore_bradykinesia_right",
                         "subscore_bradykinesia_left",
                         "subscore_bradykinesia_total",
                         'Tremor right', 'Tremor left', 'Tremor total',
                         'Rigidity right', 'Rigidity left', 'Rigidity total',
                         'Bradykinesia right', 'Bradykinesia left',
                         'Bradykinesia total',
                         'UPDRS-III total']  # don't use 'MDS UPDRS-III total'
            drop_cols = [col for col in updrs_table.columns
                         if col not in keep_cols]
            updrs_table.drop(columns=drop_cols, inplace=True)

            # rename columns
            suffix = '_pre' if timepoint == 'preop' else '_post'
            if timepoint == 'recording':
                rename_cols = {col: col.replace("subscore_", "")
                               for col in keep_cols}
                rename_cols['UPDRS_III'] = 'III'
            elif timepoint == 'preop':
                rename_cols = {col: "_".join(col.split(" ")).lower()
                               for col in keep_cols}
                rename_cols['UPDRS-III total'] = 'III'
            rename_cols = {key: f"UPDRS{suffix}_{val}"
                           for key, val in rename_cols.items()}
            updrs_table.rename(columns=rename_cols, inplace=True)

            if timepoint == 'recording':
                if multiple_scores == "average_sessions":  # and len(updrs) > 1:
                    updrs_dic_timepoint = updrs_table.mean().to_dict()
                elif multiple_scores == "latest":
                    # The "Date" column is not always filled in, so we use the last
                    # index as the sessions appear chronologically sorted. If only
                    # one entry, iloc[-1] is the same as iloc[0].
                    updrs_dic_timepoint = updrs_table.iloc[-1].to_dict()
                elif multiple_scores == 'single':
                    assert len(updrs_table) == 1
                col = f'patient_recording_date_{cond.upper()}'
                updrs_dic_timepoint[col] = recording_date
                updrs_dic.update(updrs_dic_timepoint)
            elif timepoint == 'preop':
                assert len(updrs_table) == 1
                updrs_dic_timepoint = updrs_table.iloc[0].to_dict()
                updrs_dic.update(updrs_dic_timepoint)
        updrs_dic["cond"] = cond_path
    return updrs_dic


def _cond_from_path(bids_path):
    """Extract condition from bids path. Important for Neumann data where
    condition is not obvious from session name."""
    session = bids_path.session
    acquisition = bids_path.acquisition
    if "MedOn0" in session:
        cond_path = "on"
    elif "MedOff0" in session:
        cond_path = "off"
    elif 'MedOffDys' in session:
        cond_path = 'off'
    elif 'MedOnDys' in session:
        cond_path = 'on'
    elif 'MedOffOnDys' in session:
        if acquisition == 'StimOffDopaPre':
            cond_path = 'off'
        else:
            assert acquisition[-2:].isnumeric(), 'Dopa Dose not a number'
            cond_path = 'on'
    elif session in ["TMSiSAGA20220916", "ElektaNeuromag20240208",
                     "TMSiSAGA20240122", 'TMSiSAGA20240212',
                     'TMSiPorti20240212']:
        cond_path = np.nan
    else:
        raise ValueError(f"Unknown session {session}.")
    return cond_path


def _info_from_ch_names(ch_names):
    # replace this function with _pick_ch_names_by_type below. Gives much
    # more flexibility.

    # Extract different channel types by name to obtain reference contacts
    RSTN_chs = set(filter(re.compile(".*(_RSTN)$").match, ch_names))
    LSTN_chs = set(filter(re.compile(".*(_LSTN)$").match, ch_names))
    ECOG_chs = set(filter(re.compile(".*(_ECOG)$").match, ch_names))

    # include bipolar directional channels:
    dir_chs = set(filter(re.compile(r".*_\d[a-c]").match, ch_names))
    ints = "|".join(set(re.search(r"\d", ch)[0] for ch in dir_chs))
    # Find ints in LFP chs to obtain combined LFPs. Attention: Bipolar
    # combined channels "LFP_L_2-3" are included here!
    combined_rings1 = set(filter(re.compile(f"^LFP_._({ints})_").match,
                                 ch_names))
    # Also include combined bipolar channels:
    combined_rings2 = set(filter(re.compile(f"^LFP_._({ints})-._").match,
                                 ch_names))
    combined_rings3 = set(filter(re.compile(f"^LFP_._.-({ints})_").match,
                                 ch_names))
    combined_rings = combined_rings1 | combined_rings2 | combined_rings3

    combined_rings_amp1 = set(filter(re.compile(f"^Amp_._({ints})$").match,
                                     ch_names))
    combined_rings_amp2 = set(filter(re.compile(f"^Amp_._({ints})-._").match,
                                     ch_names))
    combined_rings_amp3 = set(filter(re.compile(f"^Amp_._.-({ints})$").match,
                                     ch_names))
    combined_rings_amp = (combined_rings_amp1 | combined_rings_amp2
                          | combined_rings_amp3)

    combined_rings = combined_rings | combined_rings_amp

    distant_bip_chs = _distant_contacts(ch_names)
    return (RSTN_chs, LSTN_chs, ECOG_chs, combined_rings, dir_chs,
            distant_bip_chs)


def _get_mni_coords(raw_info):
    """Get MNI coordinates for each channel."""
    mni_locs = {}
    for ch_dic in raw_info["chs"]:
        ch_name = ch_dic["ch_name"]
        x, y, z = ch_dic["loc"][:3]
        mni_coords = (x*1000, y*1000, z*1000)  # convert to mm
        mni_locs[ch_name] = mni_coords
    if raw_info['proj_name'] == 'Litvak':
        subj_new = raw_info['subject_info']['his_id']
        subj_old = cfg.LITVAK_SUBJECT_MAP_INV[subj_new]
        mni_locs_mono = _litvak_mono_coords(subj_old)
        mni_locs.update(mni_locs_mono)
    return mni_locs


def _get_sweetspot_chs(mni_locs):
    mono_chs = [ch for ch in mni_locs if "-" not in ch]
    mono_chs = [ch for ch in mono_chs if "a" not in ch and "b" not in ch
                and "c" not in ch]
    df = {ch: mni_locs[ch] for ch in mono_chs}
    df = pd.DataFrame(df).T
    df.rename(columns={0: "mni_x", 1: "mni_y", 2: "mni_z"}, inplace=True)
    df['ch_hemisphere'] = [ch.split("_")[1] for ch in mono_chs]

    if df.dropna().empty:
        return set()

    # set 0 0 0 coordinates to NaN
    all_zero = (df.mni_x == 0) & (df.mni_y == 0) & (df.mni_z == 0)
    df.loc[all_zero, "mni_x"] = np.nan
    df.loc[all_zero, "mni_y"] = np.nan
    df.loc[all_zero, "mni_z"] = np.nan

    if df.dropna().empty:
        return set()

    # Dembek, et al. "Probabilistic Sweet Spots Predict Motor Outcome for Deep
    # Brain Stimulation in Parkinson Disease." ANN NEUROL 2019;86:527–538.
    stn_motor_R = np.array([12.5, -12.72, -5.38])
    stn_motor_L = np.array([-12.68, -13.53, -5.38])

    cond_L = df.ch_hemisphere == "L"
    cond_R = df.ch_hemisphere == "R"

    df_L = df[cond_L]
    df_R = df[cond_R]

    coords_L = np.array([df_L.mni_x, df_L.mni_y, df_L.mni_z])
    coords_R = np.array([df_R.mni_x, df_R.mni_y, df_R.mni_z])

    dist_L = np.linalg.norm(coords_L.T - stn_motor_L, axis=1)
    dist_R = np.linalg.norm(coords_R.T - stn_motor_R, axis=1)

    # create mask selecting minimum distance
    sweet_L = dist_L == dist_L.min()
    sweet_R = dist_R == dist_R.min()

    df.loc[cond_L, "sweetspot"] = sweet_L
    df.loc[cond_R, "sweetspot"] = sweet_R

    sweetspot_chs = df[df.sweetspot].index
    return set(sweetspot_chs)


def _litvak_mono_coords(subj_old):
    lead_path = f"{cfg.SOURCEDATA}/BIDS_Litvak_MEG_LFP/meta_infos/lead_reconstruction"
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

    ch_names = ["LFP_R_1_STN_MT", "LFP_R_2_STN_MT",
                "LFP_R_3_STN_MT", "LFP_R_4_STN_MT",
                "LFP_L_1_STN_MT", "LFP_L_2_STN_MT",
                "LFP_L_3_STN_MT", "LFP_L_4_STN_MT"]
    coords = [LFP_R_1, LFP_R_2, LFP_R_3, LFP_R_4,
              LFP_L_1, LFP_L_2, LFP_L_3, LFP_L_4]
    mono_coords = dict(zip(ch_names, coords))
    return mono_coords


def _normalize_psd(psds, freqs, method="Neumann"):
    """Normalize PSDs according to Neumann et al. 2017 Clin. Neurophys."""
    if method == "Neumann":
        # mask = ((freqs >= 5) & (freqs <= 45)) | ((freqs >= 55) & (freqs <= 95))
        mask = (freqs >= 5) & (freqs <= 95)  # simplify
        factor = 100  # percentage
    elif method == "Litvak":
        mask = (freqs >= 4) & (freqs <= 48)
        factor = 1
    elif method == "Tan":
        mask = ((freqs >= 1) & (freqs <= 47)) | ((freqs >= 53) & (freqs <= 90))
        factor = 100  # percentage
    elif method == "Ince":
        mask = (freqs >= 120) & (freqs <= 160)
        factor = 100  # percentage
    elif method == "Plateau":
        mask = (freqs >= 100) & (freqs <= 200)
        mini = psds[:, mask].min(1)
        avoid_zero = 1e-16  # add 10 nV/sqrt(Hz) to avoid 0
        psds_norm = psds - mini[:, None] + avoid_zero
        return psds_norm
    else:
        raise ValueError(f"Unknown normalization method {method}.")
    psd_sum = psds[:, mask].sum(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ignore_warnings()
        psds_norm = psds / psd_sum[:, None] * factor  # percentage
    return psds_norm


def _get_dbs_info(info):
    lead_num = info["subject_info"]["middle_name"]
    lead_dic = cfg.DBS_LEADS[int(lead_num)]
    dbs_dic = {"DBS_num": lead_num,
               "DBS_model": lead_dic['DBS_model'],
               "DBS_description": lead_dic['DBS_description'],
               "DBS_directional": lead_dic['DBS_directional']}
    return dbs_dic


def _get_ref_info(ch_name, matched_names):
    ch_split = ch_name.split("_")
    rSTN_chs, lSTN_chs, ecog_chs, combined_rings, _, _ = matched_names
    # Bipolar channels
    if "-" in ch_name:
        ref = "bipolar"
        # Extract ch1 and ch2 from bipolar ch. For example
        # "LFP_R_1_STN_BS" and "LFP_R_2_STN_BS" from "LFP_R_1-2_STN_BS".
        ch1 = "_".join([split.split("-")[0] for split in ch_split])
        ch2 = "_".join([split.split("-")[-1] for split in ch_split])
        ref_contacts = [ch1, ch2]
    # Left STN LAR
    elif ch_name.endswith("_LSTN"):
        ref = "LAR"
        ref_contacts = list(lSTN_chs - {ch_name} - combined_rings)
    # Right STN LAR
    elif ch_name.endswith("_RSTN"):
        ref = "LAR"
        ref_contacts = list(rSTN_chs - {ch_name} - combined_rings)
    # ECOG LAR
    elif ch_name.endswith("_ECOG"):
        ref = "LAR"  # remove ref and only keep ref_plot and rename to ref!
        ref_contacts = list(ecog_chs - {ch_name})
    elif ch_name == "ECOG_CZ":
        ref = "LAR"
        ref_contacts = list(ecog_chs - {ch_name})
    elif ch_name.startswith("MEG"):
        ref = "None"
        ref_contacts = "n/a"  # starts with "Ref-"
    elif 'WIEST' in ch_name:
        ref = 'bipolar'
        ref_contacts = [np.nan]  # starts with "Ref-"
    # Monopolar reference
    else:
        ref = "monopolar"
        # ref_plot = ref
        ref_contacts = [np.nan]  # starts with "Ref-"
    return ref, ref_contacts


def _get_fooof(freqs, psd, fname, fit_range, params=None, fit_path=None,
               load_fits=True):
    """Fit fooof, return results, and save as json files and pdf plots.

    Using FOOOFGroup not possible due to different fit parameters for
    each channel type."""
    # load if file exists
    # Loading FOOOF is pretty risky. It assumes that the file name is unique.
    # Furthermore, changing parameters in spectral computation can lead to
    # inconsistencies.
    if Path(join(fit_path, fname)).exists() and load_fits:
        fm = load_model(fname, file_path=fit_path, regenerate=True)
    else:
        fm = _fit_and_save_fooof(freqs, psd, fname, fit_path, params,
                                 fit_range)
    return fm


def _extract_fooof_params(fm, fname):
    if fm is None or not fm.has_model or fm._ap_fit is None:
        failed_fits = FailedFits()
        failed_fits.add_failed_fit(fname)
        fooof_dic = {"fm_exponent": np.nan,
                     "fm_offset": np.nan,
                     "fm_offset_log": np.nan,
                     "fm_knee": np.nan,
                     "fm_knee_fit": np.nan,
                     "fm_r_squared": np.nan,
                     "fm_center_freqs": np.nan,
                     "fm_powers": np.nan,
                     "fm_standard_devs": np.nan,
                     "fm_freqs": np.nan,
                     "fm_psd_peak_fit": np.nan,
                     "fm_psd_ap_fit": np.nan,
                     "fm_fit_range": np.nan,
                     "fm_freq_res": np.nan,
                     "fm_has_model": False,
                     "fm_info": 'fit unsuccessful'}
        return fooof_dic
    assert fm.has_model
    exponent = fm.get_params("aperiodic_params", "exponent")
    offset_log = fm.get_params("aperiodic_params", "offset")
    offset = 10**offset_log
    if fm.aperiodic_mode == "knee":
        knee = fm.get_params("aperiodic_params", "knee")
        # according to Gao (2020) and Bush (2022) the knee frequency
        # is the knee to the power of the inverse exponent. For the
        # formular, see Gao (2020) page 14 in "Materials and methods -
        # Inferring timescale from autocorrelation and PSD".
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            knee_fit = knee**(1/exponent)
    elif fm.aperiodic_mode == "fixed":
        knee = np.nan  # fit without knee
        knee_fit = 0  # Hz
    elif fm.aperiodic_mode == "lorentzian":
        # fm.get_params("aperiodic_params", "knee") does not work
        _, knee, _ = fm.aperiodic_params_
        knee_fit = 10**knee  # lorentzian uses log units for knee
    r_squared = fm.r_squared_
    # Power in linear uV**2/Hz units (instead log10(uV**2/Hz)) can only be
    # obtained by subtracting the linear aperiodic power from the linear total
    # power due to the nonlinear log-scale. The aperiodic power however
    # can be calculated without the periodic power because it
    # starts from 0 uV**2/Hz.
    peak_fit_lin = 10**fm.modeled_spectrum_ - 10**fm._ap_fit
    peak_fit2 = 10**(fm._peak_fit + fm._ap_fit) - 10**fm._ap_fit
    peak_fit3 = (10**fm._peak_fit - 1) * 10**fm._ap_fit
    assert np.allclose(peak_fit2, peak_fit_lin), 'Not identical'
    assert np.allclose(peak_fit3, peak_fit_lin), 'Not identical'

    peak_power_linear, gauss_power_linear = _get_per_lin_pwr(fm)
    fm_info = (fm.get_results(), fm.get_settings(), fm.get_meta_data())
    fit_range = (round(fm.freq_range[0]), round(fm.freq_range[1]))

    fooof_dic = {"fm_exponent": exponent,
                 "fm_offset": offset,
                 "fm_offset_log": offset_log,
                 "fm_knee": knee,
                 "fm_knee_fit": knee_fit,
                 "fm_r_squared": r_squared,
                 "fm_error": fm.error_,
                 "fm_center_freqs": fm.get_params("gaussian_params", "CF"),
                 "fm_powers": peak_power_linear,
                 "fm_powers_log": fm.get_params("peak_params", "PW"),
                 "fm_gauss_powers": gauss_power_linear,
                 "fm_gauss_powers_log": fm.get_params("gaussian_params", "PW"),
                 "fm_standard_devs": fm.get_params("gaussian_params", "BW"),
                 "fm_freqs": fm.freqs,
                 "fm_psd_peak_fit": peak_fit_lin,
                 "fm_psd_peak_fit_log": fm._peak_fit,
                 "fm_psd_ap_fit": 10**fm._ap_fit,
                 "fm_psd_ap_fit_log": fm._ap_fit,
                 "fm_fooofed_spectrum": 10**fm.modeled_spectrum_,
                 "fm_fooofed_spectrum_log": fm.modeled_spectrum_,
                 "fm_fit_range": fit_range,
                 "fm_freq_res": fm.freq_res,
                 "fm_has_model": fm.has_model,
                 "fm_info": fm_info}
    return fooof_dic


def _fit_and_save_fooof(freqs, psd, fname, fit_path, params, fit_range):
    fm = SpectralModel(**params)
    failed_fits = FailedFits()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ignore_warnings()
        assert freqs[0] == 0, f"First freq must be 0 Hz, is {freqs[0]} Hz."

        mask_fit = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
        psd_fit = psd[mask_fit]
        freqs_fit = freqs[mask_fit]
        psd_log = np.log10(psd_fit)
        if np.any(np.isnan(psd_log)) or np.any(np.isinf(psd_log)):
            # remove negative PSD values
            psd_fit[psd_fit <= 0] = 1e-12
        try:
            fm.fit(freqs_fit, psd_fit, fit_range)
        except DataError:
            failed_fits.add_failed_fit(fname)
            return None
    Path(fit_path).mkdir(parents=True, exist_ok=True)
    fm.save(fname, file_path=fit_path, save_results=True, save_settings=True,
            save_data=True)
    return fm


def _get_per_lin_pwr(fm):
    """Power in linear uV**2/Hz units (instead log10(uV**2/Hz)) can only be
    obtained by subtracting the linear aperiodic power from the linear total
    power due to the nonlinear log-scale. The aperiodic power however
    can be calculated without the periodic power because it
    starts from 0 uV**2/Hz."""
    if fm.n_peaks_ == 0:
        return np.nan, np.nan

    # fm.get_params("gaussian_params", "PW") is different from
    # "peak_params" because peak params are almost twice as high if
    # two neighboring peaks strongly overlap (peak params is sum of
    # gaussian peaks at a given frequency). Peak params are required when
    # investigated in isolation whereas gaussian params are more meaningful
    # in combination with peak width, for example in area-under-curve power.
    peak_log = fm.get_params("peak_params", "PW")
    gauss_log = fm.get_params("gaussian_params", "PW")
    center_freqs = fm.get_params("gaussian_params", "CF")

    # I have to loop over each peak frequency instead of applying a
    # mask which would be more elegant and faster. However, the
    # frequencies are floats, e.g. 3.56 and 4.2 Hz. Rounding them
    # yields two times 4: [4, 4]. In a mask, 4 Hz is only counted once,
    # therefore I get the wrong number of powers in the end.
    peak_lin_pwrs = []
    gauss_lin_pwrs = []

    # make iterable:
    center_freqs_lst = [center_freqs] if fm.n_peaks_ == 1 else center_freqs
    peak_log_lst = [peak_log] if fm.n_peaks_ == 1 else peak_log
    gauss_log_lst = [gauss_log] if fm.n_peaks_ == 1 else gauss_log
    for idx, cf in enumerate(center_freqs_lst):
        peak_log_pwr = peak_log_lst[idx]
        gauss_log_pwr = gauss_log_lst[idx]
        cf_idx = np.abs(fm.freqs - cf).argmin()
        msg = "Mask is wrong."
        assert np.isclose(fm._peak_fit[cf_idx], peak_log_pwr), msg

        # get aperiodic power in log units at center frequencies
        aperiodic_log_pwr = fm._ap_fit[cf_idx]
        assert isinstance(aperiodic_log_pwr, float)

        # convert log10(muV^2/Hz) to muV^2/Hz
        aperiodic_pwr = 10**aperiodic_log_pwr

        # get full power in linear units
        total_pwr = 10**(peak_log_pwr + aperiodic_log_pwr)
        total_gauss_pwr = 10**(gauss_log_pwr + aperiodic_log_pwr)

        # get periodic power in linear units
        peak_lin_pwr = total_pwr - aperiodic_pwr
        gauss_lin_pwr = total_gauss_pwr - aperiodic_pwr
        msg = "Linear transformation of log power to linear power is wrong."
        assert np.allclose(aperiodic_pwr + peak_lin_pwr,
                           10**fm.modeled_spectrum_[cf_idx]), msg

        peak_lin_pwrs.append(peak_lin_pwr)
        gauss_lin_pwrs.append(gauss_lin_pwr)
    peak_lin_pwrs = np.array(peak_lin_pwrs).flatten()
    gauss_lin_pwrs = np.array(gauss_lin_pwrs).flatten()
    return peak_lin_pwrs, gauss_lin_pwrs


def _add_minmax_noise(ch_dics):
    """Add min and max noise levels for given reference scheme for plotting."""
    df_noise = pd.DataFrame(ch_dics)  # convert to df for easy grouping
    df_noise = df_noise[df_noise.subject == "NeuEmptyroom"]
    for ref in df_noise.ch_reference.unique():
        df_ref = df_noise[df_noise.ch_reference == ref]
        psds = df_ref.psd.to_numpy()
        psds = np.vstack(psds)

        # Change values and add to dic
        for func in [np.mean, np.min, np.max]:
            dic = df_ref.iloc[0].copy()
            val = func(psds, 0)
            func_str = func.__name__.strip("a")  # -> "amax" -> "max"
            dic["psd"] = val
            dic["psd_min"] = val.min()
            dic["psd_max"] = val.max()
            dic["ch_nme"] = func_str
            ch_dics.append(dic.to_dict())


if __name__ == "__main__":
    dataframe = psd_fooof_to_dataframe(load_fits=False)
