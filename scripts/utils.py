"""Helping functions."""
import re
import warnings
from os import PathLike, remove
from os.path import isdir, isfile, join
from pathlib import Path
from shutil import copy, rmtree  # copytree
from warnings import warn
from functools import lru_cache
from distutils.dir_util import copy_tree
from typing import List, Tuple
import pandas as pd

import numpy as np
import scipy as sp
import scipy.signal as sig
from numpy.fft import irfft, rfftfreq

import nibabel as nib
from mne.io import Raw
from mne_bids import find_matching_paths, write_raw_bids
from nilearn import image

from scripts.corr_stats import _corr_results
import scripts.config as cfg


def _check_duplicated_df_rows(df):
    """Sometimes the same row is duplicated in a dataframe for example
    if fm_params == broad and fm_params == narrow in the same dataframe while
    working with the psd column which is the same for both. This function
    helps to spot the duplicates."""
    for col in df.columns:
        if not df.duplicated(subset=col, keep=False).all():
            print(col)
            try:
                print(df[col].unique())
            except TypeError:
                pass


def _delete_dirty_files(bids_path):
    """Files without annotations are saved in rawdata/ with the descriptions
    tag 'uncleaned'. Once annotations are added and bidsify_sourcedata is run
    again, this function automatically checks whether an uncleaned version is
    present and deletes it to save up space and to avoid duplicate processing
    of cleaned and uncleaned files.
    """
    if bids_path.description == "uncleaned":
        return None
    # get all uncleaned files of this file
    bids_dic = bids_path.entities
    # add plural to each key for find_matching_paths to work
    bids_dic = {key + "s": value for key, value in bids_dic.items()}
    bids_dic['descriptions'] = "uncleaned"
    bids_dic['root'] = bids_path.root
    uncleaned_files = find_matching_paths(**bids_dic)

    # delete all uncleaned files
    for dirty_file in uncleaned_files:
        dirty_file.fpath.unlink()  # delete file


def _save_bids(raw: Raw, bids_path: PathLike) -> None:
    """Save files both using save_raw_bids and raw.save to get the best of
    both worlds: save_raw_bids creates necessary meta files such as
    participants.tsv whereas raw.save saves the raw file correctly including
    it's info and dbs channel types."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ignore_warnings()
            write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True,
                           format="BrainVision", verbose=False)
    except ValueError:
        pass  # space (CapTrak) is not valid for datatype (ieeg).
    bids_path.update(extension=".fif")
    Path(bids_path.directory).mkdir(parents=True, exist_ok=True)
    raw.save(bids_path.fpath, split_naming="bids", overwrite=True)
    remove_extensions = [".eeg", ".vhdr", ".vmrk"]
    for ext in remove_extensions:
        bids_path.update(extension=ext)
        if isfile(bids_path.fpath):
            remove(bids_path.fpath)


def _copy_files_and_dirs(source_root: PathLike, raw_root: PathLike,
                         files_dirs_to_copy: list) -> None:
    """Copy all files or directories listed in files_dirs_to_copy from
    source_root to raw_root."""
    Path(raw_root).mkdir(parents=True, exist_ok=True)
    for file in files_dirs_to_copy:
        source = join(source_root, file)
        destination = join(raw_root, file)
        if isfile(source):
            copy(source, destination)
        elif isdir(source):
            # copytree(source, destination, dirs_exist_ok=True)
            copy_tree(source, destination)
    return None


def _path_from_df(df_ml, extension='.pkl'):
    # extract variables from df_ml
    power_features = df_ml.power_features.unique()[0]
    kinds = df_ml.kind_short.unique()
    chs = df_ml.ch.unique()
    chs = sorted(list(chs))
    conds = df_ml.cond.unique()
    groups = df_ml.groups.unique()[0]

    ml_params = df_ml.params_model[0]
    model = ml_params['ml_model_full']
    model = model.replace('ExtraTreesRegressor', 'Extra')
    model = model.replace('random_state', 'seed')
    speed = df_ml.speed.unique()[0]
    speed_dict = {'fast': 'f', 'slow': 's', 'medium': 'm', 'nan': 'nan'}
    speed = speed_dict[str(speed)]
    repeats = df_ml.params_model.values[0]['repeats']
    repeats_final = df_ml.params_model.values[0]['repeats_final']

    kinds = ''.join(sorted(kinds))
    chs = ''.join([ch.replace('_', '').replace('-', '') for ch in chs])
    chs = chs.replace('LFP13LFP24LFPmean', 'all')
    conds = ''.join([cond.capitalize() for cond in conds])
    conds = conds.replace('OffonOffOn', 'all')

    # Save path
    directory = join('MachineLearning', f'ML_{power_features}')
    fname = (f'features-{kinds}_chs-{chs}_conds-{conds}_'
             f'model-{model}_speed-{speed}_'
             f'repeats-({repeats}-{repeats_final})_groups-{groups}{extension}')
    return directory, fname


def _delete_files_and_dirs(raw_root: PathLike,
                           files_dirs_to_copy: list) -> None:
    """Delete meta data."""
    for file in files_dirs_to_copy:
        destination = raw_root + file
        if isfile(destination):
            remove(destination)
        elif isdir(destination):
            rmtree(destination)
    return None


def _ignore_warnings():
    """Ignore warnings related to zero reference channels when computing
    PSD or FOOOF."""
    warn("deprecated", RuntimeWarning)
    warn("deprecated", UserWarning)


def _get_ref_from_info(info):
    if info["proj_name"] in ["Litvak", "Hirschmann", "Hirschmann2", 'Florin']:
        reference = "mastoid"
    elif info["proj_name"] == "Tan":
        reference = "bipolar?"
    elif info["proj_name"] == "Neumann":
        if info['subject_info']['his_id'] == 'NeuEmptyroom':
            reference = "1 kOhm"
        else:
            start = info["description"].find("Ref-") + 4
            end = info["description"].find(".")
            reference = info["description"][start:end]
        # test if LFP + ECOG reference were used (2 instead of 1)
        if "+" in reference:
            ref1, ref2 = reference.split(" + ")
            reference = ref1 if ref1.startswith("LFP") else ref2
            assert reference.startswith("LFP")
    return reference


def _distant_contacts(ch_names):
    """Return distant bipolar contacts.

    e.g. 'LFP_R_2a-4_STN_BS' but not 'LFP_R_3a-4_STN_BS'."""
    bip_chs = set(filter(re.compile("^.*-.*").match, ch_names))
    bipolar_distant = []
    for ch in bip_chs:
        nums = re.sub("\D", "", ch.split("_")[2])  # get 3a-4
        assert len(nums) == 2
        nums = int(nums[0]), int(nums[1])
        distance = abs(np.diff(nums))
        if distance > 1:
            bipolar_distant.append(ch)
    return bipolar_distant


@lru_cache(maxsize=None)  # load mask only once
def _load_stn_masks(threshold=0.1):
    """Return binary stn_mask_LR and inverse matrix for transformation."""
    # Paths to MNI_ICBM_2009b_NLIN_ASYM template
    path_leadDBS = join('..', 'MATLAB', 'leadDBS')
    path_mni2009b = join(path_leadDBS, 'templates', 'space',
                         'MNI152NLin2009bAsym')
    path_ewert = join(path_mni2009b, 'atlases', 'DISTAL Minimal (Ewert 2017)')
    path_stn_L = join(path_ewert, 'lh', 'STN.nii.gz')
    path_stn_R = join(path_ewert, 'rh', 'STN.nii.gz')

    # Load STN nifti mask
    stn_nifti_L = nib.load(path_stn_L)
    stn_nifti_R = nib.load(path_stn_R)

    # Resample STN mask to match MNI_ICBM_2009b_NLIN_ASYM template
    stn_img_L = stn_nifti_L.get_fdata()
    stn_img_R = stn_nifti_R.get_fdata()

    stn_mask_L = nib.Nifti1Image(stn_img_L, stn_nifti_L.affine)
    stn_mask_R = nib.Nifti1Image(stn_img_R, stn_nifti_R.affine)

    template = nib.load(join(path_mni2009b, 't1.nii'))
    stn_volume_L = image.resample_img(stn_mask_L,
                                      target_affine=template.affine,
                                      target_shape=template.shape)
    stn_volume_R = image.resample_img(stn_mask_R,
                                      target_affine=template.affine,
                                      target_shape=template.shape)

    # Plotting
    # from nilearn import plotting
    # plotting.plot_img(stn_mask_R, colorbar=True, threshold=threshold)
    # plotting.plot_roi(stn_volume_R, threshold=threshold, colorbar=True)

    # Combine left and right STN masks and make binary using threshold
    stn_volume_LR = stn_volume_L.get_fdata() + stn_volume_R.get_fdata()
    stn_volume_LR[stn_volume_LR > threshold] = 1
    stn_volume_LR[stn_volume_LR <= threshold] = 0
    stn_mask_LR = nib.Nifti1Image(stn_volume_LR, stn_volume_L.affine)

    # Transform mni coords to image coords
    inverse_matrix = np.linalg.inv(template.affine)
    return stn_mask_LR, inverse_matrix


class FailedFits:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FailedFits, cls).__new__(cls)
            cls._instance.data = []
        return cls._instance

    def add_failed_fit(self, info):
        self.data.append(info)


def elec_phys_signal(exponent: float,
                     periodic_params: List[Tuple[float, float, float]] = None,
                     offset=1,
                     nlv: float = None,
                     highpass: bool = False,
                     sample_rate: float = 2400,
                     duration: float = 180,
                     random_ap_phases=True,
                     random_per_phases=True,
                     seed: int = 1):
    """Generate 1/f noise with optionally added oscillations.

    Parameters
    ----------
    exponent : float
        Aperiodic 1/f exponent.
    periodic_params : list of tuples
        Oscillations parameters as list of tuples in form of
                [(center_frequency1, peak_amplitude1, peak_width1),
                (center_frequency2, peak_amplitude2, peak_width2)]
        for two oscillations.
    offset : float
        Offset of the aperiodic signal. The default is 1.
    nlv : float, optional
        Level of white noise. The default is None.
    highpass : bool, optional
        Whether to apply a 4th order butterworth highpass filter at 1Hz.
        The default is False.
    sample_rate : float, optional
        Sample rate of the signal. The default is 2400Hz.
    duration : float, optional
        Duration of the signal in seconds. The default is 180s.
    random_ap_phases : bool, optional
        Whether to add random phases to aperiodic signal. The default is True.
    random_per_phases : bool, optional
        Whether to add random phases to periodic signal. The default is True.
    seed : int, optional
        Seed for reproducibility. The default is 1.

    Returns
    -------
    aperiodic_signal : ndarray
        Aperiodic 1/f activity without oscillations.
    full_signal : ndarray
        Aperiodic 1/f activity with added oscillations.
    """
    if seed:
        np.random.seed(seed)
    # Initialize
    n_samples = int(duration * sample_rate)
    amps = np.ones(n_samples//2, complex) * offset
    freqs = rfftfreq(n_samples, d=1/sample_rate)
    freqs = freqs[1:]  # avoid division by 0

    # Create random phases
    fixed_phases = np.exp(1j * 1)
    rand_phases = np.exp(1j * np.random.uniform(0, 2*np.pi, size=amps.shape))
    if random_ap_phases:
        amps *= rand_phases
    else:
        amps *= fixed_phases

    # Multiply phases to amplitudes and create power law
    amps /= freqs ** (exponent / 2)

    # Add oscillations
    amps_osc = amps.copy()
    if periodic_params:
        for osc_params in periodic_params:
            freq_osc, amp_osc, width = osc_params
            amp_dist = sp.stats.norm(freq_osc, width).pdf(freqs)
            # add same random phases
            if random_per_phases:
                amp_dist = amp_dist * rand_phases
            else:
                amp_dist = amp_dist * fixed_phases
            amps_osc += amp_osc * amp_dist  # * offset

    # Create colored noise time series from amplitudes
    aperiodic_signal = irfft(amps)
    full_signal = irfft(amps_osc)

    # Add white noise
    if nlv:
        w_noise = np.random.normal(scale=nlv, size=n_samples-2)
        aperiodic_signal += w_noise
        full_signal += w_noise

    # Highpass filter
    if highpass:
        sos = sig.butter(4, 1, btype="hp", fs=sample_rate, output='sos')
        aperiodic_signal = sig.sosfilt(sos, aperiodic_signal)
        full_signal = sig.sosfilt(sos, full_signal)

    return aperiodic_signal, full_signal


def _extract_ml_performance(df_ml):
    errors = df_ml.error.values[0]
    errors_dummy = df_ml.error_dummy.values[0]
    errors_all = np.concatenate([errors, errors_dummy])

    model_nme = df_ml.params_model.values[0]['ml_model']
    scoring = df_ml.params_model.values[0]['scoring']
    scoring = 'RMSE' if scoring == 'neg_root_mean_squared_error' else scoring
    benchmark = df_ml.benchmark.unique()[0]
    dummy_nme = benchmark.capitalize() + 'Regressor'
    classifiers = [model_nme] * len(errors) + [dummy_nme] * len(errors_dummy)

    df_all = pd.DataFrame({'classifier': classifiers, 'errors': errors_all})
    df_model = df_all[df_all.classifier == model_nme]
    df_dummy = df_all[df_all.classifier == dummy_nme]
    order = [dummy_nme, model_nme]

    rmse = df_model['errors'].mean()
    rmse_dummy = df_dummy['errors'].mean()
    error_percentage = 100 * (rmse_dummy - rmse) / rmse_dummy
    rep_std = df_ml.rep_std.unique()[0]
    rep_std_perc = 100 * (rmse_dummy - rmse - rep_std) / rmse_dummy
    rep_std_perc = 100 * rmse_dummy / (rmse_dummy - rep_std) - 100
    performance_dict = {'df_all': df_all, 'scoring': scoring,
                        'errors_all': errors_all, 'order': order,
                        'error_percentage': error_percentage,
                        'rep_std_perc': rep_std_perc}
    return performance_dict


def _average_hemispheres(df, x, y):
    df = df.copy()
    group = df.groupby(["subject", "cond"])
    df[x] = group[x].transform("mean")
    df[y] = group[y].transform("mean")
    # select left hemisphere
    df = df.drop_duplicates(subset=["subject", "cond"])
    df = df.dropna(subset=[y, x])
    return df


def get_correlation_df(df_plot, y, total_power=True, average_hemispheres=False,
                       use_peak_power=True, corr_method='spearman', bands=None,
                       band_nmes=None,
                       add_high_beta_cf=False, band_cols=None,
                       n_perm=10000, output_file=None):
    """Get dataframe to plot barplot."""
    if y == 'UPDRS_III':
        average_hemispheres = True
    psd_kind = df_plot.psd_kind.unique()[0]
    fm_params = df_plot.fm_params.unique()[0]
    cond = df_plot.cond.unique()[0]
    projects = [proj for proj in cfg.PROJECT_ORDER_SLIM
                if proj in df_plot.project.unique()]
    df_corrs = []
    if use_peak_power:
        pwr = '_abs_max_log' if total_power else '_fm_powers_max_log'
        pwr_kind = 'max'
        freq = '_abs_max_freq' if total_power else '_fm_centerfreqs_max'
    else:
        pwr = '_abs_mean_log' if total_power else '_fm_auc_log'
        pwr_kind = 'mean'
        freq = '_abs_max_freq' if total_power else '_fm_centerfreqs_max'
    if band_cols is None:
        if bands is None:
            bands = cfg.BANDS
        band_cols = [band + pwr for band in bands]
        if not total_power:
            band_cols += ['fm_offset_log', 'fm_exponent']
    if band_nmes is None:
        band_dic = cfg.BAND_NAMES_GREEK_SHORT
        plot_dic = cfg.PLOT_LABELS_SHORT
        bands = [band.replace(pwr, '') for band in band_cols]
        band_nmes = [band_dic[band] if band in band_dic else plot_dic[band]
                     for band in bands]

    # Add High Beta Center Frequency
    if add_high_beta_cf:
        band_cols += ['beta_high' + freq]
        band_nmes += ['High Beta Freq.']

        if fm_params == 'broadLor':
            band_cols += ['fm_knee_fit']
            band_nmes += ['Knee [Hz]']

    for i, band in enumerate(band_cols):
        for project in projects:
            df_sub = df_plot[(df_plot.project == project)]
            if average_hemispheres:
                df_sub = _average_hemispheres(df_sub, band, y,)
            if (corr_method == 'withinRank') and ('_rank' in band):
                # use within method for ranked data
                use_corr_method = 'within'
                use_y = y + '_rank'
            else:
                use_corr_method = corr_method
                use_y = y

            rho, sample_size, label, _, _ = _corr_results(df_sub, band, use_y,
                                                          use_corr_method,
                                                          n_perm=n_perm)
            if rho is None:
                continue
            pval = float(label.split(' ')[-1].strip('p='))
            dic = {'project': project, 'rho': rho, 'band': band,
                   'band_nme': band_nmes[i], 'pwr_kind': pwr_kind,
                   'sample_size': sample_size, 'pval': pval,
                   'psd_kind': psd_kind, 'total_power': total_power,
                   'fm_params': fm_params, 'n_perm': n_perm,
                   'cond': cond,
                   'corr_method': corr_method, 'y': y}
            df_corrs.append(dic)
            proj_nme = df_sub.project_nme.unique()[0]
            if proj_nme == 'all':
                print(
                    f"{proj_nme} {band_nmes[i]}: rho={rho:.2f}, p={pval:.2f}",
                    file=output_file
                )
    df_corrs = pd.DataFrame(df_corrs)
    return df_corrs
