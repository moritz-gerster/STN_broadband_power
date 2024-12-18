"""Calc and save PSDs."""
import warnings
from os.path import basename, join
from pathlib import Path
import numpy as np
from mne import make_fixed_length_epochs, set_log_level
from mne.io import read_raw
from mne_bids import find_matching_paths, get_entity_vals
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scripts.utils import (_copy_files_and_dirs, _delete_dirty_files,
                           _ignore_warnings)

import scripts.config as cfg


def save_spectra(subjects=None, sessions=None, recordings=cfg.RECORDINGS,
                 verbose="error", processings=None, freq_res=1,
                 acquisitions=None, tasks=None,
                 method='welch', interpolate_line_noise=True, fillnan=True,
                 fmax=None, description_new=None,
                 descriptions="cleaned") -> None:
    """Calc PSDs and save in derivatives as hdf5 files."""
    set_log_level(verbose)
    root = cfg.PREPROCESSED
    spec_paths = find_matching_paths(root, subjects=subjects,
                                     sessions=sessions, extensions=".fif",
                                     recordings=recordings,
                                     acquisitions=acquisitions,
                                     tasks=tasks,
                                     descriptions=descriptions,
                                     processings=processings)

    for spec_path in tqdm(spec_paths, desc="Calc Spectra: "):
        spectrum = _calc_psd(spec_path, method=method, freq_res=freq_res,
                             fmax=fmax)
        if spectrum is None:
            continue
        if interpolate_line_noise:
            _interpolate_spectrum(spectrum, spec_path, freq_res)
        _smooth_emptyroom(spectrum, spec_path)
        if fillnan:
            _fill_up_nans(spectrum)
        _save_spectrum(spectrum, spec_path, description_new=description_new)
    # Copy meta info as is
    recordings = get_entity_vals(root, "recording")  # does not work yet
    for recording in recordings:
        _copy_files_and_dirs(join(root, f"meta_infos_{recording}"),
                             join(cfg.SPECTRA, f"meta_infos_{recording}"),
                             cfg.BIDS_FILES_TO_COPY)
    print(f"PSDs saved to {cfg.SPECTRA}")
    print(f"{basename(__file__).strip('.py')} done.")


def _fill_up_nans(spectrum):
    """Fill up missing frequencies up to cfg.LOWPASS with nans."""
    if spectrum.info['lowpass'] == cfg.LOWPASS:
        return spectrum

    # get spec data
    data = spectrum._data
    freqs = spectrum.freqs
    resolution = np.unique(np.diff(freqs))[0]

    # fill up missing frequencies
    freqs_full = np.arange(0, cfg.LOWPASS + 1, resolution)
    data_full = np.full((data.shape[0], len(freqs_full)), np.nan)
    data_full[:, np.searchsorted(freqs_full, freqs)] = data

    # add to spectrum
    spectrum._freqs = freqs_full
    spectrum._data = data_full
    return spectrum


def _interpolate_spectrum(spectrum, bids_path, freq_res,
                        #   bandwidth=None
                          ):
    # Set width of line noise to remove
    assert freq_res == 1, 'Saved settings dont work for this freq_res'
    if bids_path.basename in cfg.LINE_NOISE_BROAD:
        line_width = 2
    elif bids_path.subject == 'FloML013':
        line_width = 3
    elif bids_path.subject == 'TanEmptyroom':
        line_width = 4
    else:
        line_width = 1

    # Set line noise frequencies to remove
    if bids_path.basename in cfg.LINE_NOISE_FREQS:
        line_freqs = cfg.LINE_NOISE_FREQS[bids_path.basename]
    else:
        line_freqs = 50

    spectrum = _interpolate_line_noise(spectrum, line_width=line_width,
                                       line_freqs=line_freqs)
    bids_path.processing += 'NoLine'
    return spectrum


def _smooth_spectra(spectrum, bids_path, polyorder=3, window_length=100,
                    fmin=50):
    """Do not smooth frequencies below 100 Hz. Savgol filter cannot handle
    power law data."""
    psds, freqs = spectrum.get_data(picks='data', exclude=[],
                                    return_freqs=True)
    # # Calculate sigma for FWHM Gaussian smoothing
    # sigma = FWHM / (np.sqrt(2 * np.log(2)) * 2)
    # psds_smoothed = gaussian_filter1d(psds, sigma=sigma, axis=1)
    mask_fmin = freqs > fmin
    psds_smoothed = savgol_filter(psds, window_length, polyorder)
    spectrum._data[:, mask_fmin] = psds_smoothed[:, mask_fmin]
    import matplotlib.pyplot as plt
    psds_smoothed[:, ~mask_fmin] = psds[:, ~mask_fmin]
    plt.figure(figsize=(20, 9))
    # i = -1
    # mask = freqs > 95
    mask = freqs > 0
    for i in range(len(psds)):
        plt.loglog(freqs[mask], psds[i, mask], color='grey')
        plt.loglog(freqs[mask], psds_smoothed[i, mask], '--', color='green')
    plt.title(f"polyorder={polyorder}, window_length={window_length}")
    plt.tight_layout()
    bids_path.processing += 'Smooth'


def _smooth_emptyroom(spectrum, bids_path, polyorder=3, win_len_high=100,
                      fsplit=8):
    """Use different smoothing for low frequencies. Savgol filter cannot handle
    power law data, only smooth above 50 Hz."""
    if not bids_path.subject.endswith('Emptyroom'):
        return
    psds, freqs = spectrum.get_data(picks='data', exclude=[],
                                    return_freqs=True)

    # High frequency Savitzky-Golay smoothing
    mask_high = freqs >= fsplit
    psds_smoothed = savgol_filter(psds[:, mask_high], win_len_high, polyorder)
    spectrum._data[:, mask_high] = psds_smoothed

    # # Gaussian smoothing
    # mask_low = (freqs >= 6)
    # sigma = 20 / (np.sqrt(2 * np.log(2)) * 2)
    # psds_smoothed = gaussian_filter1d(spectrum._data[:, mask_low], sigma=sigma, axis=1)
    # spectrum._data[:, mask_low] = psds_smoothed

    # # Edge smoothing
    # win_edge = 25
    # mask_edge = (freqs <= fsplit + win_edge) & (freqs >= fsplit - win_edge)
    # psds_smoothed = gaussian_filter1d(psds, sigma=win_edge, axis=1)
    # spectrum._data[:, mask_edge] = psds_smoothed[:, mask_edge]

    # import matplotlib.pyplot as plt
    # psds_smoothed = spectrum._data
    # plt.figure(figsize=(20, 9))
    # # i = -1
    # # mask = freqs > 95
    # # mask = freqs > 0
    # for i in range(0, len(psds), 6):
    #     plt.loglog(freqs[:], psds[i, :], '--', color='grey')
    #     plt.loglog(freqs[:], psds_smoothed[i, :], '--', color='green',
    #                lw=4)
    # # plt.title(f"polyorder={polyorder}, window_length={win_edge}")
    # # plt.title(f"sigma={sigma}")
    # plt.tight_layout()
    bids_path.processing += 'Smooth'


def _calc_psd(bids_path, check_srate=True, method='welch', freq_res=1,
              fmax=None):
    """Load concats, calc PSD, change units."""
    raw = read_raw(bids_path.fpath)
    bids_path.update(root=cfg.SPECTRA)
    if check_srate:
        msg = "Consistent srate important for exact same PSD frequencies"
        assert raw.info["sfreq"] == cfg.RESAMPLE_FREQ, msg
    n_fft = int(raw.info["sfreq"] / freq_res)
    fmax = fmax if fmax is not None else raw.info["lowpass"]
    picks = raw.copy().pick(picks="data", exclude=()).ch_names
    nan_chs = [ch for ch in picks if np.isnan(raw.get_data(ch)).all()]
    picks = list(set(picks) - set(nan_chs))
    if not len(picks):
        return None
    if method == 'welch':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ignore_warnings()
            spectrum = raw.compute_psd(method="welch",
                                       n_fft=n_fft,  # numbers per segment
                                       picks=picks,  # include bads
                                       average="mean",
                                       fmax=fmax,
                                       verbose=False)
    elif method == 'multitaper':
        # freq_res=1 make same bins as welch for easier dataframe handling
        epochs = make_fixed_length_epochs(raw, duration=1/freq_res).load_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ignore_warnings()
            spectrum = epochs.compute_psd(method="multitaper",
                                          bandwidth=freq_res,
                                          low_bias=True,  # False bad at 1 Hz
                                          adaptive=False,  # no difference
                                          # "length" wrong
                                          normalization="full",
                                          picks=picks,  # include bads
                                          fmax=fmax,
                                          verbose=False)
    bids_path.processing += spectrum.method.capitalize()
    return spectrum


def _save_spectrum(spectrum, bids_path, description_new=None):
    """Save power spectral densities."""
    Path(bids_path.directory).mkdir(parents=True, exist_ok=True)
    spectrum.save(bids_path.fpath, overwrite=True)

    # PLOT LOW FREQUENCY PSDS #################################################
    # if description_new is not None:
    #     bids_path.description = description_new
    # bip_channels = [ch for ch in spectrum.ch_names if '-' in ch]
    # fig = spectrum.plot(picks=bip_channels, xscale='log', exclude='bads')
    # root = f'results/plots/power_spectra/{description_new}'
    # basename = bids_path.basename
    # basename = basename.replace('fif', 'pdf')
    # fig_path = bids_path.copy().update(root=root)
    # Path(fig_path.root).mkdir(parents=True, exist_ok=True)
    # fig.savefig(join(fig_path.root, basename))
    # PLOT LOW FREQUENCY PSDS #################################################
    _delete_dirty_files(bids_path)


def _interpolate_line_noise(spectrum, line_freqs=50, line_width=3):
    # Get data
    freqs = spectrum.freqs
    psds = spectrum.get_data(picks='data', exclude=[])
    if spectrum.method == 'multitaper':
        psds = psds.mean(0)
    freq_res = np.unique(np.diff(freqs))[0]
    assert np.allclose(np.diff(freqs), freq_res)

    # Make new spectrum object
    psds_interpolated = psds.copy()

    # Allow multiple line frequencies
    if isinstance(line_freqs, (int, float)):
        line_freqs = [line_freqs]
    line_noises = []
    for line_freq in line_freqs:
        line_noise = np.arange(line_freq, freqs[-1] + 1, line_freq)
        line_noises.append(line_noise)
    line_noises = np.concatenate(line_noises)

    # Get indices for interpolation in case freq_res is not 1
    line_noise_idcs = np.searchsorted(freqs, line_noises)

    # Expand the indices to cover the width of the line noise
    bin_width = int(line_width / freq_res)
    idx_to_replace = []
    for i in range(-bin_width, bin_width + 1):
        idx_to_replace.append(line_noise_idcs + i)
    idx_to_replace = np.concatenate(idx_to_replace)

    # remove indices that are out of bounds
    in_bounds = (idx_to_replace >= 0) & (idx_to_replace <= freqs[-1])
    idx_to_replace = idx_to_replace[in_bounds]

    # Linear interpolation on 1 Hz resolution better than kind='cubic'
    valid_idx = np.delete(np.arange(len(freqs)), idx_to_replace)
    interpolation = interp1d(valid_idx, psds[:, valid_idx], kind='linear',
                             fill_value="extrapolate")
    psds_interpolated[:, idx_to_replace] = interpolation(idx_to_replace)

    if spectrum.method == 'multitaper':
        # give back missing dimension
        psds_interpolated = psds_interpolated[None]
    spectrum._data = psds_interpolated
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(15, 9))
    # # i = 2
    # # i = 2  # good channels for first Litvak subject
    # # i = 3  # good channels for first Hirschmann subject
    # mask = freqs > 0
    # # mask = freqs > 450
    # for ch in range(0, psds.shape[0], 6):
    #     ch_bad = spectrum.ch_names[ch] in spectrum.info['bads']
    #     if ch_bad:
    #         continue
    #     plt.loglog(freqs[mask], psds[ch, mask], color='grey')
    #     plt.loglog(freqs[mask], psds_interpolated[ch, mask], '-',
    #                color='green')
    # # plt.loglog(freqs[mask], psds[i, mask], color='grey', label='original')
    # # plt.loglog(freqs[mask], psds_interpolated[i, mask], '--',
    #               color='green', label='linear')
    # plt.tight_layout()
    return spectrum


if __name__ == "__main__":
    save_spectra()