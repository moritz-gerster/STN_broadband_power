from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

import scripts.config as cfg
from specparam import SpectralModel
from specparam.analysis import get_band_peak
from scripts.plot_figures.settings import XTICKS_FREQ_low
from scripts.utils import elec_phys_signal
from scripts.utils_plot import _save_fig


def _normalize(psd, freqs):
    mask = ((freqs >= 5) & (freqs <= 95))
    factor = 100  # percentage
    psd_sum = psd[mask].sum()
    psds_norm = psd / psd_sum * factor  # percentage
    return psds_norm


def simulate_all(fig_dir=None, output_file=None):
    # %% Params

    # Calc Welch
    sample_rate = 2400
    welch_params = dict(fs=sample_rate, nperseg=sample_rate)

    # Plot settings
    c_rel = cfg.COLOR_DIC['normalized']
    c_rel2 = '#136f53'
    c_abs = cfg.COLOR_DIC['absolute']
    c_abs2 = '#4b4785'
    c_per = cfg.COLOR_DIC['periodic']

    # %% Simulation parameters

    # Aperiodic params
    sim_exponent = 1.5
    offset = 1
    aperiodic_params = dict(exponent=sim_exponent, offset=offset, nlv=1e-5)

    # Periodic params
    beta_freq = 17
    beta_normal_power = 1
    beta_normal_width = 2.5
    beta_normal = (beta_freq, beta_normal_power, beta_normal_width)
    beta_low = (beta_freq, beta_normal_power / 2.3, beta_normal_width / 1.75)

    theta_freq = 7.5
    theta_power = .6
    theta_width = 2
    theta = (theta_freq, theta_power, theta_width)
    beta_strong = (beta_freq, beta_normal_power * .79, beta_normal_width * .79)

    beta_narrow = (beta_freq,
                   beta_normal_power * 1.7,
                   beta_normal_width * 1.75)
    off_scaling = .5  # offset impacts oscillatory power nonlinearly
    beta_large_offset = (beta_freq,
                         beta_normal_power / 2.3 * off_scaling,
                         beta_normal_width / 2.3)

    sim_exponent_small = .3
    off_scaling = .6  # exponent impacts oscillatory power nonlinearly
    beta_small_exponent = (beta_freq,
                           beta_normal_power / 2.3 * off_scaling,
                           beta_normal_width / 1.9)

    # %% Power spectra simulations

    beta_normal_sim_ap, beta_normal_sim = elec_phys_signal(
        **aperiodic_params, periodic_params=[beta_normal]
        )
    beta_theta_broad_sim = elec_phys_signal(
        **aperiodic_params, periodic_params=[theta, beta_normal]
        )[1]
    beta_narrow_sim = elec_phys_signal(**aperiodic_params,
                                       periodic_params=[beta_narrow])[1]
    beta_strong_sim = elec_phys_signal(**aperiodic_params,
                                       periodic_params=[beta_strong])[1]
    beta_large_offset_sim_ap, beta_large_offset_sim = elec_phys_signal(
        exponent=.8, offset=.83, nlv=1e-5, periodic_params=[beta_large_offset]
        )
    beta_small_exponent_sim_ap, beta_small_exponent_sim = elec_phys_signal(
        exponent=sim_exponent_small, offset=.4, nlv=1e-5,
        periodic_params=[beta_small_exponent]
        )
    beta_low_sim_ap, beta_low_sim = elec_phys_signal(
        exponent=.8, offset=.4, nlv=1e-5, periodic_params=[beta_low]
        )

    # Create reasonable uV^2/Hz units
    scaling_factor = 50000
    beta_normal_sim *= scaling_factor
    beta_theta_broad_sim *= scaling_factor
    beta_narrow_sim *= scaling_factor
    beta_strong_sim *= scaling_factor
    beta_large_offset_sim *= scaling_factor
    beta_low_sim *= scaling_factor
    beta_low_sim_ap *= scaling_factor
    beta_small_exponent_sim *= scaling_factor
    beta_large_offset_sim_ap *= scaling_factor
    beta_small_exponent_sim_ap *= scaling_factor
    beta_normal_sim_ap *= scaling_factor

    # PSD
    freqs, beta_normal_sim = sig.welch(beta_normal_sim, **welch_params)
    freqs, beta_theta_broad_sim = sig.welch(beta_theta_broad_sim,
                                            **welch_params)
    freqs, beta_narrow_sim = sig.welch(beta_narrow_sim, **welch_params)
    freqs, beta_strong_sim = sig.welch(beta_strong_sim, **welch_params)
    freqs, beta_large_offset_sim = sig.welch(beta_large_offset_sim,
                                             **welch_params)
    freqs, beta_low_sim = sig.welch(beta_low_sim, **welch_params)
    freqs, beta_low_sim_ap = sig.welch(beta_low_sim_ap, **welch_params)
    freqs, beta_small_exponent_sim = sig.welch(beta_small_exponent_sim,
                                               **welch_params)
    freqs, beta_large_offset_sim_ap = sig.welch(beta_large_offset_sim_ap,
                                                **welch_params)
    freqs, beta_small_exponent_sim_ap = sig.welch(beta_small_exponent_sim_ap,
                                                  **welch_params)
    freqs, beta_normal_sim_ap = sig.welch(beta_normal_sim_ap, **welch_params)

    # %% Fit FOOOF
    beta_borders = cfg.BANDS['beta_low']
    beta_low_borders = cfg.BANDS['beta_low']
    beta_high_borders = cfg.BANDS['beta_high']

    # Periodic power
    fm = SpectralModel(verbose=False, max_n_peaks=1)
    fit_range = [1, 100]
    fm.fit(freqs, beta_normal_sim, fit_range)
    kwargs = dict(band=beta_borders, select_highest=True)
    cf, beta_normal_sim_per_pwr_max, _ = get_band_peak(fm, **kwargs)
    beta_mask_fm = ((fm.freqs >= beta_borders[0])
                    & (fm.freqs <= beta_borders[1]))
    beta_normal_sim_per_pwr = fm._peak_fit[beta_mask_fm].mean()

    # Get aperidiodic power at peak frequency
    freq_idx = np.argmin(np.abs(fm.freqs - cf))
    ap_pwr = fm._ap_fit[freq_idx]

    fm.fit(freqs, beta_theta_broad_sim, fit_range)
    beta_theta_broad_sim_per_pwr = fm._peak_fit[beta_mask_fm].mean()
    fm.fit(freqs, beta_narrow_sim, fit_range)
    beta_narrow_sim_per_pwr = fm._peak_fit[beta_mask_fm].mean()
    fm.fit(freqs, beta_strong_sim, fit_range)
    beta_strong_sim_per_pwr = fm._peak_fit[beta_mask_fm].mean()
    fm.fit(freqs, beta_large_offset_sim, fit_range)
    cf, beta_large_offset_per_pwr_max, _ = get_band_peak(fm, **kwargs)
    beta_large_offset_per_pwr = fm._peak_fit[beta_mask_fm].mean()
    fm_per_large_offset = 10**fm.modeled_spectrum_[beta_mask_fm]
    ap_pwr_large = fm._ap_fit[freq_idx]

    kwargs_high = kwargs.copy()
    kwargs_high['band'] = beta_high_borders
    fm.fit(freqs, beta_low_sim, fit_range)
    cf, beta_low_sim_per_pwr_max, _ = get_band_peak(fm, **kwargs)
    beta_low_sim_per_pwr = fm._peak_fit[beta_mask_fm].mean()
    fm_per_low = 10**fm.modeled_spectrum_[beta_mask_fm]
    freq_idx = np.argmin(np.abs(fm.freqs - cf))
    ap_pwr_low = fm._ap_fit[freq_idx]

    fm.fit(freqs, beta_small_exponent_sim, fit_range)
    cf, beta_small_exponent_sim_per_pwr_max, _ = get_band_peak(fm, **kwargs)
    beta_small_exponent_sim_per_pwr = fm._peak_fit[beta_mask_fm].mean()
    fm_per_small_exponent = 10**fm.modeled_spectrum_[beta_mask_fm]
    freq_idx = np.argmin(np.abs(fm.freqs - cf))
    ap_pwr_small_exponent = fm._ap_fit[freq_idx]

    # %% Normalize

    # Normalize
    beta_normal_sim_norm = _normalize(beta_normal_sim, freqs)
    beta_theta_broad_sim_norm = _normalize(beta_theta_broad_sim, freqs)
    beta_narrow_sim_norm = _normalize(beta_narrow_sim, freqs)
    beta_strong_sim_norm = _normalize(beta_strong_sim, freqs)
    beta_large_offset_sim_norm = _normalize(beta_large_offset_sim, freqs)
    beta_low_sim_norm = _normalize(beta_low_sim, freqs)
    beta_small_exponent_sim_norm = _normalize(beta_small_exponent_sim, freqs)
    # %% Shrink data

    # Mask relevant frequency range
    highpass = 3
    lowpass = 40
    filt = (freqs >= highpass) & (freqs <= lowpass)

    # Mask above highpass and below lowpass
    freqs = freqs[filt]
    beta_normal_sim = beta_normal_sim[filt]
    beta_theta_broad_sim = beta_theta_broad_sim[filt]
    beta_narrow_sim = beta_narrow_sim[filt]
    beta_strong_sim = beta_strong_sim[filt]
    beta_large_offset_sim = beta_large_offset_sim[filt]
    beta_low_sim = beta_low_sim[filt]
    beta_low_sim_ap = beta_low_sim_ap[filt]
    beta_small_exponent_sim = beta_small_exponent_sim[filt]
    beta_small_exponent_sim_ap = beta_small_exponent_sim_ap[filt]
    beta_large_offset_sim_ap = beta_large_offset_sim_ap[filt]
    beta_normal_sim_ap = beta_normal_sim_ap[filt]
    beta_normal_sim_norm = beta_normal_sim_norm[filt]
    beta_theta_broad_sim_norm = beta_theta_broad_sim_norm[filt]
    beta_narrow_sim_norm = beta_narrow_sim_norm[filt]
    beta_strong_sim_norm = beta_strong_sim_norm[filt]
    beta_large_offset_sim_norm = beta_large_offset_sim_norm[filt]
    beta_small_exponent_sim_norm = beta_small_exponent_sim_norm[filt]
    beta_low_sim_norm = beta_low_sim_norm[filt]

    # %% Band power calculation

    beta_mask = (freqs >= beta_borders[0]) & (freqs < beta_borders[1])
    beta_low_mask = ((freqs >= beta_low_borders[0])
                     & (freqs < beta_low_borders[1]))
    beta_mask = beta_low_mask

    # Absolute power
    func = np.mean
    beta_normal_sim_pwr = func(beta_normal_sim[beta_mask])
    beta_theta_broad_sim_pwr = func(beta_theta_broad_sim[beta_mask])
    beta_narrow_sim_pwr = func(beta_narrow_sim[beta_mask])
    beta_strong_sim_pwr = func(beta_strong_sim[beta_mask])
    beta_large_offset_sim_pwr = func(beta_large_offset_sim[beta_mask])
    beta_small_exponent_sim_pwr = func(beta_small_exponent_sim[beta_mask])

    beta_low_sim_pwr = func(beta_low_sim[beta_low_mask])

    # Relative power
    beta_normal_sim_norm_pwr = func(beta_normal_sim_norm[beta_mask])
    beta_theta_broad_sim_norm_pwr = func(beta_theta_broad_sim_norm[beta_mask])
    beta_narrow_sim_norm_pwr = func(beta_narrow_sim_norm[beta_mask])
    beta_strong_sim_norm_pwr = func(beta_strong_sim_norm[beta_mask])
    beta_large_offset_sim_norm_pwr = func(
        beta_large_offset_sim_norm[beta_mask]
        )
    beta_small_exponent_sim_norm_pwr = func(
        beta_small_exponent_sim_norm[beta_mask]
        )

    beta_low_sim_norm_pwr = func(beta_low_sim_norm[beta_low_mask])
    beta_normal_sim_per_pwr = (10**(beta_normal_sim_per_pwr + ap_pwr)
                               - 10**ap_pwr)
    beta_theta_broad_sim_per_pwr = (10**(beta_theta_broad_sim_per_pwr
                                         + ap_pwr) - 10**ap_pwr)
    beta_narrow_sim_per_pwr = (10**(beta_narrow_sim_per_pwr + ap_pwr)
                               - 10**ap_pwr)
    beta_strong_sim_per_pwr = (10**(beta_strong_sim_per_pwr + ap_pwr)
                               - 10**ap_pwr)
    beta_large_offset_per_pwr = (10**(beta_large_offset_per_pwr
                                      + ap_pwr_large) - 10**ap_pwr_large)
    beta_low_sim_per_pwr = (10**(beta_low_sim_per_pwr + ap_pwr_low)
                            - 10**ap_pwr_low)
    beta_low_sim_per_pwr_max = (10**(beta_low_sim_per_pwr_max + ap_pwr_low)
                                - 10**ap_pwr_low)
    beta_large_offset_per_pwr_max = (10**(beta_large_offset_per_pwr_max
                                          + ap_pwr_large) - 10**ap_pwr_large)
    beta_normal_sim_per_pwr_max = (10**(beta_normal_sim_per_pwr_max
                                        + ap_pwr) - 10**ap_pwr)
    beta_small_exponent_sim_per_pwr = (10**(beta_small_exponent_sim_per_pwr
                                            + ap_pwr_small_exponent)
                                       - 10**ap_pwr_small_exponent)
    sum_ = beta_small_exponent_sim_per_pwr_max + ap_pwr_small_exponent
    beta_small_exponent_sim_per_pwr_max = (10**(sum_)
                                           - 10**ap_pwr_small_exponent)
    ap_pwr = 10**ap_pwr
    ap_pwr_large = 10**ap_pwr_large
    ap_pwr_low = 10**ap_pwr_low
    ap_pwr_small_exponent = 10**ap_pwr_small_exponent

    # replace nan (no peak found) with 0
    beta_low_sim_per_pwr = np.nan_to_num(beta_low_sim_per_pwr)

    msg = ('Beta must be same in both conds but is '
           f'{beta_narrow_sim_norm_pwr - beta_strong_sim_norm_pwr:.2f}')
    assert np.allclose(beta_narrow_sim_norm_pwr, beta_strong_sim_norm_pwr,
                       atol=0.01), msg

    msg = ('Max. Beta must be same in both conds but is '
           f'{beta_low_sim_per_pwr_max - beta_large_offset_per_pwr_max:.2f}')
    assert np.allclose(beta_low_sim_per_pwr_max, beta_large_offset_per_pwr_max,
                       atol=0.01), msg
    msg = ('Mean Beta must be same in both conds but is '
           f'{beta_low_sim_per_pwr - beta_large_offset_per_pwr:.2f}')
    assert np.allclose(beta_low_sim_per_pwr, beta_large_offset_per_pwr,
                       atol=0.01), msg

    diff = beta_low_sim_per_pwr_max - beta_small_exponent_sim_per_pwr_max
    msg = f'Max Beta must be same in both conds but is {diff:.2f}'
    assert np.allclose(beta_low_sim_per_pwr_max,
                       beta_small_exponent_sim_per_pwr_max, atol=0.01), msg
    msg = ('Mena Beta must be same in both conds but is '
           f'{beta_low_sim_per_pwr - beta_small_exponent_sim_per_pwr:.2f}')
    assert np.allclose(beta_low_sim_per_pwr, beta_small_exponent_sim_per_pwr,
                       atol=0.02), msg

    # %% All simulations combined

    # Settings
    yticks_abs = [0, .3, .6]
    ylim_abs = [yticks_abs[0], .7]
    yticks_norm = [0, 5, 10]
    ylim_norm = [yticks_norm[0], 11.68]

    xticks = XTICKS_FREQ_low
    xticklabels = ['', 4, '', 13, 20, 30, 45]
    lw1 = 1
    lw2 = .5
    lw3 = .7
    lw_tiny = .2
    lw1_per = .7
    lw2_per = .4

    # %% Theta increased
    figsize = (1.65, .9)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # Spectra
    ax = axes[0]
    ax.plot(freqs, beta_normal_sim, c_abs, label='spec 1', lw=lw1)
    ax.plot(freqs, beta_theta_broad_sim, 'w', lw=lw3)
    ax.plot(freqs, beta_theta_broad_sim, c_abs2, label='spec 2',
            lw=lw2)

    y_data = [beta_normal_sim_pwr, beta_normal_sim_pwr]
    ax.plot(beta_borders, y_data, c_abs, markersize=1, lw=lw1)
    y_data = [beta_theta_broad_sim_pwr, beta_theta_broad_sim_pwr]
    ax.plot(beta_borders, y_data, 'w', markersize=1, lw=lw3)
    y_data = [beta_theta_broad_sim_pwr, beta_theta_broad_sim_pwr]
    ax.plot(beta_borders, y_data, c_abs2, markersize=1, lw=lw2)

    # Axes
    ax.set_ylim(ylim_abs)
    ax.set_xticks(xticks, labels=xticklabels)
    ax.set_label(None)
    ax.set_xlim([highpass, 35])
    ax.legend(loc='upper right', borderaxespad=0.3, handlelength=1)
    ax.set_yticks(yticks_abs)
    ax.tick_params(axis='y', length=0, pad=1)

    # # Annotation
    beta_abs_diff = np.abs(beta_normal_sim_pwr - beta_theta_broad_sim_pwr)
    text = r'$\Delta \beta=$'f'{beta_abs_diff:.0f}'
    print(text, file=output_file)

    # Spectra Normalized
    ax = axes[1]
    ax.plot(freqs, beta_normal_sim_norm, c_rel, label='PSD 1', lw=lw1)

    y_data = [beta_normal_sim_norm_pwr, beta_normal_sim_norm_pwr]
    ax.plot(beta_borders, y_data, c_rel, markersize=1, lw=lw1)

    ax.plot(freqs, beta_theta_broad_sim_norm, 'w', lw=lw3)
    ax.plot(freqs, beta_theta_broad_sim_norm, c_rel2, label='PSD 2', lw=lw2)

    y_data = [beta_theta_broad_sim_norm_pwr, beta_theta_broad_sim_norm_pwr]
    ax.plot(beta_borders, y_data, 'w', markersize=1, lw=lw3)
    y_data = [beta_theta_broad_sim_norm_pwr, beta_theta_broad_sim_norm_pwr]
    ax.plot(beta_borders, y_data, c_rel2, markersize=1, lw=lw2)

    # Axes
    ax.set_ylim(ylim_norm)
    ax.set_xticks(xticks, labels=xticklabels)
    ax.set_label(None)
    ax.set_xlim([highpass, 35])
    ax.set_yticks(yticks_norm)
    ax.yaxis.tick_right()
    ax.tick_params(axis='y', length=0, pad=1)

    # Annotation
    beta_rel_diff = beta_normal_sim_norm_pwr - beta_theta_broad_sim_norm_pwr
    text = r'$\Delta \beta=$'f'{beta_rel_diff:.1f}'
    print(text, file=output_file)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    _save_fig(fig, join(fig_dir, "B1__sim_theta_beta"),
              cfg.FIG_PAPER, bbox_inches=None,
              facecolor=(1, 1, 1, 0))

    # %% Vary Peak width

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # Spectra
    ax = axes[0]
    ax.plot(freqs, beta_narrow_sim, c_abs, label='spec1', lw=lw1)

    y_data = [beta_narrow_sim_pwr, beta_narrow_sim_pwr]
    ax.plot(beta_borders, y_data, c_abs, markersize=1, lw=lw1)

    ax.plot(freqs, beta_strong_sim, 'w', lw=lw3)
    ax.plot(freqs, beta_strong_sim, c_abs2, label='spec2', lw=lw2)

    y_data = [beta_strong_sim_pwr, beta_strong_sim_pwr]
    ax.plot(beta_borders, y_data, 'w', markersize=1, lw=lw3)
    y_data = [beta_strong_sim_pwr, beta_strong_sim_pwr]
    ax.plot(beta_borders, y_data, c_abs2, markersize=1, lw=lw2)

    # Axes
    ax.set_ylim(ylim_abs)
    ax.set_xticks(xticks, labels=xticklabels)
    ax.set_label(None)
    ax.set_xlim([highpass, lowpass])
    ax.set_yticks(yticks_abs)
    ax.tick_params(axis='y', length=0, pad=1)

    # Annotation
    beta_abs_diff = beta_narrow_sim_pwr - beta_strong_sim_pwr
    text = r'$\Delta \beta=$'f'{beta_abs_diff:.2f}'
    print(text, file=output_file)

    # Spectra Normalized
    ax = axes[1]
    ax.plot(freqs, beta_narrow_sim_norm, c_rel, label='PSD 1', lw=lw1)
    ax.plot(freqs, beta_strong_sim_norm, 'w', lw=lw3)
    ax.plot(freqs, beta_strong_sim_norm, c_rel2, label='PSD 2', lw=lw2)

    y_data = [beta_narrow_sim_norm_pwr, beta_narrow_sim_norm_pwr]
    ax.plot(beta_borders, y_data, c_rel, markersize=1, lw=lw1)
    y_data = [beta_strong_sim_norm_pwr, beta_strong_sim_norm_pwr]
    ax.plot(beta_borders, y_data, 'w', markersize=1, lw=lw3)
    y_data = [beta_strong_sim_norm_pwr, beta_strong_sim_norm_pwr]
    ax.plot(beta_borders, y_data, c_rel2, markersize=1, lw=lw2)

    # Axes
    ax.set_ylim(ylim_norm)
    ax.set_xticks(xticks, labels=xticklabels)
    ax.set_label(None)
    ax.set_xlim([highpass, lowpass])
    ax.set_yticks(yticks_norm)
    ax.yaxis.tick_right()
    ax.tick_params(axis='y', length=0, pad=1)

    # Annotation
    beta_rel_diff = beta_narrow_sim_norm_pwr - beta_strong_sim_norm_pwr
    text = r'$\Delta \beta=$'f'{beta_rel_diff:.1f}'
    print(text, file=output_file)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    _save_fig(fig, join(fig_dir, "B2__sim_width_beta"), cfg.FIG_PAPER,
              bbox_inches=None, facecolor=(1, 1, 1, 0))

    # %% Elevated broadband

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # Spectra
    ax = axes[0]
    ax.plot(freqs, beta_low_sim_ap, 'k-', label=None, lw=lw_tiny)
    ax.plot(freqs, beta_low_sim, c_abs, label='spec1', lw=lw1)
    ax.plot(freqs, beta_large_offset_sim_ap, 'k-', label=None, lw=lw_tiny)
    ax.plot(freqs, beta_large_offset_sim, 'w', lw=lw3)
    ax.plot(freqs, beta_large_offset_sim, c_abs2, label='spec2',
            lw=lw2)

    y_data = [beta_low_sim_pwr, beta_low_sim_pwr]
    ax.plot(beta_borders, y_data, c_abs, markersize=1, lw=lw1)
    y_data = [beta_large_offset_sim_pwr, beta_large_offset_sim_pwr]
    ax.plot(beta_borders, y_data, 'w', markersize=1, lw=lw3)
    y_data = [beta_large_offset_sim_pwr, beta_large_offset_sim_pwr]
    ax.plot(beta_borders, y_data, c_abs2, markersize=1, lw=lw2)
    ax.plot(fm.freqs[beta_mask_fm], fm_per_low, c_per, lw=lw1_per)
    ax.plot(fm.freqs[beta_mask_fm], fm_per_large_offset, c_per,
            lw=lw2_per)
    # Plot periodic power
    freq1 = freq2 = beta_freq
    y_data = [ap_pwr_low, fm_per_low.max()]
    ax.plot([freq1, freq1], y_data, c_per, lw=lw1_per)
    y_data = [ap_pwr_large, fm_per_large_offset[beta_borders[1] - beta_freq]]
    ax.plot([freq2, freq2], y_data, c_per, lw=lw1_per)

    # Axes
    ax.set_ylim(ylim_abs)
    ax.set_xticks(xticks, labels=xticklabels)
    ax.set_label(None)
    ax.set_xlim([highpass, lowpass])
    ax.set_yticks(yticks_abs)
    ax.tick_params(axis='y', length=0, pad=1)

    # Annotation
    beta_abs_diff = beta_low_sim_pwr - beta_large_offset_sim_pwr
    text = r'Absolute: $\Delta \beta=$'f'{beta_abs_diff:.2f}'
    print(text, file=output_file)
    beta_per_diff = beta_low_sim_per_pwr_max - beta_large_offset_per_pwr_max
    text = r'Periodic: $\Delta \beta=$'f'{beta_per_diff:.2f}'
    print(text, file=output_file)

    # Spectra Normalized
    ax = axes[1]
    ax.plot(freqs, beta_low_sim_norm, c_rel, label='PSD 1', lw=lw1)
    ax.plot(freqs, beta_large_offset_sim_norm, 'w', lw=lw3)
    ax.plot(freqs, beta_large_offset_sim_norm, c_rel2, label='PSD 2', lw=lw2)

    y_data = [beta_low_sim_norm_pwr, beta_low_sim_norm_pwr]
    ax.plot(beta_borders, y_data, c_rel, markersize=1, lw=lw1)
    y_data = [beta_large_offset_sim_norm_pwr, beta_large_offset_sim_norm_pwr]
    ax.plot(beta_borders, y_data, 'w', markersize=1, lw=lw3)
    y_data = [beta_large_offset_sim_norm_pwr, beta_large_offset_sim_norm_pwr]
    ax.plot(beta_borders, y_data, c_rel2, markersize=1, lw=lw2)

    # Axes
    ax.set_ylim(ylim_norm)
    ax.set_xticks(xticks, labels=xticklabels)
    ax.set_label(None)
    ax.set_xlim([highpass, lowpass])
    ax.set_yticks(yticks_norm)
    ax.yaxis.tick_right()
    ax.tick_params(axis='y', length=0, pad=1)

    # Annotation
    beta_rel_diff = beta_low_sim_norm_pwr - beta_large_offset_sim_norm_pwr
    text = r'Relative: $\Delta \beta=$'f'{beta_rel_diff:.1f}'
    print(text, file=output_file)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    _save_fig(fig, join(fig_dir, "F1__sim_broadband_shift"), cfg.FIG_PAPER,
              bbox_inches=None, facecolor=(1, 1, 1, 0))

    # %% Reduced 1/f exponent

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # Spectra
    ax = axes[0]
    ax.plot(freqs, beta_low_sim_ap, 'k-', label=None, lw=lw_tiny)
    ax.plot(freqs, beta_low_sim, c_abs, label='spec1', lw=lw1)
    ax.plot(freqs, beta_small_exponent_sim_ap, 'k-', label=None, lw=lw_tiny)
    ax.plot(freqs, beta_small_exponent_sim, 'w', lw=lw3)
    ax.plot(freqs, beta_small_exponent_sim, c_abs2, label='spec2', lw=lw2)

    y_data = [beta_low_sim_pwr, beta_low_sim_pwr]
    ax.plot(beta_borders, y_data, c_abs, markersize=1, lw=lw1)
    y_data = [beta_small_exponent_sim_pwr, beta_small_exponent_sim_pwr]
    ax.plot(beta_borders, y_data, 'w', markersize=1, lw=lw3)
    y_data = [beta_small_exponent_sim_pwr, beta_small_exponent_sim_pwr]
    ax.plot(beta_borders, y_data, c_abs2, markersize=1, lw=lw2)
    ax.plot(fm.freqs[beta_mask_fm], fm_per_low, c_per, lw=lw1_per)
    ax.plot(fm.freqs[beta_mask_fm], fm_per_small_exponent, c_per, lw=lw2_per)

    y_data = [ap_pwr_low, fm_per_low.max()]
    ax.plot([beta_freq, beta_freq], y_data, c_per, lw=lw1_per)
    y_data = [ap_pwr_small_exponent, fm_per_small_exponent.max()]
    ax.plot([beta_freq, beta_freq], y_data, c_per, lw=lw1_per)

    # Axes
    ax.set_ylim(ylim_abs)
    ax.set_xticks(xticks, labels=xticklabels)
    ax.set_label(None)
    ax.set_xlim([highpass, lowpass])
    ax.set_yticks(yticks_abs)
    ax.tick_params(axis='y', length=0, pad=1)

    # Annotation
    beta_abs_diff = beta_low_sim_pwr - beta_small_exponent_sim_pwr
    text = r'Absolute: $\Delta \beta=$'f'{beta_abs_diff:.2f}'
    print(text, file=output_file)
    beta_per_diff = (beta_low_sim_per_pwr_max
                     - beta_small_exponent_sim_per_pwr_max)
    text = r'Periodic: $\Delta \beta=$'f'{beta_per_diff:.2f}'
    print(text, file=output_file)

    # Spectra Normalized
    ax = axes[1]
    ax.plot(freqs, beta_low_sim_norm, c_rel, label='PSD 1', lw=lw1)
    ax.plot(freqs, beta_small_exponent_sim_norm, 'w', lw=lw3)
    ax.plot(freqs, beta_small_exponent_sim_norm, c_rel2, label='PSD 2', lw=lw2)

    y_data = [beta_low_sim_norm_pwr, beta_low_sim_norm_pwr]
    ax.plot(beta_borders, y_data, c_rel, markersize=1, lw=lw1)
    y_data = [beta_small_exponent_sim_norm_pwr,
              beta_small_exponent_sim_norm_pwr]
    ax.plot(beta_borders, y_data, 'w', markersize=1, lw=lw3)
    y_data = [beta_small_exponent_sim_norm_pwr,
              beta_small_exponent_sim_norm_pwr]
    ax.plot(beta_borders, y_data, c_rel2, markersize=1, lw=lw2)

    # Axes
    ax.set_ylim(ylim_norm)
    ax.set_xticks(xticks, labels=xticklabels)
    ax.set_label(None)
    ax.set_xlim([highpass, lowpass])
    ax.set_yticks(yticks_norm)
    ax.yaxis.tick_right()
    ax.tick_params(axis='y', length=0, pad=1)

    # Annotation
    beta_rel_diff = beta_low_sim_norm_pwr - beta_small_exponent_sim_norm_pwr
    text = r'Relative: $\Delta \beta=$'f'{beta_rel_diff:.1f}'
    print(text, file=output_file)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    _save_fig(fig, join(fig_dir, "F2__sim_exponent_beta"), cfg.FIG_PAPER,
              bbox_inches=None, facecolor=(1, 1, 1, 0))
