"""Helping functions."""
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.ticker import ScalarFormatter
from mne.io import read_raw
from mne_bids import find_matching_paths

import scripts.config as cfg
from scripts.plot_figures.settings import (BANDS, FONTSIZE_S, LINEWIDTH_AXES,
                                           XTICKS_FREQ_low,
                                           XTICKS_FREQ_low_labels)
from scripts.utils_plot import (_add_band, _add_band_annotations, _save_fig,
                                explode_df)


def exemplary_spectrum_mini(df_abs, fig_dir=None, prefix=''):
    df_abs = df_abs[(df_abs.project != 'all')
                    & (df_abs.cond.isin(['off', 'on']))
                    & (df_abs.sub_hemi == 'NeuEL031_R')]
    assert len(df_abs) == 2, 'Ambiguous selection'
    psd_off = df_abs[df_abs.cond == 'off'].psd.iloc[0]
    psd_on = df_abs[df_abs.cond == 'on'].psd.iloc[0]
    freqs = df_abs.psd_freqs.iloc[0]
    mask = (freqs >= 2) & (freqs <= 35)
    fig, ax = plt.subplots(figsize=(.7, .6))
    ax.plot(freqs[mask], psd_off[mask], color=cfg.COLOR_OFF, lw=.75)
    ax.plot(freqs[mask], psd_on[mask], color=cfg.COLOR_ON, lw=.75)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)

    # Plot scale y-scale
    x = 0
    y = 0
    yscale = 1  # mu V^2/Hz
    xscale = 15  # Hz
    lw = 0.5
    ax.vlines(x, ymin=y, ymax=yscale + y, color='k', lw=lw,
              transform=ax.transData)

    # Plot scale x-scale
    ax.hlines(y, x, x + xscale, color='k', lw=lw,
              transform=ax.transData)

    plt.tight_layout()
    fpath = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, f'{prefix}exemplary_spectrum', fpath, transparent=True,
              bbox_inches=None)


def exemplary_spectrum_mini_kinds(dataframes, fig_dir=None, prefix=''):
    df_norm = dataframes['df_norm']
    df_abs = dataframes['df_abs']
    df_per = dataframes['df_per']

    c_rel = cfg.COLOR_DIC['normalized']
    c_abs = cfg.COLOR_DIC['absolute']
    c_per = cfg.COLOR_DIC['periodic']
    c_ap = cfg.COLOR_DIC['periodicAP']

    freq_low = 2
    freq_high = 35

    df_abs = df_abs[(df_abs.project != 'all')
                    & (df_abs.cond.isin(['off']))
                    & (df_abs.sub_hemi == 'NeuEL031_R')]
    assert len(df_abs) == 1, 'Ambiguous selection'
    psd_abs = df_abs.psd.iloc[0]
    freqs = df_abs.psd_freqs.iloc[0]
    mask = (freqs >= freq_low) & (freqs <= freq_high)
    freqs = freqs[mask]
    psd_abs = psd_abs[mask]

    df_norm = df_norm[(df_norm.project != 'all')
                      & (df_norm.cond.isin(['off']))
                      & (df_norm.sub_hemi == 'NeuEL031_R')]
    assert len(df_norm) == 1, 'Ambiguous selection'
    psd_norm = df_norm.psd.iloc[0]
    psd_norm = psd_norm[mask]

    df_per = df_per[(df_per.project != 'all')
                    & (df_per.cond.isin(['off']))
                    & (df_per.sub_hemi == 'NeuEL031_R')]
    assert len(df_per) == 1, 'Ambiguous selection'
    per_fit = df_per.fm_psd_peak_fit.iloc[0]
    ap_fit = df_per.fm_psd_ap_fit.iloc[0]
    fm_freqs = df_per.fm_freqs.iloc[0]
    mask_fm = (fm_freqs >= freq_low) & (fm_freqs <= freq_high)
    fm_freqs = fm_freqs[mask_fm]
    ap_fit = ap_fit[mask_fm]
    per_fit = per_fit[mask_fm]

    lw_lines = .75

    fig, ax = plt.subplots(figsize=(1.6, 1.5))

    ax.plot(freqs, psd_abs, color='k', lw=lw_lines)
    ax.plot(fm_freqs, ap_fit + per_fit, c=c_per, ls="-",
            label="Periodic fit", lw=.5)
    ax.plot(fm_freqs, ap_fit, c=c_ap, ls="-",
            label="Aperiodic fit", lw=.5)

    # Fill between aperiodic and zero
    # Add white background to enable correct colors
    ax.fill_between(fm_freqs, ap_fit + per_fit, 0, color='w', alpha=1)
    # Add periodic and aperiodic shades
    ax.fill_between(fm_freqs, ap_fit, 0, color=c_ap, alpha=.3)
    ax.fill_between(fm_freqs, ap_fit + per_fit, ap_fit, color=c_per, alpha=.3)

    ax_twin = ax.twinx()
    ax_twin.plot(freqs, psd_norm, color=c_rel, lw=0)

    ax.axis('off')
    ax_twin.axis('off')
    ymin_norm = -0.45
    ax_twin.set_ylim([ymin_norm, 9.5])

    # Plot absolute scale y-scale
    x_min = 0
    yscale_abs = 1  # mu V^2/Hz
    lw_scale = .75
    ax.vlines(x_min, ymin=0, ymax=yscale_abs, color=c_abs,
              lw=lw_scale, transform=ax.transData)

    # Plot relative scale y-scale
    yscale_norm = 7  # %
    ax_twin.vlines(freq_high + 3, ymin=0, ymax=yscale_norm, color=c_rel,
                   lw=lw_scale, transform=ax_twin.transData)

    # # Plot x-scale
    # ax.hlines(0, freq_low, freq_high, color='k',
    #           lw=lw_scale, transform=ax.transData)

    plt.tight_layout()
    fpath = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, f'{prefix}exemplary_spectrum_kinds', fpath,
              transparent=True, bbox_inches=None)


def exemplary_time_series(fig_dir=None, prefix=''):
    # Load preprocessed data
    kwargs = dict(subjects='NeuEL031', root='derivatives/preprocessed',
                  extensions='.fif')
    bids_path_off = find_matching_paths(**kwargs, sessions='EcogLfpMedOff01')
    bids_path_on = find_matching_paths(**kwargs, sessions='EcogLfpMedOn01')
    assert len(bids_path_off) == len(bids_path_on) == 1, 'Ambiguous selection'
    bids_path_off = bids_path_off[0]
    bids_path_on = bids_path_on[0]

    # Load
    raw_off = read_raw(bids_path_off.fpath, verbose=0)
    raw_on = read_raw(bids_path_on.fpath, verbose=0)

    raw_off.pick(['LFP_R_2-4_STN_MT'])
    raw_on.pick(['LFP_R_2-4_STN_MT'])

    # Select data
    duration = 3  # seconds
    start_off = 19.5  # seconds
    start_on = 1  # seconds
    stop_off = start_off + duration
    stop_on = start_on + duration
    raw_off.crop(tmin=start_off, tmax=stop_off)
    raw_on.crop(tmin=start_on, tmax=stop_on)
    data_off, times = raw_off.get_data(return_times=True)
    data_off, times = np.squeeze(data_off), np.squeeze(times)
    data_on, _ = raw_on.get_data(return_times=True)
    data_on = np.squeeze(data_on)

    # Convert V to muV
    data_off *= (1e6)
    data_on *= (1e6)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(4.7, .7), sharex=True, sharey=True)

    ax = axes[0]
    ax.axis('off')
    # c_record = '#2e3180'  # same as MATLAB
    ax.plot(times, data_off, color=cfg.COLOR_OFF, lw=.15)

    ax = axes[1]
    ax.axis('off')
    ax.plot(times, data_on, color=cfg.COLOR_ON, lw=.15)

    # Plot scale
    x = -.03
    y = -15
    yscale = 20  # muV
    xscale = .5  # s
    # Plot scale y-scale
    ax.vlines(x, ymin=y, ymax=yscale + y, color='k', lw=LINEWIDTH_AXES,
              transform=ax.transData)
    ax.text(x - 0.01,
            # y + yscale/3,
            y + yscale,
            rf'{yscale} $\mu V$', ha='right',
            va='center', fontsize=4)

    # Plot scale x-scale
    ax.hlines(y, x, x + xscale, color='k', lw=LINEWIDTH_AXES,
              transform=ax.transData)
    ax.text(
        # np.mean([x, x + xscale]),
        x + xscale,
        y - 3,
        # rf'{xscale} $s$',
        rf'{xscale*1000:.0f} $ms$',
        ha='center', va='top', fontsize=4)
    plt.tight_layout()
    plt.subplots_adjust(hspace=-.3)
    fpath = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, f'{prefix}exemplary_time_series', fpath, transparent=True,
              bbox_inches='tight')


def representative_spectrum(df, kind, fig_dir=None,
                            use_peak_power=False,
                            xscale="linear", yscale="log", ylabel=None,
                            xlabel=None, figsize=(1.5, 1.5),
                            leg_kwargs={}, output_file=None,
                            legend=True, ylim=None, sub_hemis=['NeuEL031_R'],
                            prefix=''):
    """Find a nice exemplary power spectrum for paper and talk.

    Should fulfill in ON condition:
    - enhanced theta
    - strongly reduced low beta
    - slightly reduced high beta
    - beta OFF-ON somewhat different in normalized and absolute
    """
    fm_params = False if kind == 'normalized' else 'broad'
    df = df[df.cond.isin(['off', 'on']) & (df.project != 'all')]
    pwr = 'max' if use_peak_power else 'mean'
    pwr_cols = [f'{band}_abs_{pwr}' for band in BANDS]
    pwr_cols += [f'{band}_fm_mean' for band in BANDS]
    pwr_cols += [f'{band}_fm_band_aperiodic_log' for band in BANDS]
    freq_cols = [f'{band}_abs_max_freq' for band in BANDS]
    keep_cols = ['fm_freqs', 'fm_psd_ap_fit', 'fm_psd_peak_fit',
                 'fm_offset', 'fm_exponent'] + pwr_cols + freq_cols
    normalization_range = (5, 95)
    if xscale == 'log':
        xlim = (1, 100) if kind == 'periodic' else normalization_range
        if kind == 'normalized':
            ylim = None
        elif kind == 'absolute':
            ylim = (0.007, 1.5)
        elif kind == 'periodic':
            ylim = (0.007, 5)
        yticks = None
    else:
        xmin = cfg.BANDS[BANDS[0]][0]  # otherwise delta band is cut off
        xmax = XTICKS_FREQ_low[-1]  # ok to cut off low gamma
        xlim = (xmin, xmax)
        if kind == 'normalized':
            ylim = (0, 10)
            yticks = None
        elif kind == 'absolute':
            ylim = (0, 1.2)
            yticks = np.arange(0, 1.3, .2)
        elif kind == 'periodic':
            ylim = (0, 1.2)
            yticks = np.arange(0, 1.3, .2)
    for sub_hemi in sub_hemis:
        fname = f'{prefix}{kind}_{sub_hemi}'
        df_filt = df[(df.sub_hemi == sub_hemi)]
        if len(df_filt) < 2:
            continue  # both conditions must be present
        if xscale == 'log':
            if kind == 'periodic':
                assert df_filt.fm_fit_range.nunique() == 1
                fit_range = df_filt.fm_fit_range.unique()[0]
                xticks = fit_range
            else:
                xticks = normalization_range
        else:
            xticks = XTICKS_FREQ_low

        df_filt = explode_df(df_filt, psd='psd', fm_params=fm_params,
                             keep_cols=keep_cols)
        fpath = join(fig_dir, fname + '_' + xscale)
        plot_psd_df_annotated(df_filt, psd='psd', save_name=fpath,
                              use_peak_power=use_peak_power,
                              xlim=xlim, leg_kwargs=leg_kwargs,
                              ylim=ylim, figsize=figsize,
                              ylabel=ylabel, xlabel=xlabel,
                              output_file=output_file,
                              xscale=xscale, yscale=yscale, legend=legend,
                              bands=BANDS, xticks=xticks, yticks=yticks,
                              kind=kind)


def plot_psd_df_annotated(df, freqs="psd_freqs", psd="asd", hue="cond",
                          xscale="log", yscale="log",
                          use_peak_power=False, yticks=None,
                          bands=None, xlabel=None, ylabel=None, xlim=None,
                          ylim=None, ax_kwargs={}, kind=None,
                          save_name=None, col=None, legend=True, xticks=None,
                          col_order=None, row=None, add_band_colors=False,
                          output_file=None, leg_kwargs={}, figsize=(1.5, 1.5)):
    if ylabel is None:
        if kind in ['absolute', 'periodic']:
            ylabel = r"Spectra [$\mu V^2/Hz$]"
        else:
            ylabel = "Normalized spectra [%]"
    elif ylabel is False:
        ylabel = None
    if xlim:
        xmask = ((df[freqs] >= xlim[0]) & (df[freqs] <= xlim[1]))
        df = df.loc[xmask]
    global c_off, c_on
    if kind == 'normalized':
        c_off = cfg.COLOR_DIC['all']
        c_on = cfg.COLOR_DIC['all2']
    elif kind == 'absolute':
        c_off = cfg.COLOR_DIC[kind]
        c_on = cfg.COLOR_DIC[kind + '2']
    elif kind in ['periodic']:
        c_off = cfg.COLOR_DIC['all']
        c_on = cfg.COLOR_DIC['all2']
    if kind == 'periodic':
        hue_order = ['off']
        palette = [c_off]
    else:
        hue_order = ['off', 'on']
        palette = [c_off, c_on]
    g = sns.relplot(data=df, x=freqs, y=psd, hue=hue, hue_order=hue_order,
                    kind="line", palette=palette, col=col, col_order=col_order,
                    row=row, zorder=5)
    g.figure.set_size_inches(figsize)
    ax = g.axes[0, 0]
    g._legend.remove()
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        labels = [cfg.COND_DICT[l] for l in labels]
        ax.legend(handles, labels, title='Levodopa', loc='upper right',
                  handlelength=1, **leg_kwargs)

    assert len(g.axes) == 1, "Only one ax supported"
    if xscale == "log":
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        xticklabels = [f"{x} Hz" for x in xticks]
        # Add frequency bands
        for xtick in XTICKS_FREQ_low:
            ax.axvline(x=xtick, color='w', lw=LINEWIDTH_AXES)
        ax.tick_params(axis='y', pad=0)
    else:
        xticklabels = XTICKS_FREQ_low_labels
    if xlabel is None:
        xlabel = "Frequency [Hz]"
    elif xlabel is False:
        xlabel = None
    add_integral_str(df, freqs, psd, ax, kind, xscale, c_off, c_on)
    g.set(xscale=xscale, yscale=yscale, **ax_kwargs,
          xticks=xticks, xticklabels=xticklabels,
          ylabel=ylabel, xlabel=xlabel, ylim=ylim, xlim=xlim)
    if yticks is not None:
        ax.set_yticks(yticks)
    annotate_peaks(df, ax, xscale, kind, use_peak_power=use_peak_power,
                   output_file=output_file)
    if add_band_colors:
        _add_band(bands, g)
    if xscale == 'linear':
        _add_band_annotations(bands, g)
    else:
        if kind == 'periodic':
            _add_band_annotations(bands, g, short=True)
        else:
            _add_band_annotations(bands[1:], g, short=True)
    plt.tight_layout()
    _save_fig(g, save_name, cfg.FIG_PAPER, transparent=False,
              bbox_inches=None)


def annotate_peaks(df, ax, xscale, kind, use_peak_power=False,
                   output_file=None):
    textcolor = '#262626'
    text_box = dict(facecolor='#eaeaf2', pad=.4, boxstyle='round',
                    edgecolor='#cccccc')
    z_front = 6
    z_back_middle = 2
    dotted_line = dict(color=textcolor, lw=LINEWIDTH_AXES, ls=':',
                       zorder=z_front)
    text_kwargs = dict(bbox=text_box, color=textcolor, va='center')
    if xscale == 'log':
        if kind in ['normalized', 'absolute']:
            _add_broadband_shading(df, ax, kind)
        elif kind == 'periodic':
            _annotate_log_periodic(df, ax, textcolor, text_box, z_front,
                                   dotted_line)
    elif xscale == 'linear':
        arrow_range = dict(facecolor=textcolor, shrinkA=0, shrinkB=0,
                           arrowstyle='|-|', mutation_scale=1,
                           edgecolor=textcolor)
        if kind == 'normalized':
            _annotate_norm_and_abs(df, ax, kind, textcolor, z_front,
                                   dotted_line, arrow_range,
                                   ['beta_low'], use_peak_power=use_peak_power,
                                   output_file=output_file)
        elif kind == 'absolute':
            _annotate_norm_and_abs(df, ax, kind, textcolor, z_front,
                                   dotted_line, arrow_range,
                                   ['beta_low'], output_file=output_file)
        elif kind == 'periodic':
            _annotate_lin_per(df, ax, textcolor, z_front, z_back_middle,
                              text_kwargs, use_peak_power=False)


def _add_broadband_shading(df, ax, kind):
    # add shading
    freqs = df[df.cond == 'off'].psd_freqs.to_numpy(dtype=float)
    broadband_mask = ((freqs >= 5) & (freqs <= 95))
    psd_off = df[df.cond == 'off'].psd.to_numpy(dtype=float)
    psd_on = df[df.cond == 'on'].psd.to_numpy(dtype=float)
    # Fill broadband
    # Add white background to enable correct colors
    ax.fill_between(freqs[broadband_mask], psd_off[broadband_mask], 0,
                    color='w', alpha=1, zorder=0)
    # Add off and on shades
    if kind == 'normalized':
        alpha_off = 0.15
        alpha_on = 0.3
        lower = 0
    elif kind == 'absolute':
        alpha_off = alpha_on = 0.2
        lower = psd_on[broadband_mask]
    # off shading
    ax.fill_between(freqs[broadband_mask], psd_off[broadband_mask], lower,
                    color=c_off, alpha=alpha_off)
    # on shading background
    ax.fill_between(freqs[broadband_mask], psd_on[broadband_mask], 0,
                    color='w', alpha=1, zorder=0)
    # on shading
    ax.fill_between(freqs[broadband_mask], psd_on[broadband_mask], 0,
                    color=c_on, alpha=alpha_on)
    # xticks
    xmin, xmax = ax.get_xlim()
    log_center = 10**((np.log10(xmin) + np.log10(xmax)) / 2 )
    ax.set_xticks([log_center], minor=True)
    ax.set_xticklabels([r'$-$ Normalization Range $-$'], minor=True, y=-.02,
                       color='grey')
    # yticks
    yticks = ax.get_yticks()
    ylim = ax.get_ylim()
    yticks = yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])]
    for ytick in yticks:
        ax.axhline(y=ytick, color='w', lw=LINEWIDTH_AXES)


def _annotate_norm_and_abs(df, ax, kind, textcolor, z_front, dotted_line,
                           arrow_range, bands, use_peak_power=False,
                           output_file=None):
    # Extract peak points
    if use_peak_power:
        pwr = 'max'
        pwr_str = 'Total'
    else:
        pwr = 'mean'
        pwr_str = 'Mean'
    for band in bands:
        peak_power_off = df[df.cond == 'off'][f'{band}_abs_{pwr}'].values[0]
        peak_power_on = df[df.cond == 'on'][f'{band}_abs_{pwr}'].values[0]
        if use_peak_power:
            peak_freq_off = df[df.cond == 'off'][
                f'{band}_abs_max_freq'].values[0]
            peak_freq_on = df[df.cond == 'on'][
                f'{band}_abs_max_freq'].values[0]
        else:
            band_borders = cfg.BANDS[band]
            peak_freq_off = peak_freq_on = band_borders[-1]

        # Indicate Peaks as points
        if use_peak_power:
            peak_points = dict(marker='o', markersize=.2, c=textcolor,
                               zorder=z_front)
            ax.plot(peak_freq_off, peak_power_off, **peak_points)
            ax.plot(peak_freq_on, peak_power_on, **peak_points)

        # Annotate power

        # Text
        units = '\n'r'$\mu V^2/Hz$' if kind == 'absolute' else '%'
        if band == 'beta_low':
            if output_file:
                power_diff_str = f'{peak_power_off - peak_power_on:.1f}{units}'
                pwr_str = (f'{pwr_str} {cfg.BAND_NAMES_GREEK[band]} '
                           f'power off-on={power_diff_str}')
                print(pwr_str, file=output_file)
            # Vertical bracket
            x_arrow_pwr = 29.5 if kind == 'absolute' else 28.8
            ax.annotate('', xy=(x_arrow_pwr, peak_power_off),
                        xytext=(x_arrow_pwr, peak_power_on),
                        arrowprops=arrow_range)
            # Dashed lines
            ax.hlines(peak_power_off, peak_freq_off, x_arrow_pwr,
                      **dotted_line)
            ax.hlines(peak_power_on, peak_freq_on, x_arrow_pwr, **dotted_line)
            # Mean power line
            if not use_peak_power:
                ax.hlines(peak_power_off, *band_borders,
                          color=c_off,
                          lw=0.75)
                ax.hlines(peak_power_on, *band_borders,
                          color=c_on,
                          lw=0.75)
                # add shading
                freqs = df[df.cond == 'off'].psd_freqs.to_numpy(dtype=float)
                beta_mask = ((freqs >= band_borders[0])
                             & (freqs <= band_borders[1]))
                psd_off = df[df.cond == 'off'].psd.to_numpy(dtype=float)
                psd_on = df[df.cond == 'on'].psd.to_numpy(dtype=float)
                # Fill beta
                # Add white background to enable correct colors
                ax.fill_between(freqs[beta_mask], psd_off[beta_mask], 0,
                                color='w', alpha=1)
                # Add off and on shades
                alpha = 0.2 if kind == 'normalized' else .3
                ax.fill_between(freqs[beta_mask], psd_off[beta_mask], 0,
                                color=c_off, alpha=alpha)
                ax.fill_between(freqs[beta_mask], psd_on[beta_mask], 0,
                                color=c_on, alpha=alpha)


def _annotate_lin_per(df, ax, textcolor, z_front, z_back_middle, text_kwargs,
                      use_peak_power=False, annotate_text=False):
    band = 'beta_low'
    cond = 'off'
    df_cond = df[df.cond == cond]
    c_str = df.project.unique()[0]
    c_str += '2' if cond == 'on' else ''

    fm_freqs = df_cond.fm_freqs.iloc[0]
    ap_fit = df_cond.fm_psd_ap_fit.iloc[0]
    per_fit = df_cond.fm_psd_peak_fit.iloc[0]
    total_fit = ap_fit + per_fit

    # Plot 1/f exponent
    c_ap = cfg.COLOR_DIC['periodicAP']
    ax.plot(fm_freqs, ap_fit, c=c_ap, ls="-", lw=.5,
            label="Aperiodic fit", zorder=z_front)
    # Plot periodic fit
    c_per = cfg.COLOR_DIC['periodic']
    ax.plot(fm_freqs, total_fit, c=c_per, ls="-", lw=.5,
            zorder=z_front, label="Periodic fit")

    # Total power
    x_text = 32  # Hz
    peak_freq = df_cond[f'{band}_abs_max_freq'].values[0]
    band_borders = cfg.BANDS[band]
    if use_peak_power:
        x_arrow = peak_freq
    else:
        x_arrow = band_borders[1]

    beta_mask = (fm_freqs >= band_borders[0]) & (fm_freqs <= band_borders[1])
    # Fill between aperiodic and zero
    # Add white background to enable correct colors
    ax.fill_between(fm_freqs[beta_mask], total_fit[beta_mask], 0,
                    color='w', alpha=1)
    # Add periodic and aperiodic shades
    ax.fill_between(fm_freqs[beta_mask], ap_fit[beta_mask], 0,
                    color=c_ap, alpha=.3)
    ax.fill_between(fm_freqs[beta_mask], total_fit[beta_mask],
                    ap_fit[beta_mask], color=c_per, alpha=.3)

    freq_mask = fm_freqs == peak_freq
    arr_pwr = dict(facecolor=textcolor, shrinkA=0, shrinkB=1.5,
                   edgecolor=textcolor, arrowstyle='-|>',
                   mutation_scale=3)
    arr_kwargs = dict(arrowprops=arr_pwr, zorder=z_back_middle)

    # Periodic power
    if use_peak_power:
        total_pwr = ap_fit[freq_mask] + per_fit[freq_mask]
        ax.hlines(total_pwr, *band_borders, color=c_off, lw=LINEWIDTH_AXES)
        ax.annotate('', xy=(x_arrow, total_pwr), xytext=(x_text, total_pwr),
                    **arr_kwargs)
        text_kwargs_tot = text_kwargs.copy()
        text_kwargs_tot['color'] = 'k'
        ax.text(s='Total\npower', x=x_text, y=total_pwr, **text_kwargs_tot)
        height_ap = ap_fit[freq_mask] / 2
        y_text_ap = height_ap * 6
        y_periodic = (total_pwr + y_text_ap) / 2
        y_text_per = y_periodic
        text_kwargs_per = text_kwargs.copy()
        text_kwargs_per['color'] = c_per
        ax.text(s='Periodic\npower', x=x_text, y=y_text_per, **text_kwargs_per)
        ax.annotate("", xy=(x_arrow, y_periodic), xytext=(x_text, y_periodic),
                    **arr_kwargs)
        ax.vlines(peak_freq, ap_fit[freq_mask], total_pwr, color=c_per)

        # Aperiodic power
        text_kwargs_ap = text_kwargs.copy()
        text_kwargs_ap['color'] = c_ap
        ax.annotate('Aperiodic\npower', xytext=(x_text, y_text_ap),
                    xy=(x_arrow, height_ap), **text_kwargs_ap, **arr_kwargs)
        ax.vlines(peak_freq, 0, ap_fit[freq_mask], color=c_ap, lw=1.5)

    else:
        y_periodic = df_cond[f'{band}_fm_mean'].values[0]
        height_ap = 10**df_cond[f'{band}_fm_band_aperiodic_log'].values[0]
        total_pwr = height_ap + y_periodic
        ax.hlines(total_pwr, *band_borders, color=c_per, lw=.75)
        ax.hlines(height_ap, *band_borders, color=c_ap, lw=.75)
        if annotate_text:
            text_kwargs_tot = text_kwargs.copy()
            text_kwargs_tot['color'] = 'k'
            y_text_tot = total_pwr * 1.5
            ax.text(s=f'Total\n{cfg.BAND_NAMES_GREEK[band]}',
                    x=x_text, y=y_text_tot, **text_kwargs_tot)
            y_text_ap = height_ap * 2.7
            y_text_per = y_periodic * 1.1
            text_kwargs_per = text_kwargs.copy()
            text_kwargs_per['color'] = c_per
            ax.text(s=f'Periodic\n{cfg.BAND_NAMES_GREEK[band]}', x=x_text,
                    y=y_text_per, **text_kwargs_per)

            # Aperiodic power
            text_kwargs_ap = text_kwargs.copy()
            text_kwargs_ap['color'] = c_ap
            ax.text(s=f'Aperiodic\n{cfg.BAND_NAMES_GREEK[band]}', x=x_text,
                    y=y_text_ap, **text_kwargs_ap)
            ax.annotate(f'Aperiodic\n{cfg.BAND_NAMES_GREEK[band]}',
                        xytext=(x_text, y_text_ap), xy=(x_arrow, height_ap),
                        **text_kwargs_ap, **arr_kwargs)


def _annotate_log_periodic(df, ax, textcolor, text_box, z_front, dotted_line,
                           annotate_text=False):
    cond = 'off'
    df_cond = df[df.cond == cond]
    c_str = df.project.unique()[0]
    c_str += '2' if cond == 'on' else ''
    c_per = cfg.COLOR_DIC['periodic']

    fm_freqs = df_cond.fm_freqs.iloc[0]
    ap_fit = df_cond.fm_psd_ap_fit.iloc[0]
    per_fit = df_cond.fm_psd_peak_fit.iloc[0]
    total_fit = ap_fit + per_fit

    # Plot 1/f exponent
    c_ap = cfg.COLOR_DIC['periodicAP']
    ax.plot(fm_freqs, ap_fit, c=c_ap, ls="-", zorder=z_front,
            label="Aperiodic fit")
    # Plot periodic fit
    c_per = cfg.COLOR_DIC['periodic']
    ax.plot(fm_freqs, total_fit, c=c_per, ls="-", lw=.5,
            zorder=z_front, label="Periodic fit")
    # Annotate aperiodic broadband shading
    ax.fill_between(fm_freqs, ap_fit, 0, color='w', alpha=1)
    ax.fill_between(fm_freqs, ap_fit, 0, color=c_ap, alpha=0.3)

    # Interpolate offset
    offset = df_cond.fm_offset.unique()[0]
    exponent = df_cond.fm_exponent.unique()[0]
    x = np.arange(1, fm_freqs[0] + 1)
    ax.plot(x, offset * 1 / x**exponent, c=c_ap, ls=":",
            zorder=z_front)

    # 1/f Offset
    arr_offset = dict(facecolor=textcolor, shrinkA=0, shrinkB=1.5,
                      edgecolor=textcolor, arrowstyle='-|>', mutation_scale=3)
    kwargs_text = dict(bbox=text_box, color=textcolor, va='center', ha='left')
    freq_exp = (22 + 14) / 2
    if annotate_text:
        ax.annotate('Offset',
                    xy=(x[0], offset),
                    xytext=(2.5, offset),
                    arrowprops=arr_offset, **kwargs_text)

        # 1/f Exponent
        arr_exponent = arr_offset.copy()
        arr_exponent.update(dict(shrinkA=.5, shrinkB=.5))
        ax.annotate('1/f Exponent',
                    xy=(14, ap_fit[fm_freqs == 22]),
                    xytext=(2.5, ap_fit[fm_freqs == 35]),
                    arrowprops=arr_exponent, **kwargs_text)
        ax.hlines(ap_fit[fm_freqs == 22], 14, 22, **dotted_line)
        ax.vlines(14, ap_fit[fm_freqs == 14], ap_fit[fm_freqs == 22],
                  **dotted_line)
        kwargs_text2 = dict(fontsize=FONTSIZE_S-2, color=textcolor)
        ax.text(14, ap_fit[fm_freqs == 20], r'$\Delta y$',
                **kwargs_text2, va='center', ha='right')
        ax.text(freq_exp, ap_fit[fm_freqs == 23], r'$\Delta x$',
                **kwargs_text2, va='top', ha='center',)

    band_borders = cfg.BANDS['beta_low']
    beta_mask = (fm_freqs >= band_borders[0]) & (fm_freqs <= band_borders[1])
    # Fill between aperiodic and zero
    # Add white background to enable correct colors
    ax.fill_between(fm_freqs[beta_mask], total_fit[beta_mask],
                    ap_fit[beta_mask], color='w', alpha=1)
    # Add periodic and aperiodic shades
    ax.fill_between(fm_freqs[beta_mask], total_fit[beta_mask],
                    ap_fit[beta_mask], color=c_per, alpha=.3)

    # Fit range
    fit_low, fit_high = fm_freqs[0], fm_freqs[-1]
    log_center = 10**((np.log10(fit_low) + np.log10(fit_high)) / 2 )
    ax.set_xticks([log_center], minor=True)
    ax.set_xticklabels([r'$-$   Fit Range  $-$'], minor=True, y=-.02,
                       color='grey')

    # Dashed fit border lines
    ymin, _ = ax.get_ylim()
    ax.vlines(fit_low, ymin, ap_fit[0], **dotted_line)
    ax.vlines(fit_high, ymin, ap_fit[-1], **dotted_line)

    # yticks
    yticks = ax.get_yticks()
    ylim = ax.get_ylim()
    yticks = yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])]
    for ytick in yticks:
        ax.axhline(y=ytick, color='w', lw=LINEWIDTH_AXES)
    return


def add_integral_str(df, freqs, psd, ax, kind, xscale, c_off, c_on):
    if xscale == 'linear':
        return None
    if kind == 'periodic':
        return None
    freq_mask = ((df[freqs] >= 5) & (df[freqs] <= 95))
    units = '%' if kind == 'normalized' else r' $\mu V^2$'
    int_off = df[(df.cond == 'off') & freq_mask][psd].sum()
    int_on = df[(df.cond == 'on') & freq_mask][psd].sum()
    int_str = r'$\int_{5\text{ Hz}}^{95\text{ Hz}}'
    off_str = r'\text{PSD}_{\text{off}} \,\mathrm{d}f=$'
    on_str = r'\text{PSD}_{\text{on}} \,\mathrm{d}f=$'
    int_str_off = int_str + off_str
    int_str_on = int_str + on_str
    if kind == 'normalized':
        int_off_res = f'{int_off:.0f}{units}'
        int_on_res = f'{int_on:.0f}{units}'
    else:
        int_off_res = f'{int_off:.1f}{units}'
        int_on_res = f'{int_on:.1f}{units}'
    int_off = int_str_off + int_off_res
    int_on = int_str_on + int_on_res
    if kind == 'normalized':
        pos_off = (.03, .045)
        pos2 = (0, 0)
        fs = FONTSIZE_S
        pad = 0
        sep = 3
        texts = [TextArea(int_off, textprops=dict(color=c_off, fontsize=fs)),
                 TextArea(int_on, textprops=dict(color=c_on, fontsize=fs))]
        texts_vbox = VPacker(children=texts, pad=pad, sep=sep)
        ann = AnnotationBbox(texts_vbox, pos_off, xycoords=ax.transAxes,
                             box_alignment=pos2,
                             bboxprops=dict(facecolor='w', boxstyle='round',
                                            edgecolor='#cccccc'))
        ax.add_artist(ann)
    elif kind == 'absolute':
        pos_x = 0.025
        pos_off = (pos_x, .66)
        pos2 = (0, 1)
        fs = FONTSIZE_S - 1
        pad = -1
        sep = 1.5
        text_off = [TextArea(int_off,
                             textprops=dict(color=c_off, fontsize=fs))]
        text_off_vbox = VPacker(children=text_off, pad=pad, sep=sep)
        anno_kwargs = dict(xycoords=ax.transAxes, box_alignment=pos2,
                           bboxprops=dict(facecolor='w', boxstyle='round',
                                          edgecolor='#cccccc'), zorder=10)
        ann_off = AnnotationBbox(text_off_vbox, pos_off, **anno_kwargs)

        pos_on = (pos_x, .14)
        c_on_strong = '#918DC2'  # c_on too little contrast
        text_on = [TextArea(int_on, textprops=dict(color=c_on_strong,
                                                   fontsize=fs))]
        text_on_vbox = VPacker(children=text_on, pad=pad, sep=sep)
        ann_on = AnnotationBbox(text_on_vbox, pos_on, **anno_kwargs)

        ax.add_artist(ann_off)
        ax.add_artist(ann_on)
