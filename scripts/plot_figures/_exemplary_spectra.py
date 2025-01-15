"""Helping functions."""
from os.path import join

import numpy as np
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker
from matplotlib.ticker import ScalarFormatter

import scripts.config as cfg
from scripts.plot_figures.settings import *
from scripts.utils_plot import (_add_band, _add_band_annotations, _save_fig,
                                explode_df)


def representative_spectrum(df, kind, fig_dir=None,
                            use_peak_power=False, height=1.5, aspect=1,
                            xscale="linear", yscale="log", ylabel=None,
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
            ylim = (0.007, 3)
        elif kind == 'periodic':
            ylim = (0.007, 5)
    else:
        xmin = cfg.BANDS[BANDS[0]][0]  # otherwise delta band is cut off
        xmax = XTICKS_FREQ_low[-1]  # ok to cut off low gamma
        xlim = (xmin, xmax)
        if kind == 'normalized':
            ylim = (0, 10)
        elif kind == 'absolute':
            ylim = (0, 1.3)
        elif kind == 'periodic':
            ylim = (0, 1.3)
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
                              xlim=xlim,
                              ylim=ylim,
                              ylabel=ylabel,
                              xscale=xscale, yscale=yscale, legend=legend,
                              bands=BANDS, xticks=xticks, kind=kind,
                              **{'height': height, 'aspect': aspect})


def plot_psd_df_annotated(df, freqs="psd_freqs", psd="asd", hue="cond",
                          xscale="log", yscale="log",
                          use_peak_power=False,
                          bands=None, ylabel=None, xlim=None,
                          ylim=None, ax_kwargs={}, kind=None,
                          save_name=None, col=None, legend=True, xticks=None,
                          col_order=None, row=None, add_band_colors=False,
                          **rel_kwargs):
    if ylabel is None:
        if kind in ['absolute', 'periodic']:
            ylabel = r"PSD [$\mu V^2/Hz$]"
        else:
            ylabel = "Normalized PSD [%]"
    if xlim:
        xmask = (df[freqs] >= xlim[0]) & (df[freqs] <= xlim[1])
        df = df.loc[xmask]
    global c_off, c_on
    if kind == 'normalized':
        c_off = cfg.COLOR_DIC['all']
        c_on = cfg.COLOR_DIC['all2']
    elif kind == 'absolute':
        c_off = cfg.COLOR_DIC[kind]
        c_on = cfg.COLOR_DIC[kind + '2']
    elif kind in ['periodic', 'lorentzian']:
        c_off = cfg.COLOR_DIC['all']
        c_on = cfg.COLOR_DIC['all2']
    if kind == 'periodic' and xscale == 'linear':
        hue_order = ['on', 'off']
        palette = [c_on, c_off]
    else:
        hue_order = ['off', 'on']
        palette = [c_off, c_on]
    g = sns.relplot(data=df, x=freqs, y=psd, hue=hue, hue_order=hue_order,
                    kind="line", palette=palette, col=col, col_order=col_order,
                    row=row, **rel_kwargs, zorder=5)
    ax = g.axes[0, 0]
    g._legend.remove()
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        labels = [cfg.COND_DICT[l] for l in labels]
        ax.legend(handles, labels, title='Levodopa', loc='upper right',
                  handlelength=1)

    assert len(g.axes) == 1, "Only one ax supported"
    if xscale == "log":
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        xticklabels = [f"{x} Hz" for x in xticks]
        # Add frequency bands
        for xtick in XTICKS_FREQ_low:
            ax.axvline(x=xtick, color='w', lw=LINEWIDTH_AXES)
    else:
        xticklabels = XTICKS_FREQ_low_labels
    xlabel = "Frequency [Hz]"
    add_integral_str(df, freqs, psd, ax, kind, xscale, c_off, c_on)
    g.set(xscale=xscale, yscale=yscale, **ax_kwargs,
          xticks=xticks, xticklabels=xticklabels,
          ylabel=ylabel, xlabel=xlabel, ylim=ylim, xlim=xlim)
    annotate_peaks(df, ax, xscale, kind, use_peak_power=use_peak_power)
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


def annotate_peaks(df, ax, xscale, kind, use_peak_power=False):
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
            xmin, xmax = ax.get_xlim()
            log_center = 10**((np.log10(xmin) + np.log10(xmax)) / 2 )
            ax.set_xticks([log_center], minor=True)
            ax.set_xticklabels([r'$-$ Normalization Range $-$'], minor=True,
                                y=-.02, color=textcolor)
        elif kind == 'periodic':
            _annotate_log_periodic(df, ax, textcolor, text_box, z_front,
                                   dotted_line)
    elif xscale == 'linear':
        arrow_range = dict(facecolor=textcolor, shrinkA=0, shrinkB=0,
                           arrowstyle='|-|', mutation_scale=1,
                           edgecolor=textcolor)
        if kind == 'normalized':
            _annotate_norm_and_abs(df, ax, kind, textcolor, z_front,
                                   dotted_line,
                                   text_kwargs, arrow_range,
                                   ['beta_low'],
                                   use_peak_power=use_peak_power)
        elif kind == 'absolute':
            _annotate_norm_and_abs(df, ax, kind, textcolor, z_front,
                                   dotted_line,
                                   text_kwargs, arrow_range, ['beta_low'])
        elif kind == 'periodic':
            _annotate_lin_per(df, ax, textcolor, z_front, z_back_middle,
                              text_kwargs, use_peak_power=True)


def _annotate_norm_and_abs(df, ax, kind, textcolor, z_front, dotted_line,
                           text_kwargs, arrow_range, bands,
                           use_peak_power=False):
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
        x_center = (peak_freq_off + peak_freq_on) / 2
        y_center = (peak_power_off + peak_power_on) / 2

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
            power_diff_str = f'\n= {peak_power_off - peak_power_on:.1f}{units}'
            pwr_str = (f'{pwr_str}\n{cfg.BAND_NAMES_GREEK[band]}\n'
                       f'power\noff-on{power_diff_str}')
            x_text_pwr = 34
            ax.annotate(pwr_str, xy=(x_center, y_center),
                        xytext=(x_text_pwr, y_center), ha='left',
                        **text_kwargs)
            # Vertical bracket
            x_arrow_pwr = 32
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
                          lw=LINEWIDTH_AXES)
                ax.hlines(peak_power_on, *band_borders,
                          color=c_on,
                          lw=LINEWIDTH_AXES)


def _annotate_lin_per(df, ax, textcolor, z_front, z_back_middle, text_kwargs,
                      use_peak_power=False):
    band = 'beta_low'
    cond = 'off'
    df_cond = df[df.cond == cond]
    c_str = df.project.unique()[0]
    c_str += '2' if cond == 'on' else ''

    fm_freqs = df_cond.fm_freqs.iloc[0]
    ap_fit = df_cond.fm_psd_ap_fit.iloc[0]
    per_fit = df_cond.fm_psd_peak_fit.iloc[0]

    # Plot 1/f exponent
    c_ap = cfg.COLOR_DIC['periodicAP']
    ax.plot(fm_freqs, ap_fit, c=c_ap, ls="--",
            label="Aperiodic fit", zorder=z_front)
    # Plot periodic fit
    c_per = cfg.COLOR_DIC['periodic']
    ax.plot(fm_freqs, ap_fit + per_fit, c=c_per, ls="--",
            zorder=z_front, label="Periodic fit")

    # Total power
    x_text = 32  # Hz
    peak_freq = df_cond[f'{band}_abs_max_freq'].values[0]
    if use_peak_power:
        x_arrow = peak_freq
    else:
        band_borders = cfg.BANDS[band]
        x_arrow = band_borders[1]

    freq_mask = fm_freqs == peak_freq
    arr_pwr = dict(facecolor=textcolor, shrinkA=0, shrinkB=1.5,
                   edgecolor=textcolor, arrowstyle='-|>',
                   mutation_scale=3)
    arr_kwargs = dict(arrowprops=arr_pwr, zorder=z_back_middle)
    if use_peak_power:
        total_pwr = ap_fit[freq_mask] + per_fit[freq_mask]
    else:
        total_pwr = df_cond[f'{band}_abs_mean'].values[0]
        ax.hlines(total_pwr, *band_borders, color=c_off, lw=LINEWIDTH_AXES)
    ax.annotate('', xy=(x_arrow, total_pwr), xytext=(x_text, total_pwr),
                **arr_kwargs)
    text_kwargs_tot = text_kwargs.copy()
    text_kwargs_tot['color'] = 'k'
    ax.text(s='Total\npower', x=x_text, y=total_pwr, **text_kwargs_tot)

    # Periodic power
    if use_peak_power:
        height_ap = ap_fit[freq_mask] / 2
        y_text_ap = height_ap * 6
        y_periodic = (total_pwr + y_text_ap) / 2
        y_text_per = y_periodic
    else:
        height_ap = 10**df_cond[f'{band}_fm_band_aperiodic_log'].values[0]
        y_periodic = df_cond[f'{band}_fm_mean'].values[0]
        ax.hlines(y_periodic, *band_borders, color=c_per, lw=LINEWIDTH_AXES)
        ax.hlines(height_ap, *band_borders, color=c_ap, lw=LINEWIDTH_AXES)
        y_text_ap = height_ap * 2
        y_text_per = y_periodic * .84
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


def _annotate_log_periodic(df, ax, textcolor, text_box, z_front, dotted_line):
    for cond in ['off']:
        df_cond = df[df.cond == cond]
        c_str = df.project.unique()[0]
        c_str += '2' if cond == 'on' else ''

        fm_freqs = df_cond.fm_freqs.iloc[0]
        ap_fit = df_cond.fm_psd_ap_fit.iloc[0]

        # Plot 1/f exponent
        c_ap = cfg.COLOR_DIC['periodicAP']
        ax.plot(fm_freqs, ap_fit, c=c_ap, ls="--", zorder=z_front,
                label="Aperiodic fit")

        # Interpolate offset
        offset = df_cond.fm_offset.unique()[0]
        exponent = df_cond.fm_exponent.unique()[0]
        x = np.arange(1, fm_freqs[0] + 1)
        ax.plot(x, offset * 1 / x**exponent, c=c_ap, ls=":",
                zorder=z_front)

        # 1/f Offset
        arr_offset = dict(facecolor=textcolor, shrinkA=0, shrinkB=1.5,
                          edgecolor=textcolor, arrowstyle='-|>',
                          mutation_scale=3)
        kwargs_text = dict(bbox=text_box, color=textcolor,
                           va='center', ha='left')
        ax.annotate('Offset',
                    xy=(x[0], offset),
                    xytext=(2.5, offset),
                    arrowprops=arr_offset, **kwargs_text)

        # 1/f Exponent
        freq_exp = (22 + 14) / 2
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
    return


def add_integral_str(df, freqs, psd, ax, kind, xscale, c_off, c_on):
    if xscale == 'linear':
        return None
    if kind == 'periodic':
        return None
    freq_mask = (df[freqs] >= 5) & (df[freqs] <= 95)
    units = '%' if kind == 'normalized' else r' $\mu V^2$'
    int_off = df[(df.cond == 'off') & freq_mask][psd].sum()
    int_on = df[(df.cond == 'on') & freq_mask][psd].sum()
    off_str = r'\text{PSD}_{\text{off}}=$'
    on_str = r'\text{PSD}_{\text{on}}=$'
    int_str = r'$\int_{5\text{ Hz}}^{95\text{ Hz}}'
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
        pos1 = (.03, .045)
        pos2 = (0, 0)
        fs = FONTSIZE_S
        pad = 0
        sep = 3
    elif kind == 'absolute':
        pos1 = (.96, .95)
        pos2 = (1, 1)
        fs = FONTSIZE_S - 1
        pad = -1
        sep = 1.5
    texts = [TextArea(int_off, textprops=dict(color=c_off, fontsize=fs)),
             TextArea(int_on, textprops=dict(color=c_on, fontsize=fs))]
    texts_vbox = VPacker(children=texts, pad=pad, sep=sep)
    ann = AnnotationBbox(texts_vbox,
                         pos1,
                         xycoords=ax.transAxes,
                         box_alignment=pos2,
                         bboxprops=dict(
                             facecolor='#eaeaf2',
                             boxstyle='round',
                             edgecolor='#cccccc'
                         ))
    ax.add_artist(ann)
