from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import bootstrap

import scripts.config as cfg
from scripts.plot_figures.settings import *
from scripts.utils_plot import _save_fig, cohen_d, equalize_x_and_y


def band_barplot(df, kind, ycols, figsize=(2, 1),
                 projects=cfg.PROJECT_ORDER_SLIM, n_boot=N_BOOT_COHEN,
                 estimator='effect_size', fig_dir='Figure_S6', prefix='',
                 xticklabels=None):
    # Set linewidths proportional to sample sizes
    line_widths = {proj: .25 for proj in projects}
    line_widths['all'] = LINEWIDTH_PLOT

    names = []
    values = []
    errors = []
    significances = []
    project_list = []
    for proj in projects:
        for ycol in ycols:
            names.append(ycol)
            project_list.append(proj)
            df_proj = df[(df.project == proj)]

            # equalize x and y to enhance statistics (paired=True)
            df_proj, n = equalize_x_and_y(df_proj, 'cond', ycol)
            off_arr = df_proj[df_proj.cond == 'off'][ycol].values
            on_arr = df_proj[df_proj.cond == 'on'][ycol].values
            if estimator == 'effect_size':
                value = cohen_d(off_arr, on_arr)
                if np.all(off_arr == 0) or np.all(on_arr == 0):
                    # effect size unreasonable if all values are zero
                    value = np.nan
                data = (off_arr, on_arr)
                func = cohen_d
            else:
                if estimator == 'mean':
                    func = np.mean
                elif estimator == 'median':
                    func = np.median
                else:
                    raise ValueError('Unknown estimator')
                value = func(off_arr - on_arr)
                data = (off_arr - on_arr,)
            if proj == 'all':
                print(f'{estimator} {ycol}: {value:.2f}')
            values.append(value)
            if n_boot is None:
                msg = 'Only effect size supports parametric'
                assert estimator == 'effect_size', msg
                # parametric (fast but more assumptions)
                ci = pg.compute_esci(value, n, paired=True)
            else:
                # nonparametric (slow but more correct)
                result = bootstrap(data, func, n_resamples=n_boot, paired=True,
                                random_state=1)
                ci = result.confidence_interval
                # if distribution is degenerate (many 0 values),
                # apply parametric
                if np.any(np.isnan(result.bootstrap_distribution)):
                    ci = pg.compute_esci(value, n, paired=True)

            # CIs
            ci_lower, ci_upper = ci
            error = [[abs(value - ci_lower)], [abs(ci_upper - value)]]
            errors.append(error)
            # make sure that np.nan values are not significant
            if value is np.nan:
                significant = False
            elif ((ci_lower < 0 and ci_upper < 0)
                  or (ci_lower > 0 and ci_upper > 0)):
                significant = True
            else:
                significant = False
            significances.append(significant)

    df_plot = pd.DataFrame({'project': project_list, 'band': names,
                            'value': values, 'ci': errors,
                            'significant': significances})
    capsize = 1 if len(projects) > 1 else 0
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.axhline(0, color="k", lw=LINEWIDTH_AXES, ls='--')
    for i, proj in enumerate(projects):
        df_proj = df_plot[df_plot.project == proj]
        # Plot CIs
        yerr = [item for sublist in df_proj.ci.values for item in sublist]
        yerr = np.array(yerr).reshape(len(ycols), 2).T
        values = df_proj.value.values

        xticks = np.arange(len(ycols))# + 0.03 * i
        color = (cfg.COLOR_DIC[proj] if len(projects) > 1
                 else cfg.COLOR_DIC[kind])
        sns.barplot(x=xticks, y=values, yerr=yerr, ax=ax, capsize=capsize,
                    color=color,
                    err_kws={'linewidth': line_widths[proj]})
        _, ymax = ax.get_ylim()
        for j, sig in enumerate(significances):
            if sig:
                ax.text(xticks[j], ymax*.95, '*', ha='center',
                        fontsize=FONTSIZE_ASTERISK, color='k')

    if xticklabels is None:
        labels = [cfg.PLOT_LABELS[y_col] for y_col in ycols]
    else:
        labels = xticklabels
    ax.set_xticks(xticks, labels=labels)
    ax.tick_params(axis='x', length=0)
    ylabel = 'Cohen\'s d'f' {cfg.COND_DICT['off']}-{cfg.COND_DICT['on']}'
    ax.set_ylabel(ylabel)
    ax.set_xlabel(None)
    sns.despine(bottom=True)

    plt.tight_layout()
    fpath = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, f'{prefix}broadband_OFFON_{kind}', fpath,
              transparent=False, facecolor='white',
              bbox_inches=None)


def band_labels(y_col):
    for band in BANDS:
        if band in y_col:
            band = cfg.BAND_NAMES_GREEK[band]
            return band.replace(' ', '\n')
    # Aperiodic names
    band = cfg.PLOT_LABELS[y_col]
    return band.replace(' ', '\n')