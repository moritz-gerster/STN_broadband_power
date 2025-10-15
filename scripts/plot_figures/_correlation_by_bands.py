"""Helping functions."""
from os.path import join

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

import scripts.config as cfg
from scripts.plot_figures.settings import FONTSIZE_ASTERISK
from scripts.utils_plot import _save_fig, convert_pvalue_to_asterisks


def barplot_UPDRS_bands(df_corrs, fig_dir='Figure1', title=False,
                        palette_barplot=None, prefix='', xlabel=True,
                        figsize=(1.9, 1.3), fontsize_stat=10, stat_height=0.8):
    kind = df_corrs.kind.unique()[0]
    band_cols = df_corrs.band_nme.unique()
    projects = [proj for proj in cfg.PROJECT_ORDER_SLIM
                if proj in df_corrs.project.unique() and proj != 'all']
    df_single = df_corrs[(df_corrs.project != 'all')]
    df_all = df_corrs[(df_corrs.project == 'all')]
    if palette_barplot is None:
        if df_corrs.project.nunique() == 1:
            color_all = cfg.COLOR_DIC[kind]
        else:
            color_all = cfg.COLOR_DIC["all3"]
        plot_all = {'data': df_all, 'color': color_all, 'width': 0.6}
    else:
        plot_all = {'data': df_all, 'palette': palette_barplot, 'width': 0.6}

    palette = [cfg.COLOR_DIC[proj + "3"] for proj in projects]

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharey=True)

    plot_kwargs = {'ax': ax, 'x': 'band_nme', 'y': 'rho', "order": band_cols,
                   'legend': False}
    plot_single = {'data': df_single, 'hue': 'project',
                   'palette': palette, 'alpha': 1, 'width': 0.4}

    sns.barplot(**plot_all, **plot_kwargs)
    sns.barplot(**plot_single, **plot_kwargs)
    # add significance star for bars where pval < 0.05
    if len(ax.containers) > 2:
        bars_pooled = ax.containers[0]  # select all project
    else:
        bars_pooled = ax.containers
    ymin, ymax = ax.get_ylim()
    for bar, band in zip(bars_pooled, band_cols):
        df_band = df_all[(df_all.band_nme == band)]
        pvalue = df_band.pval.values[0]
        text = convert_pvalue_to_asterisks(pvalue)
        if isinstance(bar, list):
            bar = bar[0]
        x_bar = bar.get_x() + bar.get_width() / 2
        ax.annotate(text, xy=(x_bar, ymax*stat_height), ha='center',
                    va='bottom', fontsize=fontsize_stat)
    ax.set_ylim(ymin, ymax*1.12)
    xticklabels = []
    for xticklabel in df_all.band_nme.unique():
        xticklabels.append(xticklabel)
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels)
    if xlabel:
        xlabel = 'Frequency band'
    else:
        xlabel = None
    ax.set_xlabel(xlabel, labelpad=1.5)
    corr_method = df_all.corr_method.unique()[0]
    y = df_corrs['y'].unique()[0]
    if y == 'patient_days_after_implantation':
        kind_str = 'Relative' if kind == 'normalized' else 'Absolute'
        ylabel = f'{kind_str} power vs. days after surgery'
    elif corr_method == 'spearman':
        ylabel = r"$\rho$"
    elif corr_method == 'within':
        ylabel = r"$r_{rm}$"
    elif corr_method == 'withinRank':
        ylabel = r"$r_{\text{rank rm}}$"
    if title == True:
        ax.set_title(ylabel)
        unit_alone = r"$\rho$"
        ax.set_ylabel(unit_alone)
    elif isinstance(title, str):
        ax.set_title(title)
    else:
        ax.set_ylabel(ylabel)
    cond = df_all.cond.unique()[0]
    updrs = df_all.y.unique()[0]
    pwr_kind = df_all.pwr_kind.unique()[0]
    fname = (f'{prefix}band_UPDRS_{kind}_{cond}_'
             f'{updrs}_{corr_method}_{pwr_kind}')
    plt.tight_layout()
    _save_fig(fig, f'{fig_dir}/{fname}', cfg.FIG_PAPER,
              transparent=True, bbox_inches=None)


def barplot_UPDRS_periodic(df_corrs, save_dir='Figure5'):
    kind = df_corrs.kind.unique()[0]
    band_cols = df_corrs.band_nme.unique()
    cond = df_corrs.cond.unique()[0]
    palette = [(sns.color_palette()[0])] + list(sns.color_palette("flare", 3))

    figsize = ((len(band_cols)) / 16, .4)
    _, ax = plt.subplots(1, 1, figsize=figsize, sharey=True)

    plot_kwargs = {'ax': ax, 'x': 'band_nme', 'y': 'rho', "order": band_cols,
                   'legend': False, 'data': df_corrs, 'width': 0.6,
                   'palette': palette}
    sns.barplot(**plot_kwargs)
    ax.hlines(0, *ax.get_xlim(), lw=0.1, color='k')

    # add significance star for bars where pval < 0.05
    bars_pooled = ax.containers
    ymin, ymax = ax.get_ylim()
    ydiff = ymax - ymin
    for bar, band in zip(bars_pooled, band_cols):
        df_band = df_corrs[(df_corrs.band_nme == band)]
        pvalue = df_band.pval.values[0]
        text = convert_pvalue_to_asterisks(pvalue, stack_vertically=True)
        x_bar = bar[0].get_x() + bar[0].get_width() / 2
        y_bar = bar[0].get_height()
        offset = max(y_bar, 0) - 0.15 * ydiff
        va = 'bottom'
        ax.annotate(text, xy=(x_bar, offset), ha='center', va=va,
                    fontsize=FONTSIZE_ASTERISK, linespacing=.15)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel(None)
    # hide x-axis and move y-axis to right
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    corr_method = df_corrs.corr_method.unique()[0]
    if corr_method == 'spearman':
        ylabel = r"Spearman $\rho$"
    elif corr_method == 'within':
        ylabel = r"$r_{rm}$"
    elif corr_method == 'withinRank':
        ylabel = r"$r_{\text{rank rm}}$"
    ax.set_ylabel(ylabel)
    cond = df_corrs.cond.unique()[0]
    updrs = df_corrs.y.unique()[0]
    pwr_kind = df_corrs.pwr_kind.unique()[0]
    fname = f'band_UPDRS_{kind}_{cond}_{updrs}_{corr_method}_{pwr_kind}'
    plt.tight_layout()
    if cond == 'on':
        export_legend(save_dir, band_cols, palette, fname)


def export_legend(save_dir, band_cols, palette, fname):
    handles = [Patch(color=color) for color in palette]
    labels = [band_nme for band_nme in band_cols]
    legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True,
                        ncol=1)
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted())
    filename = join(cfg.FIG_PAPER, save_dir, f'{fname}_legend.pdf')
    fig.savefig(filename, dpi="figure", bbox_inches=bbox, transparent=True)
    plt.show()
