import warnings
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import scripts.config as cfg
from scripts.plot_figures._correlation_by_bands import barplot_UPDRS_bands
from scripts.plot_figures.settings import *
from scripts.utils import get_correlation_df
from scripts.utils_plot import _corr_results, _save_fig, _stat_anno, plot_corr


def updrs_pre_post(df, fig_dir='Figure_S5', prefix=''):
    df = df[df.project_nme != 'all']

    conds = ['off', 'on', 'offon_abs']

    fig, axes = plt.subplots(2, len(conds), figsize=(2, 2.1), sharey='col')

    for col, cond in enumerate(conds):

        df_cond = df[df.cond == cond]

        # Filter to include only subjects with both pre and post scores
        df_both = df_cond.drop_duplicates(subset=['subject'])
        df_both = df_both[(df_both['UPDRS_pre_III'].notna())
                          & (df_both['UPDRS_post_III'].notna())]

        # Reshape data to long format for plotting
        df_long = pd.melt(df_both,
                          id_vars=['project_nme', 'subject'],
                          value_vars=['UPDRS_pre_III', 'UPDRS_post_III'],
                          var_name='Assessment',
                          value_name='UPDRS_III_Score')

        # Map assessment names to 'Pre' and 'Post' for easier plotting
        rename = {'UPDRS_pre_III': 'Pre', 'UPDRS_post_III': 'Post'}
        df_long['Assessment'] = df_long['Assessment'].map(rename)
        projects = [proj for proj in cfg.PROJECT_NAMES
                    if proj in df_long.project_nme.unique()]

        # Loop through each project and plot on separate axes
        for row, project in enumerate(projects):
            ax = axes[row, col]
            df_proj = df_long[df_long['project_nme'] == project]

            # Plot boxplot for Pre and Post scores
            width = 0.5
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sns.boxplot(data=df_proj, x='Assessment', y='UPDRS_III_Score',
                            ax=ax, color=cfg.COLOR_DIC[project],
                            showfliers=False, linewidth=0.2, width=width)

            # Add lines connecting paired pre and post scores for each subject
            for sub in df_proj.subject.unique():
                df_sub = df_proj[df_proj['subject'] == sub]
                ax.plot([0, 1], df_sub['UPDRS_III_Score'], marker='o',
                        markersize=.00005, color='k', lw=0.05)

            # Run Wilcoxon signed-rank test and annotate p-value
            _stat_anno(ax, df_proj, 'Assessment', 'UPDRS_III_Score',
                       y_line=None)

            # Customize each axis
            ax.set_ylabel(f'UPDRS-III {project}' if col == 0 else '')
            ax.set_xlabel(None)
            if row == 0:
                ax.set_title(f'{cfg.COND_DICT[cond]}', y=1.)
                ax.set_xticks([])
            else:
                ax.set_title(None)
            ax.set_xlim(-.5, 1.5)

    plt.tight_layout()
    fpath = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, f'{prefix}stun_effect_UPDRS_III_{'_'.join(conds)}.pdf',
              fpath, bbox_inches=None)


def pre_post_vs_recovery(df, fig_dir='Figure_S5', prefix=''):
    df = df[~df.project.isin(['all'])].copy()
    df["UPDRS_III_prepost_diff"] = df["UPDRS_pre_III"] - df["UPDRS_post_III"]
    x = 'patient_days_after_implantation'
    y = 'UPDRS_III_prepost_diff'
    subset = ['subject', 'cond', 'project']
    df = df.dropna(subset=[x, y]).drop_duplicates(subset=subset)

    conds = ['off', 'on', 'offon_abs']
    corr_method = 'spearman'
    fig, axes = plt.subplots(1, len(conds), figsize=(4.7, 2.25), sharey=True)

    for col, cond in enumerate(conds):
        ax = axes[col]
        ylabel = 'UPDRS-III post-pre' if col == 0 else None
        df_plot = df[df.cond.isin([cond])]
        plot_corr(ax, df_plot, x, y, hue="project",
                  corr_method=corr_method, title=cfg.COND_DICT[cond], ci=95,
                  xlabel='Days after surgery', scatter_kws={'s': 1})
        # modify legend entries
        weights = []
        for project in cfg.PROJECT_NAMES:
            if project in df_plot.project_nme.unique():
                df_proj = df_plot[df_plot.project_nme == project]
                corr_results = _corr_results(df_proj, x, y, corr_method, None,
                                             n_perm=N_PERM_CORR)
                rho, sample_size, label, weight, _ = corr_results
                weights.append(weight)
        handles, labels = ax.get_legend_handles_labels()
        labels = [label.replace('Berlin ', '').replace('Düsseldorf1 ', '')
                  for label in labels]
        leg = ax.legend(handles, labels, loc='upper left',
                        bbox_to_anchor=(0, 1))
        [t.set_fontweight(w) for t, w in zip(leg.get_texts(), weights)]
        ax.set_title(cfg.COND_DICT[cond], fontsize=FONTSIZE_M)
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    fname = f'{prefix}stun_effect_UPDRS_vs_days_{'_'.join(conds)}.pdf'
    fpath = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, fname, fpath, bbox_inches=None, transparent=True)


def pre_post_vs_symptoms(df_norm, fig_dir='Figure_S5', prefix=''):
    df_norm = df_norm[~df_norm.project.isin(['all'])].copy()
    df_norm_off = df_norm[df_norm.cond.isin(['off'])]
    df_norm_offon = df_norm[df_norm.cond.isin(['offon_abs'])]
    # average hemispheres
    keep = ['subject', 'cond', 'project', 'color']
    df_plot = df_norm_off.groupby(keep).mean(numeric_only=True).reset_index()

    # get subjects who have both pre and post scores
    y_pre = 'UPDRS_pre_III'
    y_post = 'UPDRS_post_III'
    x = 'beta_low_abs_max_log'
    df_plot = df_plot.dropna(subset=[x, y_pre, y_post])
    corr_method = 'spearman'

    _, _, label_pre, weight, _ = _corr_results(df_plot, x, y_pre, corr_method,
                                               None, n_perm=N_PERM_CORR)
    _, _, label_post, weight, _ = _corr_results(df_plot, x, y_post,
                                                corr_method, None,
                                                n_perm=N_PERM_CORR)

    fig, axes = plt.subplots(1, 4, figsize=(4.7, 2), sharey="row")

    ax = axes[0]
    sns.regplot(ax=ax, data=df_plot, y=y_pre, x=x, ci=95,
                scatter_kws=dict(s=1), color='k', label=label_pre, marker='.',
                n_boot=1000)

    y_title = 0.9
    ax.text(0.5, 1.1, 'Pre', ha='center', fontsize=FONTSIZE_M,
            transform=ax.transAxes, fontweight='bold')
    ax.set_title(label_pre, weight=weight, y=y_title)
    ax.set_xlabel(None)
    ax.set_ylabel('UPDRS-III off')

    ax = axes[1]
    sns.regplot(ax=ax, data=df_plot, y=y_post, x=x, ci=95,
                scatter_kws=dict(s=1), color='k', label=label_post, marker='.',
                n_boot=1000)

    ax.text(0.5, 1.1, 'Post', ha='center', fontsize=FONTSIZE_M,
            transform=ax.transAxes, fontweight='bold')
    ax.set_title(label_post, weight=weight, y=y_title)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    # average hemispheres
    keep = ['subject', 'cond', 'project', 'color']
    df_plot = df_norm_offon.groupby(keep).mean(numeric_only=True).reset_index()
    df_plot = df_plot.dropna(subset=[x, y_pre, y_post])

    _, _, label_pre, weight, _ = _corr_results(df_plot, x, y_pre, corr_method,
                                               None, n_perm=N_PERM_CORR)
    _, _, label_post, weight, _ = _corr_results(df_plot, x, y_post,
                                                corr_method, None,
                                                n_perm=N_PERM_CORR)
    ax = axes[2]
    sns.regplot(ax=ax, data=df_plot, y=y_pre, x=x, ci=95,
                scatter_kws=dict(s=1), color='k', label=label_pre, marker='.',
                n_boot=1000)

    ax.text(0.5, 1.1, 'Pre', ha='center', fontsize=FONTSIZE_M,
            transform=ax.transAxes, fontweight='bold')
    ax.set_title(label_pre, weight=weight, y=y_title)
    band_nme = x.replace('_abs_max_log', '')
    band_nme = cfg.BAND_NAMES_GREEK[band_nme]
    ax.set_xlabel(None)
    ax.set_ylabel('UPDRS-III off-on')

    ax = axes[3]
    sns.regplot(ax=ax, data=df_plot, y=y_post, x=x, ci=95,
                scatter_kws=dict(s=1), color='k', label=label_post, marker='.',
                n_boot=1000)
    ax.set_title(label_post, weight=weight, y=y_title)
    ax.text(0.5, 1.1, 'Post', ha='center', fontsize=FONTSIZE_M,
            transform=ax.transAxes, fontweight='bold')
    band_nme = x.replace('_abs_max_log', '')
    band_nme = cfg.BAND_NAMES_GREEK[band_nme]
    ax.set_xlabel(None)
    fig.supxlabel(f'Relative {band_nme} [%]', y=0.04)
    ax.set_ylabel(None)

    plt.tight_layout()
    fname = f'{prefix}norm_beta_low_vs_UPDRS_offon_off_prepost.pdf'
    fpath = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, fname, fpath, bbox_inches=None,
              transparent=True)


def power_vs_recovery(df_norm, fig_dir='Figure_S5', prefix='',
                      output_file=None):
    y = "patient_days_after_implantation"
    total_power = True
    kind = 'normalized'

    exclude_projects = ['Hir', 'Lit']  # all recorded on same day

    for cond in ['off']:
        data = df_norm[(df_norm.cond == cond)]

        for exclude in exclude_projects:
            data = data[~data.subject.str.startswith(exclude)]

        for corr_method in ['spearman']:
            df_corr = get_correlation_df(data, y, total_power=total_power,
                                         n_perm=N_PERM_CORR,
                                         bands=BANDS,
                                         add_high_beta_cf=False,
                                         use_peak_power=False,
                                         output_file=output_file,
                                         corr_method=corr_method)
            df_corr['kind'] = kind
            barplot_UPDRS_bands(df_corr, fig_dir=fig_dir, title=False,
                                prefix=prefix, figsize=(2, 1.9))