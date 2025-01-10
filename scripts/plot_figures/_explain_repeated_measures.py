from os.path import join

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from pingouin import rm_corr
from scipy.stats import pearsonr, spearmanr
from statsmodels.formula.api import ols

import scripts.config as cfg
from scripts.plot_figures.settings import *
from scripts.utils_plot import _save_fig


def repeated_measures_toy_example(fig_dir='Figure6', prefix=''):
    beta_powers = [1, 2, 3, 4.5, 5.8, 6.5]
    updrs_scores = [2, 2.7, 4, 3, 5.5, 6.5]
    hemispheres = ['L', 'R'] * 3
    markers = ['o', 'v', 'v', 'o', 'o', 'v']
    patient_list = ['1', '1', '2', '2', '3', '3']
    dark = sns.color_palette("dark")
    colors = [dark[0], dark[0], dark[2], dark[2], dark[6], dark[6]]

    df1 = pd.DataFrame({'beta_power': beta_powers, 'UPDRS': updrs_scores,
                        'subject': patient_list,
                    'marker': markers,
                    'ch_hemisphere': hemispheres,
                    'color': colors})

    # calc repeated measures correlation
    x = 'beta_power'
    y = 'UPDRS'
    subject = 'subject'
    formula = f"Q('{y}') ~ C(Q('{subject}')) + Q('{x}')"
    model = ols(formula, data=df1).fit()
    df1["pred"] = model.fittedvalues

    corr_spearman = spearmanr(df1.beta_power.values, df1.UPDRS.values)
    corr_within = rm_corr(df1, x='beta_power', y='UPDRS', subject='subject')

    corr_str_spearman = r"Spearman's $\rho$"
    corr_str_rm = r"Repeated measures $r_{rm}$"
    corr_str_spearman2 = r"$\rho$"
    corr_str_rm2 = r"$r_{rm}$"
    stat_string1 = (
        f"{corr_str_spearman:<33}"r"$=$"f"{corr_spearman[0]:.2f}\n"
        f"{corr_str_rm}"r"$=$"f"{corr_within['r'].values[0]:.2f}")

    beta_powers2 = [1, 2.2, 3, 4.5, 4.7, 6]
    updrs_scores2 = [3.1, 4.7, 3.3, 3.7, 3, 4]
    markers = ['o', 'v', 'o', 'v', 'o', 'v']

    df2 = pd.DataFrame({'beta_power': beta_powers2, 'UPDRS': updrs_scores2,
                        'subject': patient_list,
                        'marker': markers,
                        'ch_hemisphere': hemispheres,
                        'color': colors})

    # calc repeated measures correlation
    x = 'beta_power'
    y = 'UPDRS'
    subject = 'subject'
    formula = f"Q('{y}') ~ C(Q('{subject}')) + Q('{x}')"
    model = ols(formula, data=df2).fit()
    df2["pred"] = model.fittedvalues

    corr_spearman = spearmanr(df2.beta_power.values, df2.UPDRS.values)
    corr_within = rm_corr(df2, x='beta_power', y='UPDRS', subject='subject')

    stat_string2 = (
        f"{corr_str_spearman2:<8}"r"$=$"f"{corr_spearman[0]:.2f}\n"
        f"{corr_str_rm2}"r"$=$"f"{corr_within['r'].values[0]:.2f}")

    dfs = [df1, df2]

    fig, axes = plt.subplots(1, 2, figsize=(2.6, 1.25), sharey=True)

    for axi in range(2):
        ax = axes[axi]
        df = dfs[axi]
        for i, subject in enumerate(df.subject.unique()):
            df_sub = df[df['subject'] == subject]
            color = df_sub.color.values[0]
            # Separate loop for each hemisphere to enable different markers
            for hemi in df_sub.ch_hemisphere.unique():
                df_sub_hemi = df_sub[df_sub['ch_hemisphere'] == hemi]
                ax.scatter(df_sub_hemi['beta_power'], df_sub_hemi['UPDRS'],
                           label=subject,
                           marker=df_sub_hemi['marker'].values[0],
                           color=color, s=7, zorder=1,
                           edgecolors='k')
            # Plot repeated measures correlation
            label = r'$r_{rm}$' if i == 2 else None
            sns.regplot(x=x, y="pred", data=df_sub, ax=ax, scatter=False,
                        ci=None, truncate=True, label=label, color=color,
                        line_kws=dict(linewidth=0.5, zorder=1))

        # Plot linear regression across all data
        coef = np.polyfit(df['beta_power'].values, df['UPDRS'].values, 1)
        poly1d_fn = np.poly1d(coef)
        ax.plot(df['beta_power'], poly1d_fn(df['beta_power']), 'k--',
                label='Lin. reg.', lw=.25, zorder=0)

        # Set axis
        ax.set_ylabel(None)
        ax.set_xlabel('Beta power [a.u.]')
        ax.set_ylim(1, 8)

    # Legend handles
    handles = [Patch(color=color) for color in df.color.unique()]
    labels = [subject for subject in df.subject.unique()]
    axes[0].legend(handles, labels, handlelength=1, ncol=3, title='Patient',
                   columnspacing=1, borderaxespad=0.2,
                   handletextpad=0.4, loc='upper left')
    axes[0].set_ylabel('Bradykinesia-Rigidity')

    more = mlines.Line2D([], [], color='k', marker='v', markersize=2, lw=0)
    less = mlines.Line2D([], [], color='k', marker='o', markersize=2, lw=0)
    rm_handle = mlines.Line2D([], [], color='k', lw=.5, linestyle='-')
    lin_reg = mlines.Line2D([], [], color='k', linestyle='--', lw=.25)
    handles = [more, less, lin_reg, rm_handle]
    labels = ['More affected', 'Less affected', 'Lin. reg.', r'$r_{rm}$']
    axes[1].legend(handles, labels, ncol=2, handlelength=.75, columnspacing=1,
                   handletextpad=0.4, borderaxespad=0.2,
                   loc='upper left')

    plt.tight_layout()
    save_dir = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, f'{prefix}repeated_measures_correlation', save_dir,
              bbox_inches=None, transparent=True)

    output_file_path = join(cfg.FIG_PAPER, fig_dir, f"{prefix}_output.txt")
    with open(output_file_path, "w") as output_file:
        print(stat_string1, file=output_file)
        print('\n', file=output_file)
        print(stat_string2, file=output_file)