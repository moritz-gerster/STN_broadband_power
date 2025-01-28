from os.path import join

import matplotlib.pyplot as plt

from scripts.plot_figures.settings import *
from scripts.utils_plot import _save_fig, plot_corr

ylabel = 'Bradykinesia-rigidity'


def _rank_df(df, x, y, repeated_m="subject", remove_ties=True):
    """Convert float values for x and y to rank integers.

    Follows rank repeated measures in
    Donna L. Mohr & Rebecca A. Marcon (2005) Testing for a  ‘within-subjects’
    association in repeated measures data, Journal of Nonparametric Statistics,
    17:3, 347-363, DOI: 10.1080/10485250500038694
    """
    df = df.copy()
    df = _correct_sample_size(df, x, y, repeated_m=repeated_m)
    df = df.dropna(subset=[x, y])
    method = 'average'  # 'Tied values are replaced by their mid-ranks.'
    df[x + '_rank'] = df.groupby(repeated_m)[x].rank(method)
    df[y + '_rank'] = df.groupby(repeated_m)[y].rank(method)

    if remove_ties:
        # Function to filter out tied ranks
        def remove_tied_ranks(df, rank_column):
            return df[df[rank_column] == df[rank_column].astype(int)]

        # Remove ties
        df = remove_tied_ranks(df, x + '_rank')
        df = remove_tied_ranks(df, y + '_rank')
    return df


def _correct_sample_size(df, x, y, repeated_m="subject"):
    """Remove subjects with less than 2 values for x, y, and hue."""
    if repeated_m == 'project':
        return df
    df_copy = df.dropna(subset=[x, y, repeated_m]).copy()

    # remove subjects with only one hemisphere
    group = [repeated_m]
    hemis_subject = df_copy.groupby(group).ch_hemisphere.nunique()
    hemi_both = hemis_subject == df_copy.ch_hemisphere.nunique()
    df_copy = df_copy.set_index(group)[hemi_both].reset_index()
    enough_subs = (df_copy.groupby(repeated_m).ch_hemisphere.nunique()
                   == 2).all()
    if not enough_subs:
        return None

    # filter subjects with less than 2 values for x, y, and hue
    df = df[df.subject.isin(df_copy.subject.unique())]
    return df


def normalized_beta_within(df_norm, fig_dir='Figure7', prefix=''):
    df_norm = df_norm[~df_norm.project.isin(['all'])]
    df_norm_off = df_norm[df_norm.cond == 'off']
    df_norm_on = df_norm[df_norm.cond.isin(['on'])
                         & df_norm.dominant_side_consistent]
    kind = 'normalized'
    corr_method = 'withinRank'
    x = 'beta_low_abs_mean_log'
    Y = 'UPDRS_bradyrigid_contra'

    consistent_str = '_consistent'
    fname = f'{prefix}corr_{corr_method}_{kind}_off&on_{x}_{Y}{consistent_str}'

    # leg_kws needs to be passed separately for some reason
    leg_kws1 = dict(handlelength=0, markerscale=0, frameon=False,
                    bbox_to_anchor=(-.1, 1))
    leg_kws2 = leg_kws1.copy()
    leg_kws2['bbox_to_anchor'] = (-.05, 1)

    plot_kwargs = dict(
        corr_method=corr_method,
        n_perm=N_PERM_CORR,
        xlabel=r'Relative Low $\beta$ power [%]',
        add_sample_size=False,
        )

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(2.5, 1.5))
    plot_corr(axes[0], df_norm_off, x, Y, leg_kws=leg_kws1, ylabel=ylabel,
              **plot_kwargs)
    plot_corr(axes[1], df_norm_on, x, Y, leg_kws=leg_kws2,
              **plot_kwargs)
    axes[1].set_ylabel(None)
    plt.tight_layout()
    _save_fig(fig, fname, join(cfg.FIG_PAPER, fig_dir),
              bbox_inches=None, transparent=True)


def periodic_gamma_within(df_per, fig_dir='Figure7', prefix='',
                          exemplary_subs=None, figsize=(1.7, 1.34),
                          bbox_to_anchor=(-0, 1)):
    df_per_on = df_per[df_per.cond.isin(['on'])
                       & df_per.dominant_side_consistent
                       & ~df_per.project.isin(['all'])]
    kind = 'periodic'
    x = 'gamma_fm_mean_log'
    Y = 'UPDRS_bradyrigid_contra'
    corr_method = 'withinRank'
    cond = 'on'
    consistent_str = '_consistent'
    fname = f'{prefix}corr_{corr_method}_{kind}_{cond}_{x}_{Y}{consistent_str}'

    # leg_kws needs to be passed separately for some reason
    leg_kws1 = dict(handlelength=0, markerscale=0, frameon=False,
                    bbox_to_anchor=bbox_to_anchor
                    )
    plot_kwargs = dict(
        corr_method=corr_method,
        n_perm=N_PERM_CORR,
        xlabel=r'$\gamma$ power',
        ylabel=ylabel,
        add_sample_size=True,
        subs_special=exemplary_subs,
    )

    # figsize_single = (1.7, 1.34)
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=figsize)
    plot_corr(ax, df_per_on, x, Y, leg_kws=leg_kws1, **plot_kwargs)
    plt.tight_layout()
    _save_fig(fig, fname, join(cfg.FIG_PAPER, fig_dir),
              bbox_inches=None, transparent=True)


def absolute_gamma_within(df_abs, fig_dir='Figure8', prefix=''):
    df_abs = df_abs[~df_abs.project.isin(['all'])]
    df_abs_off = df_abs[df_abs.cond == 'off']
    df_abs_on = df_abs[df_abs.cond.isin(['on'])
                       & df_abs.dominant_side_consistent]
    kind = 'absolute'
    corr_method = 'withinRank'
    x = 'gamma_mid_abs_max5Hz_log'
    Y = 'UPDRS_bradyrigid_contra'
    consistent_str = '_consistent'
    fname = f'{prefix}corr_{corr_method}_{kind}_off&on_{x}_{Y}{consistent_str}'

    # leg_kws needs to be passed separately for some reason
    leg_kws1 = dict(handlelength=0, markerscale=0, frameon=False,
                    bbox_to_anchor=(-.1, 1))
    leg_kws2 = leg_kws1.copy()
    leg_kws2['bbox_to_anchor'] = (-.05, 1)

    plot_kwargs = dict(
        corr_method=corr_method,
        n_perm=N_PERM_CORR,
        xlabel=cfg.PLOT_LABELS[x],
        add_sample_size=False,
        subs_special=cfg.EXEMPLARY_SUBS_APERIODIC,
    )

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(2.3, 1.3))
    plot_corr(axes[0], df_abs_off, x, Y, leg_kws=leg_kws1, ylabel=ylabel,
              **plot_kwargs)
    plot_corr(axes[1], df_abs_on, x, Y, leg_kws=leg_kws2,
              **plot_kwargs)
    axes[1].set_ylabel(None)
    plt.tight_layout()
    _save_fig(fig, fname, join(cfg.FIG_PAPER, fig_dir),
              bbox_inches=None, transparent=True)


def aperiodic_within(df_per, fig_dir='Figure8', prefix=''):
    df_per = df_per[~df_per.project.isin(['all'])]
    df_per_off = df_per[df_per.cond == 'off']
    df_per_on = df_per[df_per.cond.isin(['on'])
                       & df_per.dominant_side_consistent]
    kind = 'periodic'
    corr_method = 'withinRank'
    x = 'full_fm_band_aperiodic_log'
    Y = 'UPDRS_bradyrigid_contra'
    consistent_str = '_consistent'
    fname = f'{prefix}corr_{corr_method}_{kind}_off&on_{x}_{Y}{consistent_str}'

    # leg_kws needs to be passed separately for some reason
    leg_kws1 = dict(handlelength=0, markerscale=0, frameon=False,
                    bbox_to_anchor=(-.1, 1))
    leg_kws2 = leg_kws1.copy()
    leg_kws2['bbox_to_anchor'] = (-.05, 1)

    plot_kwargs = dict(
        corr_method=corr_method,
        n_perm=N_PERM_CORR,
        xlabel='Aperiodic power (1-60 Hz)',
        add_sample_size=False,
        subs_special=cfg.EXEMPLARY_SUBS_APERIODIC,
    )

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(2.3, 1.3))
    plot_corr(axes[0], df_per_off, x, Y, leg_kws=leg_kws1, ylabel=ylabel,
              **plot_kwargs)
    plot_corr(axes[1], df_per_on, x, Y, leg_kws=leg_kws2,
              **plot_kwargs)
    axes[1].set_ylabel(None)
    plt.tight_layout()
    _save_fig(fig, fname, join(cfg.FIG_PAPER, fig_dir),
              bbox_inches=None, transparent=True)
