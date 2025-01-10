import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import scripts.config as cfg
from scripts.plot_figures.settings import *
from scripts.utils import _average_hemispheres, _corr_results
from scripts.utils_plot import _axes2d, _save_fig, convert_pvalue_to_asterisks


def _correct_sample_size(df, x, repeated_m="subject"):
    """Remove subjects with less than 2 values for x, y, and hue."""
    if repeated_m == 'project':
        return df
    df_copy = df.dropna(subset=[x, repeated_m]).copy()

    # remove subjects with only one hemisphere
    group = [repeated_m]
    hemis_subject = df_copy.groupby(group).ch_hemisphere.nunique()
    hemi_both = hemis_subject == df_copy.ch_hemisphere.nunique()
    df_copy = df_copy.set_index(group)[hemi_both].reset_index()
    # assert no subjects with only one hemisphere
    enough_subs = (df_copy.groupby(repeated_m).ch_hemisphere.nunique() == 2).all()
    if not enough_subs:
        return None

    # filter subjects with less than 2 values for x, y, and hue
    df = df[df.subject.isin(df_copy.subject.unique())]
    return df


def _rank_df(df, rank_cols, repeated_m="subject", remove_ties=True):
    """Convert float values for x and y to rank integers.

    Follows rank repeated measures in
    Donna L. Mohr & Rebecca A. Marcon (2005) Testing for a  ‘within-subjects’
    association in repeated measures data, Journal of Nonparametric Statistics,
    17:3, 347-363, DOI: 10.1080/10485250500038694
    """
    # Function to filter out tied ranks
    def remove_ties(df, rank_column):
        return df[df[rank_column] == df[rank_column].astype(int)]

    df = df.copy()
    for cond in ['off', 'on']:
        for x in rank_cols:
            # initialize always new in case removing ties yields empty df
            df_rank = df.copy()
            df_rank_cond = df_rank[df_rank.cond == cond]
            df_rank_cond = _correct_sample_size(df_rank_cond, x,
                                                repeated_m=repeated_m)
            df_rank_cond[x +
                         '_rank'] = df_rank_cond.groupby(repeated_m)[x].rank()
            if remove_ties:
                df_rank_cond = remove_ties(df_rank_cond, x + '_rank')
            df.loc[(df.cond == cond), x + '_rank'] = df_rank_cond[x + '_rank']
    return df


def get_correlation_df_multi(dataframes,
                             kinds=['normalized', 'absolute', 'periodic'],
                             corr_methods=['spearman', 'withinRank'],
                             remove_ties=True,
                             project='all', bands=BANDS,
                             conds=['off', 'on'], n_perm=N_PERM_CORR):
    """Get dataframe to plot barplot."""
    plot_dic = cfg.PLOT_LABELS_SHORT
    band_dic = cfg.BAND_NAMES_GREEK_SHORT

    df_corrs = []
    for kind in kinds:
        if kind == 'normalized':
            df = dataframes['df_norm'].copy()
            df = df[(df.project == 'all')]

            band_cols = [f'{band}_abs_mean_log' for band in bands]
            band_nmes = [band_dic[band] for band in bands]
        elif kind == 'absolute':
            df = dataframes['df_abs'].copy()
            df = df[(df.project == 'all') & df.fm_exponent.notna()]
            band_cols = [f'{band}_abs_mean_log' for band in bands]
            band_nmes = [band_dic[band] for band in bands]
        elif kind == 'periodic':
            df = dataframes['df_per'].copy()
            df = df[(df.project == 'all')]
            band_cols = [f'{band}_fm_mean_log' for band in bands]
            band_nmes = [band_dic[band] for band in bands]
        elif kind == 'periodicAP':

            # Add apperiodic power
            def ap_pwr(df, f_low=5, f_high=95):
                # fm limits all projects
                ap_pwr = df.fm_psd_ap_fit_log
                freqs = df.fm_freqs
                mask = (freqs >= f_low) & (freqs <= f_high)
                return ap_pwr[mask].sum()
            df['ap_power'] = df.apply(ap_pwr, f_low=1, f_high=60, axis=1)

            band_cols = ['fm_exponent', 'fm_offset_log', 'ap_power']
            band_nmes = ['1/f exponent',
                         plot_dic['fm_offset_log'],
                         'Ap. pwr. 1-60 Hz']
        msg = f'{set(band_cols) - set(band_nmes)}'
        assert len(band_cols) == len(band_nmes), msg

        for cond in conds:
            df_cond = df[df.cond == cond]

            if cond == 'on':
                # only include consistent asymmetry for ON subjects to
                # exclude possible LDOPA side effects
                df_cond = df_cond[df_cond.dominant_side_consistent]

            # remove subject with only one condition
            subject_counts = df_cond.subject.value_counts()
            valid_subjects = subject_counts[subject_counts == 2].index
            df_cond = df_cond[df_cond.subject.isin(valid_subjects)]
            for i, x in enumerate(band_cols):
                if 'offset' in x or 'exponent' in x:
                    color = (cfg.COLOR_DIC['periodicAP'] if cond == 'off'
                             else cfg.COLOR_DIC['periodicAP2'])
                else:
                    color = (cfg.COLOR_DIC[kind] if cond == 'off'
                             else cfg.COLOR_DIC[f'{kind}2'])
                for corr_method in corr_methods:
                    if corr_method == 'spearman':
                        use_corr_method = corr_method
                        y = 'UPDRS_III'
                        df_rho = _average_hemispheres(df_cond, x, y)
                    elif corr_method == 'withinRank':
                        df_rho = df_cond
                        if '_rank' in x:
                            # use within method for ranked data
                            use_corr_method = 'within'
                            y = 'UPDRS_bradyrigid_contra_rank'
                        else:
                            use_corr_method = corr_method
                            y = 'UPDRS_bradyrigid_contra'
                    corr_kwargs = dict(df_rho=df_rho, x=x, y=y,
                                       remove_ties=remove_ties,
                                       corr_method=use_corr_method,
                                       n_perm=n_perm)
                    rho, sample_size, label, _, _ = _corr_results(
                        **corr_kwargs
                        )
                    if rho is None:
                        msg = f'No correlation found for {corr_kwargs}'
                        raise ValueError(msg)
                        # continue
                    pval = float(label.split(' ')[-1].strip('p='))
                    dic = {'project': project, 'rho': rho,
                           'band_col': x, 'band_nme': band_nmes[i],
                           'kind': kind, 'color': color,
                           'sample_size': sample_size, 'pval': pval,
                           'n_perm': n_perm, 'cond': cond,
                           'corr_method': corr_method, 'y': y}
                    df_corrs.append(dic)
    df_corrs = pd.DataFrame(df_corrs)
    return df_corrs


def barplot_biomarkers(df_corrs, fig_dir='Figure7', prefix='',
                       figsize=(7, 1.5), ylim=None, output_file=None):

    kinds = df_corrs.kind.unique()

    hue_order = [cond for cond in cfg.COND_ORDER
                 if cond in df_corrs.cond.unique()]

    n_cols = len(kinds)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=True)
    axes = _axes2d(axes, 1, n_cols)[0]

    for i, kind in enumerate(kinds):
        ax = axes[i]
        df_kind = df_corrs[df_corrs.kind == kind]
        band_nmes = df_kind.band_nme.unique()
        sns.barplot(df_kind, ax=ax, y='rho', x='band_nme', hue='cond',
                    hue_order=hue_order,
                    legend=False)
        ax.set_ylim(ylim)
        ax.set_xlabel(None)
        # indicate significance
        _, ymax = ax.get_ylim()
        print(f'\n{kind}:', file=output_file)
        for i, cond in enumerate(hue_order):
            print(f'{cond}:', file=output_file)
            for j, band_col in enumerate(band_nmes):
                bar = ax.containers[i][j]
                mask = (df_kind.cond == cond) & (df_kind.band_nme == band_col)
                color = df_kind[mask].color.values[0]
                bar.set_facecolor(color)
                df_band = df_kind[(df_kind.band_nme == band_col) &
                                  (df_kind.cond == cond)]
                rho = df_band.rho.values[0]
                pvalue = df_band.pval.values[0]
                text = convert_pvalue_to_asterisks(pvalue)
                x_bar = bar.get_x() + bar.get_width() / 2
                ax.annotate(text,
                            xy=(x_bar, ymax*.9),
                            va='bottom',
                            ha='center',
                            fontsize=FONTSIZE_ASTERISK,
                            color=color
                            )
                print(f'{band_col}: rho={rho:.2f}, p={pvalue}',
                      file=output_file)

    axes[0].set_ylabel(r"$r_{\text{rank rm}}$")
    n_perm = df_corrs.n_perm.unique()[0]
    kind_str = '_'.join(df_corrs.kind.unique())
    fname = f'{fig_dir}/{prefix}biomarkers_{kind_str}_nperm={n_perm}'
    plt.tight_layout()
    _save_fig(fig, fname, cfg.FIG_PAPER,
              transparent=True, bbox_inches=None)
