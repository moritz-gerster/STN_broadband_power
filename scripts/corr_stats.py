"""Provide some functions used for correlatuion statistics."""
import copy
import pingouin as pg
import pandas as pd

import numpy as np
from scipy.stats import kendalltau, norm, pearsonr, spearmanr
from tqdm import trange
from joblib import Parallel, delayed

import scripts.config as cfg

rmethod = {"pearson": lambda x, y: pearsonr(x, y)[0],
           "spearman": lambda x, y: spearmanr(x, y)[0],
           "kendall": lambda x, y: kendalltau(x, y)[0]}


def p_perm(x, y, corr_method="spearman", n_perm=10000):
    """
    Return permuted p-vales for correlation coefficient of arrays x, y.

    Parameters
    ----------
    x : np.ndarray
        x variable.
    y : np.ndarray
        y variable.
    corr_method : string, optional
        Correlation method. Options: "pearson", "spearman", or "kendall".
    p : integer, optional
        Number of permutations. The default is 10000.

    Returns
    -------
    integer
        p-value of significance of correlation coefficient.
    """
    rng = np.random.default_rng(1)  # Use Generator for reproducibility
    x = copy.deepcopy(x)
    y = copy.deepcopy(y)
    r = rmethod[corr_method](x, y)
    r_perm = np.zeros(n_perm)
    for i in trange(n_perm):
        rng.shuffle(x)
        r_perm[i] = rmethod[corr_method](x, y)
    return (len(np.where(np.abs(r_perm) >= np.abs(r))[0]) + 1) / (n_perm + 1)


def p_perm_parallel(x, y, corr_method="spearman", n_perm=10000, n_jobs=-1):
    rng = np.random.default_rng(1)  # Use Generator for reproducibility
    x = copy.deepcopy(x)
    y = copy.deepcopy(y)
    r = rmethod[corr_method](x, y)

    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_perm)

    def compute_permutation(seed):
        # Use individual seed for reproducibility
        local_rng = np.random.default_rng(seed)
        x_perm = copy.deepcopy(x)  # Ensure x is not overwritten
        local_rng.shuffle(x_perm)
        return rmethod[corr_method](x_perm, y)

    r_perm = Parallel(n_jobs=n_jobs)(delayed(compute_permutation)(seed)
                                     for seed in seeds)
    perm_larger = np.abs(r_perm) >= np.abs(r)
    p_value = (np.sum(perm_larger) + 1) / (n_perm + 1)
    return p_value


def compare_corrs_perm(x1, y1, x2, y2, corr_method="spearman", n_perm=10000,
                       n_jobs=-1, tail='two-sided'):
    x1, y1 = copy.deepcopy(x1), copy.deepcopy(y1)
    x2, y2 = copy.deepcopy(x2), copy.deepcopy(y2)

    # Calculate the original correlation coefficients
    r1 = rmethod[corr_method](x1, y1)
    r2 = rmethod[corr_method](x2, y2)

    # Compute the observed difference in correlations
    r_diff_observed = r1 - r2

    # Function to compute the permuted difference in correlations
    def compute_permutation():
        # Shuffle copies of x1 and x2 independently
        x1_perm = np.random.permutation(x1)
        x2_perm = np.random.permutation(x2)
        r1_perm = rmethod[corr_method](x1_perm, y1)
        r2_perm = rmethod[corr_method](x2_perm, y2)
        return r1_perm - r2_perm

    # Run the permutation test in parallel
    r_diff_perm = Parallel(n_jobs=n_jobs)(delayed(compute_permutation)()
                                          for _ in range(n_perm))

    # Calculate p-value based on the type of test
    if tail == 'two-sided':
        perm_larger = np.abs(r_diff_perm) >= np.abs(r_diff_observed)
    elif tail == 'r1_greater':
        perm_larger = r_diff_perm >= r_diff_observed
    elif tail == 'r2_greater':
        perm_larger = r_diff_perm <= r_diff_observed
    else:
        msg = ("Invalid value for 'tail'. Choose from 'two-sided', "
               "'r1_greater', 'r2_greater'.")
        raise ValueError(msg)
    p_value = (np.sum(perm_larger) + 1) / (n_perm + 1)
    return p_value


def p_value_df(corr_method="spearman", stat_method=None, n_perm=None):
    """Return lambda function to create a p-value dataframe.

    To use this function, use pandas .corr method and give p_value_df as
    method argument. Example:

        df.corr(p_value_df("spearman", "spearman"))

        or for permutation testing

        df.corr(p_value_df("spearman", "perm_parallel", n_perm=10000))

    Parameters
    ----------
    corr_method : string
        Which correlation method to choose for p-value calculation.
        Options: "pearson", "spearman", or "kendall".
    stat_method : string or None
        Calculate p-values parameterically (e.g., "pearson") or with
        (parallel) permutation testing (e.g., "perm_parallel").
    n_perm : int or None
        Number of permutations.

    Returns
    -------
    lambda function
        Function to calculate p-value.
    """
    if stat_method is None:
        stat_method = corr_method
    if stat_method == "pearson":
        return lambda x, y: pearsonr(x, y)[1]
    elif stat_method == "spearman":
        return lambda x, y: spearmanr(x, y)[1]
    elif stat_method == "kendall":
        return lambda x, y: kendalltau(x, y)[1]
    elif stat_method == "perm":
        return lambda x, y: p_perm(x, y, corr_method=corr_method,
                                   n_perm=n_perm)
    elif stat_method == "perm_parallel":
        return lambda x, y: p_perm_parallel(x, y, corr_method=corr_method,
                                            n_perm=n_perm)


def sample_size_df():
    """Return lambda function to create a sample size dataframes.

    To use this function, use pandas .corr method and give p_value_df as
    method argument. Example:

        df.corr(sample_size_df)

    Returns
    -------
    lambda function
        Function to calculate sample sizes.
    """
    return lambda x, y: np.isfinite(x * y).sum()


def p_perm_cond(real, perm, p):
    """Calc p-value between two conditions based on real and permuted values.

    Parameters
    ----------
    real : float
        Actual value (for example pearson's r).
    perm : 1xp ndarray
        Distribution of permuted values.
    p : integer
        Number of permutations.

    Returns
    -------
    float
        p-value.
    """
    return len(np.where(np.abs(perm) >= np.abs(real))[0] + 1) / (p + 1)


def rz_ci(r, n, conf_level=0.95):
    zr_se = (1 / (n - 3)) ** 0.5
    moe = norm.ppf(1 - (1 - conf_level) / float(2)) * zr_se
    zu = np.arctanh(r) + moe
    zl = np.arctanh(r) - moe
    return np.arctanh((zl, zu))


def independent_corr(r1, r2, n, n2=None, twotailed=True, conf_level=0.95,
                     method='fisher'):
    """Calc p-values of correlation coefficients using fisher's method.

    ___________________
    Regarding Fisher Method:

    https://onlinelibrary.wiley.com/doi/epdf/10.1002/9781118445112.stat02802
    Treating the Spearman coefficients as though they were Pearson
    coefficients and using the standard Fisher's z‐transformation and
    subsequent comparison was more robust with respect to Type I error than
    either ignoring the nonnormality and computing Pearson coefficients or
    converting the Spearman coefficients to Pearson equivalents prior to
    transformation.
    -> It is statistically OK to not use permutation testing for the p-values
    of correlation coefficients.

    ... from the abstract of

    Myers, Leann, and Maria J. Sirois. "Spearman correlation coefficients,
    differences between." Encyclopedia of statistical sciences (2004).
    ___________________

    Calculates the statistic significance between two independent correlation
    coefficients.
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between a and b
    @param n: number of elements in xy
    @param n2: number of elements in ab (if distinct from n)
    @param twotailed: whether to calculate a one or two tailed test,
    only works for 'fisher' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'fisher' or 'zou'
    @return: z and p-val
    """
    if method == 'fisher':
        xy_z = np.arctanh(r1)
        ab_z = np.arctanh(r2)
        if n2 is None:
            n2 = n

        se_diff_r = np.sqrt(1/(n - 3) + 1/(n2 - 3))
        diff = xy_z - ab_z
        z = abs(diff / se_diff_r)
        p = 1 - norm.cdf(z)
        if twotailed:
            p *= 2
        return z, p

    elif method == 'zou':
        L1 = rz_ci(r1, n, conf_level=conf_level)[0]
        U1 = rz_ci(r1, n, conf_level=conf_level)[1]
        L2 = rz_ci(r2, n2, conf_level=conf_level)[0]
        U2 = rz_ci(r2, n2, conf_level=conf_level)[1]
        lower = r1 - r2 - pow((pow((r1 - L1), 2) + pow((U2 - r2), 2)), 0.5)
        upper = r1 - r2 + pow((pow((U1 - r1), 2) + pow((r2 - L2), 2)), 0.5)
        return lower, upper
    else:
        raise Exception('Wrong method!')


def _get_freqs_correlation_stats(df, x, y, average_hemispheres=False,
                                 xmax=None, n_perm=10000,
                                 remove_ties=True,
                                 corr_method='spearman'):
    assert len(df.cond.unique()) == 1, "More than one condition"
    assert len(df.project.unique()) == 1, "More than one project"
    # Averaging only possible across subjects, not within
    if corr_method.startswith('within'):
        assert not average_hemispheres, "No averaging for within"

    df_corr = _get_freqs_correlation(df, x, y, x_max=xmax,
                                     remove_ties=remove_ties,
                                     corr_method=corr_method, n_perm=n_perm,
                                     average_hemispheres=average_hemispheres)
    # Correlation frequency smoothing in Hz. Smoothing correlations after
    # calculation similar to smoothing psd before calculation. However,
    # after calculation appears smoother.
    # if rolling_mean != 1:
    #     _smooth_corr_frequencies(x, y, rolling_mean, df_corr)
    # df_corr['rolling_mean'] = f'{rolling_mean}Hz'
    return df_corr


def _smooth_corr_frequencies(x, y, rolling_mean, df_corr):
    y_plot = f"corr_{x}_{y}"
    y_pval = f"pval_{x}_{y}"
    rol_mean = df_corr[y_plot].rolling(rolling_mean, center=True).mean()
    pval_mean = df_corr[y_pval].rolling(rolling_mean, center=True).mean()
    df_corr[y_plot] = rol_mean
    df_corr[y_pval] = pval_mean


def _smooth_psd(df, x, rolling_mean):
    df = df.copy()
    x_smooth = df[x].rolling(rolling_mean, center=True).mean()
    df[x] = x_smooth
    return df


def _get_freqs_correlation(df_merged, x, y, x_max=None,
                           remove_ties=True,
                           corr_method='spearman', n_perm=10000,
                           average_hemispheres=False):
    """Only works for single condition and single project."""
    if df_merged.project.unique() != ['all']:
        assert len(df_merged.psd_kind.unique()) == 1, "More than one psd kind"
    psd_kind = df_merged.psd_kind.unique()[0]
    assert len(df_merged.cond.unique()) == 1, "More than one condition"
    cond = df_merged.cond.unique()[0]
    assert len(df_merged.project.unique()) == 1, "More than one project"
    project = df_merged.project.unique()[0]
    assert len(df_merged.fm_params.unique()) == 1, "More than one fm params"
    fm_params = df_merged.fm_params.unique()[0]
    stat_method = 'perm_parallel' if n_perm is not None else 'spearman'
    if 'offon' in cond:
        msg = 'Always use log psd for off-on to say consistent'
        assert 'log' in x, msg

    if x in ['psd', 'asd', 'psd_log']:
        freqs = "psd_freqs"
    elif 'fm' in x:
        freqs = "fm_freqs"

    df_merged = df_merged.dropna(subset=[x, y])
    if average_hemispheres:
        group = df_merged.groupby(["subject"])
        df_merged.loc[:, x] = group[x].transform("mean")
        df_merged.loc[:, y] = group[y].transform("mean")
        df_merged = df_merged.drop_duplicates('subject')
    else:
        assert not df_merged[['subject', 'ch_hemisphere']
                             ].duplicated().sum(), "Duplicates found"

    # Explode df
    sample_size = df_merged.shape[0]

    df_freqs = df_merged.explode([freqs, x])
    df_freqs[freqs] = df_freqs[freqs].astype(int)
    df_freqs[x] = df_freqs[x].astype(float)
    if x_max:
        df_freqs = df_freqs[df_freqs[freqs] <= x_max]

    # Smooth psd after correlation calculation
    # df_freqs = _smooth_psd(df_freqs, x, rolling_mean)

    group = [freqs]
    if corr_method in ["spearman", "pearson", "kendall"]:
        # Corr dataframe
        corr_cols = [x, y]
        groupby = df_freqs[group + corr_cols].groupby(group)
        df_corr = groupby.corr(method=corr_method, numeric_only=True)
        df_corr = df_corr.reset_index()
        df_corr = df_corr[df_corr['level_1'] == y]
        df_corr.drop(columns=['level_1', y], inplace=True)
        df_corr.rename(columns={x: f"corr_{x}_{y}"}, inplace=True)

        # p-value dataframe
        pval_method = p_value_df(corr_method=corr_method,
                                 stat_method=stat_method, n_perm=n_perm)
        df_pval = groupby.corr(method=pval_method, numeric_only=True)
        df_pval = df_pval.reset_index()
        df_pval = df_pval[df_pval['level_1'] == y]
        df_pval.drop(columns=['level_1', y], inplace=True)
        df_pval.rename(columns={x: f"pval_{x}_{y}"}, inplace=True)

        # merge correlations and pvalues
        df_both = df_corr.merge(df_pval, on=group)
    elif corr_method in ["within", "withinRank"]:
        assert y != 'UPDRS_III', "Need hemisphere specific score for within"
        frequencies = []
        correlations = []
        pvalues = []
        for freq in df_freqs[freqs].unique():
            mask = df_freqs[freqs] == freq
            vals = _within_corr_from_df(df_freqs[mask], x, y, corr_method,
                                        remove_ties=remove_ties,
                                        n_perm=n_perm)
            # overwrite sample_size for within
            rho, pval, sample_size, _ = vals
            frequencies.append(freq)
            correlations.append(rho)
            pvalues.append(pval)
        df_both = pd.DataFrame({freqs: frequencies,
                                f"corr_{x}_{y}": correlations,
                                f"pval_{x}_{y}": pvalues})

    # merged dataframe
    df_both['corr_method'] = corr_method
    df_both['project'] = project
    df_both['psd_kind'] = psd_kind
    df_both['fm_params'] = fm_params
    df_both['cond'] = cond
    df_both['n_perm'] = n_perm
    df_both['sample_size'] = sample_size
    df_both['hemispheres_averaged'] = average_hemispheres
    df_both['x'] = x
    df_both['y'] = y
    # df_both['rolling_mean'] = f'{rolling_mean}Hz'

    # # I don't need this for now since I don't use condition as a hue
    #
    # n_on = len(df_merged[(df_merged.cond == "on")
    #                      & (df_merged.ch_hemisphere == "L")])
    # n_off = len(df_merged[(df_merged.cond == "off")
    #                       & (df_merged.ch_hemisphere == "L")])
    #
    # # Add condition difference
    # msg = "More than one correlation found"
    # if df_both.cond.unique().size == 2:
    #     for hemi in df_both.ch_hemisphere.unique():
    #         for freq in df_both.psd_freqs.unique():
    #             mask = ((df_both.psd_freqs == freq)
    #                     & (df_both.ch_hemisphere == hemi))
    #             corr_on = df_both[mask & (df_both.cond == "on")].corr_psd
    #             corr_off = df_both[mask & (df_both.cond == "off")].corr_psd
    #             assert len(corr_on) == 1 and len(corr_off) == 1, msg
    #             df_both.loc[mask, "pval_psd_cond"] = independent_corr(
    #                         corr_on.iloc[0],
    #                         corr_off.iloc[0], n_on, n_off)[1]

    #     pval_off = df_both[df_both.cond == "off"]["pval_psd_cond"]
    #     pval_on = df_both[df_both.cond == "on"]["pval_psd_cond"]
    #     assert np.all(pval_off.reset_index(drop=True)
    #                   == pval_on.reset_index(drop=True))

    #     # Hemisphere mean
    #     for freq in df_both.psd_freqs.unique():
    #         mask = ((df_both.psd_freqs == freq)
    #                 & (df_both.ch_hemisphere == "L"))
    #         corr_on = df_both[mask & (df_both.cond == "on")].corr_psd_mean
    #         corr_off = df_both[mask & (df_both.cond == "off")].corr_psd_mean
    #         assert len(corr_on) == 1 and len(corr_off) == 1, msg
    #         df_both.loc[mask, "pval_psd_mean_cond"] = independent_corr(
    #                 corr_on.iloc[0], corr_off.iloc[0], n_on, n_off)[1]
    #         assert len(corr_on) == 1 and len(corr_off) == 1, msg
    #     hemi = (df_both.ch_hemisphere == "L")
    #     pval_off = df_both[(df_both.cond == "off")
    #                        & hemi]["pval_psd_mean_cond"]
    #     pval_on = df_both[(df_both.cond == "on")
    #                       & hemi]["pval_psd_mean_cond"]
    #     assert np.all(pval_off.reset_index(drop=True)
    #                   == pval_on.reset_index(drop=True))
    return df_both


def corr_over_freq_pvals(ax, df_both, X, Y, y_height=1):
    freqs = df_both[X].unique()
    fmax = freqs.max()
    pval_sig = np.ones(len(freqs))

    for multiple_comparison in [False, True]:
        psig = 0.05 / fmax if multiple_comparison else 0.05
        for color, cond, height in zip([cfg.COLOR_OFF, cfg.COLOR_ON],
                                       ["off", "on"],
                                       [y_height, y_height + .05]):
            height = height + .05 if multiple_comparison else height
            color = "b" if multiple_comparison and cond == "off" else color
            color = "r" if multiple_comparison and cond == "on" else color
            significant = df_both[df_both.cond == cond][Y] < psig
            ax.plot(freqs[significant], pval_sig[significant]*height, ".",
                    c=color)

    # condition difference
    if df_both.cond.unique().size == 2:
        column = Y + "_cond"
        cond_significant = df_both[df_both.cond == "on"][column] < 0.05
        ax.plot(freqs[cond_significant], pval_sig[cond_significant], "x",
                c="g")


def corr_freq_pvals(ax, df_both, X, Y, y_height=0):
    freqs = df_both[X].unique()
    pval_sig = np.ones(len(freqs))

    psig = 0.05
    color = "r"
    significant = df_both[Y] < psig
    ax.plot(freqs[significant], pval_sig[significant]*y_height, ".", c=color,
            markersize=1, label=f"p < {psig}")


def _corr_results(df_rho, x, y, corr_method, row_idx=None, sig_threshold=0.05,
                  repeated_m="subject", add_sample_size=True, R2=False,
                  n_perm=10000, pval_string=True, remove_ties=True):
    if row_idx is None:
        row_idx = slice(None)
        hue_str = ''
    else:
        hue_str = None
    df_rho = df_rho.loc[row_idx].dropna(subset=[x, y]).copy()
    if not len(df_rho):
        return None, None, None, None, None
    if corr_method in ["spearman", "pearson", "kendall"]:
        vals = _corr_from_df(df_rho, x, y, corr_method, R2, n_perm=n_perm)
    elif corr_method in ["within", "withinRank"]:
        vals = _within_corr_from_df(df_rho, x, y, corr_method, repeated_m,
                                    n_perm=n_perm, remove_ties=remove_ties)
    else:
        raise ValueError(f"corr_method '{corr_method}' not recognized")
    rho, pval, sample_size, ci = vals

    weight = "bold" if pval < sig_threshold else "normal"

    # Legend entry:
    if corr_method.startswith('within'):
        sample_size_str = f'2x{sample_size}'
    else:
        sample_size_str = f'{sample_size}'
    if add_sample_size:
        n = f" n={sample_size_str}: "  # \n
    else:
        n = ""
    corr_coeff_descr = _corr_string(corr_method, R2)
    corr = corr_coeff_descr + f"{rho:.2f}, "
    if hue_str != '':
        try:
            hue_str = cfg.PROJECT_DICT[row_idx]
        except (KeyError, TypeError):
            try:
                hue_str = cfg.COND_DICT[row_idx] + ": "
            except (KeyError, TypeError):
                try:
                    hue_str = str(row_idx)
                except (KeyError, TypeError):
                    hue_str = ''
    hue_str = '' if hue_str == 'slice(None, None, None)' else hue_str
    if pval_string:
        pval = f"p={pval:1.0e}" if abs(pval) < 0.001 else f"p={pval:.3f}"
        pval = hue_str + n + corr + pval
    return rho, sample_size, pval, weight, ci


def _corr_string(corr_method, R2):
    """Return correlation coefficient description for legend."""
    if corr_method == "pearson":
        if R2:
            corr_coeff_descr = r"$R^2=$"
        else:
            corr_coeff_descr = r"$r=$"
    elif corr_method == "spearman":
        assert not R2, "R2 inappropriate for rank correlation"
        corr_coeff_descr = r"${\rho}$="  # save space
    elif corr_method == "kendall":
        assert not R2, "R2 inappropriate for rank correlation"
        corr_coeff_descr = r"$\tau=$"
    elif corr_method == 'withinRank':
        assert not R2, "R2 inappropriate for rank correlation"
        corr_coeff_descr = r"$r_{\text{rank rm}}$="
    elif corr_method == 'within':
        assert not R2, "R2 inappropriate for within correlation"
        corr_coeff_descr = r"$r_{\text{rm}}$="
    return corr_coeff_descr


def _within_corr_from_df(df_rho, x, y, corr_method, repeated_m='subject',
                         n_perm=10000, remove_ties=True):
    df_rho = _correct_sample_size(df_rho, x, y, remove_ties=remove_ties,
                                  repeated_m=repeated_m)
    if df_rho is None:
        return 0, 1, 0, None
    if corr_method == 'withinRank':
        df_rho = _rank_df(df_rho, x, y, repeated_m=repeated_m,
                          remove_ties=remove_ties)
    sample_size = len(df_rho[repeated_m].unique())
    xy_nonzero = (df_rho[x] != 0) & (df_rho[y] != 0)
    if df_rho[xy_nonzero].subject.nunique() < 3:
        # within requires at least 3 finite subjects
        rho = 0
        pval = 1
        ci = None
    else:
        stats = pg.rm_corr(data=df_rho, x=x, y=y, subject=repeated_m)
        rho = stats.r.iloc[0]
        if (corr_method == 'within') or (n_perm is None):
            pval = stats.pval.iloc[0]
            ci = stats["CI95%"].iloc[0]
        elif corr_method == 'withinRank':
            pval = _permutation_pvalue(df_rho, x, y, repeated_m, rho,
                                       sample_size, n_perm=n_perm)
            ci = None
            # # sanity check (removed for efficiency)
            # if corr_method == 'withinRank' and repeated_m == 'subject':
            #     rhos_pearson = []
            #     rhos_spearman = []
            #     for repeated in df_rho[repeated_m].unique():
            #         df_sub = df_rho[df_rho[repeated_m] == repeated]
            #         rho_pearson = df_sub[x].corr(df_sub[y], method='pearson')
            #         rho_spearman = df_sub[x].corr(df_sub[y],
            #                                       method='spearman')
            #         rhos_pearson.append(rho_pearson)
            #         rhos_spearman.append(rho_spearman)
            #     rho_pearson = np.mean(rhos_pearson)
            #     rho_spearman = np.mean(rhos_spearman)
            #     assert np.allclose(rho, rho_pearson), 'Correlation  wrong!'
            #     assert np.allclose(rho, rho_spearman), 'Correlation  wrong!'
    return rho, pval, sample_size, ci


def _correct_sample_size(df, x, y, repeated_m="subject", remove_ties=False):
    """Remove subjects with less than 2 values for x, y, and hue."""
    if repeated_m == 'project':
        return df
    df_copy = df.dropna(subset=[x, y, repeated_m]).copy()

    # remove subjects with only one hemisphere
    group = [repeated_m]
    hemis_subject = df_copy.groupby(group).ch_hemisphere.nunique()
    hemi_both = hemis_subject == df_copy.ch_hemisphere.nunique()

    df_copy = df_copy.set_index(group)
    df_copy = df_copy.loc[hemi_both]
    df_copy = df_copy.reset_index()

    # assert no subjects with only one hemisphere
    enough_subs = (
        df_copy.groupby(repeated_m).ch_hemisphere.nunique() == 2
    ).all()
    if not enough_subs:
        return None

    # filter subjects with less than 2 values for x, y, and hue
    df = df[df.subject.isin(df_copy.subject.unique())]

    if remove_ties:
        method = 'average'  # 'Tied values are replaced by their mid-ranks.'
        df_ = df.copy()
        group = df_.groupby(repeated_m)
        df_.loc[:, x] = group[x].rank(method)
        df_.loc[:, y] = group[y].rank(method)

        # Remove ties by checking if x and y are integers
        mask1 = df_[x] == df_[x].astype(int)
        mask2 = df_[y] == df_[y].astype(int)

        df = df[(mask1 & mask2)]
    return df


def _permutation_pvalue(df_rho, x, y, repeated_m, rho, sample_size,
                        n_perm=10000):
    # apply permutation test according to Mohr & Marcon (2005)
    n = df_rho[repeated_m].value_counts().unique()
    msg = 'Only n=2 supported. If needed, use eq. 3 in Mohr 2005'
    assert len(n) == 1, msg
    n = n[0]
    msg = "Only two levels of repeated measures implemented"
    assert n == 2, msg
    m = sample_size
    z_r = rho * np.sqrt(m * (n - 1))  # eq. 4 in Mohr 2005

    # Define the function for a single permutation
    def single_permutation(x_values, group_indices, y_values,
                           repeated_m_values, m, n):
        # Important to NOT set np.random(seed) to get different permutations
        permuted_x = x_values.copy()
        for indices in group_indices:
            permuted_x[indices] = np.random.permutation(permuted_x[indices])
        df_perm = pd.DataFrame({repeated_m: repeated_m_values,
                                'x_perm': permuted_x, 'y': y_values})
        stats = pg.rm_corr(data=df_perm, x='x_perm', y='y', subject=repeated_m)
        rho_perm = stats.r.iloc[0]
        return rho_perm * np.sqrt(m * (n - 1))

    # Prepare numpy arrays for permutation
    group_indices = [np.where(df_rho[repeated_m] == subject)[0]
                     for subject in df_rho[repeated_m].unique()]
    x_values = df_rho[x].values
    y_values = df_rho[y].values
    repeated_m_values = df_rho[repeated_m].values

    # Use joblib for parallel processing
    z_permutations = Parallel(n_jobs=-1)(delayed(single_permutation)(
                x_values, group_indices, y_values, repeated_m_values, m, n)
                                                 for _ in trange(n_perm))

    # Calculate p-value
    perm_larger = np.abs(z_permutations) >= np.abs(z_r)
    pval = (np.sum(perm_larger) + 1) / (n_perm + 1)
    return pval


def _corr_from_df(df_rho, x, y, corr_method, R2, n_perm=None, ci_bounds=95):
    corr_kwargs = dict(method=corr_method)
    rho = df_rho[x].corr(df_rho[y], **corr_kwargs)
    sample_size = df_rho[[x, y]].dropna().shape[0]

    if R2:
        if corr_method == "pearson":
            rho = rho**2
        else:
            assert not R2, "R2 inappropriate for rank correlation"

    if n_perm is None:
        # Calc stats parametically (fast, more assumptions)
        pval_method = p_value_df(corr_method, corr_method)
        pval_kwargs = dict(method=pval_method)
        pval = df_rho[x].corr(df_rho[y], **pval_kwargs)
        stderr = 1.0 / np.sqrt(sample_size - 3)
        delta = 1.96 * stderr
        lower = np.tanh(np.arctanh(rho) - delta)
        upper = np.tanh(np.arctanh(rho) + delta)
        ci = [lower, upper]
    else:
        # Calc p-value nonparametically using permutation testing
        pval_method = p_value_df(corr_method=corr_method,
                                 stat_method='perm_parallel', n_perm=n_perm)
        pval_kwargs = dict(method=pval_method)
        pval = df_rho[x].corr(df_rho[y], **pval_kwargs)

        ci = pg.compute_bootci(x=df_rho[x], y=df_rho[y], func=corr_method,
                               confidence=ci_bounds/100, n_boot=n_perm,
                               paired=True, seed=1)
    return rho, pval, sample_size, ci


def remove_tied_ranks(df, rank_column):
    """Function to filter out tied ranks."""
    return df[df[rank_column] == df[rank_column].astype(int)]


def _rank_df(df, x, y, repeated_m="subject", remove_ties=True):
    """Convert float values for x and y to rank integers.

    Follows rank repeated measures in
    Donna L. Mohr & Rebecca A. Marcon (2005) Testing for a  ‘within-subjects’
    association in repeated measures data, Journal of Nonparametric Statistics,
    17:3, 347-363, DOI: 10.1080/10485250500038694
    """
    method = 'average'  # 'Tied values are replaced by their mid-ranks.'
    group = df.groupby(repeated_m)
    df.loc[:, x] = group[x].rank(method)
    df.loc[:, y] = group[y].rank(method)

    if remove_ties:
        df = remove_tied_ranks(df, x)
        df = remove_tied_ranks(df, y)
    return df
