"""Helping functions."""
from itertools import chain, combinations
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

import scripts.config as cfg
from scripts.plot_figures.settings import *
from scripts.utils_plot import (_corr_results, _save_fig,
                                convert_pvalue_to_asterisks)


def f_test(sum_squares_resid1, sum_squares_resid2, dof1, dof2,
           model_nme1='Model 1', model_nme2='Model 2', output_file=None):
    if dof1 == dof2:
        f_stat = sum_squares_resid1 / sum_squares_resid2
        p_value = 1 - stats.f.cdf(f_stat, dof1, dof2)
    else:
        f_stat = ((sum_squares_resid1 - sum_squares_resid2)
                  / (dof1 - dof2)) / (sum_squares_resid2 / dof2)
        p_value = 1 - stats.f.cdf(f_stat, dof1 - dof2, dof2)
    print(f'{model_nme2} (same or more parameters) is better '
          f'than {model_nme1} at p={p_value}', file=output_file)
    return p_value


def _corrected_aic(model):
    """Correct AIC for small sample sizes."""
    aic = model.aic
    n_features = model.params.shape[0]
    sample_size = model.nobs
    correction = ((2 * n_features**2 + 2 * n_features)
                  / (sample_size - n_features - 1))
    AICc = aic + correction
    return AICc


def all_combinations(any_list, max_len=None):
    if max_len is None:
        max_len = len(any_list)
    return chain.from_iterable(combinations(any_list, i + 1)
                               for i in range(max_len))


def plot_all(df, X, y, kind, add_constant=True, fig_dir='Figure3', prefix='',
             output_file=None):
    if isinstance(X, str):
        X = [X]
    X = X.copy()
    df = df.copy()
    df = df.dropna(subset=X+[y])
    if y == 'UPDRS_III':
        # average hemispheres
        keep = ['subject', 'cond', 'project', 'color']
        df = df.groupby(keep).mean(numeric_only=True).reset_index()

    # Linear regression
    X_arr = df[X].values
    if add_constant:
        X_arr = sm.add_constant(X_arr)
    y_arr = df[y].values
    # Calculate AIC and BIC
    model = sm.OLS(y_arr, X_arr).fit()
    coefficients = model.params
    sum_of_squared_residuals = model.ssr
    degrees_of_freedom = model.df_resid
    AICc_linreg = _corrected_aic(model)
    BIC_linreg = model.bic
    if coefficients.ndim == 2:
        coefficients = coefficients[0]

    bands = [band.replace('_abs_mean_log', '').replace('_fm_mean_log', '')
             for band in X]
    bands = [cfg.BAND_NAMES_GREEK_SHORT[band] for band in bands]
    feature_nme = f"Lin. reg. ({', '.join(bands)})"
    X.append(feature_nme)
    full_model_abic = f'AICc: {AICc_linreg:.0f}, BIC: {BIC_linreg:.0f}'
    print(f'{feature_nme} coefficients: {coefficients}', file=output_file)
    df[feature_nme] = [np.dot(coefficients, x) for x in X_arr]


    fig, axes = plt.subplots(1, len(X), figsize=(3.5, 1.3), sharey=True,
                             width_ratios=[1] * (len(X) - 1) + [1.5])
    for i, x in enumerate(X):
        add_sample_size = True if i == len(X)-1 else False
        x_arr = df[x].values
        if add_constant:
            x_arr = sm.add_constant(x_arr)
        model = sm.OLS(y_arr, x_arr).fit()
        AICc = _corrected_aic(model)
        BIC = model.bic
        if i != len(X)-1:
            model_abic = f'AICc: {AICc:.0f}, BIC: {BIC:.0f}'
        else:
            model_abic = full_model_abic
        corr_results = _corr_results(df, x, y, 'pearson', None,
                                     add_sample_size=add_sample_size,
                                     n_perm=N_PERM_CORR)
        r_pearson, sample_size, label, weight, _ = corr_results
        pvalue = float(label.split('p=')[1])
        ax = axes[i]
        kind_color = ('periodicAP'
                      if x in ['fm_offset_log',
                               'fm_exponent',
                               'fm_exponent_narrow'] else kind)
        color = cfg.COLOR_DIC[kind_color]
        sns.regplot(ax=ax, data=df, y=y, x=x, ci=95, scatter_kws=dict(s=1),
                    color=color, label=label, marker='.', n_boot=1000)

        try:
            xlabel = cfg.PLOT_LABELS[x].replace(' mean', '')
        except KeyError:
            xlabel = x
        print(f'{xlabel}: {label}, {model_abic}', file=output_file)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(None)
        ax.set_title(label, weight=weight)
    axes[0].set_ylabel('UPDRS-III')
    plt.tight_layout()
    fname = f'{prefix}{kind}_regression_all_vs_{y}_{df.cond.unique()[0]}.pdf'
    save_dir = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, fname, save_dir, bbox_inches=None,
              transparent=True)
    return (r_pearson, pvalue, sum_of_squared_residuals, degrees_of_freedom,
            AICc_linreg, BIC_linreg)


def plot_all_ax(ax, df, X, y, kind, add_constant=True, ylabel=False,
                output_file=None):
    if isinstance(X, str):
        X = [X]
    X = X.copy()
    df = df.copy()
    df = df.dropna(subset=X+[y])
    if y == 'UPDRS_III':
        # average hemispheres
        keep = ['subject', 'cond', 'project', 'color']
        df = df.groupby(keep).mean(numeric_only=True).reset_index()

    # Linear regression
    X_arr = df[X].values
    if add_constant:
        X_arr = sm.add_constant(X_arr)
    y_arr = df[y].values
    # Calculate AIC and BIC
    model = sm.OLS(y_arr, X_arr).fit()
    coefficients = model.params
    sum_of_squared_residuals = model.ssr
    degrees_of_freedom = model.df_resid
    AICc_linreg = _corrected_aic(model)
    BIC_linreg = model.bic
    if coefficients.ndim == 2:
        coefficients = coefficients[0]

    bands = [band.replace('_abs_mean_log', '').replace('_fm_mean_log', '')
             for band in X]
    bands = [cfg.BAND_NAMES_GREEK_SHORT[band] for band in bands]
    feature_nme = f"Lin. reg. ({', '.join(bands)})"
    X.append(feature_nme)
    model_abic = f'AICc: {AICc_linreg:.0f}, BIC: {BIC_linreg:.0f}'
    print(f'{feature_nme} coefficients: {coefficients}', file=output_file)
    df[feature_nme] = [np.dot(coefficients, x) for x in X_arr]

    x = feature_nme
    add_sample_size = True
    corr_results = _corr_results(df, x, y, 'pearson', None,
                                 add_sample_size=add_sample_size,
                                 n_perm=N_PERM_CORR)
    r_pearson, sample_size, label, weight, _ = corr_results
    pvalue = float(label.split('p=')[1])
    kind_color = ('periodicAP'
                  if x in ['fm_offset_log',
                           'fm_exponent',
                           'fm_exponent_narrow'] else kind)
    color = cfg.COLOR_DIC[kind_color]
    sns.regplot(ax=ax, data=df, y=y, x=x, ci=95, scatter_kws=dict(s=1),
                color=color, label=label, marker='.', n_boot=1000)

    try:
        xlabel = cfg.PLOT_LABELS[x].replace(' mean', '')
    except KeyError:
        xlabel = x
    print(f'{xlabel}: {label}, {model_abic}', file=output_file)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel('UPDRS-III')
    else:
        ax.set_ylabel(None)
        ax.set_yticklabels([])
    ax.set_title(label, weight=weight)
    return (r_pearson, pvalue, sum_of_squared_residuals, degrees_of_freedom,
            AICc_linreg, BIC_linreg)


def find_best_model(df, y, kind, bands=cfg.BANDS.keys(), power='mean',
                    n_models=3, max_params=3, add_constant=True,
                    optimize='AIC+BIC'):
    if kind == 'periodic':
        pwr_str = f'_fm_{power}_log'
    else:
        pwr_str = f'_abs_{power}_log'
    features = [band + pwr_str for band in bands]
    if kind == 'periodic':
        features += ['fm_offset_log', 'fm_exponent']
    feature_nme = "Regression model"
    df = df.copy()
    df = df.dropna(subset=features + [y])
    if y == 'UPDRS_III':
        # average hemispheres
        keep = ['subject', 'cond', 'project', 'color']
        df = df.groupby(keep).mean(numeric_only=True).reset_index()
    y_arr = df[y].values
    combinations = [list(l) for l in all_combinations(features, max_len=max_params)]

    # Test all combinations
    metrics = []
    models = []
    for X in combinations:

        # Linear regression
        X_arr = df[X].values
        if add_constant:
            X_arr = sm.add_constant(X_arr)
        # Calculate AIC and BIC
        model = sm.OLS(y_arr, X_arr).fit()
        if optimize == 'AIC':
            AICc = _corrected_aic(model)
            metrics.append(AICc)
        elif optimize == 'BIC':
            metrics.append(model.bic)
        elif optimize == 'AIC+BIC':
            AICc = _corrected_aic(model)
            metrics.append(AICc + model.bic)
        models.append(model)

    # Sort models by metric
    sorted_idx = np.argsort(metrics)
    # Select and plot the n_models best models
    sorted_idx = sorted_idx[:n_models]
    sorted_features = [combinations[i] for i in sorted_idx]
    sorted_models = [models[i] for i in sorted_idx]
    for i, model in enumerate(sorted_models):

        coefficients = model.params
        if coefficients.ndim == 2:
            coefficients = coefficients[0]
        AICc = _corrected_aic(model)

        X = sorted_features[i]
        bands_kept = [band for band in bands if band + pwr_str in X]
        bands_kept = [cfg.BAND_NAMES_GREEK_SHORT[band] for band in bands_kept]
        feature_nme = f"Lin. reg. ({', '.join(bands_kept)})"
        full_model_abic = f'AICc: {AICc:.0f}, BIC: {model.bic:.0f}'
        print(f'{feature_nme} coefficients: {coefficients}')
        X_arr = df[X].values
        if add_constant:
            X_arr = sm.add_constant(X_arr)
        df[feature_nme] = [np.dot(coefficients, x) for x in X_arr]
        X.append(feature_nme)

        fig, axes = plt.subplots(1, len(X), figsize=(3.5, 1.3), sharey=True,
                                width_ratios=[1] * (len(X) - 1) + [1.5])
        for i, x in enumerate(X):
            add_sample_size = True if i == len(X)-1 else False
            x_arr = df[x].values
            if add_constant:
                x_arr = sm.add_constant(x_arr)
            model = sm.OLS(y_arr, x_arr).fit()
            AICc = _corrected_aic(model)
            if i != len(X)-1:
                model_abic = f'AICc: {AICc:.0f}, BIC: {model.bic:.0f}'
            else:
                model_abic = full_model_abic
            corr_results = _corr_results(df, x, y, 'pearson', None,
                                        add_sample_size=add_sample_size,
                                        n_perm=n_perm)
            rho, sample_size, label, weight, _ = corr_results
            ax = axes[i]
            kind_color = ('periodicAP'
                          if x in ['fm_offset_log',
                                   'fm_exponent',
                                   'fm_exponent_narrow'] else kind)
            color = cfg.COLOR_DIC[kind_color]
            sns.regplot(ax=ax, data=df, y=y, x=x, ci=95, scatter_kws=dict(s=1),
                        color=color, label=label, marker='.', n_boot=1000)

            try:
                xlabel = cfg.PLOT_LABELS[x].replace(' mean', '')
            except KeyError:
                xlabel = x
            print(f'{xlabel}: {label}, {model_abic}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(None)
            ax.set_title(label, weight=weight)
        axes[0].set_ylabel('UPDRS-III')
        plt.tight_layout()


def representative_scatter_plot(df_norm, x, y, cond, corr_method='spearman',
                                fig_dir='Figure2', prefix=''):
    df_plot = df_norm[(df_norm.cond == cond) & (df_norm.project == 'all')]
    if y == 'UPDRS_III':
        # average hemispheres
        keep = ['subject', 'cond', 'project', 'color']
        df_plot = df_plot.groupby(keep).mean(numeric_only=True).reset_index()

    corr_results = _corr_results(df_plot, x, y, corr_method, None,
                                 n_perm=N_PERM_CORR)
    rho, sample_size, label, weight, _ = corr_results

    df_plot = df_plot.dropna(subset=[x, y])

    fig, ax = plt.subplots(figsize=(1.3, 1.4))

    sns.regplot(ax=ax, data=df_plot, y=y, x=x, ci=95, scatter_kws=dict(s=1),
                color='k', label=label, marker='.', n_boot=1000)

    ax.set_title(label, weight=weight)
    band_nme = x.replace('_abs_mean_log', '')
    band_label = cfg.BAND_NAMES_GREEK[band_nme]
    ax.set_xlabel(f'Relative {band_label} [%]')
    ax.set_ylabel(cfg.PLOT_LABELS[y])
    plt.tight_layout()
    fname = f'{prefix}scatter_normalized_{band_nme}_vs_{y}_{cond}.pdf'
    fpath = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, fname, fpath, bbox_inches=None,
              transparent=True)


def model_comparison(dataframes, fig_dir=None, output_file=None):
    # Dataframes
    df_norm = dataframes['df_norm']
    df_abs = dataframes['df_abs']
    df_per = dataframes['df_per']
    df_norm_off = df_norm[df_norm.cond.isin(['off'])
                          & (df_norm.project != 'all')]
    df_abs_off = df_abs[df_abs.cond.isin(['off'])
                        & (df_abs.project != 'all')]
    df_per_off = df_per[df_per.cond.isin(['off'])
                        & (df_per.project != 'all')]

    # Plot model comparison
    y = 'UPDRS_III'
    kinds = ['normalized', 'absolute', 'periodic']
    xticklabels = [cfg.PLOT_LABELS[kind] for kind in kinds]
    colors = [cfg.COLOR_DIC[kind] for kind in kinds]

    fig, axes = plt.subplots(1, 3, figsize=(4.5, 2), sharey=True)

    # Normalized
    df_norm_off_ = df_norm_off[df_norm_off.sub_hemi.isin(
        df_per_off.sub_hemi.unique())]
    X = ['beta_low_abs_mean_log']
    res = plot_all_ax(axes[0], df_norm_off_, X, y,
                      'normalized', ylabel=True,
                      output_file=output_file)
    r_linreg_rel, p_linreg_rel, ssr_rel1, dof_rel1, aic_rel, bic_rel = res

    # Absolute
    X = ['theta_abs_mean_log',
         'beta_low_abs_mean_log',
         'gamma_low_abs_mean_log']
    df_abs_off_ = df_abs_off[df_abs_off.sub_hemi.isin(
        df_per_off.sub_hemi.unique())]
    res = plot_all_ax(axes[1], df_abs_off_, X, y, 'absolute',
                      output_file=output_file)
    r_linreg_abs, p_linreg_abs, ssr_abs3, dof_abs3, aic_abs, bic_abs = res

    # Periodic
    X = ['fm_offset_log', 'beta_low_fm_mean_log', 'gamma_low_fm_mean_log']
    res = plot_all_ax(axes[2], df_per_off, X, y, 'periodic',
                      output_file=output_file)
    r_linreg_per, p_linreg_per, ssr_per3, dof_per3, aic_per, bic_per = res

    # f-test model comparison
    pval_norm_abs = f_test(ssr_rel1, ssr_abs3, dof_rel1, dof_abs3,
                           model_nme1='Normalized', model_nme2='Absolute',
                           output_file=output_file)
    pval_norm_per = f_test(ssr_rel1, ssr_per3, dof_rel1, dof_per3,
                           model_nme1='Normalized', model_nme2='Periodic',
                           output_file=output_file)
    pval_abs_per = f_test(ssr_abs3, ssr_per3, dof_abs3, dof_per3,
                          model_nme1='Absolute', model_nme2='Periodic',
                          output_file=output_file)

    annotations = [('normalized', 'absolute', pval_norm_abs),
                   ('absolute', 'periodic', pval_abs_per),
                   ('normalized', 'periodic', pval_norm_per)]
    correlations = [r_linreg_rel, r_linreg_abs, r_linreg_per]
    pvalues = [p_linreg_rel, p_linreg_abs, p_linreg_per]

    axes[0].set_ylabel('UPDRS-III')
    yticks = [0, 10, 20, 30, 40, 50, 60, 70]
    axes[0].set_yticks(yticks, labels=yticks)
    plt.tight_layout()
    _save_fig(fig, f'{fig_dir}/A__lin_regs.pdf', cfg.FIG_PAPER,
              bbox_inches=None, transparent=True)

    # Barplot model comparison
    fig, ax = plt.subplots(1, 1, figsize=(2, 1.985))

    ax.bar(x=xticklabels, height=correlations, color=colors)

    # Statistics
    bars = ax.containers[0]
    ymin, ymax = ax.get_ylim()
    yscale = np.abs(ymax - ymin)
    y_buffer = 0.03*yscale
    bar_x_coords = []
    for i, bar in enumerate(bars):
        pvalue = pvalues[i]
        text = convert_pvalue_to_asterisks(pvalue)
        x_bar = bar.get_x() + bar.get_width() / 2
        bar_x_coords.append(bar.get_x())
        ax.annotate(text, xy=(x_bar, ymax + y_buffer), ha='center', va='top',
                    fontsize=FONTSIZE_ASTERISK)

    y_buffer = 0.02*yscale
    height_stat = ymax + 1*y_buffer
    # Loop through each annotation
    for (x1_label, x2_label, pvalue) in annotations:
        # Get the indices of the bars from the 'kinds' list
        x1 = kinds.index(x1_label)
        x2 = kinds.index(x2_label)

        # Determine the y position for the line and asterisk
        y_line = height_stat + 1.5*y_buffer  # Add some offset for the line
        y_text = y_line

        # Draw the line connecting the two bars
        ax.plot([x1, x1, x2, x2],
                [y_line, y_line + y_buffer/2, y_line + y_buffer/2, y_line],
                color='black')

        # Get the significance text based on the p-value
        text = convert_pvalue_to_asterisks(pvalue, print_ns=True)

        # Place the text above the line
        ax.text((x1 + x2) / 2, y_text, text,
                ha='center', va='bottom', fontsize=FONTSIZE_ASTERISK)
        height_stat += 3.5*y_buffer

    # Annotate AICc and BIC values in the center of the bars
    aic_values = [aic_rel, aic_abs, aic_per]
    bic_values = [bic_rel, bic_abs, bic_per]
    for x, (aic, bic) in enumerate(zip(aic_values, bic_values)):
        corr = correlations[x]
        ax.text(x, .09,
                f'r={corr:.2f}', ha='center', va='bottom',
                fontsize=5.5, fontweight='bold', color='w')
        ax.text(x, .008,
                f'AIC: {aic:.0f}\nBIC: {bic:.0f}',
                ha='center',
                va='bottom', fontsize=FONTSIZE_S,
                color='w')

    ax.set_ylabel(r"Pearson's $r$")
    ax.set_title(f'Linear regression {cfg.COND_DICT['off']} ~ UPDRS-III',
                fontweight='bold', y=1)
    ax.set_xlabel(r'$\alpha$ placeholder', alpha=0)
    plt.tight_layout()
    _save_fig(fig, f'{fig_dir}/B__model_comparison.pdf', cfg.FIG_PAPER,
              bbox_inches=None, transparent=True)


def corr_offset_theta(df_per, fig_dir='Figure_S6', prefix=''):
    y = 'fm_offset_log'
    x = 'theta_abs_mean_log'
    corr_method = 'spearman'

    df_per = df_per[~df_per.project.isin(['all'])]
    df_per = df_per[df_per.cond.isin(['off'])]

    # average hemispheres
    keep = ['subject', 'cond', 'project', 'color']
    df_plot = df_per.groupby(keep).mean(numeric_only=True).reset_index()
    df_plot = df_plot.dropna(subset=[x]+[y])

    corr_results = _corr_results(df_plot, x, y, corr_method, None, n_perm=None)
    rho, sample_size, label, weight, _ = corr_results

    df_plot = df_plot.dropna(subset=[x, y])

    fig, ax = plt.subplots(figsize=(1.5, 1.5))

    sns.regplot(ax=ax, data=df_plot, y=y, x=x, ci=95, scatter_kws=dict(s=1),
                color='k', label=label, marker='.')

    ax.set_title(label, weight=weight)
    ax.set_xlabel(cfg.PLOT_LABELS[x])
    ax.set_ylabel(cfg.PLOT_LABELS_SHORT[y])
    plt.tight_layout()
    fname = f'{prefix}abs_{x}_vs_{y}_off.pdf'
    fpath = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, fname, fpath, bbox_inches=None,
              transparent=True)
