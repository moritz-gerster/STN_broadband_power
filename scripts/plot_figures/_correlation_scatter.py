"""Helping functions."""
from itertools import chain, combinations
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

import scripts.config as cfg
from scripts.plot_figures.settings import N_PERM_CORR, FONTSIZE_ASTERISK
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


def j_test(data1, data2, X1, X2, y, model_nme1='Model 1', model_nme2='Model 2',
           alpha=0.05, output_file=None):
    data1 = data1.copy()
    data1 = data1.dropna(subset=X1+[y])

    data2 = data2.copy()
    data2 = data2.dropna(subset=X2+[y])

    if y == 'UPDRS_III':
        # average hemispheres
        keep = ['subject', 'cond', 'project', 'color']
        data1 = data1.groupby(keep).mean(numeric_only=True).reset_index()
        data2 = data2.groupby(keep).mean(numeric_only=True).reset_index()

    assert len(data1) == len(data2), 'Data unequal length'
    assert (data1[y].values == data2[y].values).all(), 'Order unequal'
    y_arr = data1[y].values
    X_arr1 = data1[X1].values
    X_arr2 = data2[X2].values
    X_arr1 = sm.add_constant(X_arr1)
    X_arr2 = sm.add_constant(X_arr2)

    model1 = sm.OLS(y_arr, X_arr1).fit()
    model2 = sm.OLS(y_arr, X_arr2).fit()
    y_pred1 = model1.predict(X_arr1)
    y_pred2 = model2.predict(X_arr2)

    # Extend x_arrays by predicted values
    X_arr1_ext = np.column_stack((X_arr1, y_pred2))
    X_arr2_ext = np.column_stack((X_arr2, y_pred1))

    model1_ext = sm.OLS(y_arr, X_arr1_ext).fit()
    model2_ext = sm.OLS(y_arr, X_arr2_ext).fit()

    pval1_ext = model1_ext.pvalues[-1]
    pval2_ext = model2_ext.pvalues[-1]

    print('J-test results:', file=output_file)
    if pval1_ext < alpha and pval2_ext > alpha:
        print(f'{model_nme2} is significantly better than {model_nme1} at '
              f'p={pval1_ext}', file=output_file)
    elif pval1_ext > alpha and pval2_ext < alpha:
        print(f'{model_nme1} is significantly better than {model_nme2} at '
              f'p={pval2_ext}', file=output_file)
    elif pval1_ext < alpha and pval2_ext < alpha:
        print(f'{model_nme2} and {model_nme1} are significantly different\n'
              f'pval1_ext={pval1_ext}, pval2_ext={pval2_ext}',
              file=output_file)
    elif pval1_ext > alpha and pval2_ext > alpha:
        print(f'{model_nme2} and {model_nme1} fail to fit the data\n'
              f'pval1_ext={pval1_ext}, pval2_ext={pval2_ext}',
              file=output_file)
    return pval1_ext, pval2_ext


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
             output_file=None, ylabel=None):
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

    fig, axes = plt.subplots(1, len(X), figsize=(3.21, 1.1), sharey=True,
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
        r_pearson, _, label, weight, _ = corr_results
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
        ax.set_xlabel(xlabel, fontsize=6)
        ax.set_ylabel(None)
        ax.set_title(label, weight=weight)
    if ylabel is None:
        ylabel = 'UPDRS-III'
    else:
        ylabel = None
    axes[0].set_ylabel(ylabel)
    plt.tight_layout()
    fname = f'{prefix}{kind}_regression_all_vs_{y}_{df.cond.unique()[0]}.pdf'
    save_dir = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, fname, save_dir, bbox_inches=None,
              transparent=True)
    return (r_pearson, pvalue, sum_of_squared_residuals, degrees_of_freedom,
            AICc_linreg, BIC_linreg)


def plot_all_ax(ax, df, X, y, kind, add_constant=True, ylabel=False,
                output_file=None, title=True, fontsize=7):
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
    r_pearson, _, label, weight, _ = corr_results
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
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    if ylabel:
        ax.set_ylabel('UPDRS-III')
    else:
        ax.set_ylabel(None)
        ax.set_yticklabels([])
    if title:
        ax.set_title(label, weight=weight)
    return (r_pearson, pvalue, sum_of_squared_residuals, degrees_of_freedom,
            AICc_linreg, BIC_linreg)


def find_best_model(df, y, kind, bands=cfg.BANDS.keys(), power='mean',
                    n_models=3, max_params=3, add_constant=True,
                    n_perm=None, optimize='AIC+BIC'):
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
    combinations = [list(l) for l in
                    all_combinations(features, max_len=max_params)]

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

        _, axes = plt.subplots(1, len(X), figsize=(3.5, 1.3), sharey=True,
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
            _, _, label, weight, _ = corr_results
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
                                fig_dir='Figure2', prefix='',
                                xlabel=True, title=True, output_file=None,
                                figsize=(1.2, 1.4),
                                n_perm=N_PERM_CORR):
    df_plot = df_norm[(df_norm.cond == cond) & (df_norm.project == 'all')]
    if y == 'UPDRS_III':
        # average hemispheres
        keep = ['subject', 'cond', 'project', 'color']
        df_plot = df_plot.groupby(keep).mean(numeric_only=True).reset_index()

    corr_results = _corr_results(df_plot, x, y, corr_method, None,
                                 n_perm=n_perm)
    _, _, label, weight, _ = corr_results

    df_plot = df_plot.dropna(subset=[x, y])

    fig, ax = plt.subplots(figsize=figsize)

    sns.regplot(ax=ax, data=df_plot, y=y, x=x, ci=95, scatter_kws=dict(s=1),
                color='k', label=label, marker='.', n_boot=1000)

    if title:
        ax.set_title(label, weight=weight)
    if output_file:
        print(label, file=output_file)
    band_nme = x.replace('_abs_mean_log', '')
    band_label = cfg.BAND_NAMES_GREEK[band_nme]
    xlabel = f'Relative {band_label} [%]' if xlabel else None
    ax.set_xlabel(xlabel)
    ax.set_ylabel(cfg.PLOT_LABELS[y])
    plt.tight_layout()
    fname = f'{prefix}scatter_normalized_{band_nme}_vs_{y}_{cond}.pdf'
    fpath = join(cfg.FIG_PAPER, fig_dir)
    _save_fig(fig, fname, fpath, bbox_inches=None,
              transparent=True)


def model_comparison(dataframes, fig_dir=None, output_file=None, fontsize=7,
                     model_comparison='f_test'):
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

    fig, axes = plt.subplots(1, 3, figsize=(4.6, 2), sharey=True)

    # Normalized
    df_norm_off_ = df_norm_off[df_norm_off.sub_hemi.isin(
        df_per_off.sub_hemi.unique())]
    X_norm = ['beta_low_abs_mean_log']
    res = plot_all_ax(axes[0], df_norm_off_, X_norm, y,
                      'normalized', ylabel=True, title=False,
                      output_file=output_file)
    r_linreg_rel, p_linreg_rel, ssr_rel1, dof_rel1, aic_rel, bic_rel = res

    # Absolute
    X_abs = ['theta_abs_mean_log',
             'beta_low_abs_mean_log',
             'gamma_low_abs_mean_log']
    df_abs_off_ = df_abs_off[df_abs_off.sub_hemi.isin(
        df_per_off.sub_hemi.unique())]
    res = plot_all_ax(axes[1], df_abs_off_, X_abs, y, 'absolute', title=False,
                      output_file=output_file)
    r_linreg_abs, p_linreg_abs, ssr_abs3, dof_abs3, aic_abs, bic_abs = res

    # Periodic
    X_per = ['fm_offset_log', 'beta_low_fm_mean_log', 'gamma_low_fm_mean_log']
    res = plot_all_ax(axes[2], df_per_off, X_per, y, 'periodic', title=False,
                      output_file=output_file)
    r_linreg_per, p_linreg_per, ssr_per3, dof_per3, aic_per, bic_per = res

    # model comparison
    if model_comparison == 'f_test':
        pval_norm_abs = f_test(ssr_rel1, ssr_abs3, dof_rel1, dof_abs3,
                               model_nme1='Normalized',
                               model_nme2='Absolute',
                               output_file=output_file)
        pval_norm_per = f_test(ssr_rel1, ssr_per3, dof_rel1, dof_per3,
                               model_nme1='Normalized',
                               model_nme2='Periodic',
                               output_file=output_file)
        pval_abs_per = f_test(ssr_abs3, ssr_per3, dof_abs3, dof_per3,
                              model_nme1='Absolute',
                              model_nme2='Periodic',
                              output_file=output_file)
    elif model_comparison == 'j_test':
        pval_norm_abs, _ = j_test(df_norm_off_, df_abs_off_, X_norm, X_abs, y,
                                  model_nme1='Normalized',
                                  model_nme2='Absolute',
                                  output_file=output_file)
        pval_norm_per, _ = j_test(df_norm_off_, df_per_off, X_norm, X_per, y,
                                  model_nme1='Normalized',
                                  model_nme2='Periodic',
                                  output_file=output_file)
        pval_abs_per, _ = j_test(df_abs_off_, df_per_off, X_abs, X_per, y,
                                 model_nme1='Absolute', model_nme2='Periodic',
                                 output_file=output_file)

    annotations = [('normalized', 'absolute', pval_norm_abs),
                   ('absolute', 'periodic', pval_abs_per),
                   ('normalized', 'periodic', pval_norm_per)]
    correlations = [r_linreg_rel, r_linreg_abs, r_linreg_per]
    pvalues = [p_linreg_rel, p_linreg_abs, p_linreg_per]

    axes[0].set_ylabel('UPDRS-III', fontsize=fontsize)
    yticks = [0, 10, 20, 30, 40, 50, 60, 70]
    axes[0].set_yticks(yticks, labels=yticks, fontsize=fontsize)
    plt.tight_layout()
    _save_fig(fig, f'{fig_dir}/A__lin_regs.pdf', cfg.FIG_PAPER,
              bbox_inches=None, transparent=True)

    # Barplot model comparison
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))

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
        ax.text(x, .11,
                f'r={corr:.2f}', ha='center', va='bottom',
                fontsize=fontsize, fontweight='bold', color='w')
        ax.text(x, .008,
                f'AIC: {aic:.0f}\nBIC: {bic:.0f}',
                ha='center',
                va='bottom', fontsize=fontsize-1,
                color='w')

    ax.set_ylabel(r"Pearson's $r$", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    # ax.set_title(f'Linear regression {cfg.COND_DICT['off']} vs. UPDRS-III',
    #              fontweight='bold', y=1)
    ax.set_xlabel(r'$\alpha$ placeholder', alpha=0, fontsize=fontsize)
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
    _, _, label, weight, _ = corr_results

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