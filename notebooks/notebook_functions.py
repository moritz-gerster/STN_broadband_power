from os.path import join
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.formula.api as smf
from statannotations.Annotator import Annotator
import ptitprince as pt

import scripts.config as cfg
from scripts.corr_stats import independent_corr, p_value_df

pd.options.display.max_columns = None
# pd.set_option('display.max_rows', 50)
c_on = "#E66101"
c_off = "#5E3C99"
palette = [c_off, c_on]
order = ["off", "on"]
bip_chs = ["LFP_1-2", "LFP_2-3", "LFP_3-4"]
rain_kwargs = dict(jitter=.06, width_viol=0.5, palette=palette, pointplot=True,
                   order=order)

df = pd.read_pickle(join("../", cfg.DF_PATH, cfg.DF_FOOOF))
df = df[(df.subject != "emptyroom")
        & (df.ch_reference != "monopolar")
        # & ~df.ch_bad
        # & (df.bids_processing == "ReRef1SpectrumWelch")]
        ]


# def equalize_x_and_y(df, x, y) -> tuple[pd.DataFrame, int]:
#     """Wilcoxon requires equal sample sizes."""
#     df_finite = df.dropna(subset=[x, y])

#     # drop subjects with only one x condition
#     subs = df_finite.groupby(["subject"])[x].unique()
#     subs_both_conds = subs.apply(list).apply(len) == len(df_finite[x].unique())
#     subs_both_conds = df_finite.subject.unique()[subs_both_conds]
#     df_new = df_finite[df_finite.subject.isin(subs_both_conds)]

#     # drop subjects with only one y condition
#     sub_vals = df_new.subject.value_counts()
#     sub_max = df_new.subject.value_counts().max()
#     subs_both_ys = (sub_vals == sub_max).sort_index()
#     subs_both_ys = df_new.subject.unique()[subs_both_ys]
#     df_new = df_new[df_new.subject.isin(subs_both_ys)]

#     sample_size = len(df_new[y]) // len(df_new[x].unique())
#     return df_new, sample_size


# def plot_exponent(df, ax, x, y, test="Wilcoxon", order=[("on", "off")],
#                   hue=None):
#     if test == "Wilcoxon":
#         df_plot, n = equalize_x_and_y(df, x, y)
#     else:
#         df_plot, n = df, None
#     g = pt.RainCloud(ax=ax, data=df_plot, x=x, y=y, hue=hue, **rain_kwargs)
#     annotator = Annotator(ax=ax, pairs=order, order=rain_kwargs["order"],
#                           data=df_plot, x=x, y=y)  # test='Wilcoxon-ls'
#     annotator.configure(test=test, text_format='simple', loc='inside', verbose=True)
#     annotator.apply_and_annotate()
#     return g


# def plot_corr(ax, df, X, Y, corr_method="rmcorr", hue="cond", ci=95,
#               repeated_m="subject", color=None, x_label=None, y_label=None,
#               title=None):

#     if hue:
#         df_rho = df.set_index(hue)
#         iterate = df_rho.index.unique()
#     else:
#         df_rho = df
#         iterate = [slice(None)]

#     rhos = []
#     sample_sizes = []
#     labels = []
#     weights = []
#     for row_idx in iterate:
#         corr_results = _corr_results(df_rho, X, Y, corr_method, row_idx,
#                                      repeated_m=repeated_m)
#         rho, ci, sample_size, label, weight = corr_results

#         rhos.append(rho)
#         sample_sizes.append(sample_size)
#         labels.append(label)
#         weights.append(weight)
#         color = cfg.COLOR_DIC[row_idx] if hue == "cond" else color

#         sns.regplot(ax=ax, data=df_rho.loc[row_idx], y=Y, x=X, ci=ci,
#                     color=color, label=label)

#     _plot_legend(ax, X, Y, labels, weights, hue, rhos, sample_sizes)
#     if x_label:
#         ax.set_xlabel(x_label)
#     if y_label:
#         ax.set_ylabel(y_label)
#     if title:
#         ax.set_title(title)
#     return ax


def plot_rm_corrs(df_all, X_vals, Y, cond="off", label_colors=None,
                  title=None):
    label_dic = cfg.PLOT_LABELS_UNITS
    n_cols = len(X_vals)
    n_rows = 1

    df_rho = df_all[df_all.cond == cond].copy()
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5*1.1*n_cols, 2.5*1.1*n_rows),
                             sharey=True)
    axes = axes.flatten()
    for i, X in enumerate(X_vals):

        labels = []
        weights = []

        corr_results = _corr_results(df_rho, X, Y, "rmcorr", slice(None),
                                     repeated_m="subject")
        rho, ci, sample_size, label, weight = corr_results

        labels.append(label)
        weights.append(weight)

        sns.lineplot(ax=axes[i], data=df_rho, y=Y, x=X, ci=ci, hue="subject",
                     color=cfg.COLOR_DIC[cond], label=label)
        sns.scatterplot(ax=axes[i], data=df_rho, y=Y, x=X, ci=ci,
                        hue="subject", color=cfg.COLOR_DIC[cond], label=label)

        red_patch = patches.Patch(color='w')
        leg = axes[i].legend(loc=2, handles=[red_patch], labels=labels,
                             handletextpad=-2)
        # Set significant legend elements bold
        [t.set_fontweight(w) for t, w in zip(leg.get_texts(), weights)]
        try:
            color = label_colors[i]
        except (IndexError, TypeError):
            color = "k"
        axes[i].set_xlabel(label_dic[X], c=color) if X in label_dic else None
        axes[i].set_ylabel(None)

        axes[i].yaxis.set_tick_params(which='both', labelbottom=True)
    # make x label in bold
    axes[0].set_ylabel(label_dic[Y])
    if title:
        fig.suptitle(title, fontsize=18, y=1.05)
    plt.subplots_adjust(wspace=.4)
    plt.show()


def plot_significant_corr(ax, df, X, Y, corr_method="rmcorr", hue="cond",
                          ci=95, repeated_m="subject", color=None):
    """Only plot if significant."""
    if hue:
        df_rho = df.set_index(hue)
        iterate = df_rho.index.unique()
    else:
        df_rho = df
        iterate = [slice(None)]

    rhos = []
    sample_sizes = []
    labels = []
    weights = []
    for row_idx in iterate:
        corr_results = _corr_results(df_rho, X, Y, corr_method, row_idx,
                                     repeated_m=repeated_m)
        rho, ci, sample_size, label, weight = corr_results

        rhos.append(rho)
        sample_sizes.append(sample_size)
        labels.append(label)
        weights.append(weight)
        color = cfg.COLOR_DIC[row_idx] if hue == "cond" else color

        # if weight == "bold":
        # sns.regplot(ax=ax, data=df_rho.loc[row_idx], y=Y, x=X, ci=ci,
        #             color=color, label=label)

    if any([weight == "bold" for weight in weights]):
        # _plot_legend(ax, X, Y, labels, weights, hue, rhos, sample_sizes)
        return True
    else:
        return False


def remove_df_columns(df, updrs=True, bands=True, fooof=True,
                      patient_info=True, connectivity=True):
    drop = []
    if updrs:
        drop += [col for col in df.columns if col.startswith("UPDRS")]
    if bands:
        drop += [col for band in cfg.BANDS
                 for col in df.columns if col.startswith(band)]
    if fooof:
        drop += [col for col in df.columns if col.startswith("fm_")]
    if patient_info:
        drop += [col for col in df.columns if col.startswith("patient_")]
    if connectivity:
        drop += ["coh_freqs", "coh", "imcoh", "mic", "mim", "ppc",
                 "pli2_unbiased", "wpli2_debiased"]
    return df.drop(columns=drop)


# def _plot_legend(ax, X, Y, labels, weights, hue, rhos, sample_sizes):

#     title, title_fontproperties = _leg_titles(hue, rhos, sample_sizes)

#     # frameon = True if hue else False
#     frameon = True
#     handlelength = None if hue else 2
#     handles, _ = ax.get_legend_handles_labels()
#     loc = 2 if len(rhos) == 2 else 0
#     if len(rhos) > 6:
#         # Shrink current axis by 20%
#         box = ax.get_position()
#         ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#         # set legend outside of plot
#         bbox_to_anchor = (1, 1)
#         ncol = 2
#     else:
#         # bbox_to_anchor = (0, 1.4)  # top
#         bbox_to_anchor = (1, 1)  # right
#         ncol = 1
#     leg = ax.legend(loc=loc, ncol=ncol, title=title, handles=handles,
#                     labels=labels, frameon=frameon, handlelength=handlelength,
#                     title_fontproperties=title_fontproperties,
#                     bbox_to_anchor=bbox_to_anchor)
#     # Set significant legend elements bold
#     [t.set_fontweight(w) for t, w in zip(leg.get_texts(), weights)]
#     ax.set_xlabel(X.replace("_", " ").capitalize())
#     ax.set_ylabel(Y.replace("_", " ").capitalize())


# def _leg_titles(hue, rhos, sample_sizes):
#     if len(rhos) == 2:
#         _, p_cond = independent_corr(*rhos, *sample_sizes)
#         weight = "bold" if p_cond < 0.05 else "normal"
#         title = f"Condition: p={p_cond:.3f}"
#         title_fontproperties = {'weight': weight}
#     else:
#         title = None
#         title_fontproperties = None
#     return title, title_fontproperties


def _corr_results(df_rho, X, Y, corr_method, row_idx, repeated_m="subject"):
    if corr_method in ["spearman", "pearson", "kendall"]:
        pval_method = p_value_df(corr_method, corr_method)
        # corr_kwargs = dict(method=corr_method, numeric_only=True)
        # pval_kwargs = dict(method=pval_method, numeric_only=True)
        corr_kwargs = dict(method=corr_method)
        pval_kwargs = dict(method=pval_method)
        rho = df_rho.loc[row_idx, X].corr(df_rho.loc[row_idx, Y],
                                          **corr_kwargs)# .iloc[0, 1]
        pval = df_rho.loc[row_idx, X].corr(df_rho.loc[row_idx, Y],
                                           **pval_kwargs)# .iloc[0, 1]
        # rho = df_rho.loc[row_idx, [X, Y]].corr(**corr_kwargs).iloc[0, 1]
        # pval = df_rho.loc[row_idx, [X, Y]].corr(**pval_kwargs).iloc[0, 1]
        ci_plot = 95
        sample_size = df_rho.loc[row_idx, [X, Y]].dropna().shape[0]

        sample_size_descr = sample_size
        corr_coeff_descr = r"${ \rho=}$"
    elif corr_method == "rmcorr":
        stats = pg.rm_corr(data=df_rho.loc[row_idx], x=X, y=Y,
                           subject=repeated_m)
        rho = stats.r.iloc[0]
        pval = stats.pval.iloc[0]
        ci_plot = None
        corr_coeff_descr = r"$r_{rm}=$"
        # ci = stats["CI95%"].iloc[0]
        # power = stats.power.iloc[0]
        # dof = stats.dof.iloc[0]

        # # Calc confidence interval
        # stderr = 1.0 / math.sqrt(sample_size - 3)
        # delta = 1.96 * stderr
        # lower = math.tanh(math.atanh(rho) - delta)
        # upper = math.tanh(math.atanh(rho) + delta)
        # ci = [lower, upper]

        # sample_size = dof + 1
        data = df_rho.loc[row_idx, [X, Y, repeated_m]].dropna()
        sample_size = len(data[repeated_m].unique())
        # assert sample_size == len(data[repeated_m].unique())
        repetititions = len(data) / sample_size
        sample_size_descr = (f"{sample_size} ({repeated_m}) "
                             f"x {repetititions:.0f}")

    weight = "bold" if pval < 0.05 else "normal"

    # Legend entry:
    # if isinstance(slice(None), slice):
    #     hue_str = ""
    # else:
    #     hue_str = f"{row_idx.capitalize()} "
    if isinstance(row_idx, slice):
        hue_str = ""
    elif row_idx == "cond":
        hue_str = f"{row_idx.capitalize()} "
    # elif isinstance(row_idx, str):
    #     hue_str = f"{row_idx} "
    else:
        hue_str = f"{row_idx} "
    n = f"{hue_str}n={sample_size_descr}:\n"
    corr = corr_coeff_descr + f"{rho:.2f}, "
    pval = f"p={pval:.3f}"
    label = f"{n + corr + pval}"
    return rho, ci_plot, sample_size, label, weight