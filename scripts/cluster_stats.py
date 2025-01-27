"""Cluster stats modified from Richard Köhler.

Bases on pte_stats"""
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes, figure
from pte_stats import cluster, timeseries


def lineplot_compare(x_1: np.ndarray, x_2: np.ndarray, times: np.ndarray,
                     alpha: float = 0.05,
                     n_perm: int = 1000,
                     correction_method: str = "cluster",
                     two_tailed: bool = False, paired_x1x2: bool = False,
                     ax: axes.Axes | None = None,
                     y_lims: Sequence | None = None,
                     colors: Sequence[tuple] | None = None,
                     x_label: str | None = None,
                     y_label: str | None = None,
                     title: str | None = None,
                     legend: bool = True,
                     color_cluster='yellow',
                     leg_title=None,
                     alpha_plot=0.5,
                     plot_clusters=True,
                     show: bool = True, save_path: str | None = None
                     ) -> tuple[figure.Figure | None,
                                list[tuple[float, float]]]:
    """Plot comparison of continuous prediction arrays."""
    fig = None
    if not ax:
        fig, axis = plt.subplots(1, 1)
    else:
        axis = ax
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:2]

    # if cluster_times is None:
    cluster_times = _pval_correction_lineplot(ax=axis, x=x_1, y=x_2,
                        times=times,
                        alpha=alpha, n_perm=n_perm,
                        correction_method=correction_method,
                        two_tailed=two_tailed,
                        min_cluster_size=2,
                        color_cluster=color_cluster,
                        alpha_plot=alpha_plot,
                        plot_clusters=plot_clusters,
                        onesample_xy=paired_x1x2)
    if title:
        ax.set_title(title)
    if x_label:
        axis.set_xlabel(x_label)
    elif x_label is None:
        axis.set_xlabel(x_label)
    if legend:
        if leg_title is not None:
            leg_title += fr" ($n=2 \times {x_1.shape[1]}$)"
        axis.legend(title=leg_title, loc="upper right")
    if y_label:
        axis.set_ylabel(y_label)
    if y_lims:
        axis.set_ylim(y_lims[0], y_lims[1])
    if fig is not None:
        fig.tight_layout()
        if save_path:
            fig.savefig(str(save_path), bbox_inches="tight")
    if show:
        plt.show(block=True)
    return fig, cluster_times


def _pval_correction_lineplot(ax: axes.Axes, x: np.ndarray,
                              y: int | float | np.ndarray, times: np.ndarray,
                              alpha: float, correction_method: str,
                              n_perm: int, two_tailed: bool,
                              onesample_xy: bool,
                              one_tailed_test: Literal["larger"]
                              | Literal["smaller"] = "larger",
                              min_cluster_size: int = 2,
                              color_cluster="yellow",
                              alpha_plot=0.5,
                              plot_clusters=True,
                              stat_label=True,
                              ) -> list[tuple[float, float]]:
    """Perform p-value correction for single lineplot."""
    cluster_times = []
    if onesample_xy:
        data_a = x - y
        data_b = 0.0
    else:
        data_a = x
        data_b = y

    if not two_tailed and one_tailed_test == "smaller":
        data_a_stat = data_a * -1
    else:
        data_a_stat = data_a

    if correction_method == "cluster":
        _, clusters_ind = cluster.cluster_analysis_1d(data_a=data_a_stat.T,
                                data_b=data_b, alpha=alpha, n_perm=n_perm,
                                only_max_cluster=False, two_tailed=two_tailed,
                                min_cluster_size=min_cluster_size)
        if len(clusters_ind) == 0:
            return cluster_times
        cluster_count = len(clusters_ind)
        clusters = np.zeros(data_a_stat.shape[0], dtype=np.int32)
        for ind in clusters_ind:
            clusters[ind] = 1
    elif correction_method in ["cluster_pvals", "fdr"]:
        p_vals = timeseries.timeseries_pvals(
            x=data_a_stat, y=data_b, n_perm=n_perm, two_tailed=two_tailed)
        clusters, cluster_count = cluster.clusters_from_pvals(
            p_vals=p_vals, alpha=alpha, correction_method=correction_method,
            n_perm=n_perm, min_cluster_size=min_cluster_size)
    else:
        msg = f"Unknown cluster correction method: {correction_method}."
        raise ValueError(msg)

    if cluster_count <= 0:
        print("No clusters found.")
        return cluster_times
    if isinstance(y, (int, float)):
        y_arr = np.ones((x.shape[0], 1))
        y_arr[:, 0] = y
    else:
        y_arr = y
    if onesample_xy:
        x_arr = x
    else:
        x_arr = data_a
    if stat_label:
        label = f"p ≤ {alpha}"
    else:
        label = None
    x_labels = times.round(2)
    clusters = np.array(clusters)  # added by MG
    for cluster_idx in range(1, cluster_count + 1):
        index = np.where(clusters == cluster_idx)[0]
        if index.size == 0:
            print("No clusters found.")
            continue
        y1 = x_arr[index].mean(axis=1)
        y2 = y_arr[index].mean(axis=1)
        if plot_clusters:  # added by MG
            ax.fill_between(x=times[index], y1=y1, y2=y2, alpha=alpha_plot,
                            color=color_cluster,
                            label=label)
        time_0 = x_labels[index[0]]
        time_1 = x_labels[index[-1]]
        print(f"Cluster found between {time_0} Hz and" f" {time_1} Hz.")
        label = None  # Avoid printing label multiple times
        cluster_times.append((time_0, time_1))
    return cluster_times
