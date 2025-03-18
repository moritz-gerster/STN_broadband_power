from os import makedirs
from os.path import join

from scripts.config import FIG_PAPER
from scripts.plot_figures._sweetspot_distance import (plot_corrs_highbeta_off,
                                                      plot_sweetspot_distance)


def figure5(df_orig):
    output_file_path = join(FIG_PAPER, 'Figure5', "output.txt")
    makedirs(join(FIG_PAPER, 'Figure5'), exist_ok=True)
    with open(output_file_path, "w") as output_file:
        df_corr = plot_sweetspot_distance(df_orig, fig_dir='Figure5',
                                          output_file=output_file)
        plot_corrs_highbeta_off(df_corr, fig_dir='Figure5',
                                output_file=output_file)
