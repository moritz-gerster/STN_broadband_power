from os import makedirs
from os.path import join

from scripts.config import FIG_PAPER
from scripts.plot_figures._correlation_scatter import model_comparison
from scripts.plot_figures.settings import get_dfs


def figure4(df_orig):
    # equalize subject count for model comparisons
    dataframes_equal = get_dfs(df_orig, ch_choice='ch_dist_sweet',
                               equalize_subjects_norm_abs=True)
    output_file_path = join(FIG_PAPER, 'Figure4', "5___output.txt")
    makedirs(join(FIG_PAPER, 'Figure4'), exist_ok=True)
    with open(output_file_path, "w") as output_file:
        model_comparison(dataframes_equal,
                         fig_dir='Figure4',
                         model_comparison='j_test',
                         output_file=output_file)
