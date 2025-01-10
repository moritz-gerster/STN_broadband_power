"""Plot and save all subfigures."""
from os.path import basename, join

import pandas as pd

import scripts.config as cfg
from scripts.plot_figures import (figure1, figure2, figure3, figure4, figure5,
                                  figure6, figure7, figure8, supp_figure1,
                                  supp_figure2, supp_figure3, supp_figure4,
                                  supp_figure5, supp_figure6, supp_figure7,
                                  supp_figure8)


def all_figures():
    df_orig = pd.read_pickle(join(cfg.DF_PATH, cfg.DF_FOOOF))
    figure1(df_orig)
    figure2(df_orig)
    figure3(df_orig)
    figure4(df_orig)
    df_corr = figure5(df_orig)
    figure6(df_corr)
    figure7(df_orig)
    figure8(df_orig)

    supp_figure1(df_orig)
    supp_figure2(df_orig)
    supp_figure3(df_orig)
    supp_figure4(df_orig)
    supp_figure5(df_orig)
    supp_figure6(df_orig)
    supp_figure7(df_orig)
    supp_figure8(df_orig)
    print(f"{basename(__file__).strip('.py')} done.")
    return None


if __name__ == "__main__":
    import time
    start = time.time()
    all_figures()
    end = time.time()
    print(f"Time elapsed: {end - start:.2f} s")
