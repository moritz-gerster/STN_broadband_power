# %%
"""Runs all scripts necessary for hypothesis 1.1."""
from os.path import basename

from scripts._00_bidsify_sourcedata import bidsify_sourcedata
from scripts._01_preprocess import preprocess
from scripts._02_calc_spectra import save_spectra
from scripts._03_make_dataframe import psd_fooof_to_dataframe
from scripts._04_organize_dataframe import organize_df
from scripts._05_localization_table import export_localization_table
from scripts._06_plot_figures import all_figures

bidsify_sourcedata(neumann=True, litvak=True, hirschmann=True, tan=True,
                   florin=True)
preprocess()
save_spectra()
psd_fooof_to_dataframe(load_fits=False)
organize_df()
export_localization_table()
all_figures()
print(f"{basename(__file__).strip('.py')} done.")
