# %%
"""Runs all scripts necessary for hypothesis 1.1."""
from os.path import basename

from scripts._00_bidsify_sourcedata import bidsify_sourcedata
from scripts._01_preprocess import preprocess
from scripts._02_calc_spectra import save_spectra
from scripts._03_make_dataframe import psd_fooof_to_dataframe
from scripts._04_organize_dataframe import organize_df
from scripts._plot1_PSDs import plot_spectra
from scripts._plot3_FOOOF import plot_fooof
# from scripts._plot5_results import relevant_plots
from scripts._paper_figures_OLD import all_figures

# bidsify_sourcedata(neumann=True, litvak=True, hirschmann=True, tan=True,
#                    florin=True)
# preprocess()
# save_spectra()
psd_fooof_to_dataframe(load_fits=True)
organize_df()
# plot_spectra(psd='asd', projects='Neumann')
# plot_spectra(psd='psd')
# plot_fooof()
# relevant_plots()
# all_figures()
print(f"{basename(__file__).strip('.py')} done.")

# TODO: remove all pylint annotation with which I disagree (superflous parents,
# no lambda expressions, etc.)