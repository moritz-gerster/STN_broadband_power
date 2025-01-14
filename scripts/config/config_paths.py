'''Config paths.'''
from os.path import join

BASE_DIR = '../BIDS_STN_LFP'

# =============================================================================
# Set load paths
# =============================================================================
ANNOTATION_NOTES = join(BASE_DIR, 'sourcedata', 'BIDS_Neumann_ECOG_LFP',
                        'meta_infos', 'Data_Notes_Berlin.xlsx'), 'TO_JSON'
GOOD_FILES_JSON = join(BASE_DIR, 'sourcedata', 'BIDS_Neumann_ECOG_LFP',
                       'meta_infos', 'Berlin_MedOffOn.json')

# =============================================================================
# Save paths data
# =============================================================================
DF_PATH = join('results', 'dataframes')
DF_FOOOF = 'df_fooof.pkl'
DF_ML = 'df_ml.pkl'
DF_FOOOF_RAW = 'df_fooof_raw.pkl'
DF_COH = 'df_coherence.pkl'
DF_TRACT = 'df_tracts.pkl'
NEW_CH_NAMES = join('rawdata', 'meta_infos_Neumann', 'converted_ch_names.json')
PREPROCESSED = join('derivatives', 'preprocessed')
SPECTRA = join('derivatives', 'spectra')
RAWDATA = join('rawdata')
SOURCEDATA = join(BASE_DIR, 'sourcedata')
ANNOTATIONS = join(BASE_DIR, 'derivatives', 'annotations')

# =============================================================================
# Save paths plots
# =============================================================================

FIG_ASD = 'results/plots/amplitude_spectra/'
FIG_PSD_FOOOF = 'results/plots/power_spectra/'
FIG_FOOOF = 'results/plots/fooof_plots/'
FIG_RESULTS = 'results/plots/analysis_results/'
FIG_PAPER = 'results/plots/paper_figures/'
FOOOF_SAVE_JSON = 'derivatives/fooof_fit_data'

BIDS_FILES_TO_COPY = ['dataset_description.json', 'participants.json',
                      'participants.tsv', 'meta_infos']
