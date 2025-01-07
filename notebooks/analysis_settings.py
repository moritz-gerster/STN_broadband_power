import pandas as pd
from os.path import join
import scripts.config as cfg
import seaborn as sns
import matplotlib.pyplot as plt


# def _specify_subjects_analysis(df):
#     """Indicate subjects to use for within subjects analysis fullfilling all
#     required conditions."""
#     df_within = df.copy()
#     df_within = df_within[df_within.cond.isin(['off', 'on'])
#                           & ~df_within.project.isin(['all'])]

#     # Condition 1: Symptom dominant hemisphere consistent across conditions
#     # df_within = df_within[df_within.dominant_side_consistent]
#     # df_within = df_within[df_within.dominant_side_consistent_or_equal]

#     # Condition 2: Hemispheres are unequally affected (lateralized symptoms)
#     side_on = df_within["patient_symptom_dominant_side_BR_on"]
#     side_off = df_within["patient_symptom_dominant_side_BR_off"]
#     on_not_equal = side_on.isin(['severe side', 'mild side'])
#     off_not_equal = side_off.isin(['severe side', 'mild side'])
#     # df_within = df_within[on_not_equal & off_not_equal]
#     df_within = df_within[off_not_equal
#                           & (on_not_equal & df_within.dominant_side_consistent)]

#     # Condition 3: FOOOF did not fail
#     df_within = df_within[df_within.fm_exponent.notna()]

#     # Condition 4: Data for both hemispheres exist
#     # count = len(['off', 'on']) + len(['L', 'R'])
#     # subject_counts = df_within.groupby(['subject']).ch_hemisphere.count() == count
#     # subject_counts = df_within.groupby('subject').ch_hemisphere.nunique() == 2
#     # valid_subjects = subject_counts[subject_counts].index
#     # df_within = df_within[df_within.subject.isin(valid_subjects)]

#     # Indicate subject selection
#     final_subjects = df_within.subject.unique()
#     df["asymmetric_subjects"] = df.subject.isin(final_subjects)
#     df[df.cond == 'off'].subject.nunique()
#     return df


def get_dfs(ch_choice=None, chs=None, equalize_subjects_norm_abs=False,
            broad_params='broad'):
    df = pd.read_pickle(join('..', cfg.DF_PATH, cfg.DF_FOOOF))

    ch_mask = df.ch.isin(chs) if chs is not None else df[ch_choice]

    df = df[(df.ch_reference == 'bipolar') & ~df.ch_bad & ch_mask]

    # df = _specify_subjects_analysis(df)

    df['sub_hemi_cond'] = df.sub_hemi + '_' + df.cond

    df_norm = df[(df.psd_kind == "normalized") & (df.fm_params == False)]
    # df_normInce = df[(df.psd_kind == "normalizedInce")
    #                  & (df.fm_params == False)]
    df_abs = df[(df.psd_kind == "standard") & (df.fm_params == broad_params)]
    df_per = df_abs.copy()
    df_per = df_per[df_per.fm_exponent.notna()]

    if equalize_subjects_norm_abs:
        # Apply same subjects to norm and abs for better comparison of methods
        df_abs = df_abs[df_abs.fm_exponent.notna()]
        subs_abs_per = df_abs.sub_hemi_cond.unique()
        df_norm = df_norm[df_norm.sub_hemi_cond.isin(subs_abs_per)]

    # Sanity checks to avoid accidental duplicates
    dfs = [df_norm, df_abs, df_per]
    for df_ in dfs:
        group = df_.groupby(['project_nme', 'subject', 'cond', 'ch_nme'])
        msg = 'Some subject-hemispheres plotted twice.'
        assert (group.ch_nme.count() <= 1).all(), msg

    # remove 'all' project
    pivot = df[(df.psd_kind == "standard") & (df.fm_params == 'broad')
               & (df.project != 'all')]
    pivot = pivot[pivot.cond.isin(['on', 'off', 'offon_abs'])]
    pivot['UPDRS_exists'] = pivot.UPDRS_bradyrigid_contra.notna()
    pivot['asymmetric_on'] = pivot.patient_symptom_dominant_side_BR_on.isin(['severe side', 'mild side'])
    pivot['asymmetric_off'] = pivot.patient_symptom_dominant_side_BR_off.isin(['severe side', 'mild side'])
    pivot['has_model'] = pivot.fm_exponent.notna()

    # Step 1: Group by subject, project, and hemisphere to check condition availability
    cols = ['subject', 'project_nme', 'ch_hemisphere', 'UPDRS_exists',
            'asymmetric_on',
            'asymmetric_off', 'dominant_side_consistent', 'has_model']
    pivot = pivot.pivot_table(index=cols, columns='cond', aggfunc='size',
                               fill_value=0).reset_index()

    # Step 2: Create boolean columns for off, on conditions
    pivot['off_available'] = pivot[pivot['has_model']]['off'] > 0
    pivot['on_available'] = pivot[pivot['has_model']]['on'] > 0

    # Step 3: Aggregate the availability across hemispheres
    df_sample_sizes = pivot.groupby(['subject', 'project_nme']).agg(
        off_available=('off_available', 'any'),
        on_available=('on_available', 'any'),
        both_hemis_off_available=('off_available', lambda x: x.sum() == 2),
        both_hemis_on_available=('on_available', lambda x: x.sum() == 2),
        UPDRS_exists=('UPDRS_exists', 'any'),
        asymmetric_on=('asymmetric_on', 'any'),
        asymmetric_off=('asymmetric_off', 'any'),
        dominant_side_consistent=('dominant_side_consistent', 'any'),
        has_model=('has_model', 'any'),
    ).reset_index()
    # df_sample_sizes['UPDRS_exists'] = True

    dataframes = dict(df=df, df_norm=df_norm,
                    #   df_normInce=df_normInce,
                    df_per=df_per,
                      df_abs=df_abs, df_sample_sizes=df_sample_sizes)
    return dataframes

# Bands
# BANDS = ['delta', 'theta', "alpha", "beta_low", "beta_high", 'gamma_broad']
BANDS = ['delta', 'theta', "alpha", "beta_low", "beta_high", 'gamma_low']
# BANDS = ['delta', 'theta_alpha', 'beta_low', 'beta_high', 'gamma_low']
APERIODIC = ['fm_offset_log', 'fm_exponent', 'fm_exponent_narrow']

# Plot spectra
CI_SPECT = 'se'  # use standard error for spectra
# XTICKS_FREQ = (0, 10, 20, 30, 40)
# XTICKS_FREQ = (0, 15, 30, 45)

# Plot confidence intervals
CI = 95  # use non-parametric 95% confidence interval for results

XTICKS_FREQ = []
bands = BANDS
for band in bands:
    XTICKS_FREQ.extend(cfg.BANDS[band])
XTICKS_FREQ = sorted(list(set(XTICKS_FREQ)))
XTICKS_FREQ_low = XTICKS_FREQ.copy()
XTICKS_FREQ_low_labels = XTICKS_FREQ.copy()
XTICKS_FREQ_low_labels[0] = ''
# XTICKS_FREQ_high = XTICKS_FREQ + [60, 80, 100]
XTICKS_FREQ_high = XTICKS_FREQ + [60, 100]
XTICKS_FREQ_high_labels_skip9 = XTICKS_FREQ_high.copy()
XTICKS_FREQ_high_labels_skip13 = XTICKS_FREQ_high.copy()
XTICKS_FREQ_high_labels_skip9[0] = ''
XTICKS_FREQ_high_labels_skip13[0] = ''
XTICKS_FREQ_high_labels_skip9[2] = ''
XTICKS_FREQ_high_labels_skip13[3] = ''
del XTICKS_FREQ

# Figure directories
kinds = ['normalized',
        #  'relative',
         'absolute',
         'periodic', 'periodicAP', 'periodicFULL',
         'lorentzian', 'lorentzianAP', 'lorentzianFULL',
         'normalizedInce'
         ]
dirs = ['Figure1',
        'Figure2',
        'Figure2', 'Figure2', 'Figure2',
        'Figure2', 'Figure2', 'Figure2',
        'SuppPlateau']
KIND_DIR = dict(zip(kinds, dirs))