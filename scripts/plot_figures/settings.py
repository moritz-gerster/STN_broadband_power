import matplotlib.pyplot as plt

import scripts.config as cfg


def get_dfs(df, ch_choice=None, chs=None, equalize_subjects_norm_abs=False):
    # Select channels
    ch_mask = df.ch.isin(chs) if chs is not None else df[ch_choice]

    # Filter
    df = df[(df.ch_reference == 'bipolar') & ~df.ch_bad & ch_mask].copy()
    df.loc[:, 'sub_hemi_cond'] = df.sub_hemi + '_' + df.cond  # add column

    # Select dataframes
    df_norm = df[(df.psd_kind == "normalized") & (df.fm_params is False)]
    df_abs = df[(df.psd_kind == "standard") & (df.fm_params == 'broad')]
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
    pivot['asymmetric_on'] = pivot.patient_symptom_dominant_side_BR_on.isin(
        ['severe side', 'mild side'])
    pivot['asymmetric_off'] = pivot.patient_symptom_dominant_side_BR_off.isin(
        ['severe side', 'mild side'])
    pivot['has_model'] = pivot.fm_exponent.notna()

    # Step 1: Group by subject, project, and hemisphere to check condition
    # availability
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

    dataframes = dict(df=df, df_norm=df_norm, df_per=df_per, df_abs=df_abs,
                      df_sample_sizes=df_sample_sizes)
    return dataframes

# Bands
BANDS = ['delta', 'theta', "alpha", "beta_low", "beta_high", 'gamma_low']
APERIODIC = ['fm_offset_log', 'fm_exponent', 'fm_exponent_narrow']

# Plot spectra
CI_SPECT = 'se'  # use standard error for spectra

# Plot confidence intervals
CI = 95  # use non-parametric 95% confidence interval for results

# Get xticks and labels based on bands
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

# Permutations

# Paper settings
N_PERM_CLUSTER = int(1e6)
N_BOOT_COHEN = 10000
N_PERM_CORR = 10000

# Fast settings
# N_PERM_CLUSTER = int(1e2)
# N_BOOT_COHEN = None
# N_PERM_CORR = None

# Plot settings

# Text
FONTSIZE_S = 5
FONTSIZE_ASTERISK = 7
FONTSIZE_M = 6
FONTSIZE_L = 8
plt.rc('font', size=FONTSIZE_S)  # bold not working
plt.rc('axes', titlesize=FONTSIZE_S, labelsize=FONTSIZE_S)
plt.rc('xtick', labelsize=FONTSIZE_S)
plt.rc('ytick', labelsize=FONTSIZE_S)
plt.rc('legend', fontsize=FONTSIZE_S, title_fontsize=FONTSIZE_S, framealpha=1)
plt.rc('figure', titlesize=FONTSIZE_S)

# Graphics
LINEWIDTH_AXES = .25
TICK_SIZE = 1.5
LINEWIDTH_PLOT = .5
plt.rc('lines', linewidth=LINEWIDTH_PLOT)
plt.rc('axes', linewidth=LINEWIDTH_AXES)
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['xtick.major.size'] = TICK_SIZE
plt.rcParams['xtick.major.width'] = LINEWIDTH_AXES
plt.rcParams['ytick.major.size'] = TICK_SIZE
plt.rcParams['ytick.major.width'] = LINEWIDTH_AXES
plt.rcParams['patch.linewidth'] = LINEWIDTH_AXES
plt.rcParams['grid.linewidth'] = LINEWIDTH_AXES
