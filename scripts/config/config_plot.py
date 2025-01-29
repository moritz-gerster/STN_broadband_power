"""Plot parameters."""
from seaborn import color_palette
from matplotlib.colors import to_rgb
import numpy as np

import scripts.config as cfg

cond_palette = color_palette("colorblind")
COLOR_OFF = 'k'
COLOR_ON = 'grey'
COLOR_AMP = 'k'
COLOR_ONOFF_ABS = cond_palette[2]
SAMPLE_STN = r'$n_{\text{STN}}$'
SAMPLE_PAT = r'$n_{\text{pat}}$'
proj_palette = color_palette("Paired")


def get_mean_color(rgb1, rgb2):
    """Get mean color between two rgb colors."""
    color = tuple((np.array(rgb1) + np.array(rgb2)) / 2)
    return color


COLOR_DIC = dict(
    # LDOPA Conditions
    off=COLOR_OFF,
    on=COLOR_ON,
    offon_abs=COLOR_ONOFF_ABS,

    # PSD Kinds
    normalized='#1b9e77',
    normalized2='#49b192',
    absolute='#7570b3',
    absolute2='#aca9d1',
    periodic='#d95f02',
    periodic2='#e17f35',
    periodicAP='#e7298a',
    periodicAP2='#ff6699',
    periodicFULL='k',
    periodicFULL2='grey',
    normalizedInce='#66a61e',

    # Projects
    TMSi_amplifier=COLOR_AMP,

    Neumann=proj_palette[1],
    Berlin=proj_palette[1],
    Neumann2=proj_palette[0],
    Neumann3=get_mean_color(*proj_palette[:2]),

    Litvak=proj_palette[3],
    London=proj_palette[3],
    Litvak2=proj_palette[2],
    Litvak3=get_mean_color(*proj_palette[2:4]),

    Tan=proj_palette[5],
    Oxford=proj_palette[5],
    Tan2=proj_palette[4],
    Tan3=get_mean_color(*proj_palette[4:6]),

    Hirschmann=proj_palette[7],
    Düsseldorf2=proj_palette[7],
    Hirschmann2=proj_palette[6],
    Hirschmann3=get_mean_color(*proj_palette[6:8]),

    Florin=proj_palette[9],
    Düsseldorf1=proj_palette[9],
    Florin2=proj_palette[8],
    Florin3=get_mean_color(*proj_palette[8:10]),

    Oxford_SG_Saga2=proj_palette[10],
    Oxford_SG_Saga=proj_palette[11],
    Oxford_SG_Porti2=proj_palette[10],
    Oxford_SG_Porti=proj_palette[11],
    Oxford_K_Porti2=proj_palette[10],
    Oxford_K_Porti=proj_palette[11],
    Oxford_K_Saga2=proj_palette[10],
    Oxford_K_Saga=proj_palette[11],

    all=to_rgb('k'),
    all2=to_rgb('grey'),
    all3=get_mean_color(to_rgb('k'), to_rgb('grey')),
)

KIND_DICT = dict(normalized='Relative', absolute='Absolute',
                 periodic='Periodic', periodicAP='Aperiodic',
                 periodicBOTH='Periodic', periodicFULL='Total',
                 normalizedInce='Relative (120-160 Hz)')

PROJECT_ORDER = ["Neumann", "Litvak", 'Florin', "Hirschmann", "Tan",
                 "Oxford_SG_Saga", "Oxford_SG_Porti",
                 "Oxford_K_Saga", "Oxford_K_Porti", 'TMSi_amplifier', 'all']
PROJECT_ORDER_SLIM = ["Neumann", "Litvak", 'Florin', "Hirschmann", "Tan", 'all']
PROJECT_ORDER_PALETTE = [COLOR_DIC[proj] for proj in PROJECT_ORDER]
COND_ORDER = ["off", "on", 'offon_rel', 'offon_abs']
PROJECT_DICT_ = dict(Tan='Oxford      ',
                     Neumann='Berlin       ',
                     Hirschmann='Düsseldorf2',
                     Florin='Düsseldorf1',
                     Litvak='London     ')

PROJECT_DICT = dict(Tan='Oxford',
                    Neumann='Berlin',
                    Hirschmann='Düsseldorf2',
                    Florin='Düsseldorf1',
                    Litvak='London',
                    Oxford_SG_Saga='Oxford_SG_Saga',
                    Oxford_SG_Porti='Oxford_SG_Porti',
                    Oxford_K_Saga='Oxford_K_Saga',
                    Oxford_K_Porti='Oxford_K_Porti',
                    TMSi_amplifier='TMSi_amplifier',
                    all='all')

PROJECT_NAMES = [PROJECT_DICT[proj] for proj in PROJECT_ORDER_SLIM]
COND_DICT = dict(on='on', off='off',
                 offon_abs='off-on',
                 offon_rel='off-on [%]')

C_TOTAL_PWR = 'k'
C_PER_PWR = "#f03b20"
C_AP_PWR = "#3182bd"

EXEMPLARY_SUBS_APERIODIC = ['NeuEL020', 'FloML007', 'HirML021', 'HirML013',
                            'TanL009']  # Fig. 8
EXEMPLARY_SUBS_GAMMA = ['HirML018', 'FloML007', 'NeuEL019',
                        'NeuEL026']  # Fig. S8
dark = color_palette("dark")
COLORS_SPECIAL_SUBS = [dark[0], dark[2], dark[6], dark[8], dark[5], dark[9],
                       dark[1]]
SYMBOLS_SPECIAL_SUBS = ['o', 's', 'd', '>', 'p', '*', '<']

UPDRS_labels = ['UPDRS_bradykinesia_contra',
                'UPDRS_bradykinesia_ipsi',
                'UPDRS_bradyrigid_contra',
                'UPDRS_bradyrigid_ipsi',
                'UPDRS_tremor_contra',
                'UPDRS_tremor_ipsi',
                'UPDRS_bradyrigid_left',
                'UPDRS_bradyrigid_right',
                'UPDRS_Date',
                'UPDRS_hemibody_contra',
                'UPDRS_hemibody_ipsi',
                'UPDRS_hemibody_left',
                'UPDRS_hemibody_right',
                'UPDRS_subscore_bradykinesia_contra',
                'UPDRS_subscore_bradykinesia_ipsi',
                'UPDRS_subscore_bradykinesia_left',
                'UPDRS_subscore_bradykinesia_right',
                'UPDRS_subscore_bradykinesia_total',
                'UPDRS_subscore_rigidity_contra',
                'UPDRS_subscore_rigidity_ipsi',
                'UPDRS_subscore_rigidity_left',
                'UPDRS_subscore_rigidity_right',
                'UPDRS_subscore_rigidity_total',
                'UPDRS_subscore_tremor_contra',
                'UPDRS_subscore_tremor_ipsi',
                'UPDRS_subscore_tremor_left',
                'UPDRS_subscore_tremor_right',
                'UPDRS_subscore_tremor_total']
updrs_dic = {updrs: updrs.replace("_", " ").capitalize()
             for updrs in UPDRS_labels}
updrs_dic = {key: value.replace('Updrs', 'UPDRS')
             for key, value in updrs_dic.items()}
updrs_dic['UPDRS_III'] = 'UPDRS-III'
updrs_dic['UPDRS_pre_bradykinesia_contra'] = 'UPDRS bradykinesia contra (pre)'
updrs_dic['UPDRS_pre_bradyrigid_contra'] = 'UPDRS bradyrigid contra (pre)'

updrs_dic_short = updrs_dic.copy()
updrs_dic_short['UPDRS_bradyrigid_contra'] = 'Contra BR'

patient_dict = {'patient_age': 'Age',
                'patient_disease_duration': 'Disease duration'}

aperiodic_dic = {'fm_exponent': '1/f exponent',
                 'fm_offset': 'Offset',
                 'fm_offset_log': 'Offset',
                 'fm_knee': 'Knee',
                 'fm_knee_fit': 'Knee frequency',
                 'fm_exponent (Lor)': '1/f exponent (Lor)',
                 'fm_offset (Lor)': '1/f offset (Lor)',
                 'fm_offset_log (Lor)': '1/f offset (Lor)',
                 'fm_knee (Lor)': 'Knee (Lor)',
                 'fm_knee_fit (Lor)': 'Knee frequency (Lor)',
                 'fm_exponent (narrow)': '1/f exponent (narrow)',
                 'fm_exponent_narrow': '1/f exponent (narrow)',
                 }
aperiodic_dic_short = aperiodic_dic.copy()
aperiodic_dic_short['fm_exponent'] = '1/f'

pwr_dic_units = {}
for band, band_nme in cfg.BAND_NAMES_GREEK.items():
    pwr_dic_units[f"{band}_abs_max"] = (f"{band_nme} Max. PSD "
                                        r'[$\mu V^2/Hz$]')
    pwr_dic_units[f"{band}_abs_mean"] = (f"{band_nme} Mean PSD "
                                         r'[$\mu V^2/Hz$]')
    pwr_dic_units[f"{band}_abs_max_log"] = (f"{band_nme} Max. PSD "
                                            r'[log10$(\mu V^2/Hz)$]')
    pwr_dic_units[f"{band}_fm_powers_max_log"] = (f"{band_nme} Max. PSD "
                                                  r'[log10$(\mu V^2/Hz) - 1$]')
    pwr_dic_units[f"{band}_fm_auc_log"] = (f"{band_nme} Max. PSD "
                                           r'[log10$(\mu V^2/Hz) - 1$]')
    pwr_dic_units[f"{band}_fm_powers_max"] = (f"{band_nme} Max. PSD "
                                              r'[$\mu V^2/Hz$]')
    pwr_dic_units[f"{band}_fm_auc"] = (f"{band_nme} Max. PSD "
                                       r'[$\mu V^2/Hz$]')

band_dic = {}
# for band, band_nme in cfg.BAND_NAMES.items():
for band, band_nme in cfg.BAND_NAMES_GREEK.items():
    # total power
    band_dic[f"{band}_abs_max"] = f"{band_nme} max"
    band_dic[f"{band}_abs_max_log"] = f"{band_nme} max"
    band_dic[f"{band}_abs_max5Hz"] = f"{band_nme} 5 Hz max"
    band_dic[f"{band}_abs_mean"] = f"{band_nme} mean"
    band_dic[f"{band}_abs_mean_log"] = f"{band_nme} mean"
    band_dic[f"{band}_abs_min"] = f"{band_nme} min"
    band_dic[f"{band}_abs_min_log"] = f"{band_nme} min"
    band_dic[f"{band}_abs_max5Hz_log"] = f"{band_nme} 5 Hz max"
    band_dic[f"{band}_abs_max_freq"] = f"{band_nme} peak freq."
    band_dic[f"{band}_fm_mean"] = f"{band_nme} mean"
    band_dic[f"{band}_fm_mean_log"] = f"{band_nme} mean"
    # fooof power
    band_dic[f"{band}_fm_auc"] = f"{band_nme} AUC"
    band_dic[f"{band}_fm_auc_log"] = f"{band_nme} AUC"
    band_dic[f"{band}_fm_powers_max"] = f"{band_nme} peak"
    band_dic[f"{band}_fm_powers_max_log"] = f"{band_nme} peak"
    band_dic[f"{band}_fm_centerfreqs_max"] = f"{band_nme} max. per. freq."
    band_dic[f"{band}_fm_stds_max"] = f"{band_nme} peak width"
    band_dic[f"{band}_fm_peak_count"] = f"{band_nme} # per. peaks"
    band_dic[f"{band}_fm_band_aperiodic_log"] = f"{band_nme} Aper. power"
band_dic['psd_mean_5to95_log'] = 'Mean PSD 5-95 Hz'
band_dic['psd_sum_5to95'] = 'PSD sum 5-95 Hz'

PLOT_LABELS = {
    'patient_days_after_implantation': 'Days after surgery',
    **updrs_dic_short, **band_dic, **aperiodic_dic, **patient_dict,
    **KIND_DICT, **COND_DICT}

PLOT_LABELS_SHORT = {**aperiodic_dic_short,
                     'full_fm_band_aperiodic_log': 'Aper. power'}
for band, band_nme in cfg.BAND_NAMES_GREEK_SHORT.items():
    # total power
    PLOT_LABELS_SHORT[f"{band}_abs_max"] = f"{band_nme} max"
    PLOT_LABELS_SHORT[f"{band}_abs_max_log"] = f"{band_nme} max"
    PLOT_LABELS_SHORT[f"{band}_abs_max5Hz"] = f"{band_nme} 5 Hz max"
    PLOT_LABELS_SHORT[f"{band}_abs_mean"] = f"{band_nme} mean"
    PLOT_LABELS_SHORT[f"{band}_abs_mean_log"] = f"{band_nme} mean"
    PLOT_LABELS_SHORT[f"{band}_abs_min"] = f"{band_nme} min"
    PLOT_LABELS_SHORT[f"{band}_abs_min_log"] = f"{band_nme} min"
    PLOT_LABELS_SHORT[f"{band}_abs_max5Hz_log"] = f"{band_nme} 5 Hz max"
    PLOT_LABELS_SHORT[f"{band}_abs_max_freq"] = f"{band_nme} peak freq."
    PLOT_LABELS_SHORT[f"{band}_fm_mean"] = f"{band_nme} mean"
    PLOT_LABELS_SHORT[f"{band}_fm_mean_log"] = f"{band_nme} mean"
    # fooof power
    PLOT_LABELS_SHORT[f"{band}_fm_auc"] = f"{band_nme} AUC"
    PLOT_LABELS_SHORT[f"{band}_fm_auc_log"] = f"{band_nme} AUC"
    PLOT_LABELS_SHORT[f"{band}_fm_powers_max"] = f"{band_nme} peak"
    PLOT_LABELS_SHORT[f"{band}_fm_powers_max_log"] = f"{band_nme} peak"
    PLOT_LABELS_SHORT[f"{band}_fm_centerfreqs_max"] = f"{band_nme} max. per. freq."
    PLOT_LABELS_SHORT[f"{band}_fm_stds_max"] = f"{band_nme} peak width"
    PLOT_LABELS_SHORT[f"{band}_fm_peak_count"] = f"{band_nme} # per. peaks"
    PLOT_LABELS_SHORT[f"{band}_fm_band_aperiodic"] = f"{band_nme} aper. power"
PLOT_LABELS_SHORT['psd_mean_5to95_log'] = 'Mean PSD 5-95 Hz'
PLOT_LABELS_SHORT['psd_sum_5to95'] = 'PSD sum 5-95 Hz'


def get_channel_plot_colors(long=False):
    """
    Get colors for plotting channels.

    Left hemisphere: autumn, right hemisphere: winter, eeg: cool.
    Sequential colormaps as a function of channels for each hemisphere and
    channel type.
    """
    from matplotlib.pyplot import cm
    from numpy import linspace
    color_dic = {"L": cm.autumn, "R": cm.winter}

    plot_colors = {}
    # TMSi amplifier
    plot_colors["TMSi"] = "k"
    plot_colors['LFP_L_WIEST'] = 'k'
    plot_colors['LFP_R_WIEST'] = 'k'
    plot_colors['LFP_L_INSIDE'] = 'k'
    plot_colors['LFP_R_INSIDE'] = 'k'

    bipolar_lfp_left = ['LFP_L_1-2', 'LFP_L_2-3', 'LFP_L_3-4']
    bipolar_amp_left = ['Amp_L_1-2', 'Amp_L_2-3', 'Amp_L_3-4']
    bipolar_lfp_left_distant = ['LFP_L_1-3', 'LFP_L_2-4', 'LFP_L_3-5']
    bipolar_amp_left_distant = ['Amp_L_1-3', 'Amp_L_2-4']
    bipolar_lfp_right = ['LFP_R_1-2', 'LFP_R_2-3', 'LFP_R_3-4']
    bipolar_amp_right = ['Amp_R_1-2', 'Amp_R_2-3', 'Amp_R_3-4']
    bipolar_lfp_right_distant = ['LFP_R_1-3', 'LFP_R_2-4', 'LFP_R_3-5']
    bipolar_amp_right_distant = ['Amp_R_1-3', 'Amp_R_2-4']

    monopolar_lfp_left = ['LFP_L_1', 'LFP_L_2', 'LFP_L_3', 'LFP_L_4']

    monopolar_lfp_right = ['LFP_R_1', 'LFP_R_2', 'LFP_R_3', 'LFP_R_4']

    left_chs = [bipolar_lfp_left, bipolar_lfp_left_distant,
                monopolar_lfp_left, bipolar_amp_left, bipolar_amp_left_distant]
    right_chs = [bipolar_lfp_right, bipolar_lfp_right_distant,
                 monopolar_lfp_right, bipolar_amp_right,
                 bipolar_amp_right_distant]
    for hemi, hemi_channels in zip(["L", "R"], [left_chs, right_chs]):
        for chs in hemi_channels:
            colors = color_dic[hemi](linspace(0, 1, len(chs)))
            col_dic = dict(zip(chs, colors))
            plot_colors.update(col_dic)
    lfp_directional = (monopolar_lfp_left + monopolar_lfp_right)
    for ch_name in lfp_directional:
        color = plot_colors[ch_name]
        for letter in "abc":
            ch_dir = ch_name + letter
            plot_colors[ch_dir] = color

    if long:
        monopolar_lfp_left_sub8 = ["LFP_L_1", "LFP_L_2", "LFP_L_3", "LFP_L_4",
                                   "LFP_L_5", "LFP_L_6", "LFP_L_7", "LFP_L_8"]
        monopolar_lfp_right_sub8 = ["LFP_R_1", "LFP_R_2", "LFP_R_3", "LFP_R_4",
                                    "LFP_R_5", "LFP_R_6", "LFP_R_7", "LFP_R_8"]

        bipolar_lfp_left_sub8 = ['LFP_L_1-2', 'LFP_L_2-3', 'LFP_L_3-4',
                                 'LFP_L_4-5', 'LFP_L_5-6', 'LFP_L_6-7',
                                 'LFP_L_7-8']
        bipolar_lfp_right_sub8 = ['LFP_R_1-2', 'LFP_R_2-3', 'LFP_R_3-4',
                                  'LFP_R_4-5', 'LFP_R_5-6', 'LFP_R_6-7',
                                  'LFP_R_7-8']

        # Add LFP colors subject 8 (1abc-...-5abc-6)
        left_chs = [monopolar_lfp_left_sub8, bipolar_lfp_left_sub8]
        right_chs = [monopolar_lfp_right_sub8, bipolar_lfp_right_sub8]
        for hemi, hemi_channels in zip(["L", "R"], [left_chs, right_chs]):
            for chs in hemi_channels:
                colors = color_dic[hemi](linspace(0, 1, len(chs)))
                col_dic = dict(zip(chs, colors))
                plot_colors.update(col_dic)
        lfp_directional = (monopolar_lfp_left_sub8 + bipolar_lfp_left_sub8 +
                           monopolar_lfp_right_sub8 + bipolar_lfp_right_sub8)
        for ch_name in lfp_directional:
            color = plot_colors[ch_name]
            for letter in "abc":
                ch_dir = ch_name + letter
                plot_colors[ch_dir] = color
    return plot_colors


CHANNEL_PLOT_COLORS = get_channel_plot_colors()
CHANNEL_PLOT_COLORS_LONG = get_channel_plot_colors(long=True)

DBS_MODEL_DIC = {
    'St. Jude Infinity directional':
        r'$\bf{St. Jude}$''\nInfinity directional',
    'Boston Scientific Vercise Cartesia':
        r'$\bf{Boston Scientific}$''\nVercise Cartesia',
    'Medtronic 3389':
        r'$\bf{Medtronic}$''\n3389',
    'Boston Scientific Vercise Standard':
        r'$\bf{Boston Scientific}$''\nVercise Standard',
    'St. Jude Infinity':
        r'$\bf{St. Jude}$''\nInfinity',
    'Medtronic SenSight Short':
        r'$\bf{Medtronic}$''\nSenSight Short',
    'Boston Scientific Vercise Cartesia X':
        r'$\bf{Boston Scientific}$''\nVercise Cartesia X',
    'Boston Scientific Vercise Cartesia HX':
        r'$\bf{Boston Scientific}$''\nVercise Cartesia HX'
        }