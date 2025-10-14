"""Config File setting all parameters and paths."""
import numpy as np

# Task and conds
TASKS = ["Rest"]
ACQUISITIONS = (["StimOff"])

RESAMPLE_FREQ = 2000
HIGHPASS = 1
LOWPASS = 795 # avoid line noise
RECORDINGS = ["Neumann", "Litvak", 'Florin', "Hirschmann", "Tan"]

DBS_LEADS = {
    1: {# 'DBS_manufacturer': 'Medtronic',
        'DBS_model': 'Medtronic SenSight Short',
        'DBS_description': '8-contact, 4-level, directional DBS lead',
        'DBS_directional': True},

    2: {# 'DBS_manufacturer': 'Boston Scientific',
        'DBS_model': 'Boston Scientific Vercise Cartesia',
        'DBS_description': '8-contact, 4-level, directional DBS lead',
        'DBS_directional': True},

    3: {# 'DBS_manufacturer': 'Medtronic',
        'DBS_model': 'Medtronic 3389',
        'DBS_description': '4-contact, 4-level, non-directional DBS lead',
        'DBS_directional': False},

    4: {# 'DBS_manufacturer': 'Boston Scientific',
        'DBS_model': 'Boston Scientific Vercise Cartesia X',
        'DBS_description': '16-contact, 5-level, directional DBS lead',
        'DBS_directional': True},

    5: {# 'DBS_manufacturer': 'St. Jude',
        'DBS_model': 'St. Jude Infinity directional',
        'DBS_description': '8-contact, 4-level, directional DBS lead',
        'DBS_directional': True},

    6: {# 'DBS_manufacturer': 'Boston Scientific',
        'DBS_model': 'Boston Scientific Vercise Standard',
        'DBS_description': '8-contact, 8-level, non-directional DBS lead',
        'DBS_directional': False},

    7: {# 'DBS_manufacturer': 'St. Jude',
        'DBS_model': 'St. Jude Infinity',
        'DBS_description': '4-contact, 4-level, non-directional DBS lead',
        'DBS_directional': False},

    8: {# 'DBS_manufacturer': 'Boston Scientific',
        'DBS_model': 'Boston Scientific Vercise Cartesia HX',
        'DBS_description': '16-contact, 8-level, directional DBS lead',
        'DBS_directional': True},

    0: {# 'DBS_manufacturer': np.nan,
        'DBS_model': np.nan,
        'DBS_description': np.nan,
        'DBS_directional': np.nan}
    }

LINE_NOISE_BROAD = [
    'sub-TanL004_ses-LfpMedOn01_task-RestLdopa_acq-StimOnLeft_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif',
    'sub-TanL005_ses-LfpMedOff01_task-RestLdopa_acq-StimOnLeft_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif',
    'sub-TanL005_ses-LfpMedOff01_task-RestLdopa_acq-StimOnRight_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif',
    'sub-TanL006_ses-LfpMedOn01_task-RestLdopa_acq-StimOnLeft_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif',
    'sub-TanL007_ses-LfpMedOn01_task-RestLdopa_acq-StimOnLeft_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif',
    'sub-TanL010_ses-LfpMedOn01_task-RestLdopa_acq-StimOnLeft_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif',
    'sub-TanL010_ses-LfpMedOn01_task-RestLdopa_acq-StimOnRight_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif',
    'sub-TanL011_ses-LfpMedOn01_task-RestLdopa_acq-StimOnLeft_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif',
    'sub-TanL011_ses-LfpMedOff01_task-RestLdopa_acq-StimOnLeft_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif',
    'sub-TanL011_ses-LfpMedOff01_task-RestLdopa_acq-StimOnRight_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif',
    'sub-LitML013_ses-MegLfpMedOn01_task-Rest_acq-StimOff_run-01_proc-HighpassWelch_rec-Litvak_desc-cleaned_ieeg.fif',
    'sub-LitML001_ses-MegLfpMedOff01_task-Rest_acq-StimOff_run-01_proc-HighpassWelch_rec-Litvak_desc-cleaned_ieeg.fif',
    'sub-LitML002_ses-MegLfpMedOff01_task-Rest_acq-StimOff_run-01_proc-HighpassWelch_rec-Litvak_desc-cleaned_ieeg.fif',
    'sub-LitML008_ses-MegLfpMedOff01_task-Rest_acq-StimOff_run-01_proc-HighpassWelch_rec-Litvak_desc-cleaned_ieeg.fif',
    'sub-LitML010_ses-MegLfpMedOff01_task-Rest_acq-StimOff_run-01_proc-HighpassWelch_rec-Litvak_desc-cleaned_ieeg.fif',
    'sub-LitML011_ses-MegLfpMedOff01_task-Rest_acq-StimOff_run-01_proc-HighpassWelch_rec-Litvak_desc-cleaned_ieeg.fif',
    'sub-LitML013_ses-MegLfpMedOff01_task-Rest_acq-StimOff_run-01_proc-HighpassWelch_rec-Litvak_desc-cleaned_ieeg.fif',
    'sub-LitML014_ses-MegLfpMedOff01_task-Rest_acq-StimOff_run-01_proc-HighpassWelch_rec-Litvak_desc-cleaned_ieeg.fif',
    'sub-FloML028_ses-LfpMedOff01_task-Rest_acq-StimOff_run-01_proc-HighpassWelch_rec-Florin_desc-cleaned_ieeg.fif'
                     ]

LINE_NOISE_FREQS = {
    'sub-TanL006_ses-LfpMedOn01_task-RestLdopa_acq-StimOnLeft_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif': [50, 847],
    'sub-TanL005_ses-LfpMedOff01_task-RestLdopa_acq-StimOnRight_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif': [50, 260, 390, 910, 951],
    'sub-TanL004_ses-LfpMedOn01_task-RestLdopa_acq-StimOnLeft_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif': [50, 216],
    'sub-FloML013_ses-LfpMedOff01_task-Rest_acq-StimOff_run-01_proc-HighpassWelch_rec-Florin_desc-cleaned_ieeg.fif': [35.84, 50],
    'sub-FloML013_ses-LfpMedOn01_task-Rest_acq-StimOff_run-01_proc-HighpassWelch_rec-Florin_desc-cleaned_ieeg.fif': [35.84, 50],
    'sub-TanL011_ses-LfpMedOff01_task-RestLdopa_acq-StimOnLeft_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif': [50, 853, 953, 997],
    'sub-TanEmptyroom_ses-TMSiSAGA20240212_task-noise_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif': [50, 240, 800, 855, 880, 885, 946],
    'sub-TanEmptyroom_ses-TMSiPorti20240212_task-noise_run-01_proc-HighpassWelch_rec-Tan_desc-cleaned_ieeg.fif': [50, 486, 535, 558, 567, 612, 674, 711, 717, 763, 770, 780, 785, 790, 830, 836, 840, 865, 871, 874, 882, 888, 944, 957, 962, 967, 1000,
                                                                                                                  522, 528, 544, 508, 513, 532, 492]
    }