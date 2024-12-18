"""Config File setting all paramters and paths."""

from os.path import join
from scripts.config import SOURCEDATA

SOURCEDATA_NEU = join(SOURCEDATA, 'BIDS_Neumann_ECOG_LFP')

# Channels
POSSIBLE_CH_NAMES = {
    # Standard LFP
    "LFP_R_01_STN_BS", "LFP_R_02_STN_BS", "LFP_R_03_STN_BS",
    "LFP_R_04_STN_BS", "LFP_R_05_STN_BS", "LFP_R_06_STN_BS",
    "LFP_R_07_STN_BS", "LFP_R_08_STN_BS",
    "LFP_L_01_STN_BS", "LFP_L_02_STN_BS", "LFP_L_03_STN_BS",
    "LFP_L_04_STN_BS", "LFP_L_05_STN_BS", "LFP_L_06_STN_BS",
    "LFP_L_07_STN_BS", "LFP_L_08_STN_BS",

    "LFP_R_01_STN_MT", "LFP_R_02_STN_MT", "LFP_R_03_STN_MT",
    "LFP_R_04_STN_MT", "LFP_R_05_STN_MT", "LFP_R_06_STN_MT",
    "LFP_R_07_STN_MT", "LFP_R_08_STN_MT",
    "LFP_L_01_STN_MT", "LFP_L_02_STN_MT", "LFP_L_03_STN_MT",
    "LFP_L_04_STN_MT", "LFP_L_05_STN_MT", "LFP_L_06_STN_MT",
    "LFP_L_07_STN_MT", "LFP_L_08_STN_MT",

    # Standard ECOG
    "ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT",
    "ECOG_R_03_SMC_AT", "ECOG_R_04_SMC_AT",
    "ECOG_R_05_SMC_AT", "ECOG_R_06_SMC_AT",

    "ECOG_L_01_SMC_AT", "ECOG_L_02_SMC_AT",
    "ECOG_L_03_SMC_AT", "ECOG_L_04_SMC_AT",
    "ECOG_L_05_SMC_AT", "ECOG_L_06_SMC_AT",

    # Double sided ECOG
    "ECOG_R_07_SMC_AT", "ECOG_R_08_SMC_AT",
    "ECOG_R_09_SMC_AT", "ECOG_R_10_SMC_AT",
    "ECOG_R_11_SMC_AT", "ECOG_R_12_SMC_AT",

    # Directional leads already averaged
    "LFP_R_020304_STN_BS", "LFP_R_050607_STN_BS",
    "LFP_L_020304_STN_BS", "LFP_L_050607_STN_BS",
    "LFP_R_020304_STN_MT", "LFP_R_050607_STN_MT",
    "LFP_L_020304_STN_MT", "LFP_L_050607_STN_MT",
    "LFP_L_234_STN_MT", "LFP_R_234_STN_MT",
    "LFP_L_567_STN_MT", "LFP_R_567_STN_MT",

    # Boston Scientific Vercise Cartesia X
    # 16-contact, 5-level, directional DBS lead
    "LFP_R_09_STN_BS", "LFP_R_10_STN_BS", "LFP_R_11_STN_BS",
    "LFP_R_12_STN_BS", "LFP_R_13_STN_BS", "LFP_R_14_STN_BS",
    "LFP_R_15_STN_BS", "LFP_R_16_STN_BS",
    "LFP_L_09_STN_BS", "LFP_L_10_STN_BS", "LFP_L_11_STN_BS",
    "LFP_L_12_STN_BS", "LFP_L_13_STN_BS", "LFP_L_14_STN_BS",
    "LFP_L_15_STN_BS", "LFP_L_16_STN_BS",

    # EEG
    "EEG_CZ_AO", "EEG_FZ_AO", "EEG_CZ_TM", "EEG_FZ_TM",
    # "EEG_CZ_MT", "EEG_FZ_MT",

    # Accelerometer
    "ACC_R_X_D2_TM", "ACC_R_Y_D2_TM", "ACC_R_Z_D2_TM",
    "ACC_L_X_D2_TM", "ACC_L_Y_D2_TM", "ACC_L_Z_D2_TM",
    "ACC_R_X_D2_AO", "ACC_R_Y_D2_AO", "ACC_R_Z_D2_AO",
    "ACC_L_X_D2_AO", "ACC_L_Y_D2_AO", "ACC_L_Z_D2_AO",

    "ANALOG",  "UNI_33", "TRIGGERS", # no idea

    # Refs
    "REF_EARLOBE", "REF_MASTOID", "REF_COMMONAVG", "REF_GREF",

    # EMG and ECG
    "EMG_R_BR_TM", "EMG_L_BR_TM",
    "EMG_R_BR_AO", "EMG_L_BR_AO",
    'EMG_1_BR_AO', 'EMG_1_FDI_AO',
    'EMG_2_BR_AO', 'EMG_2_FDI_AO',
    "ECG",

    # Sub 16
    'EMG_L_R1C1_BR_TM', 'EMG_L_R1C2_BR_TM', 'EMG_L_R1C3_BR_TM',
    'EMG_L_R1C4_BR_TM', 'EMG_L_R1C5_BR_TM', 'EMG_L_R1C6_BR_TM',
    'EMG_L_R1C7_BR_TM', 'EMG_L_R1C8_BR_TM', 'EMG_L_R2C1_BR_TM',
    'EMG_L_R2C2_BR_TM', 'EMG_L_R2C3_BR_TM', 'EMG_L_R2C4_BR_TM',
    'EMG_L_R2C5_BR_TM', 'EMG_L_R2C6_BR_TM', 'EMG_L_R2C7_BR_TM',
    'EMG_L_R2C8_BR_TM', 'EMG_L_R3C1_BR_TM', 'EMG_L_R3C2_BR_TM',
    'EMG_L_R3C3_BR_TM', 'EMG_L_R3C4_BR_TM', 'EMG_L_R3C5_BR_TM',
    'EMG_L_R3C6_BR_TM', 'EMG_L_R3C7_BR_TM', 'EMG_L_R3C8_BR_TM',
    'EMG_L_R4C1_BR_TM', 'EMG_L_R4C2_BR_TM', 'EMG_L_R4C3_BR_TM',
    'EMG_L_R4C4_BR_TM', 'EMG_L_R4C5_BR_TM', 'EMG_L_R4C6_BR_TM',
    'EMG_L_R4C7_BR_TM', 'EMG_L_R4C8_BR_TM',

    # Amplifier recording
    'Impedance_1kOhm', 'Impedance_1.5kOhm',
    'Impedance_2.2kOhm', 'Impedance_2.7kOhm',
    'Impedance_3.3kOhm', 'Impedance_3.9kOhm',
    'Impedance_4.7kOhm', 'Impedance_5.6kOhm'
}

DIRECTIONAL_LEADS_AVERAGED = [
    "sub-EL002_ses-EcogLfpMedOff02_task-Rest_acq-StimOff_run-3",
    "sub-EL005_ses-EcogLfpMedOff02_task-Rest_acq-StimOff_run-1",
    "sub-EL005_ses-EcogLfpMedOn01_task-Rest_acq-StimOff_run-1",
    "sub-EL005_ses-EcogLfpMedOn02_task-Rest_acq-StimOff_run-1",
    "sub-EL007_ses-EcogLfpMedOn01_task-Rest_acq-StimOff_run-1",
    "sub-L012_ses-LfpMedOn01_task-Rest_acq-StimOff_run-1",
]

SUB_PREFIX_NEU = "Neu"
# not very elegant because hard coded and assumes no more than 5 sessions
NEUMANN_SESSIONS = ([f"EcogLfpMedOff0{i}" for i in range(1, 6)] +
                    [f"EcogLfpMedOn0{i}" for i in range(1, 6)] +
                    [f"LfpMedOff0{i}" for i in range(1, 6)] +
                    [f"LfpMedOn0{i}" for i in range(1, 6)])

DBS_LEAD_MAP_NEU = {'EL001': 2,
                    'EL002': 2,
                    'EL003': 3,
                    'EL004': 2,
                    'EL005': 1,
                    'EL006': 1,
                    'EL007': 1,
                    'EL008': 4,
                    'EL009': 1,
                    'EL010': 1,
                    'EL011': 1,
                    'EL012': 1,
                    'EL013': 1,
                    'EL014': 1,
                    'EL015': 1,
                    'EL016': 1,
                    'EL017': 1,
                    'EL018': 1,
                    'EL019': 1,
                    'EL020': 1,
                    'EL021': 1,
                    'EL022': 1,
                    'EL023': 1,
                    'EL024': 1,
                    'EL025': 1,
                    'EL026': 1,
                    'EL027': 1,
                    'EL028': 1,

                    'EL029': 1,
                    'EL030': 1,
                    'EL031': 1,


                    'L001': 1,
                    'L002': 1,
                    'L003': 1,
                    'L004': 1,
                    'L005': 1,
                    'L006': 1,
                    'L007': 1,
                    'L008': 1,
                    'L009': 1,
                    'L010': 1,
                    'L011': 1,
                    'L012': 1,
                    'L013': 1,
                    'L014': 1,
                    'L015': 1,
                    'L016': 1,
                    'L017': 1,
                    'L018': 1,
                    'L019': 1,

                    'L020': 1,
                    'L022': 1,
                    'L024': 1,
                    'L025': 1,
                    'L026': 1,
                    'Emptyroom': 0}


NEUMANN_CHNAME_MAP_EMPTY = {'Ch1': 'Amp_R_1',
                            'Ch2': 'Amp_R_2a',
                            'Ch3': 'Amp_R_3a',
                            'Ch4': 'Amp_R_2b',
                            'Ch5': 'Amp_R_4',
                            'Ch6': 'Amp_R_3b',
                            'Ch7': 'Amp_R_2c',
                            'Ch8': 'Amp_R_3c',

                            'Ch9': 'Amp_L_1',
                            'Ch10': 'Amp_L_2a',
                            'Ch11': 'Amp_L_2b',
                            'Ch12': 'Amp_L_2c',
                            'Ch13': 'Amp_L_3a',
                            'Ch14': 'Amp_L_3b',
                            'Ch15': 'Amp_L_3c',
                            'Ch16': 'Amp_L_4',
                            }