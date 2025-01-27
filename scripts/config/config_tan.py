from os.path import join
from scripts.config import SOURCEDATA

SOURCEDATA_TAN = join(SOURCEDATA, 'BIDS_Tan_EEG_LFP')

#%% Subject Info
subjects_old = [
                # LDOPA subjects
                'G10', 'G23', 'G24', 'G25', 'G27', 'G28', 'G30', 'G31', 'G32',
                'G33', 'G34', 'K6', 'K7', 'K8', 'K11', 'XG37', 'XG39',
                # exclude DBS only subjects
                # 'G1', 'G22', 'G4', 'G5', 'K4', 'M5', 'M6'
]
subjects_new = [f"TanL{sub:03d}" for sub in range(1, len(subjects_old) + 1)]
TAN_SUBJECT_MAP = dict(zip(subjects_old, subjects_new))
TAN_SUBJECT_MAP_REV = dict(zip(subjects_new, subjects_old))

DBS_LEAD_MAP_TAN = {'G27': 8, 'G28': 2, 'G30': 4, 'G31': 1, 'G32': 4,
                    'G34': 1, 'K11': 1, 'K8': 3, 'XG37': 1, 'XG39': 1,
                    'K13': 1, 'K7': 3, 'K9': 5, 'K14': 1, 'G10': 2, 'G23': 2,
                    'G24': 2, 'G13': 2, 'G1': 2, 'G22': 2, 'G4': 2, 'G5': 2,
                    'G6': 3, 'K4': 3, 'M5': 3, 'M6': 5, 'K10': 5, 'G25': 0,
                    'G33': 0, 'K6': 0}

DBS_MATRIX_INFO = ['G10LOFF',
                   'G10LON',
                   'G10ROFF',
                   'G1LOFF',
                   'G22LOFF',
                   'G23ROFF',
                   'G24LOFF',
                   'G24LON',
                   'G24ROFF',
                   'G24RON',
                   'G27RON',
                   'G28LON',
                   'G28RON',
                   'G30LON',
                   'G30RON',
                   'G31LON',
                   'G31RON',
                   'G32LON',
                   'G32RON',
                   'G4ROFF',
                   'G5ROFF',
                   'K4LOFF',
                   'M5LOFF',
                   'M6ROFF',
                   'XG37LON',
                   'XG39RON']

#%% Channel info
# This is a huge mess. Plenty of different channel names. Sometimes, DBS ch
# index starts at 0, sometimes at 1.
# Strategy: Drop all misc and eeg channels, keep DBS channels. If one DBS
# channel starts with "0" use CH_NME_MAP_TAN_IDX0 to rename channels. If one
# channels ends with "4" use CH_NME_MAP_TAN_IDX1 to rename channels. If one
# channel starts with 'Virtual' use CH_NME_MAP_TAN_FNAME to rename channels
# which have been figured out. These files remain ambiguous:
# ['G27_leSTN_ON.mat', 'G27_riSTN_ON.mat',
# 'G31_leSTN_ON.mat', 'G31_riSTN_ON.mat', 'G32_riSTN_ON.mat',
# 'XG39_ERNA_riSTN_ON.mat']. Here, I simply assume that the index starts at 1
# since it does not really matter for the analysis and since for the same
# subjects the index started at 1 for the OFF condition. Wiest picks are
# called 'LFP_lr_WIEST_STN_MT' because only in that case the channels can be
# chosen for both conditions even if they change.

#%% Channel types
acceloremter_chs = ['AcX', 'AcY', 'AcZ',
                    'Ac1X', 'Ac1Y', 'Ac1Z', 'Ac2X', 'Ac2Y', 'Ac2Z',
                    'AcXF', 'AcYF', 'AcZF', 'AcXL', 'AcYL', 'AcZL', 'AcXH',
                    'AcYH', 'AcZH', 'AcrX', 'AcrY', 'AcrZ', 'Acrx', 'Acry',
                    'Acrz', 'Aclx', 'Acly', 'Aclz',
                    'Accx', 'Accy', 'Accz', 'AccX', 'AccY', 'AccZ',
                    'AcXR', 'AcYR', 'AcZR', 'S-AcZH', 'F-AcZH',
                    'Ax1', 'Ax2', 'AxL', 'AxR'  # <- also accelerometer?
                    ]

stimulation_intensity = ['Am1', 'Am2', 'Amp1', 'Amp2']

processed_chs = ['Bet1', 'Bet2', 'Bet3']

filtered_chs = ['F-C3', 'F-CREF', 'F-Cz', 'F-CPz', 'F-Cz-0', 'F-Cz-1',
                'F-CPz-0', 'F-CPz-1', 'F-CREF-0', 'F-CREF-1', 'F_R02',
                'F-R1', 'F-R2', 'F-R3', 'F-Rv1', 'F-Rv3', 'F-R13', 'F_L02',
                'F_L13', 'F_Le02', 'F-L0', 'F-L1', 'F-L2', 'F-L123', 'F-L13',
                'F-L789', 'F-R13-0', 'F-R13-1', 'F-L24-0', 'F-L24-1']

emg_ecg_chs = ['EMG1', 'EMG2', 'EMGR', 'ECG', 'LFfl',
               # second row also emg?
               'LFE', 'LFF', 'LFex', 'RFE', 'RFF', 'RFex', 'RFfl']

resistance_chs = ['Rs1', 'Rs2']  # plot to verify

# what are these S-channels?
s_chs = ['S-C3', 'S-CREF', 'S-Cz', 'S-CPz',  'S-Cz-0', 'S-Cz-1',  'S-CPz-0',
         'S-CPz-1', 'S-CREF-0', 'S-CREF-1', 'S-L0', 'S-L02', 'S-L13',
         'S-L123', 'S-L789', 'S-L2', 'S-L0', 'S-L1', 'S_L02', 'S_L13',
         'S_Le02', 'S_R02', 'S-R13-0', 'S-R13-1', 'S-R1', 'S-R2', 'S-R3',
         'S-R13', 'S-Rv1', 'S-Rv3', 'S-L24-0', 'S-L24-1']

# what are these?
misc_chs = ['Ch1', 'Ch2', 'Code', 'Snd',
            'FcLe', 'FcRi', 'Ph', 'Pho']

misc_chs = (acceloremter_chs + stimulation_intensity + processed_chs
            + filtered_chs + emg_ecg_chs + resistance_chs + s_chs + misc_chs)

monopolar_dbs_chs = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8',
                     'R9', 'R10', 'R11', 'R12',
                      #      'R13',
                     'R14', 'R15', 'R16',
                     'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8',
                     'L9', 'L10', 'L11', 'L12',
                      #      'L13',
                     'L14', 'L15', 'L16',]

dc_removed_dbs_chs = ['R02-DcRem', 'R02_DcRem']

dbs_levels = ['L123', 'L789', 'Lv4', 'Lv5', 'Lv6', 'Lv7', 'Lv8',
              'Rv1', 'Rv3', 'Rv4', 'Rv5', 'Rv6', 'Rv7', 'Rv8']

bipolar_dbs_chs = [
           'R02', 'R13', 'R23', 'R24', 'Rv13',
           'L01', 'L02', 'L13', 'L24', 'Le02', 'Lv13',
           'Virtual', 'Virtual-0', 'Virtual-1', 'Virtual-2', 'Virtual-3']

bipolar_dbs_chs_neighbors = ['R23', 'L01']

dbs_chs = monopolar_dbs_chs + dc_removed_dbs_chs + dbs_levels + bipolar_dbs_chs

eeg_chs = ['C3', 'C4', 'Cz', 'CP3', 'CP4', 'CPz', 'CREF', 'F3', 'F4', 'FC3',
           'FC4', 'FCz', 'Fz', 'O1', 'O2', 'Oz', 'P3', 'P4', 'Pz']

#%% Channel type dictionary
dbs_chs_dic = {ch: "ecog" for ch in dbs_chs}  # only set his choice as dbs
misc_chs_dic = {ch: "misc" for ch in misc_chs}
eeg_chs_dic = {ch: "eeg" for ch in eeg_chs}

CH_TYPE_MAP_TAN = {**dbs_chs_dic, **misc_chs_dic, **eeg_chs_dic}

#%% Channel names
DROP_CHANNELS = (bipolar_dbs_chs_neighbors + dbs_levels + monopolar_dbs_chs
                 + dc_removed_dbs_chs)

Wiest_dic = {'LFP_R_WIEST_STN_MT': 'LFP_R_WIEST_STN_MT',
             'LFP_L_WIEST_STN_MT': 'LFP_L_WIEST_STN_MT'}

CH_NME_MAP_TAN_IDX0 = {
        'R02': 'LFP_R_1-3_STN_MT',
        'R13': 'LFP_R_2-4_STN_MT',
        'Rv13': 'LFP_R_2-4_STN_MT',

        'L02': 'LFP_L_1-3_STN_MT',
        'Le02': 'LFP_L_1-3_STN_MT',
        'L13': 'LFP_L_2-4_STN_MT',
        'Lv13': 'LFP_L_2-4_STN_MT',
        **Wiest_dic}

CH_NME_MAP_TAN_IDX1 = {
        'R13': 'LFP_R_1-3_STN_MT',
        'Rv13': 'LFP_R_1-3_STN_MT',
        'R24': 'LFP_R_2-4_STN_MT',

        'L13': 'LFP_L_1-3_STN_MT',
        'Lv13': 'LFP_L_1-3_STN_MT',
        'L24': 'LFP_L_2-4_STN_MT',
        **Wiest_dic}

virtual_dic = {'Virtual-0': 'LFP_L_1-3_STN_MT',
               'Virtual-1': 'LFP_L_2-4_STN_MT',
               'Virtual-2': 'LFP_R_1-3_STN_MT',
               'Virtual-3': 'LFP_R_2-4_STN_MT',
               **Wiest_dic}

dic_13 = {'R13': 'LFP_R_1-3_STN_MT',
          'Rv13': 'LFP_R_1-3_STN_MT',
          'Lv13': 'LFP_L_1-3_STN_MT',
          'L13': 'LFP_L_1-3_STN_MT',
          **Wiest_dic}

CH_NME_MAP_TAN_FNAME = {
                # guessed:
                'G27_leSTN_ON.mat': dic_13,
                'G27_riSTN_ON.mat': dic_13,
                'G31_leSTN_ON.mat': dic_13,
                'G31_riSTN_ON.mat': dic_13,
                'G32_riSTN_ON.mat': dic_13,
                'XG39_ERNA_riSTN_ON.mat': dic_13,
                # checked:
                'G33_leSTN_ON.mat': virtual_dic,
                'G33_riSTN_ON.mat': virtual_dic,
                'G34_leSTN_ON.mat': virtual_dic,
                'G34_riSTN_ON.mat': virtual_dic,
                'K6_leSTN_ON.mat': virtual_dic,
                'K6_riSTN_ON.mat': virtual_dic,
                'G33_leSTN_OFF.mat': virtual_dic,
                'G33_riSTN_OFF.mat': virtual_dic,
                'G34_leSTN_OFF.mat': virtual_dic,
                'G34_riSTN_OFF.mat': virtual_dic,
                'K6_leSTN_OFF.mat': virtual_dic,
                'K6_riSTN_OFF.mat': virtual_dic}

TAN_CHNAME_MAP_PORTI = {'Ch1': 'Amp_R_1',
                        'Ch2': 'Amp_R_2',
                        'Ch3': 'Amp_R_3',
                        'Ch4': 'Amp_R_4',

                        'Ch5': 'Amp_L_1',
                        'Ch6': 'Amp_L_2',
                        'Ch7': 'Amp_L_3',
                        'Ch8': 'Amp_L_4',
                        }

TAN_CHNAME_MAP_SAGA = {'UN01': 'Amp_R_1',
                       'UN02': 'Amp_R_2',
                       'UN03': 'Amp_R_3',
                       'UN04': 'Amp_R_4',

                       'UN05': 'Amp_L_1',
                       'UN06': 'Amp_L_2',
                       'UN07': 'Amp_L_3',
                       'UN08': 'Amp_L_4',
                       }


#%% Channel picks and times selected in Wiest et al.
CH_TIME_SELECTION_WIEST = {
    'G10_leSTN_ON.mat': {'idx': 9, 'ch_nme': 'Le02', 'time': (1, 61)},
    'G10_riSTN_ON.mat': {'idx': 10, 'ch_nme': 'R02', 'time': (1, 61)},
    'G23_leSTN_ON.mat': {'idx': 10, 'ch_nme': 'L13', 'time': (70, 130)},
    'G23_riSTN_ON.mat': {'idx': 11, 'ch_nme': 'R02', 'time': (1460, 1520)},
    'G24_leSTN_ON.mat': {'idx': 9, 'ch_nme': 'L02', 'time': (1, 61)},
    'G24_riSTN_ON.mat': {'idx': 11, 'ch_nme': 'R02', 'time': (1, 61)},
    'G25_leSTN_ON.mat': {'idx': 1, 'ch_nme': 'L24', 'time': (1, 61)},
    'G25_riSTN_ON.mat': {'idx': 1, 'ch_nme': 'R24', 'time': (1, 61)},
    'G27_leSTN_ON.mat': {'idx': 30, 'ch_nme': 'Lv13', 'time': (277, 337)},
    'G27_riSTN_ON.mat': {'idx': 30, 'ch_nme': 'Rv13', 'time': (1, 61)},
    'G28_leSTN_ON.mat': {'idx': 14, 'ch_nme': 'L02', 'time': (1, 61)},
    'G28_riSTN_ON.mat': {'idx': 14, 'ch_nme': 'R24', 'time': (230, 290)},
    'G30_leSTN_ON.mat': {'idx': 18, 'ch_nme': 'L24', 'time': (1, 61)},
    'G30_riSTN_ON.mat': {'idx': 18, 'ch_nme': 'R13', 'time': (1, 61)},
    'G31_leSTN_ON.mat': {'idx': 15, 'ch_nme': 'L13', 'time': (1, 59)},
    'G31_riSTN_ON.mat': {'idx': 15, 'ch_nme': 'R13', 'time': (1, 61)},
    'G32_leSTN_ON.mat': {'idx': 23, 'ch_nme': 'L13', 'time': (1, 61)},
    'G32_riSTN_ON.mat': {'idx': 23, 'ch_nme': 'R13', 'time': (1, 61)},
    'G33_leSTN_ON.mat': {'idx': 25, 'ch_nme': 'Virtual', 'time': (1, 61)},
    'G33_riSTN_ON.mat': {'idx': 27, 'ch_nme': 'Virtual', 'time': (1, 61)},
    'G34_leSTN_ON.mat': {'idx': 24, 'ch_nme': 'Virtual', 'time': (200, 260)},
    'G34_riSTN_ON.mat': {'idx': 26, 'ch_nme': 'Virtual', 'time': (200, 260)},
    'K11_riSTN_ON.mat': {'idx': 8, 'ch_nme': 'R13', 'time': (1, 61)},
    'K6_leSTN_ON.mat': {'idx': 16, 'ch_nme': 'Virtual', 'time': (1, 61)},
    'K6_riSTN_ON.mat': {'idx': 19, 'ch_nme': 'Virtual', 'time': (1, 61)},
    'K7_leSTN_ON.mat': {'idx': 8, 'ch_nme': 'L13', 'time': (1, 61)},
    'K8_leSTN_ON.mat': {'idx': 14, 'ch_nme': 'L02', 'time': (1, 61)},
    'K8_riSTN_ON.mat': {'idx': 14, 'ch_nme': 'R02', 'time': (1, 61)},
    'XG37_ERNA_leSTN_ON.mat': {'idx': 20, 'ch_nme': 'L24', 'time': (1, 61)},
    'XG39_ERNA_riSTN_ON.mat': {'idx': 18, 'ch_nme': 'R13', 'time': (1, 61)},

    'G10_leSTN_OFF.mat': {'idx': 9, 'ch_nme': 'Le02', 'time': (760, 820)},
    'G10_riSTN_OFF.mat': {'idx': 11, 'ch_nme': 'R02', 'time': (2360, 2420)},
    'G23_leSTN_OFF.mat': {'idx': 10, 'ch_nme': 'L13', 'time': (1, 61)},
    'G23_riSTN_OFF.mat': {'idx': 11, 'ch_nme': 'R02', 'time': (1550, 1610)},
    'G24_leSTN_OFF.mat': {'idx': 9, 'ch_nme': 'L02', 'time': (1, 61)},
    'G24_riSTN_OFF.mat': {'idx': 11, 'ch_nme': 'R02', 'time': (1, 61)},
    'G25_leSTN_OFF.mat': {'idx': 1, 'ch_nme': 'L24', 'time': (1, 61)},
    'G25_riSTN_OFF.mat': {'idx': 1, 'ch_nme': 'R24', 'time': (1, 61)},
    'G27_leSTN_OFF.mat': {'idx': 1, 'ch_nme': 'L13',
                          'time': (1, 61)},  # this time includes DBS
    'G27_riSTN_OFF.mat': {'idx': 1, 'ch_nme': 'R24', 'time': (1, 61)},
    'G28_leSTN_OFF.mat': {'idx': 7, 'ch_nme': 'L02', 'time': (1, 61)},
    'G28_riSTN_OFF.mat': {'idx': 10, 'ch_nme': 'R13', 'time': (1, 61)},
    'G30_leSTN_OFF.mat': {'idx': 9, 'ch_nme': 'L02', 'time': (1, 61)},
    'G30_riSTN_OFF.mat': {'idx': 7, 'ch_nme': 'R02', 'time': (1, 61)},
    'G31_leSTN_OFF.mat': {'idx': 7, 'ch_nme': 'L02', 'time': (1, 61)},
    'G31_riSTN_OFF.mat': {'idx': 10, 'ch_nme': 'R13', 'time': (1, 61)},
    'G32_leSTN_OFF.mat': {'idx': 6, 'ch_nme': 'L13', 'time': (1, 61)},
    'G32_riSTN_OFF.mat': {'idx': 7, 'ch_nme': 'R02', 'time': (1, 61)},
    'G33_leSTN_OFF.mat': {'idx': 20, 'ch_nme': 'Virtual', 'time': (1, 61)},
    'G33_riSTN_OFF.mat': {'idx': 22, 'ch_nme': 'Virtual', 'time': (1, 61)},
    'G34_leSTN_OFF.mat': {'idx': 24, 'ch_nme': 'Virtual', 'time': (200, 260)},
    'G34_riSTN_OFF.mat': {'idx': 26, 'ch_nme': 'Virtual', 'time': (200, 260)},
    'K11_riSTN_OFF.mat': {'idx': 3, 'ch_nme': 'R02', 'time': (1, 61)},
    'K6_leSTN_OFF.mat': {'idx': 16, 'ch_nme': 'Virtual', 'time': (1, 61)},
    'K6_riSTN_OFF.mat': {'idx': 19, 'ch_nme': 'Virtual', 'time': (1, 61)},
    'K7_leSTN_OFF.mat': {'idx': 12, 'ch_nme': 'L13', 'time': (1, 61)},
    'K8_leSTN_OFF.mat': {'idx': 7, 'ch_nme': 'L02', 'time': (70, 130)},
    'K8_riSTN_OFF.mat': {'idx': 9, 'ch_nme': 'R02', 'time': (70, 130)},
    'XG37_leSTN_OFF.mat': {'idx': 9, 'ch_nme': 'L24', 'time': (1, 61)},
    'XG39_riSTN_OFF.mat': {'idx': 8, 'ch_nme': 'R13', 'time': (1, 61)}}

#%% Channel times selected by myself
TIME_SELECTION_OWN = {
                'G10_leSTN_ON.mat': {'time': (0, 796)},
                'G10_riSTN_ON.mat': {'time': (0, 796)},
                'G23_leSTN_ON.mat': {'time': (0, None)},
                'G23_riSTN_ON.mat': {'time': (0, 400)},
                'G24_leSTN_ON.mat': {'time': (0, 467)},
                'G24_riSTN_ON.mat': {'time': (0, 502)},
                'G25_leSTN_ON.mat': {'time': (0, None)},
                'G25_riSTN_ON.mat': {'time': (0, None)},
                'G27_leSTN_ON.mat': {'time': (0, None)},
                'G27_riSTN_ON.mat': {'time': (0, 628)},
                'G28_leSTN_ON.mat': {'time': (0, 616)},
                'G28_riSTN_ON.mat': {'time': (0, 592)},
                'G30_leSTN_ON.mat': {'time': (0, 318)},
                'G30_riSTN_ON.mat': {'time': (0, 330)},
                'G31_leSTN_ON.mat': {'time': (0, None)},
                'G31_riSTN_ON.mat': {'time': (0, 318)},
                'G32_leSTN_ON.mat': {'time': (0, 578)},
                'G32_riSTN_ON.mat': {'time': (0, 146)},
                'G33_leSTN_ON.mat': {'time': (0, None)},
                'G33_riSTN_ON.mat': {'time': (0, None)},
                'G34_leSTN_ON.mat': {'time': (0, None)},
                'G34_riSTN_ON.mat': {'time': (0, None)},
                'K11_riSTN_ON.mat': {'time': (0, 324)},
                'K6_leSTN_ON.mat': {'time': (0, None)},
                'K6_riSTN_ON.mat': {'time': (0, None)},
                'K7_leSTN_ON.mat': {'time': (0, 384)},
                'K8_leSTN_ON.mat': {'time': (0, None)},
                'K8_riSTN_ON.mat': {'time': (0, 328)},
                'XG37_ERNA_leSTN_ON.mat': {'time': (0, 284)},
                'XG39_ERNA_riSTN_ON.mat': {'time': (0, 376)},

                'G10_leSTN_OFF.mat': {'time': (0, 836)},
                'G10_riSTN_OFF.mat': {'time': (0, 808)},
                'G23_leSTN_OFF.mat': {'time': (0, None)},
                'G23_riSTN_OFF.mat': {'time': (0, 771)},
                'G24_leSTN_OFF.mat': {'time': (0, 469)},
                'G24_riSTN_OFF.mat': {'time': (0, 254)},
                'G25_leSTN_OFF.mat': {'time': (0, None)},
                'G25_riSTN_OFF.mat': {'time': (0, None)},
                'G27_leSTN_OFF.mat': {'time': (0, None)},
                'G27_riSTN_OFF.mat': {'time': (0, None)},
                'G28_leSTN_OFF.mat': {'time': (0, None)},
                'G28_riSTN_OFF.mat': {'time': (0, None)},
                'G30_leSTN_OFF.mat': {'time': (0, None)},
                'G30_riSTN_OFF.mat': {'time': (0, None)},
                'G31_leSTN_OFF.mat': {'time': (0, None)},
                'G31_riSTN_OFF.mat': {'time': (0, None)},
                'G32_leSTN_OFF.mat': {'time': (0, None)},
                'G32_riSTN_OFF.mat': {'time': (0, None)},
                'G33_leSTN_OFF.mat': {'time': (0, None)},
                'G33_riSTN_OFF.mat': {'time': (0, None)},
                'G34_leSTN_OFF.mat': {'time': (0, None)},
                'G34_riSTN_OFF.mat': {'time': (0, None)},
                'K11_riSTN_OFF.mat': {'time': (0, None)},
                'K6_leSTN_OFF.mat': {'time': (0, None)},
                'K6_riSTN_OFF.mat': {'time': (0, None)},
                'K7_leSTN_OFF.mat': {'time': (0, None)},
                'K8_leSTN_OFF.mat': {'time': (0, None)},
                'K8_riSTN_OFF.mat': {'time': (0, None)},
                'XG37_leSTN_OFF.mat': {'time': (0, None)},
                'XG39_riSTN_OFF.mat': {'time': (0, None)}
}

#%% BIDS meta data
TAN_META = dict(
        name="BIDS_Tan_EEG_LFP",
        dataset_type="raw",
        data_license="CC Attribution-ShareAlike (CC BY-SA)",
        authors=['Wiest, C.', 'Torrecillos, F.', 'Pogosyan, A.', 'Baig, F.',
                 'Pereira, E.', 'Morgante, F.', 'Ashkan, K.', 'Tan, H.'],
        how_to_acknowledge="n/a",
        references_and_links=['Wiest C, Torrecillos F, Pogosyan A, Bange M, Muthuraman M, Groppa S, et al. The aperiodic exponent of subthalamic field potentials reflects excitation/inhibition balance in Parkinsonism. Elife. 2023;12. doi:10.7554/eLife.82467'],
        acknowledgements=None,
        funding=None,
        doi='https://doi.org/10.5287/bodleian:mzJ7YwXvo')
