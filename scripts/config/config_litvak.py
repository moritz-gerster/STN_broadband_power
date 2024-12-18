from os.path import join
from scripts.config import SOURCEDATA

SOURCEDATA_LIT = join(SOURCEDATA, 'BIDS_Litvak_MEG_LFP')

# dictionary mapping conditions to session BIDS names
LITVAK_SESSION_MAP = dict(zip(["on", "off"],
                              ["MegLfpMedOn01", "MegLfpMedOff01"]))
# dictioanry for renaming subjects
subjects_old = [f"subj{sub}" for sub in range(1, 15)]
SUB_PREFIX_LIT = "LitML"
subjects_new = [f"{SUB_PREFIX_LIT}{sub:03d}" for sub in range(1, 15)]
LITVAK_SUBJECT_MAP = dict(zip(subjects_old, subjects_new))
LITVAK_SUBJECT_MAP_INV = dict(zip(subjects_new, subjects_old))

# dictionary for renaming channels
old_ch_names = ["SMA", "leftM1", "rightM1",
                "STN_R01", "STN_R12", "STN_R23",
                "STN_L01", "STN_L12", "STN_L23",
                "EMG_R", "EMG_L", "HEOG", "VEOG", "event"]
new_ch_names = ["MEG_Z_SMA", "MEG_L_M1", "MEG_R_M1",
                "LFP_R_1-2_STN_MT", "LFP_R_2-3_STN_MT", "LFP_R_3-4_STN_MT",
                "LFP_L_1-2_STN_MT", "LFP_L_2-3_STN_MT", "LFP_L_3-4_STN_MT",
                "EMG_R", "EMG_L", "HEOG", "VEOG", "event"]
CH_NME_MAP_LITVAK = dict(zip(old_ch_names, new_ch_names))

# dictionary for resetting LFP channel types
new_ch_types = ["mag", "mag", "mag",
                "dbs", "dbs", "dbs", "dbs", "dbs", "dbs",
                "emg", "emg", "eog", "eog", "stim"]
LITVAK_CHTYPE_MAP = dict(zip(new_ch_names, new_ch_types))

LITVAK_META = dict(
        name="BIDS_Litvak_MEG_LFP",
        dataset_type="raw",
        data_license="n/a",
        authors=["Vladimir Litvak", "Moritz Gerster"],
        how_to_acknowledge="n/a",
        references_and_links=[
            "Litvak, Vladimir, Alexandre Eusebio, Ashwani Jha, Robert Oostenveld, Gareth R. Barnes, William D. Penny, Ludvic Zrinzo, et al. 2010. “Optimized Beamforming for Simultaneous MEG and Intracranial Local Field Potential Recordings in Deep Brain Stimulation Patients.” NeuroImage 50 (4): 1578–88.",
            "Litvak, Vladimir, Ashwani Jha, Alexandre Eusebio, Robert Oostenveld, Tom Foltynie, Patricia Limousin, Ludvic Zrinzo, Marwan I. Hariz, Karl Friston, and Peter Brown. 2011. “Resting Oscillatory Cortico-Subthalamic Connectivity in Patients with Parkinson’s Disease.” Brain: A Journal of Neurology 134 (Pt 2): 359–74."],
        acknowledgements=None,
        funding=None,
        doi=None
                )
