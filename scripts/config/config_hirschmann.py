from os.path import join
from scripts.config import SOURCEDATA

SOURCEDATA_HIR = join(SOURCEDATA, 'BIDS_Hirschmann_MEG_LFP')

subjects_old = ['0cGdk9', '2IhVOz', '2IU8mi', '6m9kB5', '8RgPiG', 'AB2PeX',
                'AbzsOg', 'BgojEx', 'BYJoWR', 'dCsWjQ', 'FIyfdR', 'FYbcap',
                'gNX5yb', 'hnetKS', 'i4oK0F', 'iDpl28', 'jyC0j3', 'oLNpHd',
                'PuPVlx', 'QZTsn6', 'VopvKx', 'zxEhes']
subjects_new = [f"HirML{sub + 1:03d}" for sub in range(len(subjects_old))]

SUB_MAPPING_HIR = dict(zip(subjects_old, subjects_new))
SUB_MAPPING_HIR_REV = dict(zip(subjects_new, subjects_old))

# 4 channels per hemisphere: Medtronic 3389 (3) or St. Jude Infinity (7)
# 8 channels per hemisphere: Boston Scientific The Vercise Standard Lead (6)
# Subs with 8 leads per hemisphere: 6m9kB5, 8RgPiG
# Sub with St. Jude Leads: oLNpHd
# Subs with Medtronic 3389: all others
DBS_LEAD_MAP_HIR = {'0cGdk9': 3,
                    '2IhVOz': 3,
                    '2IU8mi': 3,
                    'AB2PeX': 3,
                    'AbzsOg': 3,
                    'BYJoWR': 3,
                    'dCsWjQ': 3,
                    'FIyfdR': 3,
                    'FYbcap': 3,
                    'gNX5yb': 3,
                    'hnetKS': 3,
                    'i4oK0F': 3,
                    'iDpl28': 3,
                    'jyC0j3': 3,
                    'PuPVlx': 3,
                    'QZTsn6': 3,
                    'zxEhes': 3,
                    'VopvKx': 3,
                    'BgojEx': 3,

                    '6m9kB5': 6,
                    '8RgPiG': 6,

                    'oLNpHd': 7,
                    }

# dictionary mapping conditions to session BIDS names
conds = ["MedOn", "MedOff"]
sessions = ["MegLfpMedOn01", "MegLfpMedOff01"]
SESSION_MAPPING_HIR = dict(zip(conds, sessions))