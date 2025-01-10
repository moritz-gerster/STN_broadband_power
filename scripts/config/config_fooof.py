"""FOOOF Parameters."""
# Params
broad_fixed = {"peak_width_limits": (2, 12),
               "max_n_peaks": 4,
               "min_peak_height": 0.1,
               "aperiodic_mode": "fixed",
               "verbose": False}
broad = {"standard": ([2, 60], broad_fixed)}

# Exclusion criteria
broad_error = {'r_squared_threshold': 0.85}
FIT_ERROR_THRESHOLDS = dict(broad=broad_error)

# All
FOOOF_DICT = dict(Neumann=dict(broad=broad),
                  Florin=dict(broad=broad),
                  Litvak=dict(broad=broad),
                  Hirschmann=dict(broad=broad),
                  Tan=dict(broad=broad))
