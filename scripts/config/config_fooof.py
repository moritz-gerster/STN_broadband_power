"""FOOOF Parameters."""

# %% Finely Tuned Gamma
FTG = {"peak_width_limits": (1, 4),
       "max_n_peaks": 1,
       "min_peak_height": 0.075,
       "peak_threshold": 3.5,
       "aperiodic_mode": "fixed",
       "verbose": False}

FTG = {"standard": ([58, 100], FTG)}

# %% Narrow
# narrow = {"peak_width_limits": (1, 4),
#           "max_n_peaks": 1,
#           "min_peak_height": 0.075,
#           "peak_threshold": 3.5,
#           "aperiodic_mode": "fixed",
#           "verbose": False}

# narrow = {"standard": ([55, 75], narrow)}

# %% HFOs

HFO_params_flo = {"peak_width_limits": (30, 120),
                  "peak_threshold": 2.6,
                  "max_n_peaks": 2,
                  "min_peak_height": 0.075,
                  'aperiodic_mode': 'fixed',
                  "verbose": False}
HFO_flo = ([105, 550], HFO_params_flo)
HFO_florin = {"standard": HFO_flo}
# for these Florin parameters,
# peaks below 125 and above 400 Hz need to be excluded

HFO_params_tan = {"peak_width_limits": (30, 80),
                  "peak_threshold": 2.8,
                  "max_n_peaks": 2,
                  "min_peak_height": 0.09,
                  'aperiodic_mode': 'knee',
                  "verbose": False}
HFO_tan_range = ([125, 600], HFO_params_tan)
HFO_tan = {"standard": HFO_tan_range}
# for these Tan parameters,
# peaks below 200 and above 420 Hz need to be excluded

HFO_params_neu = {"peak_width_limits": (25, 100),
                  "peak_threshold": 2.8,
                  "max_n_peaks": 2,
                  "min_peak_height": 0,  # 0 and .065 almost same. 0.7 bad
                  'aperiodic_mode': 'knee',
                  "verbose": False}
HFO_neu = ([105, 995], HFO_params_neu)  # lower range 90 and 105 same good
HFO_neumann = {"standard": HFO_neu}
# for these Neumann parameters,
# peaks below 160 and above 500 Hz need to be excluded

HFO_params_lit = {"peak_width_limits": (30, 120),
                  "peak_threshold": 2.6,
                  "max_n_peaks": 2,
                  # min_peak_height 0.085 and 0 almost same. However, 0.085
                  # causes some false negative and keeps many false positives
                  "min_peak_height": 0,
                  'aperiodic_mode': 'fixed',
                  "verbose": False}

HFO_lit = ([105, 600], HFO_params_lit)
HFO_litvak = {"standard": HFO_lit}
# for these litvak parameters,
# peaks below 230 and above 420 Hz need to be excluded


# %% Broad params

broad_fixed = {"peak_width_limits": (2, 12),
               "max_n_peaks": 4,
               "min_peak_height": 0.1,
               "aperiodic_mode": "fixed",
               "verbose": False}

broad = {"standard": ([2, 60], broad_fixed)}

# %% Exclusion criteria

broad_error = {'r_squared_threshold': 0.85}
FIT_ERROR_THRESHOLDS = dict(broad=broad_error)

FIT_OSCILLATION_EXCLUSION = dict(
                        Neumann=(160, 500),
                        Florin=(125, 400),
                        Hirschmann=(125, 400),  # Check again
                        Litvak=(230, 420),
                        Tan=(200, 420)
                                )

# %% All params

neumann_fooof = dict(
        # gamma=FTG,
        # narrow=narrow,
        broad=broad,
        # HFO=HFO_neumann,
        )
florin_fooof = dict(
        # gamma=FTG,
        # narrow=narrow,
        broad=broad,
        # HFO=HFO_florin,
                    )
litvak_fooof = dict(
        # gamma=FTG,
        # narrow=narrow,
        broad=broad,
        # HFO=HFO_litvak,
        )
hirschmann_fooof = dict(
        # gamma=FTG,
        # narrow=narrow,
        broad=broad,
        # HFO=HFO_florin,
        )
tan_fooof = dict(
        # gamma=FTG,
        # narrow=narrow,
        broad=broad,
        # HFO=HFO_tan,
        )

FOOOF_DICT = dict(Neumann=neumann_fooof,
                  Florin=florin_fooof,
                  Litvak=litvak_fooof,
                  Hirschmann=hirschmann_fooof,
                  Tan=tan_fooof)
