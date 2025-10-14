"""Band definitions."""
import seaborn as sns

BANDS = {
    "delta": (2, 4),
    "theta": (4, 9),
    "delta_theta": (2, 9),
    "alpha": (9, 13),
    "theta_alpha": (4, 13),
    "beta_low": (13, 20),
    "beta_high": (20, 30),
    "beta": (13, 30),
    "alpha_beta": (8, 35),
    "gamma_low": (30, 45),
    "gamma_mid": (45, 60),
    "gamma": (30, 60),
    "full": (2, 60),
    }

NOISE_FLOORS = {
    "Neumann": (95, 295),
    "Florin": (95, 595),
    "Litvak": (95, 595),
    "Hirschmann": (95, 495),
    "Tan": (95, 250)
    }

BAND_NAMES_GREEK = {
    'delta': r'$\delta$',
    'theta': r'$\theta$',
    'delta_theta': r'$\delta \theta$',
    'alpha': r'$\alpha$',
    'theta_alpha': r'$\theta \alpha$',
    'alpha_beta': r'$\alpha \beta$',
    'beta': r'$\beta$',
    'beta_low': r'Low $\beta$',
    'beta_high': r'High $\beta$',
    'gamma_low': r'Low $\gamma$',
    "gamma_mid": r'Mid $\gamma$',
    "gamma": r'$\gamma$ (30-60 Hz)',
    'full': '2-60 Hz',
    'fm_exponent': '1/f exponent',
    'fm_offset_log': 'Offset',
    'fm_offset': 'Offset',
    'full_fm_band_aperiodic_log': 'Aperiodic broadband power',
    'patient_age': 'Patient age'
}

BAND_NAMES_GREEK_SHORT = {
    'delta': r'$\delta$',
    'delta_theta': r'$\delta \theta$',
    'theta': r'$\theta$',
    'alpha': r'$\alpha$',
    'theta_alpha': r'$\theta \alpha$',
    'alpha_beta': r'$\alpha \beta$',
    'beta': r'$\beta$',
    'beta_low': r'L$\beta$',
    'beta_high': r'H$\beta$',
    'gamma_low': r'L$\gamma$',
    "gamma_mid": r'M$\gamma$',
    'gamma': r'$\gamma$',
    'full': '2-60 Hz',
    'fm_exponent': '1/f',
    'fm_offset_log': 'Offset',
    'fm_offset': 'Offset',
}

colors = sns.color_palette("Set2", n_colors=7)
c1, c4, c2, c5, c3, c6, c7 = colors

BAND_COLORS = {
    "delta": c1,
    'delta_theta': c2,
    'theta_alpha': c2,
    "theta": c2,
    "alpha": c3,
    "beta_low": c4,
    "beta_high": c5,
    "beta": c4,
    "alpha_beta": c3,
    "gamma_low": c6,
    "gamma_mid": c7,
    "gamma": c7,
    "FTG": c7,
    }

BAND_LOCALIZATION = [
    'alpha_beta',  # reproduce Shreve 2017
    'beta_low', 'beta_high', 'gamma_low', 'gamma_mid', 'gamma']
