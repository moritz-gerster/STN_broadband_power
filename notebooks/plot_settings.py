from os.path import join
import scripts.config as cfg
import matplotlib.pyplot as plt

# Text
FONTSIZE_S = 5
FONTSIZE_ASTERISK = 7
FONTSIZE_M = 6
FONTSIZE_L = 8
# plt.rc('font', family='Helvetica', size=FONTSIZE_S)  # bold Helvetica not working
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

# Paths
SAVE_DIR = join('..', cfg.FIG_PAPER)