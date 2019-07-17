"""
Simple styling used for matplotlib figures
"""

from matplotlib import pyplot as plt

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 17
plt.rcParams['lines.linewidth'] = 3

def light_axis():
    "Hide the top and right spines"
    ax = plt.gca()
    for s in ('top', 'right'):
        ax.spines['top'].set_visible(False)
