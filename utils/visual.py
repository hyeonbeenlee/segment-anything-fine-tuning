import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import os

def plot_template(fontsize=15):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes3d.grid'] = True
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.labelsize'] = fontsize + 2
    plt.rcParams['axes.titlesize'] = fontsize + 4
    plt.rcParams['xtick.labelsize'] = fontsize - 4
    plt.rcParams['ytick.labelsize'] = fontsize - 4


def increase_leglw(leg, linewidth: float = 3):
    for legobj in leg.legendHandles:
        legobj.set_linewidth(linewidth)


def add_textbox(ax, string, loc: int = 3, fontsize: int = 12):
    artist = AnchoredText(string, loc=loc, prop={'fontsize': fontsize})
    ax.add_artist(artist)


def save_allfigs(Prefix: str = "Fig", subFolder: str = None):
    path = "figures"
    if subFolder:
        path = f"{path}/{subFolder}"
    try:
        os.makedirs(path)
    except:
        pass

    for fignum in plt.get_fignums():
        plt.figure(fignum)
        plt.savefig(f"{path}/{Prefix}_{fignum:02d}.png",
                    dpi=400)
        print(f"figures/{Prefix}_{fignum:02d}.png Saved.")
        plt.close(plt.figure(fignum))