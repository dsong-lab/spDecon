"""Plotting helpers shared by the six tutorials."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Wedge
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377',
          '#BBBBBB', '#332288', '#88CCAA', '#CC6677', '#882255', '#999933']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'svg.fonttype':'none', 'pdf.fonttype':42,
                     'figure.dpi':110, 'savefig.dpi':180})


def correlation(a, b):
    a, b = a-a.mean(0), b-b.mean(0)
    return a.T @ b / np.maximum(np.linalg.norm(a,axis=0)[:,None]*np.linalg.norm(b,axis=0),1e-15)


def align(p, truth):
    scores = correlation(p, truth)
    i, j = linear_sum_assignment(-scores)
    order = i[np.argsort(j)]
    return p[:,order], order


def spatial(ax, xy, values, title, *, categorical=False, size=6, vmax=1):
    if categorical:
        colors = [COLORS[int(i)%len(COLORS)] if i >= 0 else '#e5e5e5' for i in values]
        art = ax.scatter(*xy.T,c=colors,s=size,linewidths=0,rasterized=True)
    else:
        art = ax.scatter(*xy.T,c=values,s=size,cmap='viridis',vmin=0,vmax=vmax,linewidths=0,rasterized=True)
    ax.set(title=title,aspect='equal'); ax.invert_yaxis(); ax.axis('off')
    return art


def pies(ax, xy, p, title, radius):
    patches, colors = [], []
    for center, row in zip(xy,p):
        boundaries = np.r_[0,np.cumsum(row)]*360
        for j in range(len(row)):
            patches.append(Wedge(center,radius,boundaries[j],boundaries[j+1]))
            colors.append(COLORS[j%len(COLORS)])
    ax.add_collection(PatchCollection(patches,facecolors=colors,edgecolors='none',rasterized=True))
    ax.autoscale_view(); ax.set(title=title,aspect='equal'); ax.invert_yaxis(); ax.axis('off')


def legend(fig, labels, *, columns=4, y=0):
    fig.legend([Patch(color=COLORS[i%len(COLORS)]) for i in range(len(labels))],labels,
               loc='lower center',bbox_to_anchor=(.5,y),ncol=columns,frameon=False,fontsize=9)


def save_figure(fig, name):
    out = ROOT / 'outputs/figures'
    out.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg','pdf']:
        fig.savefig(out/f'{name}.{ext}',bbox_inches='tight')
    plt.show()


def save_table(table, name):
    out = ROOT / 'outputs/tables'
    out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/f'{name}.csv',index=False)
    return table
