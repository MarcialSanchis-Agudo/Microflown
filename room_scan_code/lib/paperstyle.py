"""
paperstyle.py — shared figure style for the paper (the conventions of Figs 4-5).

White background, serif text, plasma for magnitudes, a diverging map for signed
quantities, colourbars matched to the axes they describe with the label as a
title, panel tags in the upper-left corner, and explicit axes placement so
neighbouring labels never collide.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'savefig.dpi': 300, 'figure.dpi': 130,
})

# palette
BLUE, RED, ORANGE, GREEN, GREY = '#1f5fbf', '#d42b2b', '#d9531e', '#2e8b57', '0.72'
SEQ = 'plasma'          # magnitudes
DIV = 'RdBu_r'          # signed quantities (blue < 0 < red)
TEXTW = 7.2             # full text width [in]


def tag(ax, t, x=0.03, y=0.97):
    ax.text(x, y, f'({t})', transform=ax.transAxes, va='top', ha='left', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.2), zorder=20)


def cbar(fig, mappable, cax, title, nticks=4, ticks=None):
    """Colourbar in a pre-placed axis (so it matches its map), label as a title."""
    cb = fig.colorbar(mappable, cax=cax)
    cb.ax.set_title(title, fontsize=8, pad=4)
    if ticks is not None:
        cb.set_ticks(ticks)
    else:
        cb.locator = mticker.MaxNLocator(nticks); cb.update_ticks()
    cb.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)
    return cb


def source(ax, x=0.0, y=0.0, s=110):
    ax.plot(x, y, marker='*', ms=np.sqrt(s), mfc='white', mec='k', mew=0.9, zorder=15, ls='none')


def box(ax, x0, y0, w, h, **kw):
    kw = {'facecolor': '0.85', 'edgecolor': 'k', 'lw': 0.9, 'zorder': 10, **kw}
    ax.add_patch(Rectangle((x0, y0), w, h, **kw))


class Row:
    """
    Explicit single-row layout in inches.  add_map() places an equal-aspect map of
    a given data extent at a given width; add_cbar() reserves a colourbar strip of
    the same height; add_plot() places an ordinary axis.  Gaps are explicit.
    """
    def __init__(self, W=TEXTW, H=2.4, left=0.50, bottom=0.42):
        self.W, self.H, self.x, self.y0 = W, H, left, bottom
        self.fig = plt.figure(figsize=(W, H))
        self.h = None

    def _ax(self, w, h):
        a = self.fig.add_axes([self.x / self.W, self.y0 / self.H, w / self.W, h / self.H])
        self.x += w
        return a

    def gap(self, g):
        self.x += g

    def add_map(self, xlim, ylim, width):
        h = width * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
        self.h = h if self.h is None else self.h
        ax = self._ax(width, h)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        return ax

    def add_cbar(self, width=0.09, pad=0.06):
        self.gap(pad)
        return self._ax(width, self.h)

    def add_plot(self, width, height=None):
        return self._ax(width, height if height is not None else self.h)

    def save(self, path, preview=None):
        self.fig.savefig(path)
        if preview:
            self.fig.savefig(preview, dpi=130)
        plt.close(self.fig)
