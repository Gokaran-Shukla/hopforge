"""Plotting for transport systems: geometry with leads, site maps, currents."""
from __future__ import annotations

import numpy as np


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_system(sys, ax=None, lead_cells=2, s=None, bonds=True, title=None):
    """Draw the scattering region (grey) and the first cells of every lead."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    P = sys.positions()
    m = sys.model
    if bonds:
        H = sys.hamiltonian().tocoo()
        ns = sys.ns
        a, b = H.row // ns, H.col // ns
        keep = a < b
        segs = np.stack([P[a[keep], :2], P[b[keep], :2]], 1)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segs, colors="0.75", linewidths=0.6, zorder=1))
    size = s or max(4, min(60, 4000 / max(1, len(P))))
    ax.scatter(P[:, 0], P[:, 1], s=size, c="0.35", zorder=2, label="scattering region")
    A = m.lat.vectors
    for n, L in enumerate(sys.leads):
        c = f"C{n % 10}"
        shift = np.array(L.vector) @ A
        Q = np.array([L.model.lat.positions[o] + np.array(R) @ A for o, R in L.sites0])
        for j in range(lead_cells):
            QQ = Q + j * shift
            ax.scatter(QQ[:, 0], QQ[:, 1], s=size, c=c, alpha=1 - 0.35 * j, zorder=2,
                       label=f"lead {n}" if j == 0 else None)
        cen = Q[:, :2].mean(0) + (lead_cells - 0.5) * shift[:2]
        ax.annotate("", xy=cen + shift[:2] * 1.5, xytext=cen,
                    arrowprops=dict(arrowstyle="->", color=c, lw=2))
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.set_title(title or f"{sys.nsites} sites, {len(sys.leads)} leads")
    return ax


def plot_map(sys, values, ax=None, cmap="viridis", s=None, title="", colorbar=True, vmin=None,
             vmax=None):
    """Colour every site by a value (e.g. LDOS or density from a lead)."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    P = sys.positions()
    v = np.asarray(values)
    if v.size == sys.nbasis and sys.ns > 1:
        v = v.reshape(-1, sys.ns).sum(1)
    size = s or max(4, min(80, 5000 / max(1, len(P))))
    sc = ax.scatter(P[:, 0], P[:, 1], c=v, s=size, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(sc, ax=ax)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def plot_current(sys, bonds, J, ax=None, cmap="inferno", title="current", density=None,
                 arrows=True, s=None):
    """Bond currents as coloured lines plus site-averaged current arrows."""
    plt = _plt()
    from matplotlib.collections import LineCollection
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    P = sys.positions()[:, :2]
    a, b = bonds[:, 0], bonds[:, 1]
    segs = np.stack([P[a], P[b]], 1)
    w = np.abs(J) / (np.abs(J).max() or 1)
    lc = LineCollection(segs, array=np.abs(J), cmap=cmap, linewidths=0.3 + 3.5 * w)
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label="|J|")
    if arrows:
        vec = np.zeros_like(P)
        d = P[b] - P[a]
        dn = d / np.linalg.norm(d, axis=1)[:, None]
        np.add.at(vec, a, 0.5 * J[:, None] * dn)
        np.add.at(vec, b, 0.5 * J[:, None] * dn)
        mag = np.linalg.norm(vec, axis=1)
        sel = mag > 0.05 * mag.max()
        ax.quiver(P[sel, 0], P[sel, 1], vec[sel, 0], vec[sel, 1], color="c", scale=None,
                  width=0.003, zorder=3)
    ax.set_xlim(P[:, 0].min() - 1, P[:, 0].max() + 1)
    ax.set_ylim(P[:, 1].min() - 1, P[:, 1].max() + 1)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax
