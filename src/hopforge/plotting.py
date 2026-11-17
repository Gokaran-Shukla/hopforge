"""Matplotlib helpers: bands, DOS, lattices, Berry curvature maps, WCC flow."""
from __future__ import annotations

import numpy as np


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_bands(model, path=None, n=200, ax=None, color="k", weights=None, cmap="coolwarm",
               ylim=None, lw=1.2, title=None, fermi=None, **kw):
    """Plot the band structure. ``weights`` may be an operator (matrix): the
    bands are then coloured by its expectation value (e.g. spin_operator('z'))."""
    from .solvers import bands as _bands
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    b = _bands(model, path, n, vectors=weights is not None)
    x = b.x
    if weights is None:
        ax.plot(x, b.energies, color=color, lw=lw, **kw)
    else:
        w = b.weights(weights)
        vmax = np.abs(w).max() or 1
        for nb in range(b.energies.shape[1]):
            sc = ax.scatter(x, b.energies[:, nb], c=w[:, nb], s=4, cmap=cmap, vmin=-vmax, vmax=vmax)
        plt.colorbar(sc, ax=ax)
    for t in b.path.ticks:
        ax.axvline(t, color="0.7", lw=0.6)
    ax.set_xticks(b.path.ticks)
    ax.set_xticklabels(b.path.labels)
    ax.set_xlim(x[0], x[-1])
    if fermi is not None:
        ax.axhline(fermi, color="r", ls="--", lw=0.8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel("Energy")
    ax.set_title(title or model.name)
    return ax


def plot_dos(d, ax=None, label="total", **kw):
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    ax.plot(d.energies, d.dos, "k", label=label, **kw)
    for name, y in d.pdos.items():
        ax.plot(d.energies, y, label=name)
    ax.set_xlabel("Energy")
    ax.set_ylabel("DOS (states / cell / energy)")
    if d.pdos:
        ax.legend()
    return ax




SHELL_STYLES = [  # colour, linestyle, linewidth for 1st, 2nd, 3rd, ... neighbours
    ("#d62728", "-", 2.4), ("#1f77b4", "--", 1.7), ("#2ca02c", ":", 1.7),
    ("#9467bd", "-.", 1.3), ("#ff7f0e", (0, (5, 2, 1, 2)), 1.2), ("#8c564b", (0, (1, 3)), 1.1),
]
_ORD = {1: "1st", 2: "2nd", 3: "3rd"}


def _site_representatives(model):
    """One orbital per distinct atomic site (site label + position)."""
    reps, seen = [], {}
    for i in range(model.norb):
        key = (model.lat.site(i), tuple(np.round(model.lat.positions[i], 6)))
        if key not in seen:
            seen[key] = i
            reps.append(i)
    return reps


def neighbor_shells(model, shells=3, search=3):
    """Geometric neighbour shells between atomic sites.

    Returns a list (one entry per shell) of dicts with keys ``distance``,
    ``bonds`` [(i, j, R, d)], and ``coordination`` {site_orbital: count}.
    """
    reps = _site_representatives(model)
    bonds = [b for b in model._all_bonds(search, 1e-5) if b[0] in reps and b[1] in reps]
    dists = sorted({round(b[3], 5) for b in bonds})[:shells]
    out = []
    for dist in dists:
        sel = [(i, j, R, d) for i, j, R, dd, d in bonds if round(dd, 5) == dist]
        coord = {i: 0 for i in reps}
        for i, j, R, d in sel:
            coord[i] += 1
            coord[j] += 1
        out.append(dict(distance=float(dist), bonds=sel, coordination=coord))
    return out


def plot_lattice(model, shells=3, cells=None, ax=None, bonds="shells", annotate=True,
                 show_cell=True, highlight="auto", projection=None, s=None, legend=True,
                 title=None, circles=True):
    """Draw the crystal with its 1st, 2nd, 3rd, ... nearest-neighbour bonds.

    Parameters
    ----------
    shells : number of neighbour shells to draw (each with its own colour and
             line style; the legend gives distance and coordination number).
    cells  : how many unit cells to show in each periodic direction (+-cells).
    bonds  : "shells" - geometric neighbour shells (default);
             "model"  - the hoppings actually present in the model, coloured
             by the shell they belong to;  None - no bonds.
    highlight : orbital name/index: draw only the bonds of that site (home cell),
             a clean way to see its 1st/2nd/3rd neighbours (with distance
             circles in 2D). "auto" = off in 1D/2D, first site in 3D.
    circles : draw the neighbour-distance circles in highlight mode (2D).
    projection : None (auto), "xy" or "3d".
    """
    import itertools
    plt = _plt()
    A, P, D, dim = model.lat.vectors, model.lat.positions, model.lat.D, model.dim
    use3d = projection == "3d" or (projection is None and D == 3 and not _is_flat(P, A))
    if cells is None:
        cells = 1 if D == 3 else 2
    if isinstance(highlight, str) and highlight == "auto":
        highlight = 0 if (use3d and dim == 3) else None
    if ax is None:
        fig = plt.figure(figsize=(6.4, 6.0))
        ax = fig.add_subplot(111, projection="3d") if use3d else fig.add_subplot(111)

    def pt(v):
        v = np.concatenate([np.ravel(v), np.zeros(3)])[:3]
        return v if use3d else v[:2]

    rng = range(-cells, cells + 1) if dim else [0]
    Rs = [np.array(R, int) for R in itertools.product(rng, repeat=dim)] or [np.zeros(0, int)]
    Rset = {tuple(R) for R in Rs}
    shift = (lambda R: R @ A) if dim else (lambda R: np.zeros(D))
    reps = _site_representatives(model)
    sh = neighbor_shells(model, shells)
    dist_to_shell = {round(x["distance"], 5): n for n, x in enumerate(sh)}
    handles = []
    allpts = np.vstack([P + shift(R0) for R0 in Rs])
    collinear = (not use3d) and (D == 1 or np.ptp(allpts[:, 1]) < 1e-9)
    extra_pts = []

    def draw(a, b, col, ls, lw, n, alpha=0.9, z=1):
        if collinear:
            t = np.linspace(0, 1, 40)
            h = 0.18 * (n + 1) * np.linalg.norm(b - a) * (1 if n % 2 == 0 else -1)
            x = pt(a)[0] + t * (pt(b)[0] - pt(a)[0])
            y = np.sin(np.pi * t) * h
            ax.plot(x, y, color=col, ls=ls, lw=lw, zorder=z, alpha=alpha)
        else:
            ax.plot(*np.array([pt(a), pt(b)]).T, color=col, ls=ls, lw=lw, zorder=z, alpha=alpha)

    # ---- bonds ---------------------------------------------------------
    if bonds:
        if bonds == "model":
            seen = set()
            per_shell = {n: [] for n in range(len(sh))}
            for t in model._terms:
                d = P[t.j] + shift(t.R) - P[t.i]
                n = dist_to_shell.get(round(np.linalg.norm(d), 5))
                key = (t.i, t.j, tuple(t.R))
                if n is None or key in seen:
                    continue
                seen.add(key)
                per_shell[n].append((t.i, t.j, t.R, d))
            groups = [(n, per_shell[n]) for n in range(len(sh))]
        else:
            groups = [(n, x["bonds"]) for n, x in enumerate(sh)]
        hl = model.orb(highlight) if highlight is not None else None
        if hl is not None and sh:
            for i, j, R, d in sh[0]["bonds"]:
                for R0 in Rs:
                    if tuple(R0 + R) in Rset:
                        a = P[i] + shift(R0)
                        draw(a, a + d, "0.85", "-", 0.8, 0, 1, 0)
            if circles and not use3d:
                c0 = pt(P[hl])
                for n, x in enumerate(sh):
                    col = SHELL_STYLES[n % len(SHELL_STYLES)][0]
                    ax.add_patch(plt.Circle(c0, x["distance"], fill=False, color=col, lw=0.8,
                                            ls=(0, (2, 3)), alpha=0.6, zorder=0))
        for n, blist in reversed(groups):
            col, ls, lw = SHELL_STYLES[n % len(SHELL_STYLES)]
            drawn = False
            for i, j, R, d in blist:
                for R0 in Rs:
                    if hl is not None:
                        if R0.any() or hl not in (i, j):
                            continue
                    elif tuple(R0 + R) not in Rset:
                        continue
                    a = P[i] + shift(R0)
                    if hl is not None and j == hl and i != hl:
                        a, b = P[j] + shift(R0), P[j] + shift(R0) - d
                    else:
                        b = a + d
                    if hl is not None and i == hl and j == hl:
                        # self-shell bond: draw both directions from the centre
                        for bb in (a + d, a - d):
                            draw(a, bb, col, ls, lw, n)
                            extra_pts.append((model.lat.site(hl), bb))
                        drawn = True
                        continue
                    draw(a, b, col, ls, lw, n)
                    if hl is not None:
                        other = j if i == hl else i
                        extra_pts.append((model.lat.site(other), b))
                    drawn = True
            if drawn or blist:
                z = sorted(set(v for k, v in sh[n]["coordination"].items()))
                zs = "/".join(str(v) for v in z)
                lab = f"{_ORD.get(n + 1, f'{n + 1}th')} NN  d = {sh[n]['distance']:.4g}  (z = {zs})"
                handles.insert(0, plt.Line2D([], [], color=col, ls=ls, lw=lw, label=lab))

    # ---- unit cell -------------------------------------------------------
    if show_cell and dim >= 1:
        o = np.zeros(D)
        if dim == 1:
            corners = [o, A[0]]
        elif dim == 2:
            corners = [o, A[0], A[0] + A[1], A[1], o]
        else:
            corners = None
            edges = []
            for c in itertools.product([0, 1], repeat=3):
                for a in range(3):
                    if c[a] == 0:
                        c2 = list(c)
                        c2[a] = 1
                        edges.append((np.array(c) @ A, np.array(c2) @ A))
            for p0, p1 in edges:
                ax.plot(*np.array([pt(p0), pt(p1)]).T, color="0.45", lw=0.8, ls="--", zorder=0)
        if corners is not None:
            ax.plot(*np.array([pt(c) for c in corners]).T, color="0.45", lw=0.9, ls="--", zorder=0)
        for a in range(dim):
            v = pt(A[a])
            if use3d:
                ax.quiver(0, 0, 0, *v, color="k", arrow_length_ratio=0.12, lw=1.2)
                ax.text(*(v * 1.05), f"$a_{a + 1}$")
            else:
                ax.annotate("", xy=v, xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.3))
                ax.annotate(f"$a_{a + 1}$", v, textcoords="offset points", xytext=(4, -10))

    # ---- sites -------------------------------------------------------------
    sites = []
    for i in reps:
        if model.lat.site(i) not in sites:
            sites.append(model.lat.site(i))
    colors = {s_: f"C{(n + 4) % 10}" for n, s_ in enumerate(sites)}
    size = s or (90 if len(Rs) < 30 else 40)
    for site in sites:
        pts = np.array([pt(P[i] + shift(R0)) for R0 in Rs for i in reps if model.lat.site(i) == site])
        home = np.array([pt(P[i]) for i in reps if model.lat.site(i) == site])
        ax.scatter(*pts.T, s=size, color=colors[site], edgecolor="none", zorder=3, alpha=0.85)
        ax.scatter(*home.T, s=size * 1.15, color=colors[site], edgecolor="k", lw=1.2, zorder=4)
        ex = [pt(p) for st, p in extra_pts if st == site]
        if ex:
            ax.scatter(*np.array(ex).T, s=size, color=colors[site], edgecolor="none", zorder=3,
                       alpha=0.85)
        handles.append(plt.Line2D([], [], marker="o", ls="", color=colors[site], markersize=8,
                                  label=f"site {site}"))
    if annotate:
        for i in reps:
            kinds = [model.lat.kind(j) for j in range(model.norb)
                     if model.lat.site(j) == model.lat.site(i) and np.allclose(P[j], P[i])]
            label = model.lat.site(i) + (f" ({','.join(kinds)})" if kinds != ["s"] else "")
            if use3d:
                ax.text(*pt(P[i]), " " + label, fontsize=8)
            else:
                ax.annotate(label, pt(P[i]), textcoords="offset points", xytext=(6, 5), fontsize=9,
                            fontweight="bold")
    if collinear:
        ax.set_yticks([])
        ax.set_xlabel("x")
        for sp in ("left", "right", "top"):
            ax.spines[sp].set_visible(False)
        ax.set_aspect("equal")
    elif not use3d:
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    if legend and handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                  frameon=False)
    ax.set_title(title or f"{model.name}: neighbour shells")
    return ax


def _is_flat(P, A):
    pts = np.vstack([P, P + (A.sum(0) if len(A) else 0)])
    return np.ptp(pts[:, 2]) < 1e-9 if pts.shape[1] == 3 else True


def plot_wcc(wcc, ax=None, kperp=None, title="Wannier charge centres"):
    """Plot hybrid Wannier charge centres (output of topology.wilson_loop)."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))
    k = np.linspace(0, 1, len(wcc)) if kperp is None else kperp
    for n in range(wcc.shape[1]):
        ax.plot(k, wcc[:, n], "o", ms=2.5, color="C0")
    ax.set_xlabel(r"$k_\perp$ (reduced)")
    ax.set_ylabel(r"WCC $\bar{x}$ (reduced)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    return ax


def plot_kmap(values, extent=(0, 1, 0, 1), ax=None, cmap="RdBu_r", title="", symmetric=True):
    """Colour map of a quantity on a 2D k grid (e.g. Berry curvature)."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    v = np.asarray(values)
    vmax = np.abs(v).max() if symmetric else None
    im = ax.imshow(v.T, origin="lower", extent=extent, cmap=cmap,
                   vmin=-vmax if symmetric else None, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(r"$k_1$")
    ax.set_ylabel(r"$k_2$")
    ax.set_title(title)
    return ax
