"""Quantum phases and phase diagrams.

* :func:`hubbard_phase_diagram`  - mean-field magnetic phase diagram versus U
  and doping (paramagnet / ferromagnet / Neel antiferromagnet), competing
  solutions compared by energy (Hirsch, PRB 31, 4403 (1985); Stoner criterion).
* :func:`quantum_critical_scan`  - exact (ED) diagnostics of a quantum phase
  transition: order parameter, Binder cumulant, fidelity susceptibility, gap,
  entanglement entropy.
* :func:`entanglement_entropy`, :func:`central_charge` - Calabrese-Cardy
  scaling S(l) = c/3 ln[(L/pi) sin(pi l / L)] + const (J. Stat. Mech. P06002 (2004)).
* :func:`mott_lobes`              - Bose-Hubbard mean-field phase diagram.
* :func:`topological_phase_diagram` - Chern / Z2 maps over two parameters.
* :func:`phase_diagram`, :func:`plot_phase_diagram` - generic 2D scans
  (parallel over grid points with ``n_jobs``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .parallel import pmap


# =============================================================================
# generic 2D scans
# =============================================================================
@dataclass
class PhaseDiagram:
    x_name: str
    y_name: str
    x: np.ndarray
    y: np.ndarray
    phase: np.ndarray                    # (ny, nx) labels (str)
    data: dict = field(default_factory=dict)  # name -> (ny, nx) arrays

    def fraction(self, label):
        return float(np.mean(self.phase == label))


def phase_diagram(func, x_name, xs, y_name, ys, n_jobs=None):
    """Evaluate ``func(x, y) -> (label, {observable: value})`` on a grid."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    pts = [(x, y) for y in ys for x in xs]
    res = pmap(lambda p: func(*p), pts, n_jobs)
    labels = np.array([r[0] for r in res], dtype=object).reshape(len(ys), len(xs))
    keys = sorted({k for r in res for k in r[1]})
    data = {k: np.array([r[1].get(k, np.nan) for r in res], float).reshape(len(ys), len(xs)) for k in keys}
    return PhaseDiagram(x_name, y_name, xs, ys, labels, data)


_COLORS = {"PM": "#d9d9d9", "FM": "#e41a1c", "AFM": "#377eb8", "Mott": "#984ea3", "SF": "#4daf4a",
           "ordered": "#377eb8", "disordered": "#d9d9d9", "trivial": "#d9d9d9"}


def plot_phase_diagram(pd: PhaseDiagram, ax=None, contour=None, title=None, cmap_cont="viridis"):
    """Categorical phase map (+ optional contour of an observable)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    labels = sorted(set(pd.phase.ravel()), key=str)
    cols = [_COLORS.get(str(l), plt.cm.tab10(i % 10)) for i, l in enumerate(labels)]
    idx = np.vectorize(lambda v: labels.index(v))(pd.phase)
    dx = (pd.x[1] - pd.x[0]) / 2 if len(pd.x) > 1 else 0.5
    dy = (pd.y[1] - pd.y[0]) / 2 if len(pd.y) > 1 else 0.5
    ax.imshow(idx, origin="lower", aspect="auto", cmap=ListedColormap(cols), vmin=-0.5,
              vmax=len(labels) - 0.5, extent=(pd.x[0] - dx, pd.x[-1] + dx, pd.y[0] - dy, pd.y[-1] + dy),
              interpolation="nearest")
    if contour is not None and contour in pd.data and len(pd.x) > 1 and len(pd.y) > 1:
        cs = ax.contour(pd.x, pd.y, pd.data[contour], colors="k", linewidths=0.6, levels=6)
        ax.clabel(cs, fontsize=7)
    ax.legend(handles=[Patch(color=c, label=str(l)) for c, l in zip(cols, labels)], loc="upper left",
              bbox_to_anchor=(1.01, 1), frameon=False)
    ax.set_xlabel(pd.x_name)
    ax.set_ylabel(pd.y_name)
    ax.set_title(title or "phase diagram")
    return ax


# =============================================================================
# Hubbard mean-field phase diagram vs U and doping
# =============================================================================
def bipartite_signs(model, shell=1):
    """+-1 sublattice signs from the NN bond graph (None if not bipartite)."""
    n = model.norb
    adj = [[] for _ in range(n)]
    for i, j, R, d in model.neighbors(shell):
        adj[i].append(j)
        adj[j].append(i)
    sign = [0] * n
    for s0 in range(n):
        if sign[s0]:
            continue
        sign[s0] = 1
        stack = [s0]
        while stack:
            a = stack.pop()
            for b in adj[a]:
                if sign[b] == 0:
                    sign[b] = -sign[a]
                    stack.append(b)
                elif sign[b] == sign[a]:
                    return None
    return np.array(sign, float)


def classify_moments(m, signs=None, tol=0.02):
    """'PM', 'FM', 'AFM' or 'other' from per-orbital moment vectors (n, 3)."""
    mag = np.linalg.norm(m, axis=1)
    if mag.max() < tol:
        return "PM"
    tot = np.linalg.norm(m.sum(0)) / len(m)
    if tot > 0.8 * mag.mean():
        return "FM"
    if signs is not None:
        stag = np.linalg.norm((signs[:, None] * m).sum(0)) / len(m)
        if stag > 0.8 * mag.mean():
            return "AFM"
    return "other"


def hubbard_phase_diagram(model, Us, fillings, nk=24, kT=0.01, n_jobs=None, tol=0.02, maxiter=400,
                          mixing=0.5, extra_inits=None):
    """Mean-field Hubbard phase diagram over U and filling.

    model    : spinful model whose unit cell can host the orders (e.g. a
               sqrt2 x sqrt2 square-lattice supercell for Neel order)
    fillings : electrons per orbital / 2 (0.5 = half filling); doping
               delta = 1 - 2 * filling
    For every point the PM, FM and AFM (bipartite) solutions are converged and
    the lowest-energy one is kept. Returns a :class:`PhaseDiagram` with x = U,
    y = doping and observables |m|, staggered m, energy.
    """
    from .meanfield import hubbard
    signs = bipartite_signs(model)
    inits = {"PM": {i: (0, 0, 1e-4) for i in range(model.norb)},
             "FM": {i: (0, 0, 0.8) for i in range(model.norb)}}
    if signs is not None:
        inits["AFM"] = {i: (0, 0, 0.8 * signs[i]) for i in range(model.norb)}
    inits.update(extra_inits or {})

    def point(U, doping):
        filling = 0.5 * (1 - doping)
        best = None
        for name, init in inits.items():
            r = hubbard(model, U, filling=filling, init=init, nk=nk, kT=kT, maxiter=maxiter, mixing=mixing)
            if best is None or r.energy < best.energy - 1e-7:
                best = r
        m = best.moments
        lab = classify_moments(m, signs, tol)
        stag = np.linalg.norm((signs[:, None] * m).sum(0)) / len(m) if signs is not None else np.nan
        return lab, {"|m|": float(np.linalg.norm(m, axis=1).mean()), "m_uniform": float(np.linalg.norm(m.sum(0)) / len(m)),
                     "m_staggered": float(stag), "energy": best.energy}
    dop = 1 - 2 * np.asarray(fillings, float)
    pd = phase_diagram(point, "U / t", Us, "doping  1 - n", dop, n_jobs)
    return pd


def stoner_critical_U(model, filling, nk=200, sigma=0.01):
    """Stoner criterion U_c = 1 / rho_sigma(E_F) (density of states per spin
    per site at the Fermi level of the paramagnet)."""
    from .solvers import dos, fermi_level
    if model.spinful:
        raise ValueError("pass the spinless template")
    base = model
    EF = fermi_level(base, filling * base.norb, nk)
    d = dos(base, nk, energies=[EF], sigma=sigma)
    rho = d.dos[0] / base.norb
    return 1.0 / rho, EF


# =============================================================================
# exact quantum critical diagnostics
# =============================================================================
def _bits_z(L):
    st = np.arange(2 ** L)
    return np.array([1 - 2 * ((st >> (L - 1 - q)) & 1) for q in range(L)])   # (L, 2^L) eigenvalues of Z_q


def order_parameter_moments(psi, L):
    """<M^2>, <M^4> of M = (1/L) sum_i Z_i for a full-basis state."""
    z = _bits_z(L).sum(0) / L
    p = np.abs(psi) ** 2
    return float(p @ z ** 2), float(p @ z ** 4)


def binder_cumulant(psi, L):
    m2, m4 = order_parameter_moments(psi, L)
    return 1 - m4 / (3 * m2 ** 2)


def entanglement_entropy(psi, L, cut):
    """von Neumann entropy of the first ``cut`` qubits (full-basis state)."""
    M = np.asarray(psi).reshape(2 ** cut, 2 ** (L - cut))
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-14]
    return float(-(s * np.log(s)).sum())


def central_charge(psi, L, cuts=None):
    """Fit S(l) = c/3 ln[(L/pi) sin(pi l/L)] + s0 (periodic chain)."""
    cuts = cuts or list(range(2, L - 1))
    S = [entanglement_entropy(psi, L, l) for l in cuts]
    x = np.log(L / np.pi * np.sin(np.pi * np.array(cuts) / L)) / 3
    c, s0 = np.polyfit(x, S, 1)
    return float(c), float(s0)


def quantum_critical_scan(builder, param, values, Ls=(8, 10, 12), n_jobs=None, dh=1e-3):
    """ED scan of a spin chain across a quantum phase transition.

    builder(L, value) -> PauliModel. Returns dict with arrays (len(Ls), len(values)):
    'binder', 'm2', 'gap', 'fidelity' (fidelity susceptibility per site), 'S_half'
    (half-chain entanglement entropy)."""
    from .manybody import PauliModel  # noqa: F401
    jobs = [(L, v) for L in Ls for v in values]

    def one(job):
        L, v = job
        m = builder(L, v)
        w, V = m.ground_state(k=2)
        psi = V[:, 0]
        m2, m4 = order_parameter_moments(psi, L)
        psi2 = builder(L, v + dh).ground_state()[1]
        F = abs(np.vdot(psi, psi2))
        chi_F = 2 * (1 - F) / dh ** 2 / L
        return dict(binder=1 - m4 / (3 * m2 ** 2), m2=m2, gap=w[1] - w[0], fidelity=chi_F,
                    S_half=entanglement_entropy(psi, L, L // 2))
    res = pmap(one, jobs, n_jobs)
    out = {k: np.array([r[k] for r in res]).reshape(len(Ls), len(values)) for k in res[0]}
    out["values"] = np.asarray(values)
    out["Ls"] = np.asarray(Ls)
    out["param"] = param
    return out


def binder_crossing(scan, i=0, j=-1):
    """Parameter value where the Binder cumulants of two system sizes cross."""
    v = scan["values"]
    d = scan["binder"][i] - scan["binder"][j]
    for a in range(len(v) - 1):
        if d[a] * d[a + 1] < 0:
            return float(v[a] - d[a] * (v[a + 1] - v[a]) / (d[a + 1] - d[a]))
    return float("nan")


def plot_critical_scan(scan, axes=None):
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    for k, ax in zip(["m2", "binder", "fidelity", "S_half"], axes):
        for L, y in zip(scan["Ls"], scan[k]):
            ax.plot(scan["values"], y, label=f"L={L}")
        ax.set_xlabel(scan["param"])
        ax.set_title({"m2": "<M_z^2>", "binder": "Binder cumulant", "fidelity": "fidelity susceptibility",
                      "S_half": "half-chain entanglement"}[k])
    axes[0].legend()
    return axes


# =============================================================================
# Bose-Hubbard Mott lobes
# =============================================================================
def mott_lobes(zt_over_U, mu_over_U, nmax=8, n_jobs=None):
    """Mean-field Bose-Hubbard phase diagram: 'Mott' (psi = 0, integer n) or
    'SF'. x = zt/U, y = mu/U. Observables: psi (condensate), n."""
    from .manybody import bose_hubbard_meanfield

    def point(zt, mu):
        psi, n, E = bose_hubbard_meanfield(1.0, zt, mu, nmax=nmax)
        lab = "SF" if psi > 1e-3 else f"Mott n={int(round(n))}"
        return lab, {"psi": psi, "n": n}
    return phase_diagram(point, "zt / U", zt_over_U, "mu / U", mu_over_U, n_jobs)


# =============================================================================
# topological phase diagrams
# =============================================================================
def topological_phase_diagram(builder, x_name, xs, y_name, ys, occ, invariant="chern", nk=30, n_jobs=None):
    """Map a topological invariant over two parameters. builder(x, y) -> Model;
    invariant 'chern' or 'z2'. Labels are 'C=..' or 'Z2=..'."""
    import hopforge as hf

    def point(x, y):
        m = builder(x, y)
        if invariant == "chern":
            c = round(hf.chern_number(m, occ, nk))
            return f"C={c}", {"C": c, "gap": hf.band_gap(m, occ if np.isscalar(occ) else len(occ), nk)}
        z = hf.z2_invariant(m, occ, nk=nk, nscan=nk + 1)
        return f"Z2={z}", {"Z2": z, "gap": hf.band_gap(m, occ, nk)}
    return phase_diagram(point, x_name, xs, y_name, ys, n_jobs)
