"""Current-voltage analysis: identify transport mechanisms and extract trap
densities and barrier/material parameters from measured (or simulated) I-V data.

Mechanisms (J current density, E = V/d field, d thickness, SI units):

    Ohmic                 J = q n0 mu V / d                     slope 1 in log J - log V
    trap-free SCLC        J = (9/8) eps mu V^2 / d^3            (Mott-Gurney; slope 2)
    trap-filled limit     sharp rise (slope > 3) at V_TFL:
                          n_t = 2 eps V_TFL / (q d^2)           (Lampert & Mark 1970)
    exponential traps     J ~ V^(l+1),  l = T_t / T              (Mark & Helfrich 1962)
    Poole-Frenkel         ln(J/E) linear in sqrt(E), slope q^1.5 / (kT sqrt(pi eps))
    Schottky emission     ln(J)   linear in sqrt(E), slope q^1.5 / (kT sqrt(4 pi eps))
    Fowler-Nordheim       ln(J/E^2) linear in 1/E, slope -8 pi sqrt(2 m*) phi^1.5 / (3 q h)
    direct tunnelling     transition-voltage spectroscopy: minimum of ln(I/V^2) vs 1/V
    hopping (sigma(T))    Arrhenius / Mott VRH (T^-1/4) / Efros-Shklovskii (T^-1/2)

>>> rep = hf.iv.analyze(V, I, thickness=500e-9, area=1e-6, eps_r=25, T=300)
>>> print(rep); rep.plot()
>>> rep.trap_density      # m^-3  (rep.trap_density_cm3 in cm^-3)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

q = 1.602176634e-19
eps0 = 8.8541878128e-12
kB = 1.380649e-23
h = 6.62607015e-34
me = 9.1093837015e-31


# =============================================================================
# forward models (for fitting checks and synthetic data)
# =============================================================================
def ohmic(V, d, n0, mu):
    return q * n0 * mu * np.asarray(V) / d


def mott_gurney(V, d, eps_r, mu, theta=1.0):
    return 9 / 8 * theta * eps0 * eps_r * mu * np.asarray(V) ** 2 / d ** 3


def lampert_vtfl(nt, d, eps_r):
    """Trap-filled-limit voltage for trap density nt (m^-3)."""
    return q * nt * d ** 2 / (2 * eps0 * eps_r)


def poole_frenkel(V, d, phi_eV, eps_r, T, C=1.0):
    E = np.asarray(V) / d
    return C * E * np.exp(-q * (phi_eV - np.sqrt(q * E / (np.pi * eps0 * eps_r))) / (kB * T))


def schottky(V, d, phi_eV, eps_r, T, A_star=1.2e6):
    E = np.asarray(V) / d
    return A_star * T ** 2 * np.exp(-q * (phi_eV - np.sqrt(q * E / (4 * np.pi * eps0 * eps_r))) / (kB * T))


def fowler_nordheim(V, d, phi_eV, m_eff=1.0):
    E = np.asarray(V) / d
    phi = phi_eV * q
    A = q ** 3 / (8 * np.pi * h * phi)
    B = 8 * np.pi * np.sqrt(2 * m_eff * me) * phi ** 1.5 / (3 * q * h)
    return A * E ** 2 * np.exp(-B / E)


def sclc_with_traps(V, d, eps_r, mu, n0, nt, theta=1e-3, n_tfl=8.0):
    """Synthetic Lampert-type SCLC curve: Ohmic / trap-limited SCLC
    (theta J_Child) below V_TFL, a steep J ~ V^n_tfl rise starting at
    V_TFL = q n_t d^2 / (2 eps), then the trap-free Child law."""
    V = np.asarray(V, float)
    Vtfl = lampert_vtfl(nt, d, eps_r)
    low = lambda v: np.maximum(ohmic(v, d, n0, mu), mott_gurney(v, d, eps_r, mu, theta))
    Jc = mott_gurney(V, d, eps_r, mu, 1.0)
    J = low(V)
    hi = V >= Vtfl
    J[hi] = np.minimum(low(np.array([Vtfl]))[0] * (V[hi] / Vtfl) ** n_tfl, Jc[hi])
    return np.maximum(J, low(V))


# =============================================================================
# analysis
# =============================================================================
def _linfit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (res[0] / ss if len(res) and ss > 0 else 0.0)
    return a, b, r2


def loglog_slope(V, J, window=5):
    """Local slope d ln J / d ln V (moving linear fit)."""
    x, y = np.log(V), np.log(J)
    n = len(x)
    s = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - window // 2), min(n, i + window // 2 + 1)
        if hi - lo >= 3:
            s[i] = _linfit(x[lo:hi], y[lo:hi])[0]
    return s


@dataclass
class IVReport:
    V: np.ndarray
    J: np.ndarray
    slope: np.ndarray
    regimes: list = field(default_factory=list)       # (Vstart, Vend, label, mean slope)
    V_TFL: float = np.nan
    trap_density: float = np.nan                       # m^-3
    mobility: float = np.nan                           # m^2/Vs (Mott-Gurney)
    mechanisms: dict = field(default_factory=dict)     # name -> dict(fit params, r2, plausible)
    best: str = ""
    params: dict = field(default_factory=dict)

    @property
    def trap_density_cm3(self):
        return self.trap_density * 1e-6

    def __repr__(self):
        lines = ["I-V analysis"]
        for a, b, lab, s in self.regimes:
            lines.append(f"  {a:9.3g} - {b:9.3g} V : {lab:24s} (slope {s:.2f})")
        if np.isfinite(self.V_TFL):
            lines.append(f"  V_TFL = {self.V_TFL:.4g} V  ->  trap density n_t = {self.trap_density_cm3:.3e} cm^-3")
        if np.isfinite(self.mobility):
            lines.append(f"  Mott-Gurney mobility = {self.mobility * 1e4:.3g} cm^2/Vs")
        for k, v in self.mechanisms.items():
            flag = "plausible" if v.get("plausible") else "unphysical"
            extra = ", ".join(f"{a}={b:.3g}" for a, b in v.items() if a not in ("r2", "plausible", "range", "kind"))
            lines.append(f"  {k:16s} R^2={v['r2']:.4f} [{flag}] {extra}")
        lines.append(f"  most consistent high-field mechanism: {self.best}")
        return "\n".join(lines)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        a0, a1 = (ax if np.ndim(ax) else (ax, None))
        a0.loglog(self.V, self.J, "k.", ms=4)
        cols = {"Ohmic": "C0", "trap-free SCLC (Child)": "C2", "trap-filled limit": "C3",
                "exponential-trap SCLC": "C1", "other": "0.6"}
        for a, b, lab, s in self.regimes:
            sel = (self.V >= a) & (self.V <= b)
            a0.loglog(self.V[sel], self.J[sel], "-", color=cols.get(lab, "0.5"), lw=3, alpha=0.6, label=f"{lab} (n={s:.1f})")
        if np.isfinite(self.V_TFL):
            a0.axvline(self.V_TFL, color="C3", ls="--", label=f"V_TFL = {self.V_TFL:.3g} V")
        a0.set_xlabel("V (V)"); a0.set_ylabel("J (A/m²)"); a0.legend(fontsize=7)
        a0.set_title(f"n_t = {self.trap_density_cm3:.2e} cm$^{{-3}}$" if np.isfinite(self.trap_density) else "I-V")
        if a1 is not None:
            a1.semilogx(self.V, self.slope, "k-")
            for y in (1, 2):
                a1.axhline(y, color="0.7", ls=":")
            a1.set_xlabel("V (V)"); a1.set_ylabel("d ln J / d ln V"); a1.set_title("local slope")
        return ax


def analyze(V, I, thickness, area=1.0, eps_r=10.0, T=300.0, m_eff=1.0, current_is_density=False,
            window=7, tfl_slope=3.0):
    """Analyse an I-V curve (positive bias branch).

    thickness : film / barrier thickness d (m); area : device area (m^2)
    eps_r     : relative permittivity (for SCLC, PF, Schottky)
    Returns an :class:`IVReport` with regimes, V_TFL, trap density, mobility
    and mechanism fits (Poole-Frenkel, Schottky, Fowler-Nordheim, transition
    voltage)."""
    V = np.asarray(V, float)
    I = np.abs(np.asarray(I, float))
    keep = (V > 0) & (I > 0)
    V, I = V[keep], I[keep]
    o = np.argsort(V)
    V, I = V[o], I[o]
    J = I if current_is_density else I / area
    d = thickness
    eps = eps0 * eps_r
    s = loglog_slope(V, J, window)
    rep = IVReport(V, J, s, params=dict(thickness=d, area=area, eps_r=eps_r, T=T))

    # --- regimes from the local slope
    def label(sl):
        if sl < 1.4:
            return "Ohmic"
        if sl < 2.6:
            return "trap-free SCLC (Child)"
        if sl < tfl_slope:
            return "exponential-trap SCLC"
        return "trap-filled limit"
    labs = [label(x) if np.isfinite(x) else "other" for x in s]
    start = 0
    for i in range(1, len(V) + 1):
        if i == len(V) or labs[i] != labs[start]:
            if i - start >= 3:
                rep.regimes.append((V[start], V[i - 1], labs[start], float(np.nanmean(s[start:i]))))
            start = i

    # --- trap-filled limit: intersection of the low-voltage (Ohmic / trapped) fit
    # and the steep TFL branch in log-log
    steep = np.flatnonzero(s >= tfl_slope)
    if len(steep):
        i0 = steep[0]
        low = np.arange(max(0, i0 - 3 * window), max(1, i0 - window // 2))
        hi_ = np.arange(i0, min(len(V), i0 + max(3, window)))
        if len(low) >= 3 and len(hi_) >= 3:
            a1_, b1_, _ = _linfit(np.log(V[low]), np.log(J[low]))
            a2_, b2_, _ = _linfit(np.log(V[hi_]), np.log(J[hi_]))
            if abs(a2_ - a1_) > 1e-6:
                rep.V_TFL = float(np.exp((b1_ - b2_) / (a2_ - a1_)))
        else:
            rep.V_TFL = float(V[i0])
        rep.trap_density = 2 * eps * rep.V_TFL / (q * d ** 2)

    # --- Mott-Gurney mobility from the Child regime (slope ~ 2), preferring the
    # highest-voltage (trap-free) segment
    child = [r for r in rep.regimes if r[2] == "trap-free SCLC (Child)"]
    if child:
        a, b = child[-1][0], child[-1][1]
        sel = (V >= a) & (V <= b)
        rep.mobility = float(np.median(J[sel] * 8 * d ** 3 / (9 * eps * V[sel] ** 2)))

    # --- high-field mechanism fits (upper half of the curve)
    sel = V >= np.median(V)
    E = V[sel] / d
    kT = kB * T
    mech = {}
    a, b, r2 = _linfit(np.sqrt(E), np.log(J[sel] / E))
    epf = q ** 3 / (np.pi * eps0 * (a * kT) ** 2) if a > 0 else np.nan
    mech["Poole-Frenkel"] = dict(r2=r2, eps_r_fit=epf, plausible=bool(a > 0 and 0.3 * eps_r < epf < 3 * eps_r))
    a, b, r2 = _linfit(np.sqrt(E), np.log(J[sel]))
    esc = q ** 3 / (4 * np.pi * eps0 * (a * kT) ** 2) if a > 0 else np.nan
    phi_s = -(b - np.log(1.2e6 * T ** 2)) * kT / q
    mech["Schottky"] = dict(r2=r2, eps_r_fit=esc, barrier_eV=phi_s,
                            plausible=bool(a > 0 and 0.3 * eps_r < esc < 3 * eps_r))
    a, b, r2 = _linfit(1 / E, np.log(J[sel] / E ** 2))
    phi_fn = ((-a) * 3 * q * h / (8 * np.pi * np.sqrt(2 * m_eff * me))) ** (2 / 3) / q if a < 0 else np.nan
    mech["Fowler-Nordheim"] = dict(r2=r2, barrier_eV=phi_fn, plausible=bool(a < 0 and 0.05 < phi_fn < 10))
    # transition voltage spectroscopy
    y = np.log(J / V ** 2)
    itv = int(np.argmin(y)) if len(y) > 2 else 0
    if 0 < itv < len(V) - 1:
        mech["transition voltage"] = dict(r2=1.0, V_trans=float(V[itv]), plausible=True)
    # power law over the same high-field range (Ohmic / SCLC / trap-limited)
    a, b, r2 = _linfit(np.log(V[sel]), np.log(J[sel]))
    kind = "Ohmic" if a < 1.4 else ("space-charge-limited (Child)" if a < 2.6 else "trap-limited SCLC")
    mech["power law"] = dict(r2=r2, exponent=a, plausible=True, kind=0.0)
    rep.mechanisms = mech
    cands = {k: v for k, v in mech.items() if k != "transition voltage" and v["plausible"]}
    best = max(cands, key=lambda k: cands[k]["r2"])
    if best == "power law":
        rep.best = kind + (" (trap-filled limit at V_TFL)" if np.isfinite(rep.V_TFL) else "")
    else:
        rep.best = best
    return rep


def hopping_analysis(T, sigma):
    """Temperature dependence of conductivity: compare Arrhenius
    (ln s ~ T^-1), Efros-Shklovskii VRH (T^-1/2) and Mott 3D VRH (T^-1/4).
    Returns dict(model -> (R^2, characteristic temperature / activation energy))."""
    T = np.asarray(T, float)
    ls = np.log(np.asarray(sigma, float))
    out = {}
    for name, p in (("Arrhenius", 1.0), ("Efros-Shklovskii VRH", 0.5), ("Mott VRH (3D)", 0.25)):
        a, b, r2 = _linfit(T ** -p, ls)
        if name == "Arrhenius":
            out[name] = dict(r2=r2, Ea_eV=-a * kB / q)
        else:
            out[name] = dict(r2=r2, T0=(-a) ** (1 / p) if a < 0 else np.nan)
    out["best"] = max((k for k in out if k != "best"), key=lambda k: out[k]["r2"])
    return out
