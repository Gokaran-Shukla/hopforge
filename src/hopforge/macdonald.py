"""Models from Allan H. MacDonald's group (UT Austin), ready to use.

MacDonald's group has not released a general tight-binding package; its
models are defined in the papers. This module implements them on top of
hopforge's engines, so every hopforge tool (bands, Berry curvature, Chern
numbers, Hubbard parameters, phase diagrams) applies to them:

Continuum moire models (plane-wave basis; lengths in nm, energies in meV)
    TMDHeterobilayer   Wu, Lovorn, Tutuc & MacDonald, PRL 121, 026402 (2018)
                       "Hubbard model physics in TMD moire bands"
    TMDHomobilayer     Wu, Lovorn, Tutuc, Martin & MacDonald, PRL 122, 086402 (2019)
                       "Topological insulators in twisted TMD homobilayers"
    BistritzerMacDonald  PNAS 108, 12233 (2011) (alias of hf.moire.BistritzerMacDonald)
    PlaneWaveContinuum   generic multi-layer engine used by the above

Lattice models (ordinary hopforge Models)
    rhombohedral_graphene  ABC N-layer graphene; chiral decomposition, E ~ k^N
                           (Min & MacDonald, PRB 77, 155416 (2008); Zhang, Sahu,
                           Min & MacDonald, PRB 82, 035409 (2010))
    graphene_qah           graphene + Rashba + exchange: QAH with C = 2
                           (Qiao et al., PRB 82, 161414(R) (2010);
                           Tse, Qiao, Yao, MacDonald & Niu, PRB 83, 155447 (2011))
    rashba_2deg            Rashba electron gas; universal intrinsic spin Hall
                           conductivity e/8pi (Sinova et al., PRL 92, 126603 (2004))

Tools
    valley_chern        Berry-curvature integral around one valley (valley Hall)
    sinova_spin_hall    Kubo spin Hall conductivity in units of e/8pi
"""
from __future__ import annotations

import numpy as np

from .model import Lattice, Model
from .moire import BistritzerMacDonald  # noqa: F401  (re-export)
from .models._util import HEX, HEX_POINTS, SQ_POINTS, set_points
from .parallel import pmap, split, chunks_for

HB2M = 38.09982   # hbar^2 / (2 m_e) in meV nm^2
COULOMB = 1439.9645  # e^2 / (4 pi eps0) in meV nm


def _rot(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# =============================================================================
# generic plane-wave continuum engine
# =============================================================================
class PlaneWaveContinuum:
    """Multi-layer continuum Hamiltonian on a triangular moire lattice.

    H_{l G, l' G'}(k) = delta_{ll'} delta_{GG'} s * (hbar^2/2m*) |k + G - kappa_l|^2
                       + sum_terms  amp  delta_{G, G' + g}

    aM      : moire period (nm)
    mass    : effective mass in units of m_e
    kappas  : list of valley offsets kappa_l (1/nm), one per layer
    terms   : list of (l, l', g, amp) meaning <l, G'+g| H |l', G'> += amp.
              Interlayer terms (l != l') get their Hermitian conjugate added
              automatically; intralayer terms must be listed for +g and -g.
    sign    : -1 for holes (valence bands, as in the MacDonald-group papers)
    shells  : plane-wave cutoff |G| <= shells * |b1|

    Moire reciprocal vectors: b1 = 4 pi/(sqrt(3) aM) x, b_j = b1 rotated by
    (j-1) 60 deg. Lattice vectors a1 = aM (sqrt3/2, -1/2), a2 = aM (0, 1).
    """

    def __init__(self, aM, mass, kappas, terms, sign=-1, shells=4.0):
        self.aM, self.mass, self.sign = float(aM), float(mass), sign
        b = 4 * np.pi / (np.sqrt(3) * aM)
        self.bvec = [_rot(np.array([b, 0.0]), j * np.pi / 3) for j in range(6)]
        self.B = np.array([self.bvec[0], self.bvec[1]])          # rows b1, b2
        self.A = aM * np.array([[np.sqrt(3) / 2, -0.5], [0.0, 1.0]])
        n = int(np.ceil(shells)) + 2
        G = [(i, j) for i in range(-2 * n, 2 * n + 1) for j in range(-2 * n, 2 * n + 1)
             if np.linalg.norm(i * self.B[0] + j * self.B[1]) <= shells * b + 1e-9]
        self.Gint = np.array(G)
        self.G = self.Gint @ self.B
        self.nG = len(G)
        self.kappas = [np.asarray(k, float) for k in kappas]
        self.nl = len(self.kappas)
        idx = {g: i for i, g in enumerate(G)}
        dim = self.nl * self.nG
        T = np.zeros((dim, dim), complex)
        Tinter = np.zeros((dim, dim), complex)
        for (l, lp, g, amp) in terms:
            gi = np.rint(np.linalg.solve(self.B.T, np.asarray(g, float))).astype(int)
            for c, Gp in enumerate(G):
                tgt = (Gp[0] + gi[0], Gp[1] + gi[1])
                r = idx.get(tgt)
                if r is None:
                    continue
                if l == lp:
                    T[l * self.nG + r, lp * self.nG + c] += amp
                else:
                    Tinter[l * self.nG + r, lp * self.nG + c] += amp
        self.T = T + Tinter + Tinter.conj().T
        if not np.allclose(self.T, self.T.conj().T):
            raise ValueError("intralayer terms must be listed for both +g and -g (Hermiticity)")
        self.dim = dim

    # ---------------------------------------------------------------- k helpers
    def frac_to_cart(self, f):
        return np.atleast_2d(f) @ self.B

    @property
    def points(self):
        """High-symmetry points of the moire BZ (cartesian, 1/nm)."""
        b1, b2 = self.B
        return {"G": np.zeros(2), "M": 0.5 * b1, "K": (b1 + b2) / 3, "K'": (2 * b1 - b2) / 3,
                "kappa+": self.kappas[0], "kappa-": self.kappas[-1]}

    def _kin(self, k):
        out = []
        for kap in self.kappas:
            q = k[:, None, :] + self.G[None] - kap
            out.append(q)
        return np.concatenate(out, 1)                      # (nk, dim, 2)

    def hamiltonian(self, k):
        """H at cartesian momenta k (nk, 2) in 1/nm -> (nk, dim, dim)."""
        k = np.atleast_2d(np.asarray(k, float))
        q = self._kin(k)
        diag = self.sign * HB2M / self.mass * (q ** 2).sum(-1)
        H = np.broadcast_to(self.T, (len(k), self.dim, self.dim)).copy()
        i = np.arange(self.dim)
        H[:, i, i] += diag
        return H

    def velocity(self, k):
        """dH/dk_x, dH/dk_y (diagonal in the plane-wave basis)."""
        q = self._kin(np.atleast_2d(np.asarray(k, float)))
        return self.sign * 2 * HB2M / self.mass * q            # (nk, dim, 2)

    def eig(self, k):
        E, V = np.linalg.eigh(self.hamiltonian(k))
        return E[:, ::-1], V[:, :, ::-1]                  # sorted from the TOP (valence convention)

    def energies(self, k, nbands=None):
        E = self.eig(k)[0]
        return E if nbands is None else E[:, :nbands]

    # ------------------------------------------------------------------ bands
    def bands(self, path=("kappa+", "G", "M", "kappa-"), n=100, nbands=6, n_jobs=None):
        """Bands along a path of point names (or cartesian points). Returns
        (s, E, ticks, labels) with E sorted from the top band down."""
        pts = [self.points[p] if isinstance(p, str) else np.asarray(p, float) for p in path]
        ks, s, ticks, s0 = [], [], [0.0], 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            t = np.linspace(0, 1, n, endpoint=False)
            seg = a + t[:, None] * (b - a)
            ks.append(seg)
            s.append(s0 + t * np.linalg.norm(b - a))
            s0 += np.linalg.norm(b - a)
            ticks.append(s0)
        ks.append(pts[-1][None]); s.append([s0])
        k = np.concatenate(ks)
        parts = split(k, chunks_for(len(k), n_jobs))
        E = np.concatenate(pmap(lambda kk: self.energies(kk, nbands), parts, n_jobs))
        labels = [p if isinstance(p, str) else "" for p in path]
        return np.concatenate(s), E, ticks, labels

    def plot_bands(self, path=("kappa+", "G", "M", "kappa-"), n=100, nbands=6, ax=None, **kw):
        import matplotlib.pyplot as plt
        s, E, ticks, labels = self.bands(path, n, nbands)
        ax = ax or plt.gca()
        ax.plot(s, E, **({"color": "k", "lw": 1.2} | kw))
        ax.set_xticks(ticks)
        ax.set_xticklabels([{"G": r"$\gamma$", "kappa+": r"$\kappa_+$", "kappa-": r"$\kappa_-$"}.get(l, l)
                            for l in labels])
        for t in ticks:
            ax.axvline(t, color="0.85", lw=0.8, zorder=0)
        ax.set_xlim(ticks[0], ticks[-1])
        ax.set_ylabel("E (meV)")
        return ax

    def kmesh(self, nk):
        i = (np.arange(nk) + 0.5) / nk
        f = np.array(np.meshgrid(i, i, indexing="ij")).reshape(2, -1).T
        return f @ self.B

    def bandwidths(self, nbands=3, nk=24):
        E = self.energies(self.kmesh(nk), nbands)
        return E.max(0) - E.min(0)

    def gaps(self, nbands=3, nk=24):
        """Gaps between consecutive bands counted from the top (meV, negative = overlap)."""
        E = self.energies(self.kmesh(nk), nbands + 1)
        return E[:, :-1].min(0) - E[:, 1:].max(0)

    # --------------------------------------------------------------- topology
    def _curv(self, k, nbands):
        E, V = self.eig(k)
        v = self.velocity(k)
        out = np.zeros((len(k), nbands))
        for i in range(len(k)):
            U = V[i]
            vx = (U.conj().T * v[i, :, 0]) @ U
            vy = (U.conj().T * v[i, :, 1]) @ U
            dE = E[i][:, None] - E[i][None, :]
            np.fill_diagonal(dE, np.inf)
            Om = -2 * np.imag(vx[:nbands] * vy[:, :nbands].T) / dE[:nbands] ** 2
            out[i] = Om.sum(1)
        return out

    def berry_curvature(self, k, nbands=2):
        """Berry curvature (nm^2) of the top ``nbands`` bands at cartesian k."""
        return self._curv(np.atleast_2d(k), nbands)

    def chern_numbers(self, nbands=3, nk=24, n_jobs=None):
        """Chern numbers of the top ``nbands`` bands (gauge-invariant Kubo
        formula integrated on an nk x nk mesh; converge with nk)."""
        k = self.kmesh(nk)
        parts = split(k, chunks_for(len(k), n_jobs))
        Om = np.concatenate(pmap(lambda kk: self._curv(kk, nbands), parts, n_jobs))
        area = abs(np.linalg.det(self.B))
        return Om.sum(0) * area / len(k) / (2 * np.pi)

    # ------------------------------------------------------- tight-binding fit
    def hubbard_hoppings(self, band=0, nk=18, shells=3):
        """Fourier transform of one band into triangular-lattice hoppings:
        E(k) = sum_R t(R) e^{i k.R}. Returns [(|R|/aM, t)] for the first
        ``shells`` neighbour shells (t real for the C3-symmetric bands here)."""
        k = self.kmesh(nk)
        E = self.energies(k, band + 1)[:, band]
        out = {}
        for n1 in range(-3, 4):
            for n2 in range(-3, 4):
                R = n1 * self.A[0] + n2 * self.A[1]
                t = np.mean(E * np.exp(-1j * k @ R))
                d = round(np.linalg.norm(R) / self.aM, 4)
                out.setdefault(d, []).append(t)
        ds = sorted(d for d in out if d > 0)[:shells]
        return [(d, float(np.mean(out[d]).real)) for d in ds], float(np.mean(out[0.0]).real)


# =============================================================================
# Wu, Lovorn, Tutuc & MacDonald (2018): TMD heterobilayer -> Hubbard model
# =============================================================================
class TMDHeterobilayer(PlaneWaveContinuum):
    """Valence-band moire Hamiltonian of a TMD heterobilayer (single valley):

        H = -hbar^2 k^2 / (2 m*) + Delta(r),   Delta(r) = sum_j V(b_j) e^{i b_j . r}
        V(b_1) = V e^{i psi}, V(b_j) fixed by C3 and V(-b) = V(b)*.

    Defaults: WSe2 on MoSe2, AA stacking, (V, psi) = (6.6 meV, -94 deg),
    m* = 0.35 m_e, a0 = 0.332 nm. The moire period is
    aM = a0 / sqrt(theta^2 + delta^2) with lattice mismatch ``delta``
    (delta = 0 for WSe2/MoSe2 as in the paper; ~0.039 for WSe2/MoS2).
    Presets: ``TMDHeterobilayer.preset("WSe2/MoS2", theta=0)``.
    """
    PRESETS = {"WSe2/MoSe2": dict(V=6.6, psi=-94.0, mass=0.35, a0=0.332, delta=0.0),
               "WSe2/MoS2": dict(V=5.1, psi=-71.0, mass=0.35, a0=0.332, delta=0.039)}

    def __init__(self, theta=2.0, V=6.6, psi=-94.0, mass=0.35, a0=0.332, delta=0.0, shells=4.0, aM=None):
        th = np.radians(theta)
        aM = aM if aM is not None else a0 / np.hypot(th, delta)
        self.theta, self.V, self.psi = theta, V, np.radians(psi)
        b = 4 * np.pi / (np.sqrt(3) * aM)
        bv = [_rot(np.array([b, 0.0]), j * np.pi / 3) for j in range(6)]
        amp = [V * np.exp(1j * self.psi * (1 if j % 2 == 0 else -1)) for j in range(6)]
        terms = [(0, 0, bv[j], amp[j]) for j in range(6)]
        super().__init__(aM, mass, [np.zeros(2)], terms, sign=-1, shells=shells)

    @classmethod
    def preset(cls, name, theta=2.0, **kw):
        return cls(theta=theta, **(cls.PRESETS[name] | kw))

    def potential(self, r):
        """Delta(r) in meV at points r (n, 2) in nm."""
        r = np.atleast_2d(r)
        return sum((self.V * np.exp(1j * self.psi * (1 if j % 2 == 0 else -1)) *
                    np.exp(1j * r @ self.bvec[j])) for j in range(6)).real

    def harmonic(self):
        """Harmonic approximation around the potential maximum (where holes
        sit): returns dict(r_max, Delta_max, hbar_omega, a_W) (meV, nm)."""
        from scipy.optimize import minimize
        best = None
        for f1 in np.linspace(0, 1, 7, endpoint=False):
            for f2 in np.linspace(0, 1, 7, endpoint=False):
                r0 = f1 * self.A[0] + f2 * self.A[1]
                res = minimize(lambda r: -self.potential(r)[0], r0, method="BFGS")
                if best is None or res.fun < best.fun:
                    best = res
        r0 = best.x
        h = 1e-3 * self.aM
        f = lambda dx, dy: self.potential(r0 + np.array([dx, dy]))[0]
        kxx = -(f(h, 0) - 2 * f(0, 0) + f(-h, 0)) / h ** 2
        kyy = -(f(0, h) - 2 * f(0, 0) + f(0, -h)) / h ** 2
        kap = 0.5 * (kxx + kyy)                                  # meV / nm^2
        hw = np.sqrt(kap * 2 * HB2M / self.mass)
        aW = (2 * HB2M / (self.mass * kap)) ** 0.25
        return dict(r_max=r0, Delta_max=-best.fun, hbar_omega=hw, a_W=aW)

    def hubbard(self, eps=10.0, nk=18):
        """Generalised triangular-lattice Hubbard parameters of the top band:
        hoppings t1, t2, t3 (Fourier fit, meV) and the on-site U0 estimated
        from the harmonic-oscillator Wannier function,
        U0 = sqrt(pi/2) e^2 / (eps a_W)."""
        hops, e0 = self.hubbard_hoppings(0, nk, 3)
        hm = self.harmonic()
        U0 = np.sqrt(np.pi / 2) * COULOMB / (eps * hm["a_W"])
        return dict(t=[t for _, t in hops], distances=[d for d, _ in hops], onsite=e0, U0=U0,
                    a_W=hm["a_W"], hbar_omega=hm["hbar_omega"], aM=self.aM)

    def to_model(self, nk=18, shells=3):
        """The fitted triangular-lattice tight-binding model (hopforge Model,
        spinless; use model.make_spinful() for the spin-valley doublet)."""
        hops, e0 = self.hubbard_hoppings(0, nk, shells)
        lat = Lattice(self.A, [("M", [0, 0])], coords="cartesian")
        m = Model(lat, name=f"TMD moire Hubbard ({self.theta} deg)")
        m.add_onsite(e0, "M")
        for sh, (d, t) in enumerate(hops, 1):
            m.add_neighbors(t, sh)
        return m


# =============================================================================
# Wu, Lovorn, Tutuc, Martin & MacDonald (2019): twisted TMD homobilayer
# =============================================================================
class TMDHomobilayer(PlaneWaveContinuum):
    """+K-valley valence-band Hamiltonian of a twisted TMD homobilayer:

        H = [[ -hbar^2 (k - kappa_+)^2 / 2m* + Delta_b(r),  Delta_T(r) ],
             [ Delta_T^dagger(r), -hbar^2 (k - kappa_-)^2 / 2m* + Delta_t(r) ]]
        Delta_{b,t}(r) = 2V sum_{j=1,3,5} cos(g_j . r +- psi)
        Delta_T(r)     = w (1 + e^{-i g_2 . r} + e^{-i g_3 . r})

    Defaults: twisted bilayer MoTe2, (V, psi, w) = (8 meV, -89.6 deg, -8.5 meV),
    m* = 0.62 m_e, a0 = 0.3472 nm, aM = a0/theta. The -K valley is the time
    reversal partner (opposite Chern numbers). ``Vz`` adds a layer potential
    (+-Vz/2, perpendicular electric field).
    """

    def __init__(self, theta=1.2, V=8.0, psi=-89.6, w=-8.5, mass=0.62, a0=0.3472, Vz=0.0, shells=4.0):
        aM = a0 / np.radians(theta)
        self.theta = theta
        b = 4 * np.pi / (np.sqrt(3) * aM)
        g = [None] + [_rot(np.array([b, 0.0]), (j - 1) * np.pi / 3) for j in range(1, 7)]
        ps = np.radians(psi)
        terms = []
        for l, sgn in ((0, +1), (1, -1)):
            for j in (1, 3, 5):
                terms.append((l, l, g[j], V * np.exp(1j * sgn * ps)))
                terms.append((l, l, -g[j], V * np.exp(-1j * sgn * ps)))
            terms.append((l, l, np.zeros(2), sgn * Vz / 2))
        # <b, G'+q| Delta_T |t, G'> with Delta_T ~ e^{-i g r}: q = -g
        for q in (np.zeros(2), -g[2], -g[3]):
            terms.append((0, 1, q, w))
        kp = (g[6] + g[1]) / 3
        km = (g[1] + g[2]) / 3
        super().__init__(aM, mass, [kp, km], terms, sign=-1, shells=shells)

    def layer_polarisation(self, k, band=0):
        """<tau_z> (bottom minus top weight) of a band at cartesian k."""
        E, V = self.eig(np.atleast_2d(k))
        w = np.abs(V[:, :, band]) ** 2
        return w[:, :self.nG].sum(1) - w[:, self.nG:].sum(1)


# =============================================================================
# lattice models
# =============================================================================
def rhombohedral_graphene(N=3, t=1.0, tperp=0.3, bias=0.0, c=1.36):
    """(Dirac valleys at fractional K = (2/3, 1/3), K' = (1/3, 2/3).)
    ABC (rhombohedral) N-layer graphene, nearest-neighbour model with the
    vertical dimer coupling ``tperp`` between B_l and A_{l+1}. ``bias`` is the
    total potential difference between the outer layers (linear profile).

    Low-energy bands follow the chiral decomposition of Min & MacDonald:
    E ~ +-(v k)^N / tperp^(N-1) near K (one chiral pseudospin doublet with
    Berry phase N pi); a bias opens a gap with valley Chern numbers +-N/2.
    Length unit: in-plane lattice constant."""
    S3 = np.sqrt(3)
    a1, a2 = np.array([1.0, 0, 0]), np.array([0.5, S3 / 2, 0])
    orbs = []
    for l in range(N):
        sh = l / 3.0
        fa = np.array([sh % 1, sh % 1]); fb = np.array([(sh + 1 / 3) % 1, (sh + 1 / 3) % 1])
        orbs.append((f"A{l + 1}", fa[0] * a1 + fa[1] * a2 + [0, 0, l * c]))
        orbs.append((f"B{l + 1}", fb[0] * a1 + fb[1] * a2 + [0, 0, l * c]))
    m = Model(Lattice([a1, a2], orbs, coords="cartesian"), name=f"ABC {N}-layer graphene")
    for l in range(1, N + 1):
        m.add_neighbors(-t, 1, [f"A{l}"], [f"B{l}"])
        if N > 1:
            m.add_onsite(bias * ((l - 1) / (N - 1) - 0.5), [f"A{l}", f"B{l}"])
    P = m.lat.positions
    names = m.lat.names
    for l in range(1, N):
        i, j = names.index(f"B{l}"), names.index(f"A{l + 1}")
        d = (P[i] - P[j])[:2]
        R = np.rint(np.linalg.solve(np.array([a1[:2], a2[:2]]).T, d)).astype(int)
        m.add_hopping(tperp, i, j, [int(R[0]), int(R[1])])
    return set_points(m, HEX_POINTS, ["G", "K", "M", "G"])


def graphene_qah(t=1.0, lr=0.05, M=0.1):
    """Graphene with Rashba SOC ``lr`` and exchange field ``M``: a quantum
    anomalous Hall insulator with Chern number C = 2 (Qiao et al. 2010;
    Tse, Qiao, Yao, MacDonald & Niu 2011)."""
    from .models.library import kane_mele
    m = kane_mele(t=t, lso=0.0, lr=lr)
    m.add_exchange(M, (0, 0, 1))
    m.name = "graphene QAH (Rashba + exchange)"
    return m


def rashba_2deg(t=1.0, alpha=0.1):
    """Low-density Rashba electron gas on a square lattice (continuum limit
    near the band bottom E = -4t)."""
    from .models.library import rashba_square
    m = rashba_square(t=t, alpha=alpha)
    m.name = "Rashba 2DEG"
    return m


# =============================================================================
# tools
# =============================================================================
def valley_chern(model, occ, valley, radius=0.35, n=161):
    """Integral of the occupied-band Berry curvature over a disk of radius
    ``radius`` (fraction of |b1|) around ``valley`` (fractional k). For a
    gapped multilayer graphene valley this approaches +-N/2."""
    B = model.lat.reciprocal[:2, :2]
    kv = np.asarray(valley, float) @ B
    r = radius * np.linalg.norm(B[0])
    x = np.linspace(-r, r, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    inside = X ** 2 + Y ** 2 <= r ** 2
    kc = np.stack([X[inside], Y[inside]], 1) + kv
    kf = np.linalg.solve(B.T, kc.T).T
    from .kubo import _band_resolved
    E, Om = _band_resolved(model, kf, 0, 1)
    dA = (x[1] - x[0]) ** 2
    return float(Om[:, :occ].sum() * dA / (2 * np.pi))


def sinova_spin_hall(mu=-3.6, alpha=0.1, nk=600, t=1.0, n_jobs=None):
    """Intrinsic spin Hall conductivity of the Rashba 2DEG (Kubo), returned in
    units of the universal value e/(8 pi) (Sinova et al. 2004). Valid when
    both Rashba bands are occupied and the density is low (continuum limit).
    hopforge spin Hall output (hbar/e)(e^2/h) with s = sigma/2: e/8pi = 1/4."""
    from .kubo import spin_hall
    m = rashba_2deg(t, alpha)
    r = spin_hall(m, mu=mu, nk=nk, n_jobs=n_jobs)
    return float(np.abs(r.sigma[0]) / 0.25)


# make the lattice models available through hf.models.get / list_models
def _register():
    from .models.library import REGISTRY
    from .models.registry import META
    for name, f, dim, desc in [
        ("rhombohedral_graphene", rhombohedral_graphene, 2, "ABC N-layer graphene, E ~ k^N (Min & MacDonald 2008)"),
        ("graphene_qah", graphene_qah, 2, "graphene + Rashba + exchange, QAH C = 2 (Tse, Qiao, Yao, MacDonald & Niu 2011)"),
        ("rashba_2deg", rashba_2deg, 2, "Rashba 2DEG, spin Hall e/8pi (Sinova et al. 2004)")]:
        REGISTRY.setdefault(name, (f, dim, desc))


_register()
