"""Library of published tight-binding models.

Every function returns a ready :class:`hopforge.Model` with special k-points
and a default band path. Unless stated otherwise the convention is
``H = -t sum <ij> c_i^dag c_j`` (t > 0) and lattice constant 1.
"""
from __future__ import annotations

import numpy as np

from .. import operators as ops
from ..model import Lattice, Model
from ._util import (HEX, HEX_POINTS, S3, SQ_POINTS, LS_matrix, cart_points, chirality,
                    set_points, spin_diag)

sx, sy, sz, s0 = ops.sx, ops.sy, ops.sz, ops.s0
LINE_POINTS = {"-X": [-0.5], "G": [0.0], "X": [0.5]}


# ===========================================================================
# 1D
# ===========================================================================
def chain(t=1.0, eps=0.0):
    """Monatomic chain: E(k) = eps - 2t cos k."""
    m = Model(Lattice([[1.0]], [("A", [0.0])]), name="chain")
    m.add_onsite(eps)
    m.add_hopping(-t, "A", "A", [1])
    return set_points(m, LINE_POINTS, ["-X", "G", "X"])


def ssh(v=0.5, w=1.0):
    """Su-Schrieffer-Heeger chain (PRL 42, 1698 (1979)). Intra-cell hopping v,
    inter-cell w; topological (winding 1) for |w| > |v|. A at 0, B at 1/2."""
    m = Model(Lattice([[1.0]], [("A", [0.0]), ("B", [0.5])]), name="SSH")
    m.add_hopping(v, "A", "B", [0])
    m.add_hopping(w, "B", "A", [1])
    m.info["chiral"] = np.diag([1.0, -1.0])
    return set_points(m, LINE_POINTS, ["-X", "G", "X"])


def rice_mele(v=0.5, w=1.0, delta=0.2):
    """Rice-Mele chain (PRL 49, 1455 (1982)): SSH plus staggered potential."""
    m = ssh(v, w)
    m.name = "Rice-Mele"
    m.add_onsite(delta, "A").add_onsite(-delta, "B")
    m.info.pop("chiral", None)
    return m


def kitaev_chain(mu=0.5, t=1.0, delta=1.0):
    """Kitaev p-wave chain in Bogoliubov-de Gennes form (Phys.-Usp. 44, 131
    (2001)). Basis (e, h). Topological for |mu| < 2|t|; chiral operator tau_x."""
    m = Model(Lattice([[1.0]], [("e", [0.0]), ("h", [0.0])]), name="Kitaev chain")
    m.add_onsite(-mu, "e").add_onsite(mu, "h")
    m.add_hopping(-t, "e", "e", [1]).add_hopping(t, "h", "h", [1])
    m.add_hopping(delta, "e", "h", [1]).add_hopping(-delta, "e", "h", [-1])
    m.info["chiral"] = np.array([[0, 1], [1, 0]], float)
    return set_points(m, LINE_POINTS, ["-X", "G", "X"])


def creutz_ladder(t=1.0, theta=np.pi / 2, mperp=0.0):
    """Creutz ladder (PRL 83, 2636 (1999)): flat bands at +-2t for theta = pi/2,
    mperp = 0 (Aharonov-Bohm caging)."""
    lat = Lattice([[1.0, 0.0]], [("A", [0, 0.5]), ("B", [0, -0.5])], coords="cartesian")
    m = Model(lat, name="Creutz ladder")
    m.add_hopping(-t * np.exp(1j * theta), "A", "A", [1])
    m.add_hopping(-t * np.exp(-1j * theta), "B", "B", [1])
    m.add_hopping(-t, "A", "B", [1]).add_hopping(-t, "B", "A", [1])
    if mperp:
        m.add_hopping(-mperp, "A", "B", [0])
    return set_points(m, LINE_POINTS, ["-X", "G", "X"])


# ===========================================================================
# 2D
# ===========================================================================
def square(t=1.0, t2=0.0, eps=0.0):
    """Square lattice with NN t and NNN t2."""
    m = Model(Lattice([[1, 0], [0, 1]], [("A", [0, 0])]), name="square")
    m.add_onsite(eps)
    m.add_neighbors(-t, 1)
    if t2:
        m.add_neighbors(-t2, 2)
    return set_points(m, SQ_POINTS, ["G", "X", "M", "G"])


def triangular(t=1.0):
    """Triangular lattice."""
    m = Model(Lattice(HEX, [("A", [0, 0])]), name="triangular")
    m.add_neighbors(-t, 1)
    return set_points(m, HEX_POINTS, ["G", "K", "M", "G"])


def honeycomb(t=1.0, delta=0.0, t2=0.0):
    """Graphene / honeycomb lattice (Wallace 1947). ``delta`` = +-delta staggered
    (hBN-like) potential, ``t2`` = NNN hopping. Dirac points at K, K'."""
    m = Model(Lattice(HEX, [("A", [1 / 3, 1 / 3]), ("B", [2 / 3, 2 / 3])]), name="honeycomb")
    m.add_neighbors(-t, 1)
    if t2:
        m.add_neighbors(-t2, 2)
    if delta:
        m.add_onsite(delta, "A").add_onsite(-delta, "B")
    return set_points(m, HEX_POINTS, ["G", "K", "M", "G"])


graphene = honeycomb


def haldane(t=1.0, t2=0.1, phi=np.pi / 2, M=0.0):
    """Haldane model (PRL 61, 2015 (1988)): Chern insulator with C = +-1 for
    |M| < 3 sqrt(3) |t2 sin phi|."""
    m = honeycomb(t)
    m.name = "Haldane"
    m.add_onsite(M, "A").add_onsite(-M, "B")
    nn = 1 / S3
    for i, j, R, d in m.neighbors(2):
        nu = chirality(m, i, j, R, nn)
        m.add_hopping(t2 * np.exp(1j * nu * phi), i, j, R)
    return m


def kane_mele(t=1.0, lso=0.06, lv=0.0, lr=0.0):
    """Kane-Mele model (PRL 95, 226801 & 146802 (2005)). Quantum spin Hall
    (Z2 = 1) for lv < 3 sqrt(3) lso when lr = 0."""
    m = Model(Lattice(HEX, [("A", [1 / 3, 1 / 3]), ("B", [2 / 3, 2 / 3])]), spinful=True,
              name="Kane-Mele")
    m.add_neighbors(-t, 1)
    m.add_onsite(lv, "A").add_onsite(-lv, "B")
    nn = 1 / S3
    for i, j, R, d in m.neighbors(2):
        nu = chirality(m, i, j, R, nn)
        m.add_hopping(1j * lso * nu * sz, i, j, R)
    if lr:
        m.add_rashba(lr, 1)
    return set_points(m, HEX_POINTS, ["G", "K", "M", "G"])


def qwz(m_=1.0, t=1.0):
    """Qi-Wu-Zhang Chern insulator (PRB 74, 085308 (2006)):
    H = t sin kx sx + t sin ky sy + (m + t cos kx + t cos ky) sz.
    C = +-1 for 0 < |m| < 2t."""
    m = Model(Lattice([[1, 0], [0, 1]], [("a", [0, 0]), ("b", [0, 0])]), name="QWZ")
    m.add_onsite_matrix(m_ * sz, ["a", "b"])
    m.add_hopping_matrix(t * (sz / 2 - 0.5j * sx), ["a", "b"], ["a", "b"], [1, 0])
    m.add_hopping_matrix(t * (sz / 2 - 0.5j * sy), ["a", "b"], ["a", "b"], [0, 1])
    return set_points(m, SQ_POINTS, ["G", "X", "M", "G"])


def bhz(A=1.0, B=1.0, C=0.0, D=0.0, M=1.0):
    """Bernevig-Hughes-Zhang lattice model of HgTe quantum wells (Science 314,
    1757 (2006)). Orbitals E (s-like) and H (p-like), spinful.
    Quantum spin Hall for 0 < M/B < 8 (M/B != 4)."""
    lat = Lattice([[1, 0], [0, 1]], [("E", [0, 0]), ("H", [0, 0])])
    m = Model(lat, spinful=True, name="BHZ")
    on = (C - 4 * D) * s0 + (M - 4 * B) * sz
    m.add_onsite_matrix(spin_diag(on, on), ["E", "H"])
    tx_up = D * s0 + B * sz - 0.5j * A * sx
    tx_dn = D * s0 + B * sz + 0.5j * A * sx
    ty = D * s0 + B * sz - 0.5j * A * sy
    m.add_hopping_matrix(spin_diag(tx_up, tx_dn), ["E", "H"], ["E", "H"], [1, 0])
    m.add_hopping_matrix(spin_diag(ty, ty), ["E", "H"], ["E", "H"], [0, 1])
    return set_points(m, SQ_POINTS, ["G", "X", "M", "G"])


def kagome(t=1.0, lso=0.0):
    """Kagome lattice: flat band at E = 2t. With ``lso`` the spinful
    Guo-Franz intrinsic SOC i lso nu sz on NN bonds (PRB 80, 113102 (2009))."""
    lat = Lattice(HEX, [("A", [0, 0]), ("B", [0.5, 0]), ("C", [0, 0.5])])
    m = Model(lat, spinful=bool(lso), name="kagome")
    m.add_neighbors(-t, 1)
    if lso:
        for i, j, R, d in m.neighbors(1):
            nu = chirality(m, i, j, R, 0.5)
            m.add_hopping(1j * lso * nu * sz, i, j, R)
    return set_points(m, HEX_POINTS, ["G", "K", "M", "G"])


def lieb(t=1.0, delta=0.0):
    """Lieb lattice (CuO2-plane-like): flat band at E = 0 touching a Dirac cone
    at M. ``delta`` shifts the corner site."""
    lat = Lattice([[1, 0], [0, 1]], [("Cu", [0, 0]), ("Ox", [0.5, 0]), ("Oy", [0, 0.5])])
    m = Model(lat, name="Lieb")
    m.add_neighbors(-t, 1)
    if delta:
        m.add_onsite(delta, "Cu")
    return set_points(m, SQ_POINTS, ["G", "X", "M", "G"])


def checkerboard(t=1.0, tp=1.0):
    """Checkerboard (planar pyrochlore) lattice: NN t and crossed-plaquette
    NNN tp (A along x, B along y). Flat band at E = 2t when tp = t."""
    lat = Lattice([[1, 0], [0, 1]], [("A", [0, 0]), ("B", [0.5, 0.5])])
    m = Model(lat, name="checkerboard")
    m.add_neighbors(-t, 1)
    m.add_hopping(-tp, "A", "A", [1, 0]).add_hopping(-tp, "B", "B", [0, 1])
    return set_points(m, SQ_POINTS, ["G", "X", "M", "G"])


def dice(t=1.0, alpha=1.0):
    """alpha-T3 / dice lattice (Raoux et al., PRL 112, 026402 (2014)): hub site
    H coupled to rim A with t and to rim B with alpha*t. Flat band at E = 0."""
    lat = Lattice(HEX, [("H", [0, 0]), ("A", [1 / 3, 1 / 3]), ("B", [2 / 3, 2 / 3])])
    m = Model(lat, name="dice")
    m.add_neighbors(-t, 1, ["H"], ["A"])
    m.add_neighbors(-alpha * t, 1, ["H"], ["B"])
    return set_points(m, HEX_POINTS, ["G", "K", "M", "G"])


def bilayer_graphene(t=1.0, tperp=0.13, U=0.0, c=1.36):
    """AB (Bernal) bilayer graphene; B1-A2 dimer coupling ``tperp`` and layer
    bias U (opens a gap). Length unit = graphene lattice constant."""
    a1, a2 = np.array([1, 0, 0.0]), np.array([0.5, S3 / 2, 0])
    f = lambda u, v, z: u * a1 + v * a2 + np.array([0, 0, z])
    orbs = [("A1", f(1 / 3, 1 / 3, 0)), ("B1", f(2 / 3, 2 / 3, 0)),
            ("A2", f(2 / 3, 2 / 3, c)), ("B2", f(0, 0, c))]
    m = Model(Lattice([a1, a2], orbs, coords="cartesian"), name="AB bilayer graphene")
    m.add_neighbors(-t, 1, ["A1"], ["B1"])
    m.add_neighbors(-t, 1, ["A2"], ["B2"])
    m.add_hopping(tperp, "B1", "A2", [0, 0])
    m.add_onsite(U / 2, ["A1", "B1"]).add_onsite(-U / 2, ["A2", "B2"])
    return set_points(m, HEX_POINTS, ["G", "K", "M", "G"])


def rashba_square(t=1.0, alpha=0.3, J=0.0, direction=(0, 0, 1)):
    """Square lattice with Rashba SOC and optional exchange J (2D Rashba
    ferromagnet: anomalous Hall, Edelstein, spin-orbit torque physics)."""
    m = Model(Lattice([[1, 0], [0, 1]], [("A", [0, 0])]), spinful=True, name="Rashba square")
    m.add_neighbors(-t, 1)
    m.add_rashba(alpha, 1)
    if J:
        m.add_exchange(J, direction)
    return set_points(m, SQ_POINTS, ["G", "X", "M", "G"])


def altermagnet_dwave(t=1.0, t1=0.6, t2=0.2, J=0.5, direction=(0, 0, 1), lso=0.0):
    """Minimal d-wave altermagnet (two sublattices with opposite moments and
    anisotropic NNN hopping, cf. Smejkal, Sinova & Jungwirth PRX 12, 031042
    (2022)). Spin splitting vanishes along G-M and is maximal along G-X / G-Y.
    Optional Rashba ``lso``."""
    lat = Lattice([[1, 0], [0, 1]], [("A", [0, 0]), ("B", [0.5, 0.5])])
    m = Model(lat, spinful=True, name="d-wave altermagnet")
    m.add_neighbors(-t, 1)
    m.add_hopping(-t1, "A", "A", [1, 0]).add_hopping(-t2, "A", "A", [0, 1])
    m.add_hopping(-t2, "B", "B", [1, 0]).add_hopping(-t1, "B", "B", [0, 1])
    m.add_exchange(J, direction, "A")
    m.add_exchange(-J, direction, "B")
    if lso:
        m.add_rashba(lso, 1)
    return set_points(m, SQ_POINTS, ["G", "X", "M", "Y", "G", "M"])


def hofstadter(p=1, q=3, t=1.0):
    """Harper-Hofstadter model (PRB 14, 2239 (1976)): square lattice with flux
    p/q per plaquette (Landau gauge, q x 1 magnetic supercell)."""
    base = square(t)
    m = base.supercell([q, 1])
    flux = 2 * np.pi * p / q
    m.apply_peierls(lambda ri, rj: flux * 0.5 * (ri[0] + rj[0]) * (rj[1] - ri[1]))
    m.name = f"Hofstadter p/q={p}/{q}"
    return set_points(m, {"G": [0, 0], "X": [0.5, 0], "Y": [0, 0.5], "M": [0.5, 0.5]},
                      ["G", "X", "M", "Y", "G"])


def p_orbital_square(pps=1.0, ppp=-0.3, pps2=0.3, ppp2=-0.1, lso=0.0, J=0.0):
    """Square lattice of px, py, pz orbitals with Slater-Koster hopping: the
    standard toy model of the orbital Hall effect (Bhowal & Vignale PRB 103,
    195309 (2021); Go et al. PRL 121, 086602 (2018)). Optional on-site
    lso L.S (spinful) and exchange J along z. NNN bonds (pps2, ppp2) are what
    hybridise px and py and give a finite orbital Hall conductivity."""
    spinful = bool(lso) or bool(J)
    lat = Lattice([[1, 0], [0, 1]], [("X:px", [0, 0]), ("X:py", [0, 0]), ("X:pz", [0, 0])])
    m = Model(lat, spinful=spinful, name="p-orbital square")
    m.add_slater_koster({"pps": pps, "ppp": ppp}, 1)
    if pps2 or ppp2:
        m.add_slater_koster({"pps": pps2, "ppp": ppp2}, 2)
    if lso:
        m.add_onsite_matrix(LS_matrix(1, lso), ["X:px", "X:py", "X:pz"])
    if J:
        m.add_exchange(J, (0, 0, 1))
    return set_points(m, SQ_POINTS, ["G", "X", "M", "G"])


# ===========================================================================
# 3D
# ===========================================================================
def cubic(t=1.0):
    """Simple cubic lattice: E = -2t (cos kx + cos ky + cos kz)."""
    m = Model(Lattice(np.eye(3), [("A", [0, 0, 0])]), name="simple cubic")
    m.add_neighbors(-t, 1)
    return set_points(m, {"G": [0, 0, 0], "X": [0.5, 0, 0], "M": [0.5, 0.5, 0], "R": [0.5] * 3},
                      ["G", "X", "M", "G", "R", "X"])


def bcc(t=1.0):
    """Body-centred cubic lattice (conventional a = 1)."""
    A = 0.5 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], float)
    m = Model(Lattice(A, [("A", [0, 0, 0])]), name="bcc")
    m.add_neighbors(-t, 1)
    return cart_points(m, {"G": [0, 0, 0], "H": [0, 0, 1], "N": [0.5, 0.5, 0], "P": [0.25] * 3},
                       ["G", "H", "N", "G", "P", "H"])


_FCC = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
_FCC_PTS = {"G": [0, 0, 0], "X": [0, 1, 0], "W": [0.5, 1, 0], "K": [0.75, 0.75, 0],
            "L": [0.5, 0.5, 0.5], "U": [0.25, 1, 0.25]}


def fcc(t=1.0):
    """Face-centred cubic lattice (conventional a = 1)."""
    m = Model(Lattice(_FCC, [("A", [0, 0, 0])]), name="fcc")
    m.add_neighbors(-t, 1)
    return cart_points(m, _FCC_PTS, ["G", "X", "W", "L", "G", "K"])


def diamond_fkm(t=1.0, dt=0.4, lso=0.125):
    """Fu-Kane-Mele diamond-lattice topological insulator (PRL 98, 106803
    (2007)). NN hopping t (t + dt on the [111] bond), NNN SOC
    i (8 lso / a^2) s . (d1 x d2). Strong TI (1;111) for dt > 0."""
    lat = Lattice(_FCC, [("A", [0, 0, 0]), ("B", [0.25, 0.25, 0.25])], coords="cartesian")
    m = Model(lat, spinful=True, name="Fu-Kane-Mele diamond")
    nn = np.sqrt(3) / 4

    def tnn(d):
        return t + dt if np.allclose(d / np.linalg.norm(d), np.ones(3) / np.sqrt(3)) or \
            np.allclose(-d / np.linalg.norm(d), np.ones(3) / np.sqrt(3)) else t
    m.add_neighbors(tnn, 1)
    import itertools
    A, P = m.lat.vectors, m.lat.positions
    sv = [sx, sy, sz]
    for i, j, R, d in m.neighbors(2):
        rj = P[j] + R @ A
        done = False
        for Rk in itertools.product(range(-2, 3), repeat=3):
            for k in range(2):
                rk = P[k] + np.array(Rk) @ A
                d1, d2 = rk - P[i], rj - rk
                if abs(np.linalg.norm(d1) - nn) < 1e-6 and abs(np.linalg.norm(d2) - nn) < 1e-6:
                    c = np.cross(d1, d2)
                    m.add_hopping(1j * 8 * lso * sum(c[a] * sv[a] for a in range(3)), i, j, R)
                    done = True
                    break
            if done:
                break
    return cart_points(m, _FCC_PTS, ["G", "X", "W", "L", "G", "K"])


def wilson_dirac_ti(A=1.0, B=1.0, M=1.0):
    """Four-band cubic-lattice 3D topological insulator (Bi2Se3-type
    Wilson-Dirac model, Zhang et al. Nat. Phys. 5, 438 (2009) regularised):
    H = A sum_i sin k_i tau_x s_i + (M - 2B sum_i (1 - cos k_i)) tau_z.
    Strong TI (1;000) for 0 < M/B < 4."""
    lat = Lattice(np.eye(3), [("P1+", [0, 0, 0]), ("P2-", [0, 0, 0])])
    m = Model(lat, spinful=True, name="Wilson-Dirac TI")
    tx = np.array([[0, 1], [1, 0]], complex)
    tz = np.diag([1.0, -1.0]).astype(complex)
    G0 = np.kron(tz, s0)
    Gs = [np.kron(tx, s) for s in (sx, sy, sz)]
    m.add_onsite_matrix((M - 6 * B) * G0, ["P1+", "P2-"])
    for a in range(3):
        R = [0, 0, 0]
        R[a] = 1
        m.add_hopping_matrix(B * G0 - 0.5j * A * Gs[a], ["P1+", "P2-"], ["P1+", "P2-"], R)
    return set_points(m, {"G": [0, 0, 0], "X": [0.5, 0, 0], "M": [0.5, 0.5, 0], "R": [0.5] * 3},
                      ["G", "X", "M", "G", "R", "X"])


def weyl_minimal(t=1.0, m_=2.0):
    """Minimal two-band Weyl semimetal with two Weyl nodes at
    k = (0, 0, +-arccos(m - 2)) (1 < m < 3):
    H = t[sin kx sx + sin ky sy + (m - cos kx - cos ky - cos kz) sz]."""
    m = Model(Lattice(np.eye(3), [("a", [0, 0, 0]), ("b", [0, 0, 0])]), name="Weyl semimetal")
    m.add_onsite_matrix(t * m_ * sz, ["a", "b"])
    for a, s in enumerate((sx, sy, None)):
        R = [0, 0, 0]
        R[a] = 1
        T = -t * sz / 2 + (-0.5j * t * s if s is not None else 0)
        m.add_hopping_matrix(T, ["a", "b"], ["a", "b"], R)
    kz = np.arccos(np.clip(m_ - 2, -1, 1)) / (2 * np.pi)
    return set_points(m, {"G": [0, 0, 0], "W+": [0, 0, kz], "W-": [0, 0, -kz], "Z": [0, 0, 0.5],
                          "X": [0.5, 0, 0], "M": [0.5, 0.5, 0]},
                      ["Z", "W+", "G", "X", "M", "G"])


def pyrochlore(t=1.0):
    """Pyrochlore lattice (corner-sharing tetrahedra): two flat bands at E = 2t."""
    pos = [("A", [0, 0, 0]), ("B", [0, 0.25, 0.25]), ("C", [0.25, 0, 0.25]), ("D", [0.25, 0.25, 0])]
    m = Model(Lattice(_FCC, pos, coords="cartesian"), name="pyrochlore")
    m.add_neighbors(-t, 1)
    return cart_points(m, _FCC_PTS, ["G", "X", "W", "L", "G", "K"])


# ===========================================================================
# registry
# ===========================================================================
REGISTRY = {
    "chain": (chain, 1, "monatomic chain"),
    "ssh": (ssh, 1, "Su-Schrieffer-Heeger chain (winding number)"),
    "rice_mele": (rice_mele, 1, "Rice-Mele chain (Thouless pumping)"),
    "kitaev_chain": (kitaev_chain, 1, "Kitaev p-wave superconducting chain (Majoranas)"),
    "creutz_ladder": (creutz_ladder, 1, "Creutz ladder (flat bands, AB caging)"),
    "square": (square, 2, "square lattice"),
    "triangular": (triangular, 2, "triangular lattice"),
    "honeycomb": (honeycomb, 2, "graphene / hBN honeycomb"),
    "haldane": (haldane, 2, "Haldane Chern insulator"),
    "kane_mele": (kane_mele, 2, "Kane-Mele quantum spin Hall insulator"),
    "qwz": (qwz, 2, "Qi-Wu-Zhang Chern insulator"),
    "bhz": (bhz, 2, "Bernevig-Hughes-Zhang HgTe quantum well"),
    "kagome": (kagome, 2, "kagome lattice (+ Guo-Franz SOC)"),
    "lieb": (lieb, 2, "Lieb lattice (flat band)"),
    "checkerboard": (checkerboard, 2, "checkerboard / planar pyrochlore"),
    "dice": (dice, 2, "alpha-T3 / dice lattice"),
    "bilayer_graphene": (bilayer_graphene, 2, "AB bilayer graphene with bias"),
    "rashba_square": (rashba_square, 2, "Rashba (ferro)magnet on a square lattice"),
    "altermagnet_dwave": (altermagnet_dwave, 2, "minimal d-wave altermagnet"),
    "hofstadter": (hofstadter, 2, "Harper-Hofstadter butterfly"),
    "p_orbital_square": (p_orbital_square, 2, "p-orbital square lattice (orbital Hall)"),
    "cubic": (cubic, 3, "simple cubic"),
    "bcc": (bcc, 3, "body-centred cubic"),
    "fcc": (fcc, 3, "face-centred cubic"),
    "diamond_fkm": (diamond_fkm, 3, "Fu-Kane-Mele diamond topological insulator"),
    "wilson_dirac_ti": (wilson_dirac_ti, 3, "4-band Bi2Se3-type 3D TI"),
    "weyl_minimal": (weyl_minimal, 3, "two-node Weyl semimetal"),
    "pyrochlore": (pyrochlore, 3, "pyrochlore lattice (flat bands)"),
}


def list_models(dim=None):
    """Print the available models (optionally only one dimension)."""
    lines = []
    for name, (_, d, desc) in REGISTRY.items():
        if dim is None or d == dim:
            lines.append(f"  {d}D  {name:20s} {desc}")
    print("hopforge model library:\n" + "\n".join(lines))
    return [n for n, (_, d, _) in REGISTRY.items() if dim is None or d == dim]


def get(name, **params):
    """Build a model by name, e.g. ``get('haldane', t2=0.2)``."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; see list_models()")
    return REGISTRY[name][0](**params)


# ===========================================================================
# Bogoliubov-de Gennes construction
# ===========================================================================
def bdg(model, mu=0.0, delta=0.0):
    """Bogoliubov-de Gennes doubling of any model: electron block H - mu,
    hole block -(H - mu)^*, on-site pairing delta between e and h of every
    orbital (the spin-degenerate s-wave toy model of Kwant's superconductor
    tutorial; for spinful models use delta * i sigma_y explicitly).

    Orbitals are renamed ``site_e`` / ``site_h`` (orbital kinds are kept)."""
    from ..model import _Term
    names, pos = [], []
    for i, n in enumerate(model.lat.names):
        site, _, kind = n.partition(":")
        for tag in ("e", "h"):
            names.append(f"{site}_{tag}" + (f":{kind}" if kind else ""))
            pos.append(model.lat.positions[i])
    lat = Lattice(model.lat.vectors, list(zip(names, pos)), coords="cartesian")
    m = Model(lat, spinful=model.spinful, name=f"BdG[{model.name}]")
    ns = model.ns
    for i in range(model.norb):
        h = model._onsite[i] - mu * np.eye(ns)
        m._onsite[2 * i] = h
        m._onsite[2 * i + 1] = -h.conj()
        if delta:
            m.add_hopping(delta * np.eye(ns), 2 * i, 2 * i + 1, None)
    for t in model._terms:
        m._terms.append(_Term(2 * t.i, 2 * t.j, t.R.copy(), t.T.copy()))
        m._terms.append(_Term(2 * t.i + 1, 2 * t.j + 1, t.R.copy(), -t.T.conj()))
    m.special_points = dict(model.special_points)
    m.info = dict(model.info)
    m.info["bdg"] = dict(mu=mu, delta=delta)
    m._changed()
    return m


REGISTRY["bdg_square"] = (lambda mu=-3.0, delta=0.2: bdg(square(), mu, delta), 2,
                          "BdG s-wave superconductor on a square lattice")


# ===========================================================================
# community registration
# ===========================================================================
def register_model(name, dim, description=""):
    """Decorator that adds a model constructor to the library so it shows up in
    ``list_models()`` and ``get(name, ...)``::

        @hf.models.register_model("my_lattice", 2, "my favourite model (ref.)")
        def my_lattice(t=1.0):
            ...
            return model
    """
    def deco(func):
        if name in REGISTRY:
            raise KeyError(f"model {name!r} already registered")
        REGISTRY[name] = (func, dim, description)
        return func
    return deco


# ===========================================================================
# B. A. Bernevig and collaborators: higher-order, superconducting, Weyl-II,
# flat Chern bands (Bistritzer-MacDonald moire bands live in hopforge.moire)
# ===========================================================================
def _gammas_bbh():
    t1 = np.array([[0, 1], [1, 0]], complex)
    t2 = np.array([[0, -1j], [1j, 0]], complex)
    t3 = np.diag([1.0, -1.0]).astype(complex)
    G0 = np.kron(t3, s0)
    G1, G2, G3 = (-np.kron(t2, s) for s in (sx, sy, sz))
    G4 = np.kron(t1, s0)
    return G0, G1, G2, G3, G4


def bbh_quadrupole(gamma=0.5, lam=1.0, delta=0.0):
    """Benalcazar-Bernevig-Hughes quadrupole insulator (Science 357, 61 (2017)):
    H = (g + l cos kx) G4 + l sin kx G3 + (g + l cos ky) G2 + l sin ky G1 + d G0.
    Quadrupole phase (q_xy = 1/2, corner charges +-e/2) for |g/l| < 1.
    Four orbitals per cell (pi flux per plaquette); delta breaks chiral symmetry."""
    G0, G1, G2, G3, G4 = _gammas_bbh()
    lat = Lattice([[1, 0], [0, 1]], [(f"o{i}", [0, 0]) for i in range(4)])
    m = Model(lat, name="BBH quadrupole")
    o = [f"o{i}" for i in range(4)]
    m.add_onsite_matrix(gamma * (G4 + G2) + delta * G0, o)
    m.add_hopping_matrix(0.5 * lam * G4 - 0.5j * lam * G3, o, o, [1, 0])
    m.add_hopping_matrix(0.5 * lam * G2 - 0.5j * lam * G1, o, o, [0, 1])
    return set_points(m, SQ_POINTS, ["G", "X", "M", "G"])


def p_ip_superconductor(t=1.0, mu=-1.0, delta=0.5):
    """Spinless chiral p+ip superconductor on the square lattice (Read & Green,
    PRB 61, 10267 (2000); Bernevig & Hughes, ch. 16), BdG basis (e, h):
    xi = -2t (cos kx + cos ky) - mu,  Delta_k = delta (sin kx + i sin ky).
    Weak pairing |mu| < 4t: C = +-1 (sign flips at mu = 0); |mu| > 4t: C = 0."""
    lat = Lattice([[1, 0], [0, 1]], [("e", [0, 0]), ("h", [0, 0])])
    m = Model(lat, name="p+ip superconductor")
    m.add_onsite(-mu, "e").add_onsite(mu, "h")
    for R in ([1, 0], [0, 1]):
        m.add_hopping(-t, "e", "e", R).add_hopping(t, "h", "h", R)
    # delta sin kx  and  i delta sin ky in the e-h block
    m.add_hopping(delta / 2j, "e", "h", [1, 0]).add_hopping(-delta / 2j, "e", "h", [-1, 0])
    m.add_hopping(delta / 2, "e", "h", [0, 1]).add_hopping(-delta / 2, "e", "h", [0, -1])
    return set_points(m, SQ_POINTS, ["G", "X", "M", "G"])


def weyl_type2(t=1.0, m_=2.0, tilt=1.5):
    """Weyl semimetal with a tilt along z (type I for tilt < 1, type II with
    electron and hole pockets touching at the node for tilt > 1; Soluyanov,
    Gresch, Wang, Wu, Troyer, Dai & Bernevig, Nature 527, 495 (2015)):
    H = weyl_minimal + t * tilt * cos(kz) * 1."""
    m = weyl_minimal(t, m_)
    m.name = f"Weyl (tilt {tilt})"
    m.add_hopping_matrix(0.5 * t * tilt * np.eye(2), ["a", "b"], ["a", "b"], [0, 0, 1])
    return m


def checkerboard_chern(t=1.0, tp=0.3, tpp=-0.2, phi=np.pi / 4):
    """Flat-band Chern insulator on the checkerboard lattice (Sun, Gu, Katsura &
    Das Sarma, PRL 106, 236803 (2011); parameters of Sheng, Gu, Sun & Sheng,
    Nat. Commun. 2, 389 (2011): t' = 0.3, t'' = -0.2, phi = pi/4, gap/bandwidth
    ~ 27); the host of fractional Chern insulators (Regnault & Bernevig, PRX 1,
    021014 (2011)). NN -t e^{i phi s_x s_y}, NNN -/+t' (staggered), diagonal -t''."""
    lat = Lattice([[1, 0], [0, 1]], [("A", [0, 0]), ("B", [0.5, 0.5])])
    m = Model(lat, name="checkerboard Chern")
    # NN A -> B bonds d = (sx, sy)/2 with amplitude -t exp(i phi sx sy)
    for sxx in (1, -1):
        for syy in (1, -1):
            R = [0 if sxx > 0 else -1, 0 if syy > 0 else -1]
            m.add_hopping(-t * np.exp(1j * phi * sxx * syy), "A", "B", R)
    m.add_hopping(-tp, "A", "A", [1, 0]).add_hopping(tp, "A", "A", [0, 1])
    m.add_hopping(tp, "B", "B", [1, 0]).add_hopping(-tp, "B", "B", [0, 1])
    for o in ("A", "B"):
        m.add_hopping(-tpp, o, o, [1, 1]).add_hopping(-tpp, o, o, [1, -1])
    return set_points(m, SQ_POINTS, ["G", "X", "M", "G"])


REGISTRY.update({
    "bbh_quadrupole": (bbh_quadrupole, 2, "Benalcazar-Bernevig-Hughes quadrupole HOTI"),
    "p_ip_superconductor": (p_ip_superconductor, 2, "chiral p+ip superconductor (Read-Green)"),
    "weyl_type2": (weyl_type2, 3, "type-I/II Weyl semimetal (Soluyanov-Bernevig)"),
    "checkerboard_chern": (checkerboard_chern, 2, "flat Chern band (FCI host, Sun/Neupert/Regnault-Bernevig)"),
})
