"""Pauli matrices, orbital angular momentum matrices and Slater-Koster tables.

Real orbital labels understood by hopforge (the part after ``:`` in an
orbital name, e.g. ``"Mo:dxy"``):

    s
    px, py, pz
    dxy, dyz, dz2, dxz, dx2-y2

Everything here works in units hbar = 1, so spin is ``sigma / 2`` and
orbital angular momentum has integer eigenvalues.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Pauli matrices
# --------------------------------------------------------------------------
s0 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = (s0, sx, sy, sz)


def pauli(which):
    """Return a Pauli matrix: ``pauli(0..3)`` or ``pauli('x'|'y'|'z'|'0')``."""
    if isinstance(which, str):
        which = {"0": 0, "i": 0, "x": 1, "y": 2, "z": 3}[which.lower()]
    return PAULI[which].copy()


def spin_block(t0=0.0, tx=0.0, ty=0.0, tz=0.0):
    """2x2 spin matrix ``t0*1 + tx*sx + ty*sy + tz*sz``."""
    return t0 * s0 + tx * sx + ty * sy + tz * sz


AXES = {"x": 0, "y": 1, "z": 2}


def _axis(a):
    return AXES[a] if isinstance(a, str) else int(a)


# --------------------------------------------------------------------------
# Real orbitals: l quantum number, bond-frame label, quadratic forms
# --------------------------------------------------------------------------
ORBITAL_KINDS = {
    "s": (0, 0),
    "px": (1, 0), "py": (1, 1), "pz": (1, 2),
    "dxy": (2, 0), "dyz": (2, 1), "dz2": (2, 2), "dxz": (2, 3), "dx2-y2": (2, 4),
}
_ALIASES = {"dx2y2": "dx2-y2", "dx2-y2": "dx2-y2", "dz^2": "dz2", "d3z2-r2": "dz2",
            "dx^2-y^2": "dx2-y2", "x": "px", "y": "py", "z": "pz"}


def normalize_kind(kind: str) -> str:
    k = kind.strip().lower()
    k = _ALIASES.get(k, k)
    if k not in ORBITAL_KINDS:
        raise ValueError(f"unknown orbital kind {kind!r}; use one of {list(ORBITAL_KINDS)}")
    return k


def _quadratic_forms():
    r2 = np.sqrt(2.0)
    Q = np.zeros((5, 3, 3))
    Q[0][0, 1] = Q[0][1, 0] = 1 / r2            # xy
    Q[1][1, 2] = Q[1][2, 1] = 1 / r2            # yz
    Q[2] = np.diag([-1, -1, 2]) / np.sqrt(6)    # z2
    Q[3][0, 2] = Q[3][2, 0] = 1 / r2            # xz
    Q[4] = np.diag([1, -1, 0]) / r2             # x2-y2
    return Q


_QD = _quadratic_forms()

# label = (|m|, 'c' or 's') used to pair functions in the bond frame (bond || z)
_BOND_LABEL = {
    0: [(0, "-")],
    1: [(1, "c"), (1, "s"), (0, "-")],                       # x, y, z
    2: [(2, "s"), (1, "s"), (0, "-"), (1, "c"), (2, "c")],   # xy, yz, z2, xz, x2-y2
}


def rotation_matrix_for_l(l: int, R: np.ndarray) -> np.ndarray:
    """Representation D^l(R) of a 3x3 rotation in the real-orbital basis.

    Column a gives the rotated orbital ``phi_a(R^T r)`` expanded in lab orbitals.
    """
    if l == 0:
        return np.ones((1, 1))
    if l == 1:
        return np.array(R, float)
    if l == 2:
        D = np.empty((5, 5))
        for a in range(5):
            Ra = R @ _QD[a] @ R.T
            for b in range(5):
                D[b, a] = np.trace(_QD[b] @ Ra)
        return D
    raise ValueError("only l = 0, 1, 2 are supported")


def _rotation_z_to(d):
    """Rotation matrix taking the unit z axis onto unit vector d (Rodrigues)."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    c = float(np.dot(z, d))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


_LNAME = "spd"


def _sk_value(l1, l2, m, params):
    """Bond-frame two-centre integral V_{l1 l2 m} with the parity sign."""
    lo, hi = sorted((l1, l2))
    key = _LNAME[lo] + _LNAME[hi] + "spd"[m]
    val = params.get(key, 0.0)
    if l1 > l2:
        val *= (-1) ** (l1 + l2)
    return val


def slater_koster_block(l1: int, l2: int, d, params: dict) -> np.ndarray:
    """Hopping block between all real orbitals of angular momentum l1 (row)
    and l2 (column) for a bond vector ``d`` (from orbital 1 to orbital 2).

    ``params`` uses the standard names ``sss, sps, pps, ppp, sds, pds, pdp,
    dds, ddp, ddd`` (s = sigma, p = pi, d = delta as the last letter).
    """
    d = np.ravel(np.asarray(d, float))
    d = np.concatenate([d, np.zeros(3 - len(d))]) if len(d) < 3 else d[:3]
    R = _rotation_z_to(d)
    lab1, lab2 = _BOND_LABEL[l1], _BOND_LABEL[l2]
    Tb = np.zeros((len(lab1), len(lab2)))
    for a, la in enumerate(lab1):
        for b, lb in enumerate(lab2):
            if la == lb:
                Tb[a, b] = _sk_value(l1, l2, la[0], params)
    D1, D2 = rotation_matrix_for_l(l1, R), rotation_matrix_for_l(l2, R)
    return D1 @ Tb @ D2.T


def slater_koster(kind1: str, kind2: str, d, params: dict) -> float:
    """Single Slater-Koster matrix element <kind1, 0 | H | kind2, d>."""
    l1, i1 = ORBITAL_KINDS[normalize_kind(kind1)]
    l2, i2 = ORBITAL_KINDS[normalize_kind(kind2)]
    return float(slater_koster_block(l1, l2, d, params)[i1, i2])


# --------------------------------------------------------------------------
# Orbital angular momentum in the real basis
# --------------------------------------------------------------------------
def _generator(axis):
    G = np.zeros((3, 3))
    eps = np.zeros((3, 3, 3))
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
    G[:, :] = -eps[axis]
    return G


def angular_momentum_matrices(l: int):
    """(Lx, Ly, Lz) for angular momentum l in the real-orbital basis
    (p: px,py,pz; d: dxy,dyz,dz2,dxz,dx2-y2)."""
    out = []
    for a in range(3):
        G = _generator(a)
        if l == 0:
            out.append(np.zeros((1, 1), complex))
        elif l == 1:
            out.append(1j * G.astype(complex))
        elif l == 2:
            dD = np.empty((5, 5))
            for i in range(5):
                Ri = G @ _QD[i] + _QD[i] @ G.T
                for j in range(5):
                    dD[j, i] = np.trace(_QD[j] @ Ri)
            out.append(1j * dD.astype(complex))
        else:
            raise ValueError("l must be 0, 1 or 2")
    return tuple(out)


def orbital_angular_momentum(kind1: str, kind2: str, axis) -> complex:
    """<kind1 | L_axis | kind2> on the same atom (zero unless same l)."""
    l1, i1 = ORBITAL_KINDS[normalize_kind(kind1)]
    l2, i2 = ORBITAL_KINDS[normalize_kind(kind2)]
    if l1 != l2:
        return 0.0
    return angular_momentum_matrices(l1)[_axis(axis)][i1, i2]
