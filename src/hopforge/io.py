"""Import / export: Wannier90 (hr.dat, tb.dat, centres.xyz, .win), VASP POSCAR.

>>> m = hf.io.read_wannier90("wannier90_hr.dat", lattice="POSCAR",
...                          positions="wannier90_centres.xyz")
>>> m = hf.io.read_wannier90_tb("wannier90_tb.dat")   # lattice + centres included
>>> m = hf.io.read_wannier90("MoS2_hr.dat", lattice=A, positions=xyz, spinful=True,
...                          spin_order="interleaved")

Conventions: H_mn(R) = <m, 0 | H | n, R> (Wannier90), divided by the
Wigner-Seitz degeneracy. Lengths in Angstrom, energies in eV.
"""
from __future__ import annotations

import numpy as np

from .model import Lattice, Model, _Term

BOHR = 0.529177210903


# ---------------------------------------------------------------------------
# structure files
# ---------------------------------------------------------------------------
def read_poscar(filename):
    """Read a VASP POSCAR/CONTCAR. Returns (lattice 3x3, species list,
    Cartesian positions (n, 3))."""
    with open(filename) as f:
        lines = [l.strip() for l in f if l.strip()]
    scale = float(lines[1].split()[0])
    A = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    A = A * scale if scale > 0 else A * (abs(scale) / abs(np.linalg.det(A))) ** (1 / 3)
    tok = lines[5].split()
    if all(t.isdigit() for t in tok):
        counts, names, i = [int(t) for t in tok], [f"X{n}" for n in range(len(tok))], 6
    else:
        names, counts, i = tok, [int(t) for t in lines[6].split()], 7
    if lines[i][0] in "sS":
        i += 1
    cart = lines[i][0] in "cCkK"
    n = sum(counts)
    P = np.array([[float(x) for x in lines[i + 1 + k].split()[:3]] for k in range(n)])
    P = P * (scale if scale > 0 else 1) if cart else P @ A
    species = [s for s, c in zip(names, counts) for _ in range(c)]
    return A, species, P


def read_win_cell(filename):
    """Read ``unit_cell_cart`` from a Wannier90 .win file (Angstrom)."""
    with open(filename) as f:
        lines = f.read().splitlines()
    for i, l in enumerate(lines):
        if l.strip().lower().startswith("begin unit_cell_cart"):
            block = []
            for l2 in lines[i + 1:]:
                if l2.strip().lower().startswith("end"):
                    break
                if l2.strip():
                    block.append(l2.split())
            unit = 1.0
            if len(block[0]) == 1:
                unit = BOHR if block[0][0].lower().startswith("bohr") else 1.0
                block = block[1:]
            return np.array([[float(x) for x in b[:3]] for b in block]) * unit
    raise ValueError("no unit_cell_cart block found")


def read_centres(filename, num_wann=None):
    """Wannier centres from a ``_centres.xyz`` file (Cartesian, Angstrom)."""
    with open(filename) as f:
        lines = f.read().splitlines()
    pts = [l.split() for l in lines[2:] if l.strip()]
    xs = [np.array([float(x) for x in p[1:4]]) for p in pts if p[0].upper() == "X"]
    if num_wann is not None:
        xs = xs[:num_wann]
    return np.array(xs)


def _lattice_from(lattice):
    if lattice is None:
        return None
    if isinstance(lattice, str):
        if lattice.lower().endswith(".win"):
            return read_win_cell(lattice)
        return read_poscar(lattice)[0]
    return np.asarray(lattice, float)


# ---------------------------------------------------------------------------
# hr.dat
# ---------------------------------------------------------------------------
def _read_hr(filename):
    with open(filename) as f:
        f.readline()
        nw = int(f.readline())
        nr = int(f.readline())
        deg = []
        while len(deg) < nr:
            deg += [int(x) for x in f.readline().split()]
        data = np.loadtxt(f, ndmin=2)
    R = data[:, :3].astype(int)
    m = data[:, 3].astype(int) - 1
    n = data[:, 4].astype(int) - 1
    H = data[:, 5] + 1j * data[:, 6]
    # repeat degeneracies per R block (nw*nw entries per R)
    H = H / np.repeat(np.array(deg, float), nw * nw)
    return nw, R, m, n, H


def _build(nw, R, m, n, H, A, positions, spinful, spin_order, names, tol, dim, name):
    if A is None:
        raise ValueError("lattice vectors are needed (array, POSCAR or .win file)")
    A = np.asarray(A, float)
    dim = dim or 3
    if positions is None:
        positions = np.zeros((nw, 3))
    positions = np.asarray(positions, float).reshape(nw, 3)
    if spinful:
        if nw % 2:
            raise ValueError("spinful needs an even number of Wannier functions")
        no = nw // 2
        if spin_order == "interleaved":            # (orb1 up, orb1 dn, orb2 up, ...)
            orb_of = np.arange(nw) // 2
            spin_of = np.arange(nw) % 2
        elif spin_order == "block":                # (all up, then all down)
            orb_of = np.arange(nw) % no
            spin_of = np.arange(nw) // no
        else:
            raise ValueError("spin_order must be 'interleaved' or 'block'")
        pos = np.array([positions[np.flatnonzero(orb_of == o)[0]] for o in range(no)])
    else:
        no = nw
        orb_of, spin_of = np.arange(nw), np.zeros(nw, int)
        pos = positions
    names = names or [f"w{i}" for i in range(no)]
    keep_dims = list(range(dim))
    lat = Lattice(A[:dim], list(zip(names, pos)), coords="cartesian")
    model = Model(lat, spinful=spinful, name=name)
    ns = model.ns
    if dim < 3 and np.any(R[:, dim:] != 0):
        mask = np.all(R[:, dim:] == 0, axis=1)
        R, m, n, H = R[mask], m[mask], n[mask], H[mask]
    Rd = R[:, :dim]
    blocks = {}
    for r, a, b, h in zip(map(tuple, Rd), m, n, H):
        if abs(h) < tol:
            continue
        oa, ob = orb_of[a], orb_of[b]
        key = (r, oa, ob)
        blk = blocks.setdefault(key, np.zeros((ns, ns), complex))
        blk[spin_of[a], spin_of[b]] += h
    zero = tuple([0] * dim)
    for (r, oa, ob), blk in blocks.items():
        if r == zero and oa == ob:
            model._onsite[oa] += 0.5 * (blk + blk.conj().T)
            continue
        rm = tuple(-x for x in r)
        # keep one orientation of each (R, a, b) / (-R, b, a) pair
        if (r, oa, ob) > (rm, ob, oa):
            continue
        model._terms.append(_Term(int(oa), int(ob), np.array(r, int), blk))
    model._changed()
    return model


def read_wannier90(hr_file, lattice=None, positions=None, spinful=False, spin_order="interleaved",
                   names=None, tol=1e-8, dim=3, name=None):
    """Build a Model from ``seedname_hr.dat``.

    lattice   : 3x3 array (rows, Angstrom), a POSCAR or a .win file
    positions : (num_wann, 3) Cartesian centres, or a ``_centres.xyz`` file
                (default: all at the origin - fine for bands, needed for Berry
                phases / polarisation / transport geometry)
    spinful   : pair Wannier functions into spin-1/2 orbitals (spinor
                projections); ``spin_order`` 'interleaved' (up, dn per
                projection; Wannier90 default) or 'block' (all up then all down)
    dim       : periodic dimension (2 for a slab/monolayer: drops R_z != 0)
    tol       : drop matrix elements smaller than this (eV)
    """
    nw, R, m, n, H = _read_hr(hr_file)
    A = _lattice_from(lattice)
    if isinstance(positions, str):
        positions = read_centres(positions, nw)
    return _build(nw, R, m, n, H, A, positions, spinful, spin_order, names, tol, dim,
                  name or str(hr_file))


def read_wannier90_tb(tb_file, spinful=False, spin_order="interleaved", names=None, tol=1e-8,
                      dim=3, use_centres=True):
    """Build a Model from ``seedname_tb.dat`` (lattice, H(R) and position
    matrix; Wannier centres are taken from the diagonal of r(R = 0))."""
    with open(tb_file) as f:
        lines = f.read().splitlines()
    A = np.array([[float(x) for x in lines[i].split()[:3]] for i in (1, 2, 3)])
    nw = int(lines[4]); nr = int(lines[5])
    deg, i = [], 6
    while len(deg) < nr:
        deg += [int(x) for x in lines[i].split()]
        i += 1
    Rs, ms, ns_, Hs = [], [], [], []
    for r in range(nr):
        while not lines[i].strip():
            i += 1
        R = [int(x) for x in lines[i].split()[:3]]
        i += 1
        for _ in range(nw * nw):
            a, b, re, im = lines[i].split()[:4]
            Rs.append(R); ms.append(int(a) - 1); ns_.append(int(b) - 1)
            Hs.append((float(re) + 1j * float(im)) / deg[r])
            i += 1
    centres = np.zeros((nw, 3))
    if use_centres:
        for r in range(nr):
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break
            R = [int(x) for x in lines[i].split()[:3]]
            i += 1
            for _ in range(nw * nw):
                p = lines[i].split()
                a, b = int(p[0]) - 1, int(p[1]) - 1
                if R == [0, 0, 0] and a == b:
                    centres[a] = [float(p[2]), float(p[4]), float(p[6])]
                i += 1
    return _build(nw, np.array(Rs), np.array(ms), np.array(ns_), np.array(Hs), A, centres, spinful,
                  spin_order, names, tol, dim, str(tb_file))


def write_poscar(filename, lattice, species, positions, comment="hopforge"):
    """Write a POSCAR (Cartesian coordinates)."""
    names = list(dict.fromkeys(species))
    counts = [species.count(s) for s in names]
    with open(filename, "w") as f:
        f.write(f"{comment}\n1.0\n")
        for v in np.asarray(lattice):
            f.write(f"  {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        f.write(" ".join(names) + "\n" + " ".join(map(str, counts)) + "\nCartesian\n")
        for s in names:
            for sp_, p in zip(species, positions):
                if sp_ == s:
                    f.write(f"  {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")
    return filename
