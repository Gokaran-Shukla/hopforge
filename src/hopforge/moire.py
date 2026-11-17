"""Continuum moire models: twisted bilayer graphene (TBG).

Bistritzer-MacDonald model (PNAS 108, 12233 (2011)) for one valley, in the
plane-wave basis of the moire reciprocal lattice, in units v = k_theta = 1
(energies in v k_theta, k_theta = 2 |K| sin(theta/2)):

    H = sum_l v sigma . (k - K_l)  +  sum_{j=1..3} T_j  (interlayer)
    T_j = w0 sigma_0 + w1 [cos(phi_j) sigma_x + sin(phi_j) sigma_y],  phi_j = 2 pi (j-1)/3

alpha = w1 / (v k_theta). In the chiral limit w0 = 0 (Tarnopolsky, Kruchkov &
Vishwanath, PRL 122, 106405 (2019)) the Dirac velocity at K_m vanishes
exactly at alpha = 0.586, 2.221, ...

>>> m = hf.moire.BistritzerMacDonald(alpha=0.586, w0_ratio=0.0)
>>> m.bands(path=["K", "G", "M", "K'"])
>>> hf.moire.magic_angles(w0_ratio=0.0)        # -> ~[0.586]
"""
from __future__ import annotations

import numpy as np

from .parallel import pmap

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
s0 = np.eye(2, dtype=complex)


def alpha_to_theta(alpha, w1_meV=110.0, vF_eVA=5.94, a_A=2.46):
    """Twist angle (degrees) for a given alpha = w1 / (v k_theta)."""
    K = 4 * np.pi / (3 * a_A)
    ktheta = w1_meV * 1e-3 / (vF_eVA * alpha)
    return np.degrees(2 * np.arcsin(ktheta / (2 * K)))


class BistritzerMacDonald:
    """Single-valley BM continuum Hamiltonian.

    alpha    : w1 / (v k_theta)
    w0_ratio : w0 / w1 (0 = chiral limit, 1 = original BM, ~0.8 realistic)
    shells   : plane-wave cutoff |G| <= shells * k_theta
    """

    def __init__(self, alpha=0.586, w0_ratio=0.0, shells=5.0):
        self.alpha, self.w0 = alpha, alpha * w0_ratio
        self.q = np.array([[0.0, -1.0], [np.sqrt(3) / 2, 0.5], [-np.sqrt(3) / 2, 0.5]])
        b1, b2 = self.q[1] - self.q[0], self.q[2] - self.q[0]
        self.b = np.array([b1, b2])
        R = int(np.ceil(shells)) + 2
        G = [i * b1 + j * b2 for i in range(-R, R + 1) for j in range(-R, R + 1)]
        G = np.array([g for g in G if np.linalg.norm(g) <= shells + 1e-9])
        self.G = G
        # layer-1 momenta G, layer-2 momenta G + q1; K_1 at 0, K_2 at -q1 -> both Dirac
        # points sit at the moire K and K' corners
        self.p1, self.p2 = G, G + self.q[0]
        self.n = len(G)
        phis = [0.0, 2 * np.pi / 3, -2 * np.pi / 3]
        self.T = [self.w0 * s0 + alpha * (np.cos(ph) * sx + np.sin(ph) * sy) for ph in phis]
        # coupling list: layer-1 index a  <->  layer-2 index b with p2[b] = p1[a] + q_j
        self.links = []
        for j, qj in enumerate(self.q):
            target = self.p1 + qj
            for a, t in enumerate(target):
                d = np.linalg.norm(self.p2 - t, axis=1)
                b = int(np.argmin(d))
                if d[b] < 1e-9:
                    self.links.append((a, b, j))
        # moire high-symmetry points (Cartesian, k_theta = 1): K_m and K'_m are the
        # two Dirac points, Gamma_m is equidistant from both, M_m is their midpoint
        Km, Kp = np.zeros(2), -self.q[0]
        self.special_points = {"K": Km, "K'": Kp, "G": Km + self.q[1], "M": 0.5 * (Km + Kp)}

    def hamiltonian(self, k):
        k = np.asarray(k, float)
        n = self.n
        H = np.zeros((4 * n, 4 * n), complex)
        for a, p in enumerate(self.p1):
            kk = k + p
            H[2 * a:2 * a + 2, 2 * a:2 * a + 2] = kk[0] * sx + kk[1] * sy
        off = 2 * n
        for b, p in enumerate(self.p2):
            kk = k + p
            H[off + 2 * b:off + 2 * b + 2, off + 2 * b:off + 2 * b + 2] = kk[0] * sx + kk[1] * sy
        for a, b, j in self.links:
            H[2 * a:2 * a + 2, off + 2 * b:off + 2 * b + 2] += self.T[j]
            H[off + 2 * b:off + 2 * b + 2, 2 * a:2 * a + 2] += self.T[j].conj().T
        return H

    def energies(self, k, nbands=None):
        w = np.linalg.eigvalsh(self.hamiltonian(k))
        if nbands is None:
            return w
        mid = len(w) // 2
        return w[mid - nbands // 2: mid + nbands // 2]

    def bands(self, path=("K", "G", "M", "K'"), n=120, nbands=8, n_jobs=None):
        """Bands along a path through moire high-symmetry points. Returns
        (x, E (nk, nbands), ticks, labels)."""
        pts = [self.special_points[p] for p in path]
        ks, xs, ticks, x0 = [], [], [0.0], 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            L = np.linalg.norm(b - a)
            m = max(2, int(n * L / 2))
            t = np.linspace(0, 1, m, endpoint=False)
            ks += [a + ti * (b - a) for ti in t]
            xs += list(x0 + t * L)
            x0 += L
            ticks.append(x0)
        ks.append(pts[-1]); xs.append(x0)
        E = np.array(pmap(lambda k: self.energies(k, nbands), ks, n_jobs))
        return np.array(xs), E, ticks, list(path)

    def dirac_velocity(self, dk=1e-4):
        """Renormalised Dirac velocity v*/v at the moire K point."""
        E0 = self.energies(np.zeros(2), 2)
        E1 = self.energies(np.array([dk, 0.0]), 2)
        return float(((E1[1] - E1[0]) - (E0[1] - E0[0])) / (2 * dk))

    def flat_bandwidth(self, nk=12):
        """Width of the two central bands over a k-mesh of the moire BZ."""
        b1, b2 = self.b
        E = []
        for i in range(nk):
            for j in range(nk):
                E.append(self.energies(i / nk * b1 + j / nk * b2, 2))
        E = np.array(E)
        return float(E.max() - E.min())


def velocity_curve(alphas, w0_ratio=0.0, shells=5.0, n_jobs=None):
    """v*/v versus alpha (parallel)."""
    return np.array(pmap(lambda a: BistritzerMacDonald(a, w0_ratio, shells).dirac_velocity(), list(alphas), n_jobs))


def magic_angles(w0_ratio=0.0, alpha_max=0.8, n=81, shells=5.0, tol=1e-6, n_jobs=None):
    """Magic values of alpha: local minima of |v*/v| that reach (numerically)
    zero, refined by golden-section search."""
    al = np.linspace(0.05, alpha_max, n)
    v = np.abs(velocity_curve(al, w0_ratio, shells, n_jobs))
    f = lambda a: abs(BistritzerMacDonald(a, w0_ratio, shells).dirac_velocity())
    out = []
    gr = (np.sqrt(5) - 1) / 2
    for i in range(1, len(al) - 1):
        if v[i] <= v[i - 1] and v[i] <= v[i + 1] and v[i] < 0.1:
            lo, hi = al[i - 1], al[i + 1]
            c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
            fc, fd = f(c), f(d)
            while hi - lo > tol:
                if fc < fd:
                    hi, d, fd = d, c, fc
                    c = hi - gr * (hi - lo)
                    fc = f(c)
                else:
                    lo, c, fc = c, d, fd
                    d = lo + gr * (hi - lo)
                    fd = f(d)
            a0 = 0.5 * (lo + hi)
            out.append((a0, f(a0)))
    return [a for a, val in out if val < 1e-2]


# =============================================================================
# real-space moire bilayers (any layer model)
# =============================================================================
def _rotation_blocks(layer, theta):
    """Per-orbital-group rotation matrices (orbital D^l x spin SU(2)) for a
    rotation by theta about z."""
    from . import operators as ops
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Us = np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]])
    ns = layer.ns
    # map each orbital to (site key, l, component)
    groups = {}
    for i in range(layer.norb):
        site = (layer.lat.site(i), tuple(np.round(layer.lat.positions[i], 6)))
        l, comp = ops.ORBITAL_KINDS[layer.lat.kind(i)]
        groups.setdefault((site, l), {})[comp] = i
    U = np.eye(layer.norb * ns, dtype=complex)
    for (site, l), comps in groups.items():
        D = ops.rotation_matrix_for_l(l, R)
        full = list(range(D.shape[0]))
        if l > 0 and set(comps) != set(full):
            # incomplete shell: rotate only within the present components
            keep = sorted(comps)
            D = D[np.ix_(keep, keep)]
            full = keep
        idx = [comps[k] for k in full]
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                U[ia * ns:(ia + 1) * ns, ib * ns:(ib + 1) * ns] = D[a, b] * (Us if ns == 2 else np.eye(1))
    return U


def rotate_model(layer, theta):
    """Copy of a 2D layer model rotated by theta about z: lattice, positions,
    orbital-dependent hoppings (p/d Slater-Koster blocks) and spin frame."""
    from .model import Lattice, Model, _Term
    c, s = np.cos(theta), np.sin(theta)
    D = layer.lat.D
    R3 = np.eye(D)
    R3[:2, :2] = [[c, -s], [s, c]]
    A = layer.lat.vectors @ R3.T
    P = layer.lat.positions @ R3.T
    m = Model(Lattice(A, list(zip(layer.lat.names, P)), coords="cartesian"), spinful=layer.spinful,
              name=f"{layer.name} rot {np.degrees(theta):.3f}")
    U = _rotation_blocks(layer, theta)
    ns = layer.ns
    blk = lambda i, j: U[i * ns:(i + 1) * ns, j * ns:(j + 1) * ns]
    # rotate onsite + hoppings as full orbital matrices: X' = U X U^dag
    no = layer.norb
    Hon = np.zeros((no * ns, no * ns), complex)
    for i in range(no):
        Hon[i * ns:(i + 1) * ns, i * ns:(i + 1) * ns] = layer._onsite[i]
    terms_by_R = {}
    for t in layer._terms:
        if not t.R.any():
            Hon[t.i * ns:(t.i + 1) * ns, t.j * ns:(t.j + 1) * ns] += t.T
            Hon[t.j * ns:(t.j + 1) * ns, t.i * ns:(t.i + 1) * ns] += t.T.conj().T
            continue
        M = terms_by_R.setdefault(tuple(t.R), np.zeros((no * ns, no * ns), complex))
        M[t.i * ns:(t.i + 1) * ns, t.j * ns:(t.j + 1) * ns] += t.T
    Hon = U @ Hon @ U.conj().T
    for i in range(no):
        m._onsite[i] = Hon[i * ns:(i + 1) * ns, i * ns:(i + 1) * ns]
        for j in range(i + 1, no):
            b = Hon[i * ns:(i + 1) * ns, j * ns:(j + 1) * ns]
            if np.abs(b).max() > 1e-14:
                m._terms.append(_Term(i, j, np.zeros(layer.dim, int), b))
    for Rv, M in terms_by_R.items():
        M = U @ M @ U.conj().T
        for i in range(no):
            for j in range(no):
                b = M[i * ns:(i + 1) * ns, j * ns:(j + 1) * ns]
                if np.abs(b).max() > 1e-14:
                    m._terms.append(_Term(i, j, np.array(Rv, int), b))
    m._changed()
    return m


def commensurate_hexagonal(m, r=1):
    """Twist angle and superlattice vectors (coefficients in the 120-degree basis
    b1 = a2, b2 = a2 - a1 of a 60-degree hexagonal lattice) of a commensurate
    twisted bilayer (Lopes dos Santos, Peres & Castro Neto, PRL 99, 256802 (2007)).
    Number of lattice sites per layer = 3m^2 + 3mr + r^2 (for gcd(r, 3) = 1)."""
    c = (3 * m ** 2 + 3 * m * r + r ** 2 / 2) / (3 * m ** 2 + 3 * m * r + r ** 2)
    theta = np.arccos(c)
    L1 = (m, m + r)
    L2 = (-(m + r), 2 * m + r)
    return theta, L1, L2


def _fill_supercell(layer, S, cutoff_search=None):
    """Orbitals of a layer inside the supercell spanned by rows of S (2xD)."""
    A = layer.lat.vectors[:2, :2]
    Sinv = np.linalg.inv(S[:, :2])
    span = int(np.ceil(np.abs(S[:, :2]).sum() / min(np.linalg.norm(A, axis=1)))) + 2
    out = []
    for n1 in range(-span, span + 1):
        for n2 in range(-span, span + 1):
            Rc = n1 * layer.lat.vectors[0] + n2 * layer.lat.vectors[1]
            for o in range(layer.norb):
                p = layer.lat.positions[o] + Rc
                f = p[:2] @ Sinv
                if np.all(f > -1e-7) and np.all(f < 1 - 1e-7):
                    out.append((o, (n1, n2), p))
    return out


def bilayer(bottom, top, S=None, d=3.35, interlayer=None, cutoff=6.0, name="bilayer"):
    """Assemble a (moire) bilayer supercell from two 2D layer models that
    share the superlattice spanned by the rows of S (Cartesian, 2 x 2).

    bottom, top : layer Models (top may be rotated with :func:`rotate_model`,
                  shifted, or a different material); the top layer sits at z = d.
    interlayer  : dict of Slater-Koster parameters, or callable
                  ``params(site_i, site_j, distance) -> dict`` (distance-dependent,
                  e.g. :func:`moon_koshino`); hoppings within ``cutoff``.
    Returns a periodic 2D Model embedded in 3D (orbital and spin operators,
    Kubo, Edelstein, torques, transport all work on it)."""
    from .model import Lattice, Model, _Term
    from . import operators as ops
    if S is None:
        S = bottom.lat.vectors[:2, :2]
    S = np.asarray(S, float)
    S3 = np.zeros((2, 3)); S3[:, :2] = S
    layers = []
    for li, (lay, z) in enumerate(((bottom, 0.0), (top, d))):
        layers.append((lay, z, _fill_supercell(lay, S)))
    names, pos, index = [], [], {}
    for li, (lay, z, orbs) in enumerate(layers):
        atom_id = {}
        for (o, Rc, p) in orbs:
            site, _, kind = lay.lat.names[o].partition(":")
            # one label per atom (site + cell), shared by all its orbitals, so that
            # orbital angular momentum operators act within each atom
            key = (site, tuple(np.round(lay.lat.positions[o], 6)), Rc)
            aid = atom_id.setdefault(key, len(atom_id))
            names.append(f"L{li}{site}_{aid}" + (f":{kind}" if kind else ""))
            pp = np.zeros(3); pp[:2] = p[:2]; pp[2] = z
            pos.append(pp)
            index[(li, o, Rc)] = len(names) - 1
    if bottom.spinful != top.spinful:
        raise ValueError("both layers must be spinful or both spinless")
    m = Model(Lattice(S3, list(zip(names, pos)), coords="cartesian"), spinful=bottom.spinful, name=name)
    Sinv = np.linalg.inv(S)
    # positions -> (orbital index, superlattice image) lookup per layer
    lookup = []
    for li, (lay, z, orbs) in enumerate(layers):
        tab = {}
        for (o, Rc, p) in orbs:
            tab[(o, tuple(np.round(p[:2] @ Sinv % 1.0, 6) % 1.0))] = index[(li, o, Rc)]
        lookup.append(tab)

    def locate(li, o, p):
        f = p[:2] @ Sinv
        cell = np.floor(f + 1e-7).astype(int)
        frac = tuple(np.round(f - cell, 6) % 1.0)
        return lookup[li].get((o, frac)), cell

    for li, (lay, z, orbs) in enumerate(layers):
        for (o, Rc, p) in orbs:
            a = index[(li, o, Rc)]
            m._onsite[a] = lay._onsite[o].copy()
        for t in lay._terms:
            for (o, Rc, p) in orbs:
                if o != t.i:
                    continue
                a = index[(li, o, Rc)]
                Rlay = np.array(Rc) + t.R[:2]
                pj = lay.lat.positions[t.j] + Rlay[0] * lay.lat.vectors[0] + Rlay[1] * lay.lat.vectors[1]
                b, cell = locate(li, t.j, pj)
                if b is None:
                    raise RuntimeError("layer is not commensurate with the supercell S")
                if a == b and not cell.any():
                    m._onsite[a] = m._onsite[a] + t.T + t.T.conj().T
                    continue
                m._terms.append(_Term(a, b, cell.astype(int), t.T.copy()))
    # interlayer couplings (bottom -> top)
    if interlayer is not None:
        bot, top_ = layers[0], layers[1]
        P = np.array(pos)
        for (ob, Rb, pb) in bot[2]:
            a = index[(0, ob, Rb)]
            for (ot, Rt, pt) in top_[2]:
                b = index[(1, ot, Rt)]
                for n1 in (-1, 0, 1):
                    for n2 in (-1, 0, 1):
                        shift = n1 * S[0] + n2 * S[1]
                        dvec = np.array([P[b, 0] + shift[0] - P[a, 0], P[b, 1] + shift[1] - P[a, 1], d])
                        dist = np.linalg.norm(dvec)
                        if dist > cutoff:
                            continue
                        si, sj = bottom.lat.site(ob), top.lat.site(ot)
                        prm = interlayer(si, sj, dist) if callable(interlayer) else interlayer
                        v = ops.slater_koster(bottom.lat.kind(ob), top.lat.kind(ot), dvec, prm)
                        if abs(v) > 1e-10:
                            m._terms.append(_Term(a, b, np.array([n1, n2]), v * np.eye(m.ns, dtype=complex)))
    m._changed()
    m.info["layers"] = {"bottom": [index[(0, o, R)] for (o, R, p) in layers[0][2]],
                        "top": [index[(1, o, R)] for (o, R, p) in layers[1][2]]}
    return m


def moon_koshino(Vpps0=0.48, Vppp0=-2.7, a0=1.42, d0=3.35, delta=0.184 * 2.46):
    """Distance-dependent pz-pz Slater-Koster parameters of Moon & Koshino,
    PRB 85, 195458 (2012) (eV, Angstrom)."""
    def params(si, sj, dist):
        return {"pps": Vpps0 * np.exp(-(dist - d0) / delta), "ppp": Vppp0 * np.exp(-(dist - a0) / delta)}
    return params


def graphene_layer(a=2.46, t=2.7):
    """Graphene pz layer in Angstrom/eV with Slater-Koster orbital labels (for moire builds)."""
    from .model import Lattice, Model
    A = a * np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    lat = Lattice(A, [("A:pz", [1 / 3, 1 / 3]), ("B:pz", [2 / 3, 2 / 3])])
    m = Model(lat, name="graphene (pz)")
    m.add_neighbors(-t, 1)
    return m


def twisted_bilayer(layer, m_index, r=1, d=3.35, interlayer=None, cutoff=6.0):
    """Commensurate twisted bilayer of a hexagonal layer model (60-degree basis):
    bottom layer unrotated, top layer rotated by the commensurate angle.
    Returns (model, theta_degrees)."""
    theta, (p1, q1), (p2, q2) = commensurate_hexagonal(m_index, r)
    a1, a2 = layer.lat.vectors[0][:2], layer.lat.vectors[1][:2]
    b1, b2 = a2, a2 - a1
    S = np.array([p1 * b1 + q1 * b2, p2 * b1 + q2 * b2])
    top = rotate_model(layer, theta)
    m = bilayer(layer, top, S, d, interlayer, cutoff, name=f"twisted bilayer {np.degrees(theta):.3f} deg")
    m.info["theta_deg"] = float(np.degrees(theta))
    return m, float(np.degrees(theta))
