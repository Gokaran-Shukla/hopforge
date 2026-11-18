"""Benchmarks against published results.

Every benchmark reproduces a number from the literature (analytic result,
quantised invariant, phase boundary, or a fitted material value) and reports
the deviation. Run all of them (in parallel) with

    python -m hopforge.benchmarks            # prints a table, writes BENCHMARKS.md
    python -m hopforge.benchmarks -j 40      # on a 40-core node

or ``hf.benchmarks.run(n_jobs=-1)``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    name: str
    quantity: str
    published: object
    value: object
    reference: str
    passed: bool
    note: str = ""
    seconds: float = 0.0


REGISTRY = []


def benchmark(name, quantity, reference):
    def deco(f):
        REGISTRY.append((name, quantity, reference, f))
        return f
    return deco


def _clean(x):
    """numpy scalars -> plain Python numbers for readable tables."""
    if isinstance(x, (np.floating, np.integer)):
        return round(float(x), 6) if isinstance(x, np.floating) else int(x)
    if isinstance(x, (list, tuple)):
        return type(x)(_clean(v) for v in x)
    if isinstance(x, np.ndarray):
        return [_clean(v) for v in x.tolist()]
    if isinstance(x, float):
        return round(x, 6)
    return x


def _close(a, b, tol):
    return bool(np.all(np.abs(np.asarray(a, float) - np.asarray(b, float)) <= tol))


# =============================================================================
# electronic structure
# =============================================================================
@benchmark("Graphene Dirac velocity", "v_F = 3 t a_cc / 2",
           "Wallace, Phys. Rev. 71, 622 (1947); Castro Neto et al., RMP 81, 109 (2009)")
def _graphene_vf():
    import hopforge as hf
    g = hf.models.honeycomb(t=1.0)
    K = g.special_points["K"] @ g.lat.reciprocal
    dk = 1e-5
    B = g.lat.reciprocal
    kred = np.linalg.solve(B.T, (K + np.array([dk, 0])))
    v = g.energies(kred)[1] / dk
    pub = 3 * 1.0 * (1 / np.sqrt(3)) / 2
    return pub, v, _close(v, pub, 1e-4)


@benchmark("AB bilayer graphene gap under bias", "E_g = |U| g1 / sqrt(U^2 + g1^2)",
           "McCann, PRB 74, 161403(R) (2006)")
def _bilayer_gap():
    import hopforge as hf
    U, g1 = 0.1, 0.4
    m = hf.models.bilayer_graphene(t=1.0, tperp=g1, U=U)
    K = m.special_points["K"]
    A = m.lat.vectors                       # (2, 3): reduced k_i = k . a_i / 2 pi
    qs = np.linspace(0, 0.3, 3001)
    best = np.inf
    for ang in np.linspace(0, 2 * np.pi, 13)[:-1]:
        dq = qs[:, None] * np.array([np.cos(ang), np.sin(ang), 0.0])[None]
        kred = K[None] + dq @ A.T / (2 * np.pi)
        E = m.energies(kred)
        best = min(best, (E[:, 2] - E[:, 1]).min())
    pub = U * g1 / np.sqrt(U ** 2 + g1 ** 2)
    return pub, best, _close(best, pub, 2e-3)


@benchmark("Silicon sp3s* (Vogl) indirect gap", "E_g(Si) [eV]",
           "Vogl, Hjalmarson & Dow, J. Phys. Chem. Solids 44, 365 (1983)")
def _silicon_vogl():
    m = silicon_vogl()
    import hopforge as hf
    p = hf.kpath(m, ["G", "X"], 600)
    E = m.energies(p.k)
    vbm = m.energies(np.zeros(3))[3]
    gap = E[:, 4].min() - vbm
    G = np.sort(m.energies(np.zeros(3)))
    gamma_ok = _close(G[[0, 1, 4, 7]], [-12.5, 0.0, 3.43, 4.1], 1e-9)
    return 1.17, round(gap, 3), _close(gap, 1.17, 0.01) and gamma_ok


@benchmark("Lieb lattice flat band", "E_flat = 0", "Lieb, PRL 62, 1201 (1989)")
def _lieb():
    import hopforge as hf
    E = hf.models.lieb().energies(np.random.default_rng(0).random((200, 2)))
    return 0.0, float(np.abs(E[:, 1]).max()), np.abs(E[:, 1]).max() < 1e-12


@benchmark("Kagome flat band", "E_flat = +2t", "Mielke, J. Phys. A 24, L73 (1991)")
def _kagome():
    import hopforge as hf
    E = hf.models.kagome().energies(np.random.default_rng(1).random((200, 2)))
    return 2.0, float(E[:, 2].mean()), np.allclose(E[:, 2], 2.0)


@benchmark("Zigzag graphene ribbon edge states", "flat E=0 band for 2pi/3 < |k a| < pi",
           "Fujita et al., JPSJ 65, 1920 (1996); Nakada et al., PRB 54, 17954 (1996)")
def _zigzag():
    import hopforge as hf
    r = hf.models.honeycomb().ribbon(1, 40)
    k_in, k_out = np.array([0.42, 0.45, 0.48]), np.array([0.05, 0.15, 0.25])
    e_in = np.abs(r.energies(k_in)).min(1).max()
    e_out = np.abs(r.energies(k_out)).min(1).min()
    return "E=0 for |k|/2pi > 1/3", f"max|E| in={e_in:.1e}, min|E| out={e_out:.2f}", e_in < 1e-6 and e_out > 0.05


@benchmark("Armchair ribbons: metallic iff N = 3p + 2", "gap(N) for N = 18, 20, 22, 24, 26",
           "Nakada et al., PRB 54, 17954 (1996); Son, Cohen & Louie, PRL 97, 216803 (2006)")
def _armchair():
    import hopforge as hf
    arm = hf.models.honeycomb().supercell([[1, 1], [-1, 1]])
    out, ok = [], True
    for W in (9, 10, 11, 12, 13):
        N = 2 * W
        r = arm.ribbon(1, W)
        E = r.energies(np.linspace(-0.5, 0.5, 401))
        gap = E[:, r.nbands // 2].min() - E[:, r.nbands // 2 - 1].max()
        out.append(round(gap, 3))
        ok &= (gap < 1e-6) == (N % 3 == 2)
    return "metallic at N = 20, 26", out, ok


# =============================================================================
# topology
# =============================================================================
@benchmark("Haldane phase boundary", "C = 1 for |M| < 3 sqrt(3) t2 sin(phi), else 0",
           "Haldane, PRL 61, 2015 (1988)")
def _haldane():
    import hopforge as hf
    t2, phi = 0.1, np.pi / 3
    Mc = 3 * np.sqrt(3) * t2 * np.sin(phi)
    c_in = hf.chern_number(hf.models.haldane(t2=t2, phi=phi, M=0.95 * Mc), 1, 60)
    c_out = hf.chern_number(hf.models.haldane(t2=t2, phi=phi, M=1.05 * Mc), 1, 60)
    hc = hf.models.haldane(t2=t2, phi=phi, M=Mc)
    gap_c = min(np.diff(hc.energies(hc.special_points[v]))[0] for v in ("K", "K'"))
    return "C: 1 -> 0 at M_c", (round(c_in), round(c_out), f"gap(M_c)={gap_c:.1e}"), \
        round(abs(c_in)) == 1 and round(c_out) == 0 and gap_c < 1e-9


@benchmark("Qi-Wu-Zhang Chern numbers", "C(m) = -sgn(m) for 0<|m|<2, 0 for |m|>2 (hopforge sign)",
           "Qi, Wu & Zhang, PRB 74, 085308 (2006)")
def _qwz():
    import hopforge as hf
    ms = [-3, -1, 1, 3]
    C = [round(hf.chern_number(hf.models.qwz(m), 1, 40)) for m in ms]
    return [0, 1, -1, 0], C, C == [0, 1, -1, 0]


@benchmark("Hofstadter TKNN Chern numbers", "Diophantine r = q s_r + p t_r, C_r = t_r - t_{r-1}",
           "Thouless, Kohmoto, Nightingale & den Nijs, PRL 49, 405 (1982)")
def _tknn():
    import hopforge as hf

    def dioph(p, q):
        t = {0: 0}
        for r in range(1, q):
            for tr in range(-q // 2, q // 2 + 1):
                if (r - p * tr) % q == 0 and abs(tr) <= q / 2:
                    t[r] = tr
                    break
        t[q] = 0
        return [t[r] - t[r - 1] for r in range(1, q + 1)]
    ok, rows = True, []
    for p, q in [(1, 3), (1, 4), (1, 5), (2, 5), (3, 7)]:
        if q % 2 == 0:
            # even q: the two central bands touch; compare the outer bands
            pass
        pub = dioph(p, q)
        m = hf.models.hofstadter(p, q)
        if q % 2 == 0:
            nb = q
            got = [round(hf.chern_number(m, [n], 48)) for n in (0, nb - 1)]
            ref = [pub[0], pub[-1]]
        else:
            got = [round(hf.chern_number(m, [n], 48)) for n in range(q)]
            ref = pub
        sign = 1 if got == ref else (-1 if got == [-x for x in ref] else 0)
        ok &= sign != 0
        rows.append(f"{p}/{q}:{got}")
    return "TKNN integers (up to the global B sign)", "; ".join(rows), ok


@benchmark("Kane-Mele QSH phase boundary", "Z2 = 1 for lambda_v < 3 sqrt(3) lambda_SO",
           "Kane & Mele, PRL 95, 146802 (2005)")
def _km():
    import hopforge as hf
    lso = 0.06
    c = 3 * np.sqrt(3) * lso
    z_in = hf.z2_invariant(hf.models.kane_mele(lso=lso, lv=0.9 * c), 2, nk=80, nscan=81)
    z_out = hf.z2_invariant(hf.models.kane_mele(lso=lso, lv=1.1 * c), 2, nk=80, nscan=81)
    return (1, 0), (z_in, z_out), (z_in, z_out) == (1, 0)


@benchmark("Kane-Mele Rashba stability", "QSH gap closes at lambda_R = 2 sqrt(3) lambda_SO (lambda_v = 0)",
           "Kane & Mele, PRL 95, 226801 (2005)")
def _km_rashba():
    import hopforge as hf
    lso = 0.06
    c = 2 * np.sqrt(3) * lso
    g_in = hf.band_gap(hf.models.kane_mele(lso=lso, lr=0.8 * c), 2, 90)
    z_in = hf.z2_invariant(hf.models.kane_mele(lso=lso, lr=0.8 * c), 2, nk=80, nscan=81)
    g_c = hf.band_gap(hf.models.kane_mele(lso=lso, lr=1.0 * c), 2, 90)
    return "gap > 0, Z2 = 1 below; gap closed at the boundary", \
        (f"gap(0.8)={g_in:.4f}", f"Z2={z_in}", f"gap(1.0)={g_c:.4f}"), g_in > 0.02 and z_in == 1 and g_c < 0.01


@benchmark("BHZ quantum spin Hall phase", "Z2 = 1 for 0 < M/B < 4 (and 4 < M/B < 8 on the lattice)",
           "Bernevig, Hughes & Zhang, Science 314, 1757 (2006)")
def _bhz():
    import hopforge as hf
    Ms = [-1.0, 1.0, 3.0, 5.0, 9.0]
    z = [hf.z2_invariant(hf.models.bhz(M=M), 2, nk=60, nscan=61) for M in Ms]
    return [0, 1, 1, 1, 0], z, z == [0, 1, 1, 1, 0]


@benchmark("Fu-Kane-Mele 3D topological insulator", "(nu0;nu1nu2nu3) = (1;111) dt>0, (0;111) dt<0",
           "Fu, Kane & Mele, PRL 98, 106803 (2007)")
def _fkm():
    import hopforge as hf
    a = hf.z2_3d(hf.models.diamond_fkm(dt=0.4), 2, nk=30, nscan=31)
    b = hf.z2_3d(hf.models.diamond_fkm(dt=-0.4), 2, nk=30, nscan=31)
    return ((1, 1, 1, 1), (0, 1, 1, 1)), (a, b), a == (1, 1, 1, 1) and b == (0, 1, 1, 1)


@benchmark("Bi2Se3-type continuum TI", "(1;000) with one surface Dirac cone at Gamma",
           "Zhang et al., Nat. Phys. 5, 438 (2009)")
def _bi2se3():
    import hopforge as hf
    from hopforge import surface
    ti = hf.discretize("M0*tau_z - B*(k_x**2+k_y**2+k_z**2)*tau_z + "
                       "A*(k_x*tau_x*sigma_x + k_y*tau_x*sigma_y + k_z*tau_x*sigma_z)",
                       "xyz", 1.0, dict(M0=1.0, B=0.5, A=1.0))
    z = hf.z2_3d(ti, 2, nk=24, nscan=25)
    E = np.linspace(-0.4, 0.4, 81)
    A = surface.spectral_at(ti, np.zeros(2), E, 2, eta=1e-3)
    return ((1, 0, 0, 0), 0.0), (z, float(E[np.argmax(A)])), z == (1, 0, 0, 0) and abs(E[np.argmax(A)]) < 1e-9


@benchmark("SSH polarisation and winding", "P = e/2 (mod e), W = 1 in the topological phase",
           "Su, Schrieffer & Heeger, PRL 42, 1698 (1979); King-Smith & Vanderbilt, PRB 47, 1651 (1993)")
def _ssh():
    import hopforge as hf
    s = hf.models.ssh(0.5, 1.0)
    zak = abs(hf.zak_phase(s, gauge="periodic"))
    W = abs(hf.winding_number(s, s.info["chiral"]))
    return (np.pi, 1), (round(zak, 6), round(W, 6)), _close([zak, W], [np.pi, 1], 1e-6)


@benchmark("Kitaev chain phase boundary", "topological (W = 1) for |mu| < 2t",
           "Kitaev, Phys.-Usp. 44, 131 (2001)")
def _kitaev():
    import hopforge as hf
    G = hf.models.kitaev_chain().info["chiral"]
    W = [round(abs(hf.winding_number(hf.models.kitaev_chain(mu=mu), G))) for mu in (1.9, 2.1, -1.9, -2.1)]
    return [1, 0, 1, 0], W, W == [1, 0, 1, 0]


@benchmark("Thouless pump", "one electron per adiabatic cycle", "Thouless, PRB 27, 6083 (1983)")
def _pump():
    import hopforge as hf
    ts = np.linspace(0, 1, 41)
    P = [hf.polarization(hf.models.rice_mele(v=1 + 0.5 * np.cos(2 * np.pi * t), w=1.0,
                                             delta=0.5 * np.sin(2 * np.pi * t))) for t in ts]
    dP = (np.unwrap(np.array(P) * 2 * np.pi) / (2 * np.pi))
    pumped = dP[-1] - dP[0]
    return 1.0, round(abs(pumped), 6), _close(abs(pumped), 1.0, 1e-6)


@benchmark("Weyl semimetal anomalous Hall", "sigma_xy = (e^2/h) dk_W / 2pi per length",
           "Burkov & Balents, PRL 107, 127205 (2011)")
def _weyl():
    import hopforge as hf
    w = hf.models.weyl_minimal(m_=2.0)
    s = abs(float(hf.anomalous_hall(w, 0.0, 60)))
    pub = 0.5 * 3.874045e-5 * 1e8
    return round(pub, 1), round(s, 1), abs(s / pub - 1) < 0.01


@benchmark("Landau levels of the square lattice", "E_n = -4t + (n + 1/2) w_c, w_c = 4 pi phi t (phi -> 0)",
           "Landau (1930); Hofstadter, PRB 14, 2239 (1976)")
def _landau():
    import hopforge as hf
    phi = 1 / 60
    m = hf.models.hofstadter(1, 60)
    E = np.sort(m.energies(np.array([0.0, 0.0])))
    wc = 4 * np.pi * phi
    pub = [-4 + 0.5 * wc, -4 + 1.5 * wc]
    return [round(x, 4) for x in pub], [round(E[0], 4), round(E[1], 4)], _close(E[:2], pub, 0.01)


@benchmark("Kagome topological insulator (Guo-Franz)", "Z2 = 1 at 1/3 filling with intrinsic SOC",
           "Guo & Franz, PRB 80, 113102 (2009)")
def _kagome_ti():
    import hopforge as hf
    k = hf.models.kagome(lso=0.1)
    z = hf.z2_invariant(k, 2, nk=80, nscan=81)
    gap = hf.band_gap(k, 2, 60)
    return 1, (z, f"gap={gap:.3f}"), z == 1 and gap > 0


# =============================================================================
# response
# =============================================================================
@benchmark("Quantised anomalous Hall", "sigma_xy = C e^2/h (Haldane, C = 1)",
           "TKNN, PRL 49, 405 (1982)")
def _qah():
    import hopforge as hf
    s = abs(float(hf.anomalous_hall(hf.models.haldane(), 0.0, 120)))
    return 1.0, round(s, 6), _close(s, 1.0, 1e-4)


@benchmark("Quantised spin Hall conductance (Kane-Mele)", "|sigma^sz_xy| = e/2pi",
           "Kane & Mele, PRL 95, 226801 (2005)")
def _qsh_kubo():
    import hopforge as hf
    s = abs(float(hf.spin_hall(hf.models.kane_mele(), 0.0, 120)))
    return 1.0, round(s, 6), _close(s, 1.0, 1e-4)


@benchmark("Graphene universal optical conductivity", "Re sigma = e^2/(4 hbar) incl. spin (1/8 spinless)",
           "Ando, Zheng & Suzuki, JPSJ 71, 1318 (2002); Nair et al., Science 320, 1308 (2008)")
def _optical():
    import hopforge as hf
    s = hf.response.optical_conductivity(hf.models.honeycomb(), [0.3], nk=400, eta=0.02).real[0]
    return 0.125, round(s, 4), abs(s / 0.125 - 1) < 0.03


@benchmark("Valley orbital moment of gapped graphene", "m(K) = e v^2 hbar / (2 Delta) (e = hbar = 1: v^2/2Delta)",
           "Xiao, Yao & Niu, PRL 99, 236809 (2007)")
def _valley():
    import hopforge as hf
    D = 0.05
    g = hf.models.honeycomb(delta=D)
    E, Om, m = hf.response.orbital_moment(g, g.special_points["K"])
    v = np.sqrt(3) / 2
    pub = v ** 2 / (2 * D)
    return round(pub, 4), round(abs(m[0, 1]), 4), abs(abs(m[0, 1]) / pub - 1) < 0.01


@benchmark("Streda formula in a Chern insulator", "dM/dmu = C e/h (per cell C A / 2pi)",
           "Streda, J. Phys. C 15, L717 (1982); Ceresoli et al., PRB 74, 024408 (2006)")
def _streda():
    import hopforge as hf
    h = hf.models.haldane(t2=0.15)
    Mu = hf.response.orbital_magnetization(h, [-0.2, 0.2], nk=150)
    pub = h.lat.cell_volume / (2 * np.pi)
    val = (Mu[1] - Mu[0]) / 0.4
    return round(pub, 6), round(val, 6), _close(val, pub, 1e-6)


@benchmark("Wiedemann-Franz law", "L = pi^2/3 (k_B/e)^2",
           "Sommerfeld (1928); Ashcroft & Mermin, ch. 13")
def _wf():
    import hopforge as hf
    r = hf.response.boltzmann(hf.models.square(), [-1.0], kT=0.01, nk=300)
    L = r.lorenz[0]
    return round(np.pi ** 2 / 3, 4), round(L, 4), abs(L / (np.pi ** 2 / 3) - 1) < 0.02


@benchmark("KPM Kubo-Bastin Hall conductivity", "sigma_xy(mu in gap) = -C e^2/h, matches k-space Kubo",
           "Garcia, Covaci & Rappoport, PRL 114, 116602 (2015)")
def _kpm():
    import hopforge as hf
    sc = hf.models.haldane(t2=0.15).supercell([24, 24])
    s = hf.kpm.conductivity(sc.hamiltonian_sparse(), sc.velocity_sparse("x"), sc.velocity_sparse("y"),
                            sc.lat.cell_volume, [0.0], num_moments=300, num_vectors=8)[0]
    return -1.0, round(s, 3), _close(s, -1.0, 0.05)


# =============================================================================
# quantum transport
# =============================================================================
@benchmark("Quantised conductance of a ballistic wire", "G = N e^2/h", "van Wees et al., PRL 60, 848 (1988)")
def _wire():
    from hopforge import transport as tr
    import hopforge as hf
    s = tr.System(hf.models.square(), lambda r: 0 <= r[0] < 10 and 0 <= r[1] < 8)
    s.attach_lead((-1, 0)); s.attach_lead((1, 0))
    T = [s.smatrix(E).transmission(1, 0) for E in (-3.8, -3.3, -2.5, -1.2)]
    N = [s.smatrix(E).num_modes(0) for E in (-3.8, -3.3, -2.5, -1.2)]
    return N, [round(t, 6) for t in T], _close(T, N, 1e-6)


@benchmark("Integer quantum Hall resistance", "R_xy = h/(nu e^2), R_xx = 0", "von Klitzing et al., PRL 45, 494 (1980)")
def _qhe():
    from hopforge import transport as tr
    import hopforge as hf
    B = 2 * np.pi * 0.05
    s = tr.System(hf.models.square(), lambda r: 0 <= r[0] < 50 and 0 <= r[1] < 24, peierls=tr.landau_gauge(B, "x"))
    s.attach_lead((-1, 0)); s.attach_lead((1, 0))
    for x0 in (10, 32):
        s.attach_lead((0, 1), shape=lambda r, x0=x0: x0 <= r[0] < x0 + 6, peierls=None)
    for x0 in (10, 32):
        s.attach_lead((0, -1), shape=lambda r, x0=x0: x0 <= r[0] < x0 + 6, peierls=None)
    R = [s.smatrix(E).four_terminal_resistance(0, 1, 2, 4) for E in (-3.4, -2.8, -2.3)]
    Rxx = abs(s.smatrix(-3.4).four_terminal_resistance(0, 1, 2, 3))
    return [1, 0.5, 0.3333], [round(r, 4) for r in R], _close(R, [1, 0.5, 1 / 3], 2e-3) and Rxx < 1e-3


@benchmark("Quantum spin Hall edge conductance", "G = 2 e^2/h, robust to disorder", "Koenig et al., Science 318, 766 (2007)")
def _qsh():
    from hopforge import transport as tr
    import hopforge as hf
    s = tr.System(hf.models.kane_mele(lso=0.1), lambda r: 0 <= r[0] < 14 and 0 <= r[1] < 15,
                  onsite=tr.anderson(1.0, seed=1))
    s.attach_lead((-1, 0)); s.attach_lead((1, 0))
    T = s.transmission(1, 0, 0.05)
    return 2.0, round(T, 6), _close(T, 2.0, 1e-4)


@benchmark("Diffusive shot noise", "Fano factor F = 1/3", "Beenakker & Buettiker, PRB 46, 1889 (1992)")
def _fano():
    from hopforge import transport as tr
    import hopforge as hf

    def make(seed):
        d = tr.System(hf.models.square(), lambda r: 0 <= r[0] < 60 and 0 <= r[1] < 30, onsite=tr.anderson(1.6, seed=seed))
        d.attach_lead((-1, 0)); d.attach_lead((1, 0))
        return d
    F = [make(sd).smatrix(-1.0).shot_noise()[1] for sd in range(10)]
    m, e = float(np.mean(F)), float(np.std(F) / np.sqrt(len(F)))
    return round(1 / 3, 4), f"{m:.3f} +- {e:.3f}", abs(m - 1 / 3) < 3 * e + 0.02


@benchmark("Andreev reflection at a transparent NS interface", "G_NS = 2 N e^2/h below the gap",
           "Blonder, Tinkham & Klapwijk, PRB 25, 4515 (1982)")
def _andreev():
    from hopforge import transport as tr
    import hopforge as hf
    Nm = hf.models.bdg(hf.models.square(), mu=-2.0, delta=0.0)
    Sm = hf.models.bdg(hf.models.square(), mu=-2.0, delta=0.1)
    s = tr.System(Nm, lambda r: 0 <= r[0] < 6 and 0 <= r[1] < 10)
    s.attach_lead((-1, 0)); s.attach_lead((1, 0), model=Sm)
    e, h = dict(orbitals=["A_e"]), dict(orbitals=["A_h"])
    N = s.smatrix(0.0).num_modes(0, e)
    G = tr.andreev_conductance(s, [0.0], 0, e, h)[0]
    return 2 * N, round(G, 3), abs(G - 2 * N) < 0.15


@benchmark("Majorana zero-bias peak", "G(V=0) = 2 e^2/h, independent of the tunnel barrier",
           "Law, Lee & Ng, PRL 103, 237001 (2009)")
def _majorana():
    from hopforge import transport as tr
    import hopforge as hf
    lead = hf.models.kitaev_chain(mu=0.5, t=1.0, delta=0.0)
    out = []
    for Vb in (1.5, 5.0):
        wire = hf.models.kitaev_chain(mu=0.5, t=1.0, delta=0.6)
        s = tr.System(wire, lambda r: 0 <= r[0] < 60,
                      onsite=lambda r, n, Vb=Vb: (Vb if r[0] < 1 else 0.0) * (1 if n == "e" else -1))
        s.attach_lead((-1,), model=lead)
        out.append(tr.andreev_conductance(s, [0.0], 0, dict(orbitals=["e"]), dict(orbitals=["h"]))[0])
    return 2.0, [round(x, 4) for x in out], _close(out, [2, 2], 1e-3)


# =============================================================================
# many-body / driven
# =============================================================================
@benchmark("Honeycomb Hubbard mean-field AFM transition", "U_c = 2.23 t",
           "Sorella & Tosatti, EPL 19, 699 (1992)")
def _hubbard():
    import hopforge as hf
    g = hf.models.honeycomb().make_spinful()
    init = {"A": (0, 0, 1), "B": (0, 0, -1)}
    m_lo = abs(hf.meanfield.hubbard(g, 2.12, filling=0.5, init=init, nk=72, kT=0.001, maxiter=1500).moments[0, 2])
    m_hi = abs(hf.meanfield.hubbard(g, 2.36, filling=0.5, init=init, nk=72, kT=0.001, maxiter=1500).moments[0, 2])
    return "m = 0 at 2.12t, m > 0 at 2.36t", (round(m_lo, 4), round(m_hi, 4)), m_lo < 0.02 and m_hi > 0.08


@benchmark("Square-lattice Hubbard mean field", "gap = U |m| (Slater AFM)", "Slater, PR 82, 538 (1951); Hirsch, PRB 31, 4403 (1985)")
def _slater():
    import hopforge as hf
    sq = hf.models.square().make_spinful().supercell([[1, 1], [1, -1]])
    r = hf.meanfield.hubbard(sq, 2.0, filling=0.5, init={0: (0, 0, 1), 1: (0, 0, -1)}, nk=60, kT=0.002, maxiter=800)
    gap = hf.band_gap(r.model, 2, 60)
    return round(2.0 * abs(r.moments[0, 2]), 4), round(gap, 4), _close(gap, 2 * abs(r.moments[0, 2]), 1e-3)


@benchmark("Light-induced Chern insulator in graphene", "circular light: C = +-1 (sign set by helicity); linear: C = 0",
           "Oka & Aoki, PRB 79, 081406 (2009); Kitagawa et al., PRB 84, 235108 (2011)")
def _floquet():
    import hopforge as hf
    F = hf.floquet
    g = hf.models.honeycomb()
    cp = round(hf.chern_number(F.EffectiveModel(g, F.circular(0.4, 10.0)), 1, 40))
    cm = round(hf.chern_number(F.EffectiveModel(g, F.circular(0.4, 10.0, -1)), 1, 40))
    cl = round(hf.chern_number(F.EffectiveModel(g, F.linear(0.4, 10.0)), 1, 40))
    K = g.special_points["K"]
    ge = np.diff(F.EffectiveModel(g, F.circular(0.4, 10.0)).energies(K))[0]
    gx = np.diff(F.FloquetModel(g, F.circular(0.4, 10.0)).quasienergies(K)[0])[0]
    return "(+-1, -+1, 0)", (cp, cm, cl, f"gap eff/exact={ge/gx:.4f}"), \
        abs(cp) == 1 and cm == -cp and cl == 0 and abs(ge / gx - 1) < 0.01


# =============================================================================
# strongly correlated systems (exact diagonalisation)
# =============================================================================
@benchmark("Hubbard dimer", "E0 = (U - sqrt(U^2 + 16 t^2)) / 2", "textbook two-site Hubbard model (e.g. Fazekas, ch. 5)")
def _dimer():
    import hopforge as hf
    out = []
    for U in (0.0, 4.0, 20.0):
        ed = hf.manybody.Hubbard(hf.models.chain().finite(2), U=U)
        out.append(ed.ground_state(1, 1)[0])
    pub = [(U - np.sqrt(U * U + 16)) / 2 for U in (0.0, 4.0, 20.0)]
    return [round(x, 6) for x in pub], [round(x, 6) for x in out], _close(out, pub, 1e-10)


@benchmark("4x4 Heisenberg antiferromagnet", "E0 = -11.228483 J (periodic)",
           "Schulz, Ziman & Poilblanc, J. Phys. I 6, 675 (1996)")
def _heis44():
    import hopforge as hf
    sq = hf.models.square().supercell([4, 4])
    E = hf.manybody.SpinModel.from_model(sq, (1.0,)).ground_state()[0]
    return -11.228483, round(E, 6), _close(E, -11.228483, 1e-6)


@benchmark("Heisenberg chain and Bethe ansatz", "E0(L=16) = -7.142296 J; e_inf = 1/4 - ln 2",
           "Hulthen (1938); Bethe (1931)")
def _heis_chain():
    import hopforge as hf
    E16 = hf.manybody.SpinModel.chain(16).ground_state()[0]
    Ls = [12, 16, 20]
    e = [hf.manybody.SpinModel.chain(L).ground_state()[0] / L for L in Ls]
    c = np.polyfit(1 / np.array(Ls) ** 2, e, 1)
    return (-7.142296, round(0.25 - np.log(2), 5)), (round(E16, 6), round(c[1], 5)), \
        _close(E16, -7.142296, 1e-6) and abs(c[1] - (0.25 - np.log(2))) < 2e-3


@benchmark("Majumdar-Ghosh point", "E0 = -3 L J1 / 8 at J2 = J1/2 (exact dimer state)",
           "Majumdar & Ghosh, J. Math. Phys. 10, 1388 (1969)")
def _mg():
    import hopforge as hf
    E = hf.manybody.SpinModel.chain(16, 1.0, 0.5).ground_state()[0]
    return -6.0, round(E, 8), _close(E, -6.0, 1e-8)


@benchmark("Kanamori atom multiplets", "two electrons in two orbitals: U-3J (x3), U-J (x2), U+J",
           "Kanamori, Prog. Theor. Phys. 30, 275 (1963); Georges, de' Medici & Mravlje, ARCMP 4, 137 (2013)")
def _kanamori():
    import hopforge as hf
    from hopforge.model import Lattice, Model
    at = Model(Lattice(None, [("a", [0, 0]), ("b", [0, 0])]), name="atom")
    U, J = 4.0, 0.8
    ed = hf.manybody.Hubbard(at, U=U, J=J, atoms=[[0, 1]])
    E = np.sort(np.concatenate([np.linalg.eigvalsh(ed.hamiltonian(1, 1).toarray()),
                                np.linalg.eigvalsh(ed.hamiltonian(2, 0).toarray()),
                                np.linalg.eigvalsh(ed.hamiltonian(0, 2).toarray())]))
    pub = [U - 3 * J] * 3 + [U - J] * 2 + [U + J]
    return pub, [round(x, 6) for x in E], _close(E, pub, 1e-9)


@benchmark("Transverse-field Ising chain", "ED = exact free-fermion energy; c = 1/2 at h = J",
           "Pfeuty, Ann. Phys. 57, 79 (1970); Calabrese & Cardy, J. Stat. Mech. P06002 (2004)")
def _tfim():
    import hopforge as hf
    MB, PH = hf.manybody, hf.phases
    E = MB.PauliModel.transverse_field_ising(12, 1.0, 0.7).ground_state()[0]
    Eex = MB.tfim_exact_energy(12, 1.0, 0.7)
    psi = MB.PauliModel.transverse_field_ising(16, 1.0, 1.0).ground_state()[1]
    c = PH.central_charge(psi, 16)[0]
    return (round(Eex, 8), 0.5), (round(E, 8), round(c, 3)), abs(E - Eex) < 1e-9 and abs(c - 0.5) < 0.02


@benchmark("Quantum Ising critical point (Binder crossing)", "h_c = J",
           "Pfeuty (1970); Sachdev, Quantum Phase Transitions (2011)")
def _binder():
    import hopforge as hf
    MB, PH = hf.manybody, hf.phases
    sc = PH.quantum_critical_scan(lambda L, h: MB.PauliModel.transverse_field_ising(L, 1.0, h), "h",
                                  np.linspace(0.8, 1.2, 17), Ls=(10, 14))
    hc = PH.binder_crossing(sc)
    return 1.0, round(hc, 3), abs(hc - 1.0) < 0.05


@benchmark("SYK ground-state energy", "E0/N -> -0.0406 J (large N, q = 4)",
           "Maldacena & Stanford, PRD 94, 106002 (2016); Garcia-Garcia & Verbaarschot, PRD 94, 126010 (2016)")
def _syk_energy():
    import hopforge as hf
    Ns = [14, 16, 18, 20, 22]
    E = [np.mean([hf.manybody.SYK(N, seed=s).ground_energy() / N for s in range(8)]) for N in Ns]
    c = np.polyfit(1 / np.array(Ns), E, 2)
    return -0.0406, round(c[-1], 4), abs(c[-1] + 0.0406) < 0.003


@benchmark("SYK level statistics", "<r>: GOE 0.531 (N=16), GUE 0.600 (N=18), GSE 0.674 (N=20)",
           "You, Ludwig & Xu, PRB 95, 115150 (2017); Atas et al., PRL 110, 084101 (2013)")
def _syk_r():
    import hopforge as hf
    r = [hf.manybody.syk_level_statistics(N, realizations=12) for N in (16, 18, 20)]
    pub = [0.5307, 0.5996, 0.6744]
    return pub, [round(x, 3) for x in r], _close(r, pub, 0.02)


@benchmark("Bose-Hubbard Mott lobe tip (mean field)", "(U/zt)_c = 3 + 2 sqrt(2) = 5.828 for n = 1",
           "Fisher, Weichman, Grinstein & Fisher, PRB 40, 546 (1989); Sachdev, QPT ch. 9")
def _bh():
    import hopforge as hf
    MB = hf.manybody
    mu = np.sqrt(2) - 1
    lo = MB.bose_hubbard_meanfield(1.0, 1 / 5.75, mu)[0]
    hi = MB.bose_hubbard_meanfield(1.0, 1 / 5.90, mu)[0]
    return round(3 + 2 * np.sqrt(2), 4), (f"psi(5.75)={lo:.3f}", f"psi(5.90)={hi:.1e}"), lo > 1e-2 and hi < 1e-4


@benchmark("Stoner criterion", "ferromagnetism for U > 1 / rho(E_F)", "Stoner, Proc. R. Soc. A 165, 372 (1938)")
def _stoner():
    import hopforge as hf
    sq = hf.models.square(t2=-0.3)
    Uc, _ = hf.phases.stoner_critical_U(sq, 0.3, nk=300, sigma=0.02)
    m = [abs(hf.meanfield.hubbard(sq.make_spinful(), f * Uc, filling=0.3, init="ferro", nk=100, kT=0.005,
                                  maxiter=3000, mixing=0.3).moments[0, 2]) for f in (0.85, 1.2)]
    return "m = 0 below U_c, m > 0 above", [round(x, 4) for x in m], m[0] < 1e-3 and m[1] > 0.1


# =============================================================================
# Bernevig / Vishwanath models
# =============================================================================
@benchmark("BBH quadrupole insulator", "nested Wilson loop p^nu = 1/2; corner charge e/2",
           "Benalcazar, Bernevig & Hughes, Science 357, 61 (2017)")
def _bbh():
    import hopforge as hf
    b = hf.models.bbh_quadrupole(gamma=0.5)
    p = hf.nested_wilson_loop(b, 2, 0, nk=60, nperp=60)
    pt = hf.nested_wilson_loop(hf.models.bbh_quadrupole(gamma=1.5), 2, 0, nk=60, nperp=60)
    f = hf.models.bbh_quadrupole(gamma=0.5, delta=1e-3).finite(16)
    E, V = np.linalg.eigh(f.hamiltonian())
    cell = (np.abs(V[:, :f.nbands // 2]) ** 2).sum(1).reshape(-1, 4).sum(1)
    Pc = f.lat.positions[::4]
    q = abs((cell[(Pc[:, 0] > 7.5) & (Pc[:, 1] > 7.5)] - 2).sum())
    return (0.5, 0.0, 0.5), (round(p, 4), round(pt, 4), round(q, 3)), \
        abs(p - 0.5) < 1e-6 and abs(pt) < 1e-6 and abs(q - 0.5) < 0.01


@benchmark("Chiral p+ip superconductor", "C = 0, +1, -1, 0 for mu = -5, -2, 2, 5 (t = 1)",
           "Read & Green, PRB 61, 10267 (2000); Bernevig & Hughes (2013), ch. 16")
def _pip():
    import hopforge as hf
    C = [round(hf.chern_number(hf.models.p_ip_superconductor(mu=mu), 1, 40)) for mu in (-5, -2, 2, 5)]
    ok = C[0] == 0 and C[3] == 0 and abs(C[1]) == 1 and C[2] == -C[1]
    return [0, 1, -1, 0], C, ok


@benchmark("Type-II Weyl semimetal", "DOS(E_W) = 0 for type I, finite for type II (tilt > velocity)",
           "Soluyanov et al. (incl. Bernevig), Nature 527, 495 (2015)")
def _weyl2():
    import hopforge as hf
    out = []
    for tilt in (0.5, 1.5):
        d = hf.dos(hf.models.weyl_type2(tilt=tilt), nk=40, energies=[0.0], sigma=0.02).dos[0]
        out.append(round(d, 4))
    return "type I ~0, type II > 0", out, out[0] < 0.2 * out[1] and out[1] > 0.05


@benchmark("Flat Chern band (checkerboard)", "C = 1, gap/bandwidth ~ 27 (nearly flat)",
           "Sun, Gu, Katsura & Das Sarma, PRL 106, 236803 (2011); Sheng et al., Nat. Commun. 2, 389 (2011)")
def _flatband():
    import hopforge as hf
    cb = hf.models.checkerboard_chern()
    E = cb.energies(hf.kmesh(cb, 80))
    fl = (E[:, 1].min() - E[:, 0].max()) / np.ptp(E[:, 0])
    C = round(hf.chern_number(cb, 1, 40))
    return (1, "~27"), (abs(C), round(fl, 1)), abs(C) == 1 and fl > 20


@benchmark("Fractional Chern insulator (ED)", "3-fold quasi-degenerate ground state at nu = 1/3",
           "Regnault & Bernevig, PRX 1, 021014 (2011); Sheng et al., Nat. Commun. 2, 389 (2011)")
def _fci():
    import hopforge as hf
    sc = hf.models.checkerboard_chern().supercell([3, 4])
    E = hf.manybody.Fermions(sc, V=[1.0]).spectrum(4, k=6)
    spread, gap = E[2] - E[0], E[3] - E[2]
    return "spread << gap", (round(spread, 4), round(gap, 4)), gap > 3 * spread


@benchmark("Chiral twisted bilayer graphene: magic angle", "first magic alpha = 0.586 (Dirac velocity vanishes)",
           "Tarnopolsky, Kruchkov & Vishwanath, PRL 122, 106405 (2019)")
def _tkv():
    import hopforge as hf
    a = hf.moire.magic_angles(0.0, alpha_max=0.7, n=19)
    return 0.586, [round(x, 4) for x in a], len(a) >= 1 and abs(a[0] - 0.586) < 0.002


@benchmark("Bistritzer-MacDonald velocity renormalisation", "v*/v = (1 - 3 a^2)/(1 + 6 a^2) for small a",
           "Bistritzer & MacDonald, PNAS 108, 12233 (2011)")
def _bm():
    import hopforge as hf
    a = 0.1
    v = hf.moire.BistritzerMacDonald(a, 1.0).dirac_velocity()
    pub = (1 - 3 * a * a) / (1 + 6 * a * a)
    return round(pub, 4), round(v, 4), abs(v - pub) < 0.005


@benchmark("Commensurate twisted bilayer graphene", "theta(m=1,2,3) = 21.787, 13.174, 9.430 deg; 4(3m^2+3m+1) atoms",
           "Lopes dos Santos, Peres & Castro Neto, PRL 99, 256802 (2007)")
def _tbg_comm():
    import hopforge as hf
    g = hf.moire.graphene_layer()
    th, n = [], []
    for mi in (1, 2, 3):
        tb, t = hf.moire.twisted_bilayer(g, mi, 1, interlayer=hf.moire.moon_koshino())
        th.append(round(t, 3)); n.append(tb.norb)
    return ([21.787, 13.174, 9.430], [28, 76, 148]), (th, n), \
        _close(th, [21.787, 13.174, 9.430], 2e-3) and n == [28, 76, 148]


@benchmark("Real-space TBG vs continuum", "Dirac point preserved; v*/v within 6% of BM (w = 0.11 eV) at 7.3 deg",
           "Moon & Koshino, PRB 85, 195458 (2012); Trambly de Laissardiere et al., Nano Lett. 10, 804 (2010)")
def _tbg_v():
    import hopforge as hf
    g = hf.moire.graphene_layer()
    tb, th = hf.moire.twisted_bilayer(g, 4, 1, interlayer=hf.moire.moon_koshino())
    v0 = 3 * 2.7 * (2.46 / np.sqrt(3)) / 2
    cands = [np.array(c) for c in [(1 / 3, 1 / 3), (2 / 3, 2 / 3), (1 / 3, 2 / 3), (2 / 3, 1 / 3)]]
    gaps = []
    for c in cands:
        E = np.linalg.eigvalsh(tb.hamiltonian(c)); n = len(E) // 2
        gaps.append(E[n] - E[n - 1])
    K = cands[int(np.argmin(gaps))]
    B = tb.lat.reciprocal
    dk = 1e-4
    Kp = np.linalg.lstsq(B.T, K @ B + np.array([dk, 0, 0]), rcond=None)[0]
    E = np.linalg.eigvalsh(tb.hamiltonian(Kp)); n = len(E) // 2
    v = (E[n] - E[n - 1]) / (2 * dk) / v0
    kth = 2 * (4 * np.pi / (3 * 2.46)) * np.sin(np.radians(th) / 2)
    a = 0.11 / (v0 * kth)
    bm = (1 - 3 * a * a) / (1 + 6 * a * a)
    return round(bm, 4), (round(v, 4), f"gap(K)={min(gaps):.1e}"), abs(v / bm - 1) < 0.06 and min(gaps) < 1e-3


# =============================================================================
# spin-orbit torques
# =============================================================================
@benchmark("Rashba spin-orbit torque symmetry", "extrinsic (Fermi surface) torque field-like, intrinsic (Fermi sea) damping-like",
           "Kurebayashi et al., Nat. Nanotech. 9, 211 (2014); Freimuth, Bluegel & Mokrousov, PRB 90, 174423 (2014)")
def _sot():
    import hopforge as hf
    fm = hf.models.rashba_square(alpha=0.3, J=0.5)
    T = hf.response.torkance(fm, (0, 0, 1), 0.5, "x", mu=-2.0, Gamma=0.02, nk=150)
    fs_fl, fs_dl = abs(T["fermi_surface_FL"][0]), abs(T["fermi_surface_DL"][0])
    se_fl, se_dl = abs(T["fermi_sea_FL"][0]), abs(T["fermi_sea_DL"][0])
    return "FS: FL >> DL; sea: DL >> FL", (round(fs_fl, 4), round(fs_dl, 4), round(se_fl, 6), round(se_dl, 4)), \
        fs_fl > 20 * fs_dl and se_dl > 20 * se_fl


@benchmark("Kubo-Streda -> Boltzmann limit", "Fermi-surface term = Boltzmann with tau = 1/(2 Gamma) (within 8%)",
           "Bastin et al., J. Phys. Chem. Solids 32, 1811 (1971); Streda (1982)")
def _ks_limit():
    import hopforge as hf
    R = hf.response
    r = hf.models.rashba_square(alpha=0.3)
    G = 0.02
    ks = R.kubo_streda(r, r.spin_operator("y"), "x", mu=-2.0, Gamma=G, nk=600, fermi_sea=False)["fermi_surface"][0]
    bz = R.edelstein(r, r.spin_operator("y"), "x", mu=-2.0, kT=0.01, tau=1 / (2 * G), nk=600)["extrinsic"][0]
    # the Boltzmann reference carries thermal smearing (kT) and mesh noise of a few %
    return 1.0, round(ks / bz, 4), abs(ks / bz - 1) < 0.08


# =============================================================================
# MacDonald group models (hopforge.macdonald)
# =============================================================================
@benchmark("TMD heterobilayer flat band (WSe2/MoSe2)", "top moire valence bandwidth at 2.0 deg ~ 11 meV",
           "Wu, Lovorn, Tutuc & MacDonald, PRL 121, 026402 (2018)")
def _md_hetero():
    import hopforge as hf
    W = hf.macdonald.TMDHeterobilayer(theta=2.0).bandwidths(1, nk=24)[0]
    return 11.0, round(W, 2), abs(W - 11.0) < 0.6


@benchmark("TMD heterobilayer WSe2/MoS2 at 0 deg", "moire period ~ 8.5 nm and bandwidth ~ 20 meV",
           "Wu, Lovorn, Tutuc & MacDonald, PRL 121, 026402 (2018)")
def _md_hetero2():
    import hopforge as hf
    h = hf.macdonald.TMDHeterobilayer.preset("WSe2/MoS2", theta=0.0)
    W = h.bandwidths(1, nk=24)[0]
    return (8.5, 20.0), (round(h.aM, 2), round(W, 2)), abs(h.aM - 8.5) < 0.1 and abs(W - 20) < 1.0


@benchmark("TMD moire harmonic potential", "Delta ~ -beta V (r/aM)^2/2 with beta = 16 pi^2 cos(psi + 120 deg)",
           "Wu, Lovorn, Tutuc & MacDonald, PRL 121, 026402 (2018)")
def _md_beta():
    import hopforge as hf
    h = hf.macdonald.TMDHeterobilayer(theta=2.0)
    hm = h.harmonic()
    beta = hm["hbar_omega"] ** 2 * h.mass / (2 * hf.macdonald.HB2M) * h.aM ** 2 / h.V
    ref = 16 * np.pi ** 2 * np.cos(np.radians(-94 + 120))
    return round(ref, 3), round(beta, 3), abs(beta / ref - 1) < 1e-4


@benchmark("Twisted MoTe2 topological bands (1.2 deg)", "+K valley Chern numbers of the top two bands (-1, +1)",
           "Wu, Lovorn, Tutuc, Martin & MacDonald, PRL 122, 086402 (2019)")
def _md_homo1():
    import hopforge as hf
    C = hf.macdonald.TMDHomobilayer(theta=1.2).chern_numbers(2, nk=24)
    return [-1, 1], [round(c, 3) for c in C], _close(C, [-1, 1], 0.02)


@benchmark("Twisted MoTe2 band inversion (2.0 deg)", "Chern numbers of the top three bands (-1, -1, +2)",
           "Wu, Lovorn, Tutuc, Martin & MacDonald, PRL 122, 086402 (2019)")
def _md_homo2():
    import hopforge as hf
    C = hf.macdonald.TMDHomobilayer(theta=2.0).chern_numbers(3, nk=24)
    return [-1, -1, 2], [round(c, 3) for c in C], _close(C, [-1, -1, 2], 0.03)


@benchmark("Twisted MoTe2 critical angles", "gap 2-3 closes at theta1* ~ 1.74 deg; gap 1-2 survives to theta2* ~ 3.1 deg",
           "Wu, Lovorn, Tutuc, Martin & MacDonald, PRL 122, 086402 (2019)")
def _md_angles():
    import hopforge as hf
    from scipy.optimize import brentq
    def gap(th, i):
        m = hf.macdonald.TMDHomobilayer(theta=th)
        k = np.vstack([m.kmesh(16), np.zeros((1, 2)), m.points["kappa+"], m.points["kappa-"]])
        E = m.energies(k, i + 2)
        return E[:, i].min() - E[:, i + 1].max()
    ths = np.linspace(1.6, 1.9, 13)
    g = [gap(t, 1) for t in ths]
    t1 = ths[int(np.argmin(g))]
    t2 = brentq(lambda t: gap(t, 0), 2.8, 3.4, xtol=1e-3)
    return (1.74, 3.1), (round(t1, 3), round(t2, 3)), abs(t1 - 1.74) < 0.03 and abs(t2 - 3.1) < 0.15


@benchmark("Universal intrinsic spin Hall conductivity", "Rashba 2DEG, both bands occupied: sigma_sH = e/8pi",
           "Sinova, Culcer, Niu, Sinitsyn, Jungwirth & MacDonald, PRL 92, 126603 (2004)")
def _md_sinova():
    import hopforge as hf
    r = hf.macdonald.sinova_spin_hall(mu=-3.6, alpha=0.1, nk=500)
    return 1.0, round(r, 4), abs(r - 1) < 0.02


@benchmark("Graphene QAH from Rashba + exchange", "Chern number C = 2 in the bulk gap",
           "Qiao et al., PRB 82, 161414(R) (2010); Tse, Qiao, Yao, MacDonald & Niu, PRB 83, 155447 (2011)")
def _md_qah():
    import hopforge as hf
    C = hf.chern_number(hf.macdonald.graphene_qah(lr=0.05, M=0.1), occ=2, nk=120)
    return 2, round(C, 4), abs(abs(C) - 2) < 1e-3


@benchmark("Chiral decomposition of ABC graphene", "E = (v k)^N / t_perp^(N-1) near K for N = 1..4",
           "Min & MacDonald, PRB 77, 155416 (2008)")
def _md_abc():
    import hopforge as hf
    ratios = []
    for N in (1, 2, 3, 4):
        m = hf.macdonald.rhombohedral_graphene(N, tperp=0.3)
        B = m.lat.reciprocal[:2, :2]
        kf = np.linalg.solve(B.T, np.array([2 / 3, 1 / 3]) @ B + np.array([1e-3, 0]))
        E = np.linalg.eigvalsh(m.hamiltonian(kf[None])[0])[N]
        ratios.append(round(E / ((np.sqrt(3) / 2 * 1e-3) ** N / 0.3 ** (N - 1)), 4))
    return [1, 1, 1, 1], ratios, _close(ratios, [1, 1, 1, 1], 2e-3)


@benchmark("Valley Chern number of gapped ABC graphene", "+-N/2 per valley (N = 2, 3)",
           "Zhang, Sahu, Min & MacDonald, PRB 82, 035409 (2010); Zhang et al., PRL 106, 156801 (2011)")
def _md_valley():
    import hopforge as hf
    v = []
    for N in (2, 3):
        m = hf.macdonald.rhombohedral_graphene(N, tperp=0.3, bias=0.02)
        v += [round(hf.macdonald.valley_chern(m, N, [2 / 3, 1 / 3], 0.3, 301), 3),
              round(hf.macdonald.valley_chern(m, N, [1 / 3, 2 / 3], 0.3, 301), 3)]
    return [1, -1, 1.5, -1.5], v, _close(v, [1, -1, 1.5, -1.5], 0.02)


# =============================================================================
# DMRG (hopforge.dmrg)
# =============================================================================
@benchmark("DMRG: Heisenberg chain L = 100 (open)", "ground-state energy E0 = -44.127739 (converged DMRG)",
           "White, PRL 69, 2863 (1992); ITensor reference, Fishman, White & Stoudenmire, SciPost Phys. Codebases 4 (2022)")
def _dmrg_heis():
    import hopforge as hf
    r = hf.dmrg.dmrg(hf.dmrg.heisenberg(100), chi=[16, 32, 64, 96], sweeps=4)
    return -44.127739, round(r.energy, 6), abs(r.energy + 44.127739) < 2e-6


@benchmark("DMRG: critical Ising chain L = 64 (open)", "E0 = exact free-fermion value; central charge c = 1/2",
           "Lieb, Schultz & Mattis, Ann. Phys. 16, 407 (1961); Calabrese & Cardy, J. Stat. Mech. P06002 (2004)")
def _dmrg_tfim():
    import hopforge as hf
    D = hf.dmrg
    r = D.dmrg(D.tfim(64, 1, 1), chi=[16, 32, 48], sweeps=4)
    ex = D.tfim_exact_obc(64, 1, 1)
    c = D.calabrese_cardy_obc(r.entropy, 64)
    return (round(ex, 6), 0.5), (round(r.energy, 6), round(c, 3)), abs(r.energy - ex) < 1e-6 and abs(c - 0.5) < 0.03


@benchmark("DMRG: Heisenberg central charge", "c = 1 from S(l) of a 60-site open chain",
           "Calabrese & Cardy, J. Stat. Mech. P06002 (2004)")
def _dmrg_c1():
    import hopforge as hf
    D = hf.dmrg
    r = D.dmrg(D.heisenberg(60), chi=[32, 64, 96], sweeps=3)
    c = D.calabrese_cardy_obc(r.entropy, 60)
    return 1.0, round(c, 3), abs(c - 1) < 0.05


@benchmark("DMRG: AKLT chain", "E0 = -2/3 per bond; string order -4/9",
           "Affleck, Kennedy, Lieb & Tasaki, PRL 59, 799 (1987); den Nijs & Rommelse, PRB 40, 4709 (1989)")
def _dmrg_aklt():
    import hopforge as hf
    D = hf.dmrg
    L = 40
    r = D.dmrg(D.aklt(L), chi=[8, 16], sweeps=3)
    S1 = D.SPIN1
    so = r.mps.correlation(S1["Sz"], S1["Sz"], 10, 30, string=S1["string"]).real
    return (round(-2 / 3 * (L - 1), 6), round(-4 / 9, 6)), (round(r.energy, 6), round(so, 6)), \
        abs(r.energy + 2 / 3 * (L - 1)) < 1e-8 and abs(so + 4 / 9) < 1e-6


@benchmark("DMRG vs exact diagonalisation (Hubbard)", "open 8-site Hubbard chain, U = 4, half filling",
           "Lieb & Wu, PRL 20, 1445 (1968) model; ED reference in hopforge.manybody")
def _dmrg_hub():
    import hopforge as hf
    L = 8
    ed = hf.manybody.Hubbard(hf.models.chain().finite(L), U=4.0).ground_state(L // 2, L // 2)
    ed = ed[0] if isinstance(ed, tuple) else ed
    r = hf.dmrg.dmrg(hf.dmrg.hubbard(L, 1, 4), chi=[32, 64, 128], sweeps=5)
    e = r.energy + 2 * L                       # undo the -U/2 N chemical potential
    return round(float(ed), 8), round(e, 8), abs(e - ed) < 1e-7


# =============================================================================
# finite-bias NEGF (hopforge.transport.negf)
# =============================================================================
@benchmark("NEGF: resonant level at finite bias", "I = (e/h) int T_BW(E) dE with energy-dependent widths (t' = 0.05, V = 1)",
           "Breit & Wigner (1936); Datta, Quantum Transport (2005), ch. 3 & 9")
def _negf_res():
    import hopforge as hf
    from hopforge import transport as tr
    L, c, tp, V = 11, 5, 0.05, 1.0

    def hop(ri, rj, T):
        xs = sorted([round(ri[0]), round(rj[0])])
        return T * tp if xs in ([c - 1, c], [c, c + 1]) else T

    s = tr.System(hf.models.chain(), lambda r: 0 <= r[0] < L, hopping=hop)
    s.attach_lead((-1,)); s.attach_lead((1,))
    I = tr.negf.solve(s, bias=V, profile="flat", n_energy=2001, currents=False).current
    E = np.linspace(-V / 2, V / 2, 200001)
    g = lambda e: (e - 1j * np.sqrt(4 - e ** 2 + 0j)) / 2
    SL, SR = tp ** 2 * g(E - V / 2), tp ** 2 * g(E + V / 2)
    GL, GR = -2 * SL.imag, -2 * SR.imag
    T = GL * GR / ((E - (SL + SR).real) ** 2 + ((GL + GR) / 2) ** 2)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    ref = trapz(T, E)
    return round(ref, 7), round(I, 7), abs(I / ref - 1) < 1e-3


@benchmark("NEGF: current conservation at finite bias", "bond currents through every cross-section = Landauer I (disordered strip)",
           "Caroli et al., J. Phys. C 4, 916 (1971); Meir & Wingreen, PRL 68, 2512 (1992)")
def _negf_cons():
    import hopforge as hf
    from hopforge import transport as tr
    s = tr.System(hf.models.square(), lambda r: 0 <= r[0] < 8 and 0 <= r[1] < 4, onsite=tr.anderson(1.5, seed=3))
    s.attach_lead((-1, 0)); s.attach_lead((1, 0))
    r = tr.negf.solve(s, bias=0.3, mu=-1.0, kT=0.01, n_energy=101)
    P = s.positions()
    cuts = []
    for x in range(7):
        J = 0.0
        for (a, b), j in zip(r.bonds, r.bond_currents):
            if min(P[a][0], P[b][0]) <= x + 0.5 < max(P[a][0], P[b][0]):
                J += np.sign(P[b][0] - P[a][0]) * j
        cuts.append(J)
    spread = max(abs(np.array(cuts) - r.current))
    return round(r.current, 6), (round(min(cuts), 6), round(max(cuts), 6)), spread < 1e-6


@benchmark("NEGF: linear-response limit", "dI/dV at V -> 0 equals the Landauer conductance T(mu)",
           "Landauer (1957); Buettiker (1986)")
def _negf_lin():
    import hopforge as hf
    from hopforge import transport as tr
    L, c, tp, mu = 11, 5, 0.1, 0.01

    def hop(ri, rj, T):
        xs = sorted([round(ri[0]), round(rj[0])])
        return T * tp if xs in ([c - 1, c], [c, c + 1]) else T

    s = tr.System(hf.models.chain(), lambda r: 0 <= r[0] < L, hopping=hop)
    s.attach_lead((-1,)); s.attach_lead((1,))
    G = tr.negf.differential_conductance(s, 0.0, dV=2e-4, mu=mu, kT=0.0, profile="flat", n_energy=41)
    T = s.smatrix(mu).transmission(1, 0)
    return round(T, 6), round(G, 6), abs(G - T) < 1e-4



# =============================================================================
# helpers
# =============================================================================
def silicon_vogl(a=5.431):
    """Bulk Si in the sp3s* model of Vogl, Hjalmarson & Dow (1983), built with
    hopforge's Slater-Koster engine (parameters in eV, lattice in Angstrom)."""
    import hopforge as hf
    from hopforge.models._util import cart_points
    fcc = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = {"A": np.zeros(3), "B": a * np.array([0.25, 0.25, 0.25])}
    orbs = []
    for S in "AB":
        orbs += [(f"{S}:s", pos[S]), (f"{S}:px", pos[S]), (f"{S}:py", pos[S]), (f"{S}:pz", pos[S]),
                 (f"{S}x:s", pos[S])]
    m = hf.Model(hf.Lattice(fcc, orbs, coords="cartesian"), name="Si sp3s* (Vogl 1983)")
    Es, Ep, Ess = -4.2, 1.715, 6.685
    for S in "AB":
        m.add_onsite(Es, f"{S}:s")
        m.add_onsite(Ep, [f"{S}:px", f"{S}:py", f"{S}:pz"])
        m.add_onsite(Ess, f"{S}x:s")
    Vss, Vxx, Vxy, Vsp, Vssp = -8.3, 1.715, 4.575, 5.7292, 5.3749
    # Vogl's diamond-structure combinations -> two-centre integrals
    pps = (3 * Vxx / 4 + 2 * 3 * Vxy / 4) / 3
    ppp = (3 * Vxx / 4 - 3 * Vxy / 4) / 3
    p = dict(sss=Vss / 4, pps=pps, ppp=ppp, sps=np.sqrt(3) * Vsp / 4)
    pstar = dict(sps=np.sqrt(3) * Vssp / 4)

    def params(si, sj, d):
        si_star, sj_star = si.endswith("x"), sj.endswith("x")
        if si_star and sj_star:
            return {}
        return pstar if (si_star or sj_star) else p
    m.add_slater_koster(params, shell=1)
    cart_points(m, {"G": [0, 0, 0], "X": [0, 1 / a, 0], "L": [0.5 / a] * 3, "K": [0.75 / a, 0.75 / a, 0]},
                ["L", "G", "X", "K", "G"])
    return m


def _run_one(entry):
    name, quantity, ref, f = entry
    t0 = time.time()
    try:
        pub, val, ok = f()
        pub, val = _clean(pub), _clean(val)
        note = ""
    except Exception as e:  # pragma: no cover
        pub, val, ok, note = "-", "-", False, f"error: {e}"
    return Result(name, quantity, pub, val, ref, bool(ok), note, time.time() - t0)


def run(n_jobs=None, names=None, verbose=True, write=None):
    """Run all benchmarks (parallel over benchmarks). Returns a list of Result."""
    from .parallel import pmap
    from .models import community_benchmarks
    entries = [e for e in REGISTRY + community_benchmarks() if names is None or e[0] in names]
    res = pmap(_run_one, entries, n_jobs)
    if verbose:
        print(table(res))
    if write:
        with open(write, "w") as fh:
            fh.write(markdown(res))
    return res


def table(results):
    w = max(len(r.name) for r in results)
    lines = [f"{'benchmark':{w}s}  pass  time   published  |  hopforge"]
    for r in results:
        lines.append(f"{r.name:{w}s}  {'PASS' if r.passed else 'FAIL'}  {r.seconds:5.1f}s  {r.published}  |  {r.value} {r.note}")
    n = sum(r.passed for r in results)
    lines.append(f"\n{n}/{len(results)} benchmarks reproduce the published results")
    return "\n".join(lines)


def markdown(results):
    import hopforge as hf
    n = sum(r.passed for r in results)
    out = [f"# hopforge benchmarks against published results\n",
           f"hopforge {hf.__version__}: **{n}/{len(results)} pass**. Regenerate with "
           "`python -m hopforge.benchmarks -j -1`.\n",
           "| # | Benchmark | Quantity | Published | hopforge | Status | Reference |",
           "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        esc = lambda x: str(x).replace("|", "\\|")
        out.append(f"| {i} | {esc(r.name)} | {esc(r.quantity)} | {esc(r.published)} | {esc(r.value)} | "
                   f"{'pass' if r.passed else 'FAIL'} | {esc(r.reference)} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Run hopforge's published-result benchmarks")
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("-o", "--output", default="BENCHMARKS.md")
    a = ap.parse_args()
    res = run(n_jobs=a.jobs, write=a.output)
    raise SystemExit(0 if all(r.passed for r in res) else 1)
