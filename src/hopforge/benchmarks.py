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
    entries = [e for e in REGISTRY if names is None or e[0] in names]
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
        out.append(f"| {i} | {r.name} | {r.quantity} | {r.published} | {r.value} | "
                   f"{'pass' if r.passed else 'FAIL'} | {r.reference} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Run hopforge's published-result benchmarks")
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("-o", "--output", default="BENCHMARKS.md")
    a = ap.parse_args()
    res = run(n_jobs=a.jobs, write=a.output)
    raise SystemExit(0 if all(r.passed for r in res) else 1)
