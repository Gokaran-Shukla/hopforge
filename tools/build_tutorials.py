"""Build (and optionally execute) the hopforge tutorial notebooks.

    python tools/build_tutorials.py            # write notebooks
    python tools/build_tutorials.py --execute  # write and run them
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "tutorials"
SETUP = "import numpy as np\nimport matplotlib.pyplot as plt\nimport hopforge as hf\nM = hf.models\n%matplotlib inline"


def quiz_cells(n, topic):
    return [
        ("md", f"""---
## Quiz {n:02d}: {topic}

Twenty multiple-choice questions on this module. Run the next cell for a clickable
version (needs `ipywidgets`), or use the text version below it and grade yourself with
`quiz.check({{1: 'a', 2: 'c', ...}})`."""),
        ("code", f"quiz = hf.Quiz({n})\nquiz.interactive()"),
        ("code", "quiz.show()"),
        ("code", "my_answers = {\n" + "\n".join(f"    {i}: '',"for i in range(1, 21)) +
         "\n}\nquiz.check(my_answers)"),
    ]


NOTEBOOKS = {}

# =============================================================================
NOTEBOOKS["01_lattices_and_models"] = [
    ("md", r"""# 01 · Lattices, orbitals and building any tight-binding model

**Goals**: define a lattice and orbitals, add on-site terms and hoppings, see 1st/2nd/3rd
neighbour shells, build spinful models with exchange and spin-orbit coupling, and use
Slater-Koster tables for s, p, d orbitals.

A tight-binding Hamiltonian is

$$H = \sum_{i} \varepsilon_i\, c_i^\dagger c_i + \sum_{i j \mathbf{R}} t_{ij}(\mathbf{R})\, c_{i,\mathbf 0}^\dagger c_{j,\mathbf R} + \text{h.c.}$$

and its Bloch form (atomic gauge) is
$H_{ij}(\mathbf k)=\sum_{\mathbf R} t_{ij}(\mathbf R)\, e^{i\mathbf k\cdot(\mathbf R+\mathbf r_j-\mathbf r_i)}$."""),
    ("code", SETUP),
    ("md", "## 1. Your first model: graphene in five lines\nPositions are fractional by default. Bonds are found automatically by distance shell."),
    ("code", """lat = hf.Lattice([[1, 0], [0.5, np.sqrt(3)/2]],
                 [("A", [1/3, 1/3]), ("B", [2/3, 2/3])])
g = hf.Model(lat, name="my graphene")
nbonds = g.add_neighbors(-1.0, shell=1)      # -t on all nearest-neighbour bonds
print(nbonds, "bonds per cell")
print(g.summary())"""),
    ("md", "## 2. Seeing the neighbour shells\n`plot_lattice` colours the 1st, 2nd and 3rd neighbour shells, and the legend gives each shell's distance $d$ and coordination number $z$."),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(14, 5))
hf.plot_lattice(g, ax=axes[0])
hf.plot_lattice(g, highlight="A", ax=axes[1], title="neighbours of site A")
plt.tight_layout()"""),
    ("code", """for n, shell in enumerate(hf.plotting.neighbor_shells(g, 3), start=1):
    print(f"shell {n}: d = {shell['distance']:.4f}, coordination = {set(shell['coordination'].values())}")"""),
    ("md", "## 3. Adding hoppings by hand\n`add_hopping(t, i, j, R)` adds $\\langle i,0|H|j,R\\rangle=t$ **and its Hermitian conjugate**, so add each bond only once."),
    ("code", """chain = hf.Model(hf.Lattice([[1.0]], [("A", [0.0])]), name="chain")
chain.add_onsite(0.0, "A")
chain.add_hopping(-1.0, "A", "A", [1])        # A(0) -> A(1); A(1)->A(0) is automatic
k = np.linspace(-0.5, 0.5, 5)
print(chain.energies(k).ravel())
print(-2*np.cos(2*np.pi*k))"""),
    ("md", "## 4. Second neighbours and site energies\nAdd an NNN hopping `t2` and a staggered potential: graphene → hBN-like."),
    ("code", """g2 = g.copy()
g2.add_neighbors(-0.1, shell=2)
g2.add_onsite(+0.3, "A").add_onsite(-0.3, "B")
g2.special_points = M.honeycomb().special_points
ax = g2.plot_bands(["G", "K", "M", "G"])"""),
    ("md", "## 5. Spinful models: exchange, Rashba, intrinsic SOC\nWith `spinful=True` every amplitude can be a 2×2 spin matrix. Basis order: orbital-major (orb0↑, orb0↓, orb1↑, …)."),
    ("code", """sq = hf.Model(hf.Lattice([[1, 0], [0, 1]], [("A", [0, 0])]), spinful=True, name="Rashba FM")
sq.add_neighbors(-1.0, 1)
sq.add_rashba(0.3)                       # i*alpha*(sigma x d).z on NN bonds
sq.add_exchange(0.5, (0, 0, 1))          # J m.sigma
sq.special_points = M.square().special_points
ax = sq.plot_bands(["G", "X", "M", "G"], weights=sq.spin_operator("z"))"""),
    ("md", "Spin-dependent hoppings can be written with Pauli matrices, e.g. the Kane-Mele term $i\\lambda\\nu_{ij}\\sigma_z$:"),
    ("code", """print(hf.pauli('z'))
print(hf.spin_block(0, 0, 0, 1j*0.06))   # t0*1 + tx*sx + ty*sy + tz*sz"""),
    ("md", "## 6. Multi-orbital models and Slater-Koster\nOrbital names `site:kind` tell hopforge the real orbital (`s, px, py, pz, dxy, dyz, dz2, dxz, dx2-y2`). `add_slater_koster` then builds all hoppings from $V_{ll'm}$ parameters."),
    ("code", """lat = hf.Lattice(np.eye(3), [("Si:s", [0,0,0]), ("Si:px", [0,0,0]), ("Si:py", [0,0,0]), ("Si:pz", [0,0,0])])
sp = hf.Model(lat, name="sp3 simple cubic")
sp.add_onsite(-2.0, "Si:s")
sp.add_onsite(+2.0, ["Si:px", "Si:py", "Si:pz"])
sp.add_slater_koster({"sss": -1.0, "sps": 1.2, "pps": 1.5, "ppp": -0.4}, shell=1)
sp.special_points = M.cubic().special_points
sp.plot_bands(["G", "X", "M", "G", "R", "X"]);"""),
    ("code", """d = [1, 1, 0]
for a, b in [("s", "px"), ("px", "py"), ("dxy", "dxy"), ("pz", "dz2")]:
    print(f"E_{a},{b}(d={d}) =", round(hf.slater_koster(a, b, d, dict(sps=1, pps=1, ppp=-0.5, pds=1, pdp=0.3, dds=-1, ddp=0.5, ddd=-0.1)), 4))"""),
    ("md", "## 7. Matrix-valued (k·p-style) lattice models\n`add_onsite_matrix` and `add_hopping_matrix` take whole blocks: ideal for models written as $d(\\mathbf k)\\cdot\\sigma$. Here is the QWZ model built by hand:"),
    ("code", """sx, sy, sz = hf.pauli('x'), hf.pauli('y'), hf.pauli('z')
q = hf.Model(hf.Lattice(np.eye(2), [("a", [0, 0]), ("b", [0, 0])]), name="QWZ by hand")
m_ = 1.0
q.add_onsite_matrix(m_ * sz, ["a", "b"])
q.add_hopping_matrix(sz/2 - 0.5j*sx, ["a", "b"], ["a", "b"], [1, 0])   # gives cos kx sz + sin kx sx
q.add_hopping_matrix(sz/2 - 0.5j*sy, ["a", "b"], ["a", "b"], [0, 1])
k = np.random.rand(3, 2)
print(np.allclose(q.hamiltonian(k), M.qwz(1.0).hamiltonian(k)))"""),
    ("md", "## 8. Export\nAny model can be written as a Wannier90 `_hr.dat` file (and so passed to WannierTools or other codes)."),
    ("code", """M.kane_mele().to_wannier90_hr("kane_mele_hr.dat")
print(open("kane_mele_hr.dat").read()[:300])"""),
    ("md", """## Exercises
1. Build the Lieb lattice yourself and check the flat band at $E=0$ against `M.lieb()`.
2. Add a 3rd-neighbour hopping to graphene and see how the particle-hole symmetry breaks.
3. Put px, py on a honeycomb lattice with Slater-Koster parameters and plot the four bands."""),
] + quiz_cells(1, "Lattices and models")

# =============================================================================
NOTEBOOKS["02_bands_and_dos"] = [
    ("md", r"""# 02 · Band structures, density of states and Fermi levels

**Goals**: k-paths through high-symmetry points, band plots with spin/orbital weights,
DOS and projected DOS, Fermi levels and gaps.

$$g(E)=\frac{1}{N_k}\sum_{n\mathbf k}\delta(E-E_{n\mathbf k}),\qquad
g_\alpha(E)=\frac{1}{N_k}\sum_{n\mathbf k}|\langle\alpha|n\mathbf k\rangle|^2\,\delta(E-E_{n\mathbf k})$$"""),
    ("code", SETUP),
    ("md", "## 1. k-paths\nEach library model carries `special_points` and a default path. Paths can mix labels and coordinates, and `'|'` breaks a path."),
    ("code", """g = M.honeycomb()
print(g.special_points)
path = hf.kpath(g, ["G", "K", "M", "G"], n=300)
print(path.labels, np.round(path.ticks, 3))"""),
    ("code", """fig, axes = plt.subplots(1, 3, figsize=(15, 4))
M.honeycomb().plot_bands(ax=axes[0])
M.honeycomb(delta=0.3).plot_bands(ax=axes[1], title="staggered potential: gap 2Δ")
M.kagome().plot_bands(ax=axes[2])
plt.tight_layout()"""),
    ("md", "## 2. DOS and projected DOS\nNotice the |E| DOS of the Dirac cone, the log van Hove peaks at ±t, and the flat-band delta peak of kagome."),
    ("code", """fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
hf.plot_dos(hf.dos(M.honeycomb(delta=0.2), nk=150, sigma=0.03, projections={"A": "A", "B": "B"}), ax=axes[0])
hf.plot_dos(hf.dos(M.square(), nk=200, sigma=0.03), ax=axes[1]); axes[1].set_title("square: log van Hove at E=0")
hf.plot_dos(hf.dos(M.kagome(), nk=120, sigma=0.03), ax=axes[2]); axes[2].set_title("kagome: flat band at +2t")
plt.tight_layout()"""),
    ("code", """d = hf.dos(M.honeycomb(), nk=100, sigma=0.05)
print("integral of DOS =", np.trapezoid(d.dos, d.energies) if hasattr(np, 'trapezoid') else np.trapz(d.dos, d.energies), "(number of bands = 2)")"""),
    ("md", "## 3. Fermi level and gaps"),
    ("code", """print("graphene E_F at half filling:", round(hf.fermi_level(M.honeycomb(), 1), 4))
print("hBN-like gap (2Δ = 0.6):", round(hf.band_gap(M.honeycomb(delta=0.3), 1, 90), 4))"""),
    ("md", "## 4. Band weights: spin texture of an altermagnet\nThe d-wave altermagnet has zero net moment but spin-split bands along Γ–X and Γ–Y with *opposite* signs, and no splitting along Γ–M."),
    ("code", """am = M.altermagnet_dwave()
am.plot_bands(weights=am.spin_operator('z'), n=400);"""),
    ("md", "## 5. 3D band structures"),
    ("code", """fig, axes = plt.subplots(1, 3, figsize=(15, 4))
M.fcc().plot_bands(ax=axes[0]); M.bcc().plot_bands(ax=axes[1]); M.pyrochlore().plot_bands(ax=axes[2])
plt.tight_layout()"""),
    ("md", "## 6. Bilayer graphene: quadratic touching and a gap with bias"),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, U in zip(axes, [0.0, 0.2]):
    m = M.bilayer_graphene(tperp=0.15, U=U)
    p = hf.kpath(m, [[0.6, 0.3], "K", [0.72, 0.36]], n=300)
    b = hf.bands(m, p)
    ax.plot(b.x, b.energies, "k"); ax.set_ylim(-0.4, 0.4); ax.set_title(f"U = {U}")
    ax.set_xticks(p.ticks); ax.set_xticklabels(p.labels)"""),
    ("md", """## Exercises
1. Plot the DOS of the cubic lattice: where are the van Hove kinks?
2. Show that the Lieb flat band carries weight only on the O sites (PDOS).
3. For `M.rashba_square(J=0.5)`, colour bands by `spin_operator('x')`."""),
] + quiz_cells(2, "Bands and DOS")

# =============================================================================
NOTEBOOKS["03_supercells_ribbons_edges"] = [
    ("md", r"""# 03 · Supercells, ribbons, slabs, flakes and edge states

**Goals**: zone folding, building ribbons with chosen edges, identifying edge states,
finite flakes, vacancies, disorder and magnetic fields (Hofstadter butterfly)."""),
    ("code", SETUP),
    ("md", "## 1. Supercells and zone folding"),
    ("code", """g = M.honeycomb()
sc = g.supercell([[1, 1], [-1, 2]])         # sqrt(3) x sqrt(3)
print(sc.norb, "orbitals; K and K' fold onto the supercell Gamma:")
print(np.round(np.sort(sc.energies(np.zeros(2))), 6))
hf.plot_lattice(sc, shells=1);"""),
    ("md", "## 2. Zigzag ribbon of graphene: flat edge band\nCutting along $a_2$ keeps the zigzag direction $a_1=\\hat x$ periodic."),
    ("code", """rib = M.honeycomb().ribbon(1, 24)
b = hf.bands(rib, ["-X", "G", "X"], n=400, vectors=True)
edge = rib.projector([0, 1, rib.norb - 2, rib.norb - 1])     # outermost orbitals
w = b.weights(edge)
fig, ax = plt.subplots(figsize=(5, 4))
for n in range(rib.nbands):
    ax.scatter(b.x, b.energies[:, n], c=w[:, n], s=2, cmap="Reds", vmin=0, vmax=0.5)
ax.set_ylim(-1.5, 1.5); ax.set_xticks(b.path.ticks); ax.set_xticklabels(b.path.labels)
ax.set_title("zigzag ribbon (red = edge weight)");"""),
    ("md", "## 3. Chern insulator ribbon: chiral edge states\nThe Haldane bulk has $C=1$, so each edge carries one chiral mode crossing the gap."),
    ("code", """hr = M.haldane(t2=0.15).ribbon(1, 30)
b = hf.bands(hr, ["-X", "G", "X"], n=400, vectors=True)
top = hr.projector(list(range(hr.norb - 4, hr.norb)))
w = b.weights(top) - b.weights(hr.projector([0, 1, 2, 3]))
fig, ax = plt.subplots(figsize=(5, 4))
for n in range(hr.nbands):
    ax.scatter(b.x, b.energies[:, n], c=w[:, n], s=2, cmap="coolwarm", vmin=-0.5, vmax=0.5)
ax.set_ylim(-2, 2); ax.set_xticks(b.path.ticks); ax.set_xticklabels(b.path.labels)
ax.set_title("Haldane ribbon: red/blue = top/bottom edge");"""),
    ("md", "## 4. Kane-Mele ribbon: helical edge states (QSH)"),
    ("code", """kr = M.kane_mele(lso=0.06).ribbon(1, 30)
b = hf.bands(kr, ["-X", "G", "X"], n=400, vectors=True)
sz_top = kr.spin_operator('z') @ kr.projector(list(range(kr.norb - 4, kr.norb)))
w = b.weights(sz_top)
fig, ax = plt.subplots(figsize=(5, 4))
for n in range(kr.nbands):
    ax.scatter(b.x, b.energies[:, n], c=w[:, n], s=2, cmap="coolwarm", vmin=-0.3, vmax=0.3)
ax.set_ylim(-1, 1); ax.set_xticks(b.path.ticks); ax.set_xticklabels(b.path.labels)
ax.set_title("top-edge spin s_z: counter-propagating spins");"""),
    ("md", "## 5. Armchair ribbon: choose the edge with a supercell first"),
    ("code", """arm = M.honeycomb().supercell([[1, 1], [-1, 1]])      # a1' along the armchair direction
for W in (9, 10, 11):
    r = arm.ribbon(1, W)
    E = r.energies(np.linspace(-0.5, 0.5, 201))
    gap = E[:, r.nbands // 2].min() - E[:, r.nbands // 2 - 1].max()
    print(f"armchair width {W}: gap = {gap:.4f}")"""),
    ("md", "## 6. Finite flakes, vacancies, disorder"),
    ("code", """flake = M.haldane(t2=0.15).finite(10)
E, V = np.linalg.eigh(flake.hamiltonian())
mid = np.argmin(np.abs(E))
P = flake.lat.positions
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(E, ".", ms=3); axes[0].set_title("flake spectrum: in-gap edge states")
axes[1].scatter(P[:, 0], P[:, 1], s=4000 * np.abs(V[:, mid])**2, c="C3")
axes[1].set_aspect("equal"); axes[1].set_title("in-gap state lives on the edge");"""),
    ("code", """sc = M.honeycomb().supercell([6, 6])
vac = sc.remove_orbitals([0])
dis = hf.add_disorder(sc, W=1.0, seed=1)
print(vac.norb, "orbitals after one vacancy;", dis.norb, "with disorder")"""),
    ("md", "## 7. Magnetic field: the Hofstadter butterfly\nThe Peierls phase $e^{i\\int A\\cdot dl}$ with flux $p/q$ per plaquette."),
    ("code", """fig, ax = plt.subplots(figsize=(6, 5))
from math import gcd
for q in range(1, 31):
    for p in range(0, q + 1):
        if gcd(p, q) != 1: continue
        m = M.hofstadter(p, q)
        E = m.energies(np.array([[0, 0], [0.5/q, 0.5]]))
        ax.plot(E.ravel(), np.full(E.size, p / q), ".k", ms=1.2)
ax.set_xlabel("E / t"); ax.set_ylabel("flux per plaquette p/q"); ax.set_title("Hofstadter butterfly");"""),
    ("md", """## Exercises
1. Plot the Fu-Kane-Mele slab (`M.diamond_fkm().slab(2, 12)`) and find the surface Dirac cones.
2. Make a disordered Haldane flake and check that the edge state survives moderate W.
3. Compute the Chern numbers of the Hofstadter bands for p/q = 1/4."""),
] + quiz_cells(3, "Supercells, ribbons and edges")

# =============================================================================
NOTEBOOKS["04_model_zoo"] = [
    ("md", r"""# 04 · The model zoo

hopforge ships 28 published models. All are ordinary `Model` objects, so everything in
the other tutorials (bands, ribbons, topology, Kubo) works on them, and you can modify them."""),
    ("code", SETUP),
    ("code", "names = M.list_models()"),
    ("md", "## 1D: SSH, Rice-Mele, Kitaev, Creutz"),
    ("code", """fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
M.ssh(0.5, 1).plot_bands(ax=axes[0])
M.rice_mele().plot_bands(ax=axes[1])
M.kitaev_chain(mu=0.5).plot_bands(ax=axes[2])
M.creutz_ladder().plot_bands(ax=axes[3], title="Creutz: flat bands")
plt.tight_layout()"""),
    ("code", """# Majorana end modes of an open Kitaev chain
for mu in (0.5, 3.0):
    E = np.sort(np.abs(M.kitaev_chain(mu=mu).finite(40).energies()))
    print(f"mu = {mu}: two smallest |E| = {E[:2]}")"""),
    ("md", "## 2D Dirac and flat-band lattices"),
    ("code", """fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, m in zip(axes.ravel(), [M.honeycomb(), M.kagome(), M.lieb(), M.dice(), M.checkerboard(), M.triangular()]):
    m.plot_bands(ax=ax)
plt.tight_layout()"""),
    ("md", "## Topological 2D models: Haldane, Kane-Mele, QWZ, BHZ"),
    ("code", """for m, occ in [(M.haldane(), 1), (M.qwz(1.0), 1)]:
    print(f"{m.name:12s}  C  = {hf.chern_number(m, occ):+.0f}")
for m in [M.kane_mele(), M.bhz()]:
    print(f"{m.name:12s}  Z2 = {hf.z2_invariant(m, 2)}")"""),
    ("md", "## Magnetism: Rashba ferromagnet and d-wave altermagnet"),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
rm = M.rashba_square(J=0.4); rm.plot_bands(ax=axes[0], weights=rm.spin_operator('y'))
am = M.altermagnet_dwave(); am.plot_bands(ax=axes[1], weights=am.spin_operator('z'))
plt.tight_layout()"""),
    ("md", "## 3D: TIs, Weyl semimetal, flat-band pyrochlore"),
    ("code", """print("Wilson-Dirac TI (nu0;nu1nu2nu3):", hf.z2_3d(M.wilson_dirac_ti(), 2, nk=30, nscan=31))
print("Fu-Kane-Mele diamond          :", hf.z2_3d(M.diamond_fkm(), 2, nk=30, nscan=31))
w = M.weyl_minimal()
print("Weyl: C(kz=0) =", round(hf.chern_number(w, 1, fixed={2: 0.0})), " C(kz=1/2) =", round(hf.chern_number(w, 1, fixed={2: 0.5})))
w.plot_bands();"""),
    ("md", "## Orbital physics: p-orbital square lattice"),
    ("code", """po = M.p_orbital_square()
po.plot_bands(weights=po.orbital_operator('z') @ po.orbital_operator('z'), cmap='viridis');"""),
    ("md", "## Build by name with custom parameters"),
    ("code", """m = M.get("haldane", t2=0.2, phi=np.pi/3, M=0.1)
print(m.summary())"""),
    ("md", """## Exercises
1. Map the Haldane phase diagram C(M, φ) on a 15×15 grid.
2. Increase the Kane-Mele Rashba term until the gap closes; what happens to Z2?
3. Find the parameter where the BHZ model changes from C_up = 1 to C_up = -1."""),
] + quiz_cells(4, "The model zoo")

# =============================================================================
NOTEBOOKS["05_berry_phase_and_chern"] = [
    ("md", r"""# 05 · Berry phase, polarisation, Berry curvature and Chern numbers

$$\mathcal A_n=i\langle u_n|\nabla_{\mathbf k}u_n\rangle,\qquad
\Omega_n^{xy}=-2\,\mathrm{Im}\sum_{m\neq n}\frac{\langle n|v_x|m\rangle\langle m|v_y|n\rangle}{(E_n-E_m)^2},\qquad
C=\frac{1}{2\pi}\int_{BZ}\Omega^{xy}\,d^2k$$"""),
    ("code", SETUP),
    ("md", "## 1. Zak phase and polarisation of the SSH chain\nPeriodic gauge: quantised 0/π. Atomic gauge: includes orbital positions, so it gives the true Wannier centre."),
    ("code", """for v, w in [(1.0, 0.5), (0.5, 1.0)]:
    s = M.ssh(v, w)
    print(f"v={v}, w={w}:  Zak(periodic) = {hf.zak_phase(s, gauge='periodic'):+.4f}   "
          f"P = {hf.polarization(s):.3f}   winding = {hf.winding_number(s, s.info['chiral']):+.0f}")"""),
    ("md", "## 2. End states of an open SSH chain (bulk-boundary correspondence)"),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, (v, w) in zip(axes, [(1.0, 0.5), (0.5, 1.0)]):
    E = M.ssh(v, w).finite(30).energies()
    ax.plot(E, "o", ms=3); ax.set_title(f"v={v}, w={w}")"""),
    ("md", "## 3. Thouless pump in the Rice-Mele model\nThe polarisation changes by one unit per adiabatic cycle: the Chern number in $(k, t)$."),
    ("code", """ts = np.linspace(0, 1, 41)
P = [hf.polarization(M.rice_mele(v=1 + 0.5*np.cos(2*np.pi*t), w=1.0, delta=0.5*np.sin(2*np.pi*t))) for t in ts]
plt.plot(ts, np.unwrap(np.array(P)*2*np.pi)/(2*np.pi), "o-"); plt.xlabel("t / T"); plt.ylabel("P (e a)");
plt.title("one electron pumped per cycle");"""),
    ("md", "## 4. Berry curvature of the Haldane model"),
    ("code", """h = M.haldane(t2=0.1, M=0.2)
k1, k2, Om = hf.berry_curvature_map(h, occ=1, nk=120)
hf.plot_kmap(Om, title="Ω(k) of the lower band: hot spots at K, K'")
print("C (FHS) =", hf.chern_number(h, 1), "   C (Kubo integral) =", 2*np.pi*Om.mean()/h.lat.cell_volume)"""),
    ("md", "## 5. Haldane phase diagram"),
    ("code", """phis = np.linspace(-np.pi, np.pi, 25); Ms = np.linspace(-0.7, 0.7, 25)
C = np.array([[hf.chern_number(M.haldane(t2=0.1, phi=p, M=m), 1, 24) for p in phis] for m in Ms])
plt.imshow(np.round(C), origin="lower", extent=[-np.pi, np.pi, -0.7, 0.7], aspect="auto", cmap="RdBu")
pp = np.linspace(-np.pi, np.pi, 200)
plt.plot(pp, 3*np.sqrt(3)*0.1*np.sin(pp), "k"); plt.plot(pp, -3*np.sqrt(3)*0.1*np.sin(pp), "k")
plt.xlabel("φ"); plt.ylabel("M"); plt.colorbar(label="C"); plt.title("Haldane phase diagram");"""),
    ("md", "## 6. QWZ Chern number versus mass"),
    ("code", """ms = np.linspace(-3, 3, 61)
plt.plot(ms, [hf.chern_number(M.qwz(m), 1, 30) for m in ms], "o-")
plt.xlabel("m"); plt.ylabel("C"); plt.title("QWZ: C = ±1 for 0 < |m| < 2");"""),
    ("md", "## 7. Chern numbers of Hofstadter bands"),
    ("code", """for p, q in [(1, 3), (1, 4), (1, 5), (2, 5)]:
    m = M.hofstadter(p, q)
    print(f"flux {p}/{q}: C =", [round(hf.chern_number(m, [n], 40)) for n in range(q)])"""),
    ("md", """## Exercises
1. Show numerically that the Berry curvature of gapped graphene (`M.honeycomb(delta=0.2)`) is opposite at K and K' (valley Hall), and that C = 0.
2. Compute the winding number of the Kitaev chain versus μ.
3. Check that the FHS Chern number converges already on a 10×10 mesh for the Haldane model."""),
] + quiz_cells(5, "Berry phase and Chern numbers")

# =============================================================================
NOTEBOOKS["06_wilson_loops_and_z2"] = [
    ("md", r"""# 06 · Wilson loops, Wannier charge centres and Z2

The Wilson loop $W(k_y)=\prod_j U(k_{x,j},k_{x,j+1})$ of occupied states has eigenvalues
$e^{-2\pi i \bar x_n(k_y)}$: the hybrid Wannier charge centres (WCC). Their winding gives
the Chern number; Kramers partner switching gives Z2."""),
    ("code", SETUP),
    ("md", "## 1. WCC of a Chern insulator: winding"),
    ("code", """fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
for ax, (m, occ, t) in zip(axes, [(M.haldane(), 1, "Haldane C=1"), (M.haldane(M=1), 1, "trivial"),
                                  (M.kane_mele(), 2, "Kane-Mele QSH")]):
    ks, wcc = hf.wilson_loop(m, occ, direction=0, nscan=61)
    hf.plot_wcc(wcc, ax=ax, kperp=ks, title=t)
plt.tight_layout()"""),
    ("md", "## 2. Z2 by the largest-gap method"),
    ("code", """for lv in [0.0, 0.2, 0.3, 0.4, 0.6]:
    print(f"Kane-Mele lv = {lv:.1f}: Z2 = {hf.z2_invariant(M.kane_mele(lv=lv), 2)}   (critical 3√3·λso = {3*np.sqrt(3)*0.06:.3f})")"""),
    ("code", """print("with Rashba 0.05:", hf.z2_invariant(M.kane_mele(lr=0.05), 2))
for Mv in [-1, 1, 3, 5, 9]:
    print(f"BHZ M/B = {Mv}: Z2 = {hf.z2_invariant(M.bhz(M=Mv), 2)}")"""),
    ("md", "## 3. Fu-Kane parity criterion (inversion-symmetric models)"),
    ("code", """par = np.kron(np.diag([1, -1]), np.eye(2))      # E (s-like) even, H (p-like) odd
for Mv in [-1, 1]:
    z2, deltas = hf.fu_kane_parity(M.bhz(M=Mv), 2, par)
    print(f"BHZ M = {Mv}: Z2 = {z2}, delta_i at TRIM = {deltas}")"""),
    ("md", "## 4. 3D topological insulators: (ν0; ν1ν2ν3)"),
    ("code", """print("Wilson-Dirac M= 1:", hf.z2_3d(M.wilson_dirac_ti(M=1), 2, nk=30, nscan=31))
print("Wilson-Dirac M= 5:", hf.z2_3d(M.wilson_dirac_ti(M=5), 2, nk=30, nscan=31))
print("Wilson-Dirac M=-1:", hf.z2_3d(M.wilson_dirac_ti(M=-1), 2, nk=30, nscan=31))
print("FKM dt=+0.4      :", hf.z2_3d(M.diamond_fkm(dt=0.4), 2, nk=30, nscan=31))
print("FKM dt=-0.4      :", hf.z2_3d(M.diamond_fkm(dt=-0.4), 2, nk=30, nscan=31))"""),
    ("md", "## 5. WCC flow on the planes of a 3D TI"),
    ("code", """wd = M.wilson_dirac_ti()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, kz in zip(axes, [0.0, 0.5]):
    ks, wcc = hf.wilson_loop(wd, 2, direction=0, scan=1, fixed={2: kz}, nscan=61)
    hf.plot_wcc(wcc, ax=ax, kperp=ks, title=f"k_z = {kz}")
plt.tight_layout()"""),
    ("md", "## 6. Surface Dirac cone of a 3D TI slab"),
    ("code", """slab = M.wilson_dirac_ti().slab(2, 14)
slab.plot_bands(["X", "G", "M"], ylim=(-1.5, 1.5), title="Wilson-Dirac slab: surface Dirac cone at Γ");"""),
    ("md", """## Exercises
1. Plot the WCC of BHZ for M = 1 and M = -1 side by side.
2. Find the Rashba strength at which the Kane-Mele bulk gap closes.
3. Build a slab of the FKM model and count the surface Dirac cones for dt > 0 and dt < 0."""),
] + quiz_cells(6, "Wilson loops and Z2")

# =============================================================================
NOTEBOOKS["07_kubo_hall_transport"] = [
    ("md", r"""# 07 · Kubo transport: anomalous, spin and orbital Hall conductivities

$$\sigma^{O}_{ab}=-\frac{e^2}{\hbar}\int\!\frac{d^dk}{(2\pi)^d}\sum_n f_n\,\Omega^{O}_{n,ab},\qquad
\Omega^{O}_{n,ab}=-2\,\mathrm{Im}\sum_{m\ne n}\frac{\langle n|J^O_a|m\rangle\langle m|v_b|n\rangle}{(E_n-E_m)^2+\eta^2},\quad J^O_a=\tfrac12\{O,v_a\}$$

$O=1$: anomalous Hall (AHC); $O=s_z$: spin Hall (SHC); $O=L_z$: orbital Hall (OHC).
Units: 2D in $e^2/h$, 3D in S/cm (Å lengths). The sign convention follows Wannier90."""),
    ("code", SETUP),
    ("md", "## 1. Quantised AHC of a Chern insulator, and its μ dependence"),
    ("code", """h = M.haldane(t2=0.1)
mus = np.linspace(-3.5, 3.5, 141)
r = hf.anomalous_hall(h, mus, nk=120)
plt.plot(mus, r.sigma); plt.axhline(-1, ls="--", c="gray")
plt.xlabel("μ"); plt.ylabel("σ_xy (e²/h)"); plt.title("Haldane: plateau -1 in the gap");
print(hf.anomalous_hall(h, 0.0, 120))"""),
    ("md", "## 2. Quantised spin Hall conductivity of Kane-Mele"),
    ("code", """km = M.kane_mele()
print(hf.spin_hall(km, 0.0, 120))
print("with Rashba 0.1:", hf.spin_hall(M.kane_mele(lr=0.1), 0.0, 120), " (no longer exactly quantised)")"""),
    ("md", "## 3. AHC of a 2D Rashba ferromagnet (metallic, non-quantised)"),
    ("code", """rf = M.rashba_square(alpha=0.3, J=0.5)
mus = np.linspace(-4.5, 4.5, 121)
plt.plot(mus, hf.anomalous_hall(rf, mus, nk=150, eta=0.01).sigma)
plt.xlabel("μ"); plt.ylabel("σ_xy (e²/h)"); plt.title("Rashba FM: Fermi-sea AHC");"""),
    ("md", "## 4. 3D: Weyl semimetal AHC in S/cm\n$\\sigma_{xy}=\\frac{e^2}{h}\\frac{\\Delta k_z}{2\\pi}$ per unit length."),
    ("code", """e2h_Scm = 3.874045e-5 * 1e8      # e^2/h per Angstrom in S/cm
for m_ in [1.5, 2.0, 2.5]:
    w = M.weyl_minimal(m_=m_)
    dk = 2*np.arccos(m_ - 2)/(2*np.pi)                     # node separation / (2 pi)
    kubo = float(hf.anomalous_hall(w, 0.0, 60))            # Kubo on a 60^3 mesh
    kz = np.linspace(0, 1, 200, endpoint=False)            # exact: sum of plane Chern numbers
    planes = -np.mean([round(hf.chern_number(w, 1, 40, fixed={2: z})) for z in kz]) * e2h_Scm
    print(f"m={m_}: Kubo {kubo:7.1f}   FHS planes {planes:7.1f}   analytic {dk*e2h_Scm:7.1f}  S/cm")"""),
    ("md", "The Kubo integral converges more slowly near the Weyl nodes (the curvature diverges there), so a dense mesh is needed. Summing FHS Chern numbers plane by plane is exact for each plane, which makes it a useful check."),
    ("md", "## 5. Orbital Hall effect without spin-orbit coupling\nThe p-orbital square lattice: the OHC is large even without SOC; adding SOC converts part of it into a spin Hall current."),
    ("code", """po = M.p_orbital_square()
mus = np.linspace(-2.5, 2.5, 101)
plt.plot(mus, hf.orbital_hall(po, mus, nk=100, eta=0.01).sigma, label="OHC, no SOC")
ps = M.p_orbital_square(lso=0.2)
plt.plot(mus, hf.orbital_hall(ps, mus, nk=100, eta=0.01).sigma, "--", label="OHC, λ=0.2")
plt.plot(mus, hf.spin_hall(ps, mus, nk=100, eta=0.01).sigma, label="SHC, λ=0.2")
plt.legend(); plt.xlabel("μ"); plt.ylabel("(ħ/e) e²/h"); plt.title("orbital vs spin Hall");"""),
    ("md", "## 6. Convergence and broadening"),
    ("code", """for nk in [20, 40, 80, 160]:
    print(nk, float(hf.anomalous_hall(M.haldane(t2=0.03), 0.0, nk)))"""),
    ("md", """## Exercises
1. Compute σ_xy of the QWZ model versus m and compare with -C.
2. Show that the AHC of `M.altermagnet_dwave()` vanishes for moments along z but not when Rashba SOC is added with in-plane moments.
3. Converge the SHC of the BHZ model and compare with Kane-Mele."""),
] + quiz_cells(7, "Kubo transport")


# =============================================================================
TR_SETUP = SETUP + "\nfrom hopforge import transport as tr\nfrom hopforge import kpm"

NOTEBOOKS["08_scattering_and_leads"] = [
    ("md", r"""# 08 · Quantum transport I: scattering regions, leads and Landauer conductance

hopforge includes a Kwant-style transport layer, written natively in hopforge.
Cut a scattering region of **any shape** out of **any** hopforge model, attach
semi-infinite leads, and compute transmissions:

$$G = \frac{e^2}{h}\,T,\qquad T_{ij}=\mathrm{Tr}\left[\Gamma_i G^r \Gamma_j G^a\right],\qquad
G^r=\left[E+i0-H_S-\sum_l\Sigma_l\right]^{-1}$$"""),
    ("code", TR_SETUP),
    ("md", "## 1. A wire and its conductance steps\nA perfect wire transmits each open channel fully, so $T(E)$ equals the number of channels."),
    ("code", """sq = M.square()
wire = tr.System(sq, lambda r: 0 <= r[0] < 20 and 0 <= r[1] < 10)
wire.attach_lead((-1, 0)); wire.attach_lead((1, 0))
tr.plot_system(wire);"""),
    ("code", """Es = np.linspace(-4, 0, 200)
T = wire.conductance(Es)
N = [wire.smatrix(E).num_modes(0) for E in Es]
plt.plot(Es, T, label="T(E)"); plt.plot(Es, N, "--", label="open channels")
plt.xlabel("E"); plt.ylabel("T"); plt.legend(); plt.title("quantised conductance steps");"""),
    ("md", "## 2. Inside a lead: bands, modes, self-energy\nThe surface Green's function of a 1D chain has the closed form $g=(E-i\\sqrt{4-E^2})/2$:"),
    ("code", """ch = tr.System(M.chain(), lambda r: 0 <= r[0] < 5); ch.attach_lead((-1,))
for E in (-1.5, 0.3, 1.9):
    print(E, np.round(ch.leads[0].surface_green(E)[0, 0], 6), np.round((E - 1j*np.sqrt(4 - E**2))/2, 6))
L = wire.leads[0]
k = np.linspace(-np.pi, np.pi, 300)
plt.plot(k, L.bands(k), "k", lw=0.8); plt.axhline(-2.5, c="r"); plt.xlabel("k (per lead cell)"); plt.title("lead bands")
print("right-moving modes at E=-2.5:", L.num_modes(-2.5))"""),
    ("md", "## 3. A quantum point contact (any shape)\nThe region is flood-filled from `start` through the hoppings while `shape(pos)` is True."),
    ("code", """def qpc(r, w=3.0):
    x, y = r[0] - 20, r[1] - 12
    return 0 <= r[0] < 40 and abs(y) < 12 and (abs(y) < w + 0.08 * x**2)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
s = tr.System(sq, qpc, start=(20, 12)); s.attach_lead((-1, 0)); s.attach_lead((1, 0))
tr.plot_system(s, ax=axes[0], bonds=False)
for w in (2.0, 3.0, 4.5):
    sw = tr.System(sq, lambda r, w=w: qpc(r, w), start=(20, 12)); sw.attach_lead((-1, 0)); sw.attach_lead((1, 0))
    Es = np.linspace(-4, -2.5, 80)
    axes[1].plot(Es, sw.conductance(Es), label=f"w={w}")
axes[1].legend(); axes[1].set_xlabel("E"); axes[1].set_ylabel("T"); axes[1].set_title("QPC steps");"""),
    ("md", "## 4. Unitarity and reciprocity"),
    ("code", """sd = tr.System(sq, lambda r: 0 <= r[0] < 20 and 0 <= r[1] < 10, onsite=tr.anderson(2.0, seed=1))
sd.attach_lead((-1, 0)); sd.attach_lead((1, 0))
S = sd.smatrix(-1.0)
print("T_10 =", S.transmission(1, 0), " T_01 =", S.transmission(0, 1))
print("T + R =", S.transmission(1, 0) + S.transmission(0, 0), " = N =", S.num_modes(0))"""),
    ("md", "## 5. Disorder averaging and Anderson localisation\n$\\langle\\ln T\\rangle$ decays linearly with length in the localised regime."),
    ("code", """lengths = [5, 10, 20, 40, 60]
for W in (1.0, 2.5):
    lnT = []
    for Lx in lengths:
        vals = []
        for seed in range(8):
            s = tr.System(sq, lambda r, Lx=Lx: 0 <= r[0] < Lx and 0 <= r[1] < 6, onsite=tr.anderson(W, seed=seed))
            s.attach_lead((-1, 0)); s.attach_lead((1, 0))
            vals.append(np.log(s.transmission(1, 0, -1.0)))
        lnT.append(np.mean(vals))
    plt.plot(lengths, lnT, "o-", label=f"W = {W}")
plt.xlabel("length L"); plt.ylabel("<ln T>"); plt.legend(); plt.title("localisation");"""),
    ("md", "## 6. Graphene: zigzag vs armchair nanoribbon junctions"),
    ("code", """g = M.honeycomb()
gs = tr.System(g, lambda r: 0 <= r[0] < 20 and 0 <= r[1] < 8)
gs.attach_lead((-1, 0)); gs.attach_lead((1, 0))
Es = np.linspace(-1.2, 1.2, 121)
plt.plot(Es, gs.conductance(Es)); plt.xlabel("E/t"); plt.ylabel("T"); plt.title("graphene ribbon device");"""),
    ("md", """## Exercises
1. Put a Gaussian potential bump in the wire (`onsite=`) and watch resonances in T(E).
2. Measure the localisation length versus width for W = 2.
3. Build a ring (Aharonov-Bohm) with a magnetic field (see notebook 09) and plot T versus flux."""),
] + quiz_cells(8, "Scattering and leads")

NOTEBOOKS["09_multiterminal_quantum_hall_qsh"] = [
    ("md", r"""# 09 · Quantum transport II: multi-terminal devices, quantum Hall and quantum spin Hall

Landauer-Büttiker: $I_i=\frac{e}{h}\sum_j G_{ij}V_j$ with $G_{ij}=-T_{ij}$ for $i\neq j$.
A magnetic field enters through Peierls phases $t_{ij}\to t_{ij}e^{i\int A\cdot dl}$."""),
    ("code", TR_SETUP),
    ("md", "## 1. A six-terminal Hall bar\nThe current leads (0, 1) carry the field in a Landau gauge that is invariant along x; the voltage probes (2–5) are field-free."),
    ("code", """B = 2*np.pi*0.05          # 0.05 flux quanta per plaquette
def hallbar(B, L=60, W=30):
    s = tr.System(M.square(), lambda r: 0 <= r[0] < L and 0 <= r[1] < W, peierls=tr.landau_gauge(B, "x"))
    s.attach_lead((-1, 0)); s.attach_lead((1, 0))
    for x0 in (12, 40): s.attach_lead((0, 1),  shape=lambda r, x0=x0: x0 <= r[0] < x0 + 8, peierls=None)
    for x0 in (12, 40): s.attach_lead((0, -1), shape=lambda r, x0=x0: x0 <= r[0] < x0 + 8, peierls=None)
    return s
hb = hallbar(B)
tr.plot_system(hb, bonds=False);"""),
    ("code", """Es = np.linspace(-3.9, -2.0, 120)
Rxy, Rxx = [], []
for E in Es:
    S = hb.smatrix(E)
    Rxy.append(S.four_terminal_resistance(0, 1, 2, 4)); Rxx.append(S.four_terminal_resistance(0, 1, 2, 3))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(Es, Rxy, label="R_xy"); ax.plot(Es, Rxx, label="R_xx")
for nu in (1, 2, 3, 4): ax.axhline(1/nu, c="gray", ls=":")
ax.set_ylim(-0.1, 1.3); ax.set_xlabel("E"); ax.set_ylabel("R (h/e²)"); ax.legend(); ax.set_title("quantum Hall plateaus");"""),
    ("md", "## 2. Edge-state current flow"),
    ("code", """S = hb.smatrix(-3.4)
bonds, J = S.current(0)
tr.plot_current(hb, bonds, J, title="current from the left lead (ν = 1): it follows one edge");"""),
    ("md", "## 3. Landau fan: T between the current leads versus B"),
    ("code", """Bs = np.linspace(0.02, 0.6, 60)
for E in (-3.5, -3.0):
    T = []
    for b in Bs:
        s = tr.System(M.square(), lambda r: 0 <= r[0] < 30 and 0 <= r[1] < 20, peierls=tr.landau_gauge(b, "x"))
        s.attach_lead((-1, 0)); s.attach_lead((1, 0))
        T.append(s.transmission(1, 0, E))
    plt.plot(Bs/(2*np.pi), T, label=f"E={E}")
plt.xlabel("flux per plaquette"); plt.ylabel("T"); plt.legend(); plt.title("edge channels drop out one by one");"""),
    ("md", "## 4. Quantum spin Hall: 2e²/h, spin-resolved, robust to disorder"),
    ("code", """km = M.kane_mele(lso=0.1)
for W in (0.0, 1.0, 2.0):
    s = tr.System(km, lambda r: 0 <= r[0] < 25 and 0 <= r[1] < 15, onsite=tr.anderson(W, seed=3) if W else None)
    s.attach_lead((-1, 0)); s.attach_lead((1, 0))
    S = s.smatrix(0.05)
    up = dict(spin="up"); dn = dict(spin="down")
    print(f"W={W}: T={S.transmission(1,0):.4f}   T_upup={S.transmission(1,0,up,up):.4f}   T_dndn={S.transmission(1,0,dn,dn):.4f}")"""),
    ("code", """s = tr.System(km, lambda r: 0 <= r[0] < 25 and 0 <= r[1] < 15)
s.attach_lead((-1, 0)); s.attach_lead((1, 0))
S = s.smatrix(0.05)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
b, Jup = S.current(0, operator=np.diag([1, 0]))
tr.plot_current(s, b, Jup, ax=axes[0], title="spin-up current: upper edge")
b, Jdn = S.current(0, operator=np.diag([0, 1]))
tr.plot_current(s, b, Jdn, ax=axes[1], title="spin-down current: lower edge");"""),
    ("md", "## 5. Breaking time reversal: a Zeeman field on the QSH edge"),
    ("code", """for J in (0.0, 0.05, 0.2):
    s = tr.System(M.kane_mele(lso=0.1), lambda r: 0 <= r[0] < 25 and 0 <= r[1] < 15,
                  onsite=lambda r, n, J=J: J * hf.pauli('x'))       # in-plane Zeeman in the region only
    s.attach_lead((-1, 0)); s.attach_lead((1, 0))
    print(f"in-plane Zeeman J={J}: T(E=0) = {s.transmission(1, 0, 0.0):.4f}")"""),
    ("md", "Breaking time reversal opens a gap at the edge Dirac point, so the edge states backscatter and the conductance at E = 0 is lost."),
    ("md", """## Exercises
1. Reverse B and verify the Onsager relation $R_{xy}(B)=-R_{xy}(-B)$.
2. Add disorder to the Hall bar: how wide do the plateaus stay?
3. Build a graphene Hall bar and find the half-integer sequence ν = ±2, ±6, …"""),
] + quiz_cells(9, "Multi-terminal, QH and QSH")

NOTEBOOKS["10_superconductors_ldos_wavefunctions"] = [
    ("md", r"""# 10 · Quantum transport III: superconducting junctions, LDOS, wavefunctions and currents

BdG: $H_{BdG}=\begin{pmatrix}H-\mu&\Delta\\\Delta^\dagger&-(H-\mu)^*\end{pmatrix}$; NS conductance (BTK):
$G = N_e - R_{ee} + R_{he}$."""),
    ("code", TR_SETUP),
    ("md", "## 1. Andreev reflection at an NS junction"),
    ("code", """mu, Delta = -2.0, 0.1
Nm = M.bdg(M.square(), mu=mu, delta=0.0)        # normal region / lead
Sm = M.bdg(M.square(), mu=mu, delta=Delta)      # superconducting lead
el, ho = dict(orbitals=["A_e"]), dict(orbitals=["A_h"])
def ns_junction(barrier):
    # tunnel barrier on the last column, right at the NS interface
    pot = lambda r, name: (barrier if r[0] >= 5 else 0.0) * (1 if name.endswith("_e") else -1)
    s = tr.System(Nm, lambda r: 0 <= r[0] < 6 and 0 <= r[1] < 10, onsite=pot)
    s.attach_lead((-1, 0)); s.attach_lead((1, 0), model=Sm)
    return s
Es = np.linspace(0, 0.3, 61)
for b in (0.0, 0.8, 1.5, 3.0):
    plt.plot(Es/Delta, tr.andreev_conductance(ns_junction(b), Es, 0, el, ho), label=f"barrier {b}")
plt.axvline(1, c="gray", ls=":"); plt.xlabel("E / Δ"); plt.ylabel("G (e²/h)"); plt.legend()
plt.title("BTK: Andreev doubling (2N = 10) → tunnelling spectroscopy");"""),
    ("code", """s = ns_junction(0.0); S = s.smatrix(0.02)
print("N_e =", S.num_modes(0, el), " R_ee =", round(S.transmission(0, 0, el, el), 4),
      " R_he =", round(S.transmission(0, 0, ho, el), 4), " -> G =", round(tr.andreev_conductance(s, 0.02, 0, el, ho)[0], 4))"""),
    ("md", "## 2. Majorana zero mode at the end of a Kitaev chain\nThe NS conductance through a topological Kitaev chain shows a quantised zero-bias peak of 2e²/h."),
    ("code", """lead = M.kitaev_chain(mu=0.5, t=1.0, delta=0.0)          # normal metallic lead
def kitaev_device(mu_wire, Vb):
    wire = M.kitaev_chain(mu=mu_wire, t=1.0, delta=0.6)
    barrier = lambda r, name: (Vb if r[0] < 1 else 0.0) * (1 if name == "e" else -1)
    s = tr.System(wire, lambda r: 0 <= r[0] < 60, onsite=barrier)
    s.attach_lead((-1,), model=lead)
    return s
Es = np.linspace(-0.4, 0.4, 321)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, mw in zip(axes, (0.5, 3.0)):
    for Vb in (1.5, 3.0, 5.0):
        G = tr.andreev_conductance(kitaev_device(mw, Vb), Es, 0, dict(orbitals=["e"]), dict(orbitals=["h"]))
        ax.plot(Es, G, label=f"barrier {Vb}")
    ax.set_title(("topological" if mw < 2 else "trivial") + f" wire (μ = {mw})")
    ax.set_xlabel("E"); ax.set_ylabel("G (e²/h)"); ax.legend(); ax.set_ylim(-0.1, 2.2)"""),
    ("md", "The topological wire shows a zero-bias peak of exactly 2e²/h (resonant Andreev reflection through the Majorana). A stronger barrier makes it narrower but not lower. The trivial wire shows no peak."),
    ("md", "## 3. LDOS of a disordered device"),
    ("code", """s = tr.System(M.square(), lambda r: (r[0]-15)**2 + (r[1]-15)**2 < 15**2, start=(15, 15),
              onsite=tr.anderson(1.5, seed=2))
s.attach_lead((-1, 0), shape=lambda r: 10 <= r[1] < 20); s.attach_lead((1, 0), shape=lambda r: 10 <= r[1] < 20)
S = s.smatrix(-1.0)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
tr.plot_system(s, ax=axes[0], bonds=False)
tr.plot_map(s, S.ldos(), ax=axes[1], title="LDOS at E = -1")
tr.plot_map(s, S.density(0), ax=axes[2], title="density injected from lead 0");"""),
    ("md", "## 4. Scattering wavefunctions and currents"),
    ("code", """psi = S.wavefunctions(0)
print("number of scattering states from lead 0:", psi.shape[1], "  open channels:", S.num_modes(0))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
tr.plot_map(s, np.abs(psi[:, 0])**2, ax=axes[0], cmap="magma", title="|ψ|² of one scattering state")
b, J = S.current(0); tr.plot_current(s, b, J, ax=axes[1], title="bond currents from lead 0");"""),
    ("code", """P = s.positions()
for xc in (8.5, 15.5, 22.5):
    cut = (P[b[:, 0], 0] - xc) * (P[b[:, 1], 0] - xc) < 0
    sign = np.sign(P[b[cut, 1], 0] - P[b[cut, 0], 0])
    print(f"current through x = {xc}: {np.sum(J[cut]*sign):.6f}   (T = {S.transmission(1, 0):.6f})")"""),
    ("md", """## Exercises
1. Plot the BTK conductance versus barrier height at E = 0 and compare with $2T_N^2/(2-T_N)^2$.
2. Show the Majorana peak height stays 2e²/h when the tunnel barrier is increased (it only narrows).
3. Map the LDOS of a Haldane flake connected to two leads at an energy in the bulk gap."""),
] + quiz_cells(10, "Superconductors, LDOS, wavefunctions")

NOTEBOOKS["11_kpm_and_continuum"] = [
    ("md", r"""# 11 · Large-scale methods: KPM and continuum discretisation

* **KPM** expands $\rho(E)$, LDOS and the Kubo-Bastin conductivity in Chebyshev polynomials:
  this scales linearly with system size (millions of orbitals).
* **Continuum discretisation** turns a $\mathbf{k}\cdot\mathbf{p}$ Hamiltonian into a
  tight-binding model on a grid, which can then go through every tool in hopforge."""),
    ("code", TR_SETUP),
    ("md", "## 1. KPM density of states of a 200×200 graphene sheet (80,000 orbitals)"),
    ("code", """big = M.honeycomb().supercell([200, 200])
H = big.hamiltonian_sparse()
print(H.shape, H.nnz, "non-zeros")
E, rho = kpm.dos(H, np.linspace(-3.2, 3.2, 600), num_moments=1000, num_vectors=10)
plt.plot(E, rho / big.norb * 2); plt.xlabel("E/t"); plt.ylabel("DOS per cell"); plt.title("KPM DOS of graphene");"""),
    ("md", "## 2. Disorder: the van Hove peaks broaden"),
    ("code", """sc = M.honeycomb().supercell([100, 100])
for W in (0.0, 1.0, 2.5):
    m = hf.add_disorder(sc, W, seed=1)
    E, rho = kpm.dos(m.hamiltonian_sparse(), np.linspace(-4, 4, 400), num_moments=600, num_vectors=4)
    plt.plot(E, rho / sc.norb * 2, label=f"W={W}")
plt.legend(); plt.xlabel("E"); plt.ylabel("DOS per cell");"""),
    ("md", "## 3. LDOS near a vacancy (exact KPM moments on chosen sites)"),
    ("code", """sc = M.honeycomb().supercell([30, 30]); vac = sc.remove_orbitals([0])
H = vac.hamiltonian_sparse(); E = np.linspace(-1, 1, 400)
P = vac.lat.positions; r0 = sc.lat.positions[0]
d = np.linalg.norm(P - r0, axis=1); near = np.argsort(d)[:3]; far = np.argsort(d)[-1]
L = kpm.ldos(H, list(near) + [far], E, num_moments=800)
for n, l in zip(["1st neighbour", "2nd", "3rd", "far away"], L): plt.plot(E, l, label=n)
plt.legend(); plt.xlabel("E"); plt.title("vacancy-induced zero mode on the opposite sublattice");"""),
    ("md", "## 4. Kubo-Bastin Hall conductivity of a large disordered Chern insulator"),
    ("code", """h = M.haldane(t2=0.15)
mus = np.linspace(-3.5, 3.5, 141)
for W in (0.0, 1.5, 4.0):
    m = hf.add_disorder(h.supercell([50, 50]), W, seed=2)
    sig = kpm.conductivity(m.hamiltonian_sparse(), m.velocity_sparse('x'), m.velocity_sparse('y'),
                           m.lat.cell_volume, mus, num_moments=256, num_vectors=16)
    plt.plot(mus, sig, lw=2 if W == 0 else 1.2, label=f"W={W}")
plt.plot(mus, hf.anomalous_hall(h, mus, 100, eta=0.02).sigma, "k--", lw=1, label="clean Kubo (k-space)")
plt.legend(); plt.xlabel("μ"); plt.ylabel("σ_xy (e²/h)"); plt.title("KPM Kubo-Bastin");"""),
    ("md", "## 5. Continuum discretisation: from k·p to tight binding"),
    ("code", """ex = "A*(k_x*sigma_x + k_y*sigma_y) + (M - B*(k_x**2 + k_y**2))*sigma_z"
p = dict(A=1.0, B=1.0, M=0.5)
f = hf.continuum.lambdify_continuum(ex, "xy", p)
k = np.linspace(-1.5, 1.5, 200)
Econt = np.array([np.linalg.eigvalsh(f(kk, 0)) for kk in k])
plt.plot(k, Econt[:, 0], "k", lw=3, label="continuum"); plt.plot(k, Econt[:, 1], "k", lw=3)
for n, a in enumerate((1.0, 0.5, 0.25)):
    m = hf.discretize(ex, "xy", a, p)
    Ed = m.energies(np.stack([k * a / (2*np.pi), 0*k], 1))
    plt.plot(k, Ed[:, 0], "--", color=f"C{n}", label=f"a = {a}"); plt.plot(k, Ed[:, 1], "--", color=f"C{n}")
plt.ylim(-3, 3); plt.legend(); plt.xlabel("k_x"); plt.title("discretised bands converge as a → 0")
print("Chern number of the discretised model:", hf.chern_number(hf.discretize(ex, "xy", 1.0, p), 1))"""),
    ("md", "## 6. A Bi2Se3-type 3D TI straight from k·p, then a slab"),
    ("code", """ti = "M0*tau_z - B*(k_x**2+k_y**2+k_z**2)*tau_z + A*(k_x*tau_x*sigma_x + k_y*tau_x*sigma_y + k_z*tau_x*sigma_z)"
mt = hf.discretize(ti, "xyz", 1.0, dict(M0=1.0, B=0.5, A=1.0))
print("Z2 indices:", hf.z2_3d(mt, 2, nk=24, nscan=25))
mt.slab(2, 16).plot_bands(["X", "G", "M"], ylim=(-1.5, 1.5), title="surface Dirac cone");"""),
    ("md", "## 7. Continuum model + transport: a disordered BHZ quantum-well wire"),
    ("code", """bhz = ("C0 - D*(k_x**2+k_y**2) + A*(k_x*tau_x*sigma_z - k_y*tau_y)"
       " + (M0 - B*(k_x**2+k_y**2))*tau_z")
qw = hf.discretize(bhz, "xy", 1.0, dict(C0=0.0, D=0.0, A=0.6, B=1.0, M0=0.3))
for W in (0.0, 1.0):
    s = tr.System(qw, lambda r: 0 <= r[0] < 30 and 0 <= r[1] < 20, onsite=tr.anderson(W, seed=5) if W else None)
    s.attach_lead((-1, 0)); s.attach_lead((1, 0))
    print(f"W={W}: T(E=0.1) = {s.transmission(1, 0, 0.1):.4f}  (QSH: two helical channels)")"""),
    ("md", "Note: exactly at the edge Dirac point (E = 0) a 20-site-wide ribbon has a tiny hybridisation gap between opposite edges, so probe slightly away from it or use wider systems."),
    ("md", """## Exercises
1. Converge the KPM DOS of graphene near E = 0 versus the number of moments.
2. Find the disorder strength at which the Haldane Hall plateau is destroyed.
3. Discretise a Rashba 2DEG `k_x**2 + k_y**2 + alpha*(k_y*sigma_x - k_x*sigma_y)` and plot its spin texture."""),
] + quiz_cells(11, "KPM and continuum")


# =============================================================================
NOTEBOOKS["12_parallel_parametric_noise"] = [
    ("md", r"""# 12 · Parallel workflows, parametric systems, magnetic gauges and shot noise

* `n_jobs=N` distributes energy and parameter sweeps, disorder realisations and k-meshes
  over N worker processes on one node, with one BLAS thread each. On a 40-core node use
  `n_jobs=40` or `export HOPFORGE_NJOBS=40`.
* Kwant-style **value functions** get their parameters at solve time: `params=dict(V=0.3)`.
* `field_gauge(B)` builds Peierls phases for any planar field profile $B(x,y)$.
* Shot noise and the Fano factor come from transmission eigenvalues: $F=\sum\tau(1-\tau)/\sum\tau$."""),
    ("code", TR_SETUP + "\nimport time\nprint('cores visible:', hf.parallel.available_cores())"),
    ("md", "## 1. Parallel energy sweep (results are identical to serial)"),
    ("code", """s = tr.System(M.square(), lambda r: 0 <= r[0] < 40 and 0 <= r[1] < 20, onsite=tr.anderson(1.0, seed=1))
s.attach_lead((-1, 0)); s.attach_lead((1, 0))
Es = np.linspace(-3.9, 0, 40)
t0 = time.time(); T1 = s.conductance(Es, n_jobs=1); t1 = time.time() - t0
t0 = time.time(); T2 = s.conductance(Es, n_jobs=-1); t2 = time.time() - t0
print(f"serial {t1:.2f}s   parallel {t2:.2f}s   identical: {np.allclose(T1, T2)}")
plt.plot(Es, T2); plt.xlabel("E"); plt.ylabel("T");"""),
    ("md", "On a 40-core Ibex/Shaheen node a SLURM script looks like `examples/slurm_40cores.sh`:"),
    ("code", "print(open('../examples/slurm_40cores.sh').read())"),
    ("md", "## 2. Parameters at solve time: a gate-defined barrier"),
    ("code", """gate = lambda r, name, Vg, w=4: Vg if abs(r[0] - 20) < w else 0.0
g = tr.System(M.square(), lambda r: 0 <= r[0] < 40 and 0 <= r[1] < 10, onsite=gate)
g.attach_lead((-1, 0)); g.attach_lead((1, 0))
Vgs = np.linspace(0, 2.5, 26)
res = tr.sweep(g, lambda S: S.transmission(1, 0), energies=[-3.0], params_list=[dict(Vg=v) for v in Vgs], n_jobs=-1)
plt.plot(Vgs, np.array(res).ravel(), "o-"); plt.xlabel("gate V_g"); plt.ylabel("T"); plt.title("pinch-off");"""),
    ("md", "## 3. Spectrum of the closed system versus a parameter"),
    ("code", """box = tr.System(M.square(), lambda r: 0 <= r[0] < 12 and 0 <= r[1] < 12,
                peierls=tr.field_gauge(lambda x, y, B: B, "x"))
Bs = np.linspace(0, 1.2, 60)
spec = tr.spectrum(box, [dict(B=b) for b in Bs], n_jobs=-1)
for n in range(25): plt.plot(Bs / (2*np.pi), [e[n] for e in spec], "k", lw=0.6)
plt.xlabel("flux per plaquette"); plt.ylabel("E"); plt.title("Fock-Darwin / Landau-level fan of a box");"""),
    ("md", "## 4. Arbitrary magnetic-field profiles: a magnetic barrier"),
    ("code", """Bprof = lambda x, y, B0: B0 * np.exp(-((x - 30) / 4)**2)          # localised field stripe
mb = tr.System(M.square(), lambda r: 0 <= r[0] < 60 and 0 <= r[1] < 20, peierls=tr.field_gauge(Bprof, "x"))
mb.attach_lead((-1, 0)); mb.attach_lead((1, 0))
B0s = np.linspace(0, 0.8, 30)
T = tr.sweep(mb, lambda S: S.transmission(1, 0), energies=[-3.0], params_list=[dict(B0=b) for b in B0s], n_jobs=-1)
plt.plot(B0s, np.array(T).ravel(), "o-"); plt.xlabel("B0"); plt.ylabel("T"); plt.title("magnetic barrier");"""),
    ("md", "## 5. Shot noise: ballistic, tunnelling and diffusive conductors"),
    ("code", """bar = tr.System(M.square(), lambda r: 0 <= r[0] < 20 and 0 <= r[1] < 10,
                onsite=lambda r, n, V: V if 9 <= r[0] < 11 else 0)
bar.attach_lead((-1, 0)); bar.attach_lead((1, 0))
print("ballistic  F =", bar.smatrix(-3.2, params=dict(V=0)).shot_noise()[1])
print("tunnelling F =", bar.smatrix(-3.2, params=dict(V=4)).shot_noise()[1])
def diffusive(seed):
    d = tr.System(M.square(), lambda r: 0 <= r[0] < 60 and 0 <= r[1] < 30, onsite=tr.anderson(1.6, seed=seed))
    d.attach_lead((-1, 0)); d.attach_lead((1, 0)); return d
Fs = hf.parallel.pmap(lambda sd: diffusive(sd).smatrix(-1.0).shot_noise()[1], range(8), n_jobs=-1)
print(f"diffusive  F = {np.mean(Fs):.3f} +- {np.std(Fs)/np.sqrt(len(Fs)):.3f}   (theory 1/3)")
tau = diffusive(0).smatrix(-1.0).transmission_eigenvalues()
plt.hist(tau, bins=20); plt.xlabel("τ"); plt.title("bimodal distribution of transmission eigenvalues");"""),
    ("md", "## 6. Parallel k-space and topology"),
    ("code", """h = M.haldane()
t0 = time.time(); a = hf.anomalous_hall(h, np.linspace(-3, 3, 61), 300, n_jobs=1).sigma; t1 = time.time() - t0
t0 = time.time(); b = hf.anomalous_hall(h, np.linspace(-3, 3, 61), 300, n_jobs=-1).sigma; t2 = time.time() - t0
print(f"AHC on 90,000 k-points: serial {t1:.2f}s, parallel {t2:.2f}s, identical {np.allclose(a, b)}")
print("FKM Z2 (parallel over the 6 planes):", hf.z2_3d(M.diamond_fkm(), 2, nk=30, nscan=31, n_jobs=-1))"""),
    ("md", """## Exercises
1. Measure the parallel speed-up versus `n_jobs` on your 40-core node.
2. Map T(E, Vg) on a 2D grid with `tr.sweep`.
3. Compute the Fano factor of a chaotic cavity (stadium shape with narrow leads): expect 1/4."""),
] + quiz_cells(12, "Parallel, parameters, gauges, noise")

NOTEBOOKS["13_wannier90_and_surface_states"] = [
    ("md", r"""# 13 · Wannier90 models and surface Green's functions

hopforge reads `seedname_hr.dat` / `seedname_tb.dat` (plus POSCAR / .win / centres), so
DFT-derived Wannier models go straight into every tool. The surface module computes
WannierTools-style surface spectral functions $A(\mathbf k_\parallel,E)=-\frac1\pi\mathrm{Im\,Tr}\,G_s$."""),
    ("code", SETUP + "\nfrom hopforge import io, surface"),
    ("md", "## 1. Import / export round trip\n(Here a hopforge model plays the role of a DFT Wannier model; with your own files just call `read_wannier90`.)"),
    ("code", """fkm = M.diamond_fkm()
fkm.to_wannier90_hr("fkm_hr.dat")
A = fkm.lat.vectors
pos = np.repeat(fkm.lat.positions, 2, axis=0)
w90 = io.read_wannier90("fkm_hr.dat", lattice=A, positions=pos, spinful=True)
k = np.random.rand(5, 3)
print(w90, " identical H(k):", np.allclose(w90.hamiltonian(k), fkm.hamiltonian(k)))
print("Z2 of the imported model:", hf.z2_3d(w90, 2, nk=30, nscan=31, n_jobs=-1))"""),
    ("md", """Typical use with real DFT output:
```python
m = hf.io.read_wannier90("Bi2Se3_hr.dat", lattice="POSCAR", positions="Bi2Se3_centres.xyz",
                         spinful=True, spin_order="interleaved")
m.add_onsite(-E_fermi)            # put E_F at zero
```"""),
    ("md", "## 2. TI surface Dirac cone"),
    ("code", """wd = M.wilson_dirac_ti()
path, E, Asurf = surface.spectral(wd, direction=2, kpath=["X", "G", "M"], energies=np.linspace(-1.5, 1.5, 200),
                                  n=120, eta=5e-3, n_jobs=-1)
_, _, Abulk = surface.spectral(wd, 2, ["X", "G", "M"], np.linspace(-1.5, 1.5, 200), n=120, eta=5e-3, bulk=True, n_jobs=-1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
surface.plot_spectral(path, E, Asurf, ax=ax[0], title="surface (top)")
surface.plot_spectral(path, E, Abulk, ax=ax[1], title="bulk");"""),
    ("md", "## 3. Spin-resolved surface spectra: helical edge of the Kane-Mele model\n`operator=` gives $-\\frac1\\pi\\mathrm{Im\\,Tr}[O\\,G_s]$. On the top edge, right movers carry spin up and left movers spin down."),
    ("code", """km = M.kane_mele(lso=0.1)
Sz = km.spin_operator('z')
path, E, Asz = surface.spectral(km, direction=1, kpath=[[-0.5, 0], [0, 0], [0.5, 0]],
                                energies=np.linspace(-0.6, 0.6, 160), n=150, eta=3e-3, operator=Sz, n_jobs=-1)
fig, ax = plt.subplots(figsize=(5, 4))
v = np.abs(Asz).max() / 20
ax.pcolormesh(path.x, E, Asz.T, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
ax.set_xticks(path.ticks); ax.set_xticklabels(path.labels); ax.set_title("edge spectral weight of s_z (red = up)");"""),
    ("md", "Note: in the idealised 4-band Wilson-Dirac TI the z-surface states are eigenstates of $\\tau_y\\sigma_z$, so the bare in-plane spin $\\langle\\tau_0\\sigma_{x,y}\\rangle$ vanishes on that surface and the locked texture is carried by orbital-spin operators. Realistic Wannier models of Bi$_2$Se$_3$ do give a finite spin polarisation."),
    ("md", "## 4. Weyl semimetal Fermi arcs"),
    ("code", """w = M.weyl_minimal()
K1, K2, Aarc = surface.fermi_arc_map(w, direction=0, energy=0.0, nk=121, eta=3e-3, n_jobs=-1)
hf.plot_kmap(np.log10(Aarc + 1e-2), extent=(-0.5, 0.5, -0.5, 0.5), symmetric=False, cmap="inferno",
             title="Fermi arc on the x-surface")
plt.xlabel("k_y"); plt.ylabel("k_z"); plt.axhline(0.25, c="c", ls=":"); plt.axhline(-0.25, c="c", ls=":");"""),
    ("md", "## 5. Kane-Mele edge spectral function (2D model, 1D edge)"),
    ("code", """km = M.kane_mele(lso=0.1)
path, E, Aedge = surface.spectral(km, direction=1, kpath=[[-0.5, 0], [0, 0], [0.5, 0]],
                                  energies=np.linspace(-1, 1, 200), n=150, eta=3e-3, n_jobs=-1)
surface.plot_spectral(path, E, Aedge, title="Kane-Mele zigzag edge: helical states");"""),
    ("md", """## Exercises
1. Compare the surface spectral function with slab bands (`wd.slab(2, 20)`).
2. Plot the bottom surface (`side="bottom"`) and explain the difference.
3. Import your own Wannier90 model and plot its surface states."""),
] + quiz_cells(13, "Wannier90 and surface states")

NOTEBOOKS["14_boltzmann_thermoelectric_edelstein"] = [
    ("md", r"""# 14 · Boltzmann transport, thermoelectrics and the Edelstein effect

$L^{(n)}_{ab}=\frac{1}{VN_k}\sum(-\partial_Ef)(E-\mu)^n v_av_b$, so
$\sigma=L^{(0)}$, $S=-\frac{1}{k_BT}L^{(0)-1}L^{(1)}$ and $\kappa=(L^{(2)}-L^{(1)}L^{(0)-1}L^{(1)})/T$.
Edelstein: $\delta\langle O\rangle = \chi E$ with extrinsic (∝τ) and intrinsic interband parts."""),
    ("code", SETUP + "\nfrom hopforge import response as R"),
    ("md", "## 1. Conductivity, Seebeck and the Wiedemann-Franz law"),
    ("code", """mus = np.linspace(-3.8, 3.8, 77)
r = R.boltzmann(M.square(), mus, kT=0.05, nk=200, n_jobs=-1)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
ax[0].plot(mus, r.sigma_over_tau[:, 0, 0]); ax[0].set_title("σ/τ")
ax[1].plot(mus, r.seebeck_uV_per_K[:, 0, 0]); ax[1].set_title("Seebeck (μV/K if E in eV)")
ax[2].plot(mus, r.lorenz / (np.pi**2/3)); ax[2].set_title("Lorenz number / L0"); ax[2].set_ylim(0.8, 1.2)
for a in ax: a.set_xlabel("μ")"""),
    ("md", "## 2. Thermoelectric power factor of a two-band semiconductor"),
    ("code", """semi = M.honeycomb(delta=0.5)
mus = np.linspace(-1.5, 1.5, 121)
for kT in (0.025, 0.05, 0.1):
    r = R.boltzmann(semi, mus, kT=kT, nk=150, n_jobs=-1)
    pf = r.seebeck[:, 0, 0]**2 * r.sigma_over_tau[:, 0, 0]
    plt.plot(mus, pf, label=f"kT={kT}")
plt.legend(); plt.xlabel("μ"); plt.ylabel("S²σ/τ (natural units)"); plt.title("power factor peaks near the band edges");"""),
    ("md", "## 3. Rashba-Edelstein effect: current-induced spin polarisation"),
    ("code", """ra = M.rashba_square(alpha=0.3)
mus = np.linspace(-4.2, 0, 60)
for a in "xyz":
    e = R.edelstein(ra, ra.spin_operator(a), "x", mu=mus, kT=0.02, nk=150, n_jobs=-1)
    plt.plot(mus, e["total"], label=f"<s_{a}> for E ∥ x")
plt.legend(); plt.xlabel("μ"); plt.ylabel("χ (per cell, τ = 1)"); plt.title("only s_y responds to E_x");"""),
    ("md", "## 4. Intrinsic (field-like vs damping-like) Edelstein in a Rashba ferromagnet"),
    ("code", """for J in (0.0, 0.3, 0.6):
    fm = M.rashba_square(alpha=0.3, J=J)
    e = R.edelstein(fm, fm.spin_operator('x'), "x", mu=-2.0, kT=0.02, nk=150, n_jobs=-1)
    print(f"J={J}: <s_x> extrinsic {e['extrinsic'][0]:+.5f}   intrinsic {e['intrinsic'][0]:+.5f}")"""),
    ("md", "## 5. Orbital Edelstein effect: the orbital Rashba model\nA surface field breaks inversion and allows bond-odd hybridisation $\\gamma\\,\\hat d_x$ between $p_z$ and $p_x$ (and $\\hat d_y$ for $p_z$-$p_y$), which is the orbital Rashba effect (Park et al., PRB 2011). It produces an orbital Edelstein response even without spin-orbit coupling."),
    ("code", """po = M.p_orbital_square()
gamma = 0.3
po.add_neighbors(lambda d: gamma * d[0], 1, ["X:pz"], ["X:px"])     # odd in d: inversion broken
po.add_neighbors(lambda d: gamma * d[1], 1, ["X:pz"], ["X:py"])
po.add_onsite(-0.5, "X:pz")                                          # crystal-field splitting
mus = np.linspace(-2.5, 2.5, 60)
for a in "xyz":
    e = R.edelstein(po, po.orbital_operator(a), "x", mu=mus, kT=0.02, nk=120, n_jobs=-1)
    plt.plot(mus, e["total"], label=f"<L_{a}>")
plt.legend(); plt.xlabel("μ"); plt.ylabel("χ (per cell, τ = 1)")
plt.title("orbital Rashba-Edelstein: E ∥ x induces L_y");"""),
    ("md", """## Exercises
1. Convert σ/τ to S/m with `r.sigma_SI(tau_fs=10)` for a 3D model.
2. Find where |S| peaks relative to the band edge of the gapped honeycomb model.
3. Compare spin and orbital Edelstein responses of `M.p_orbital_square(lso=0.2)`."""),
] + quiz_cells(14, "Boltzmann and Edelstein")

NOTEBOOKS["15_nonlinear_hall_orbital_optical"] = [
    ("md", r"""# 15 · Nonlinear Hall effect, orbital magnetism and optical conductivity"""),
    ("code", SETUP + "\nfrom hopforge import response as R"),
    ("md", "## 1. Berry curvature dipole of a tilted, strained massive Dirac model"),
    ("code", """def tilted(tilt=0.4, strain=0.3, m=1.5):
    q = M.qwz(m)
    q.add_hopping_matrix(-0.5j * tilt * np.eye(2), ["a", "b"], ["a", "b"], [1, 0])   # tilt: + tilt sin kx
    q.add_hopping_matrix(0.5 * strain * hf.pauli('z'), ["a", "b"], ["a", "b"], [1, 0])  # anisotropic mass
    return q
q = tilted()
mus = np.linspace(-1.5, 1.5, 61)
D = R.berry_curvature_dipole(q, mus, kT=0.03, nk=250, n_jobs=-1)
plt.plot(mus, D[:, 0], label="D_x"); plt.plot(mus, D[:, 1], label="D_y")
plt.legend(); plt.xlabel("μ"); plt.title("Berry curvature dipole: peaks at the band edges");"""),
    ("code", """print("no tilt  :", R.berry_curvature_dipole(tilted(tilt=0), 0.8, kT=0.03, nk=200)[0])
print("with tilt:", R.berry_curvature_dipole(tilted(), 0.8, kT=0.03, nk=200)[0])"""),
    ("md", "## 2. Valley-contrasting orbital moment of gapped graphene"),
    ("code", """g = M.honeycomb(delta=0.3)
k1, k2 = np.meshgrid(np.linspace(0, 1, 120, endpoint=False), np.linspace(0, 1, 120, endpoint=False), indexing="ij")
E, Om, m = R.orbital_moment(g, np.stack([k1.ravel(), k2.ravel()], 1))
hf.plot_kmap(m[:, 1].reshape(120, 120) * R.MU_B_PER_UNIT, title="conduction-band orbital moment (μ_B for eV, Å)");
K, Kp = g.special_points["K"], g.special_points["K'"]
print("m(K) =", R.orbital_moment(g, K)[2][0, 1]*R.MU_B_PER_UNIT, "  m(K') =", R.orbital_moment(g, Kp)[2][0, 1]*R.MU_B_PER_UNIT)"""),
    ("md", "## 3. Modern orbital magnetisation and the Středa formula"),
    ("code", """h = M.haldane(t2=0.15)
mus = np.linspace(-3.5, 3.5, 141)
Mo = R.orbital_magnetization(h, mus, kT=0.01, nk=150, n_jobs=-1)
plt.plot(mus, Mo, label="M(μ)")
plt.plot(mus, (mus) * h.lat.cell_volume / (2*np.pi) + Mo[70], "--", label="slope C·A/2π in the gap")
plt.legend(); plt.xlabel("μ"); plt.ylabel("M per cell"); plt.ylim(Mo.min()-0.1, Mo.max()+0.1);"""),
    ("md", "## 4. Optical conductivity of graphene: the universal plateau and Pauli blocking"),
    ("code", """om = np.linspace(0.05, 6.5, 200)
for mu in (0.0, 0.5, 1.0):
    s = R.optical_conductivity(M.honeycomb(), om, mu=mu, kT=0.01, nk=300, eta=0.03, n_jobs=-1)
    plt.plot(om, s.real / 0.125, label=f"μ = {mu}")
plt.axhline(1, c="gray", ls=":"); plt.legend(); plt.xlabel("ħω / t"); plt.ylabel("Re σ_xx / (e²/8ħ)")
plt.title("universal conductivity, van Hove peak at 2t, Pauli blocking below 2μ");"""),
    ("md", "## 5. Optical Hall conductivity of a Chern insulator"),
    ("code", """om = np.linspace(0.01, 4, 120)
s = R.optical_conductivity(M.haldane(t2=0.15), om, mu=0.0, nk=200, eta=0.03, component="xy", n_jobs=-1)
plt.plot(om, 2*np.pi*s.real, label="Re σ_xy (e²/h)"); plt.plot(om, 2*np.pi*s.imag, label="Im σ_xy")
plt.axhline(-1, c="gray", ls=":"); plt.legend(); plt.xlabel("ω"); plt.title("ω → 0 limit: the quantised Hall value");"""),
    ("md", """## Exercises
1. Show that the BCD vanishes for the C3-symmetric gapped honeycomb model.
2. Compute the orbital magnetisation of the d-wave altermagnet with Rashba SOC.
3. Add a Drude term (`drude_tau=`) to the doped-graphene optical conductivity."""),
] + quiz_cells(15, "Nonlinear, orbital, optical")

NOTEBOOKS["16_meanfield_hubbard_and_floquet"] = [
    ("md", r"""# 16 · Mean-field Hubbard magnetism and Floquet engineering

* Hartree-Fock: $U n_\uparrow n_\downarrow \to U(\mathrm{Tr}\,\rho_i-\rho_i)$ per site (collinear and noncollinear).
* Floquet: $A(t)$ enters as a Peierls phase, and the high-frequency Hamiltonian is
  $H_{\rm eff}=H_0+\sum_{m>0}[H_m,H_{-m}]/m\omega$."""),
    ("code", SETUP + "\nfrom hopforge import meanfield as MF, floquet as F"),
    ("md", "## 1. Antiferromagnetism of the honeycomb Hubbard model"),
    ("code", """g = M.honeycomb().make_spinful()
Us = np.linspace(1.0, 5.0, 17)
res = MF.phase_diagram(g, Us, filling=0.5, init={"A": (0, 0, 1), "B": (0, 0, -1)}, nk=48, kT=0.002, maxiter=500)
plt.plot(Us, [abs(r.moments[0, 2]) for r in res], "o-"); plt.axvline(2.23, c="gray", ls=":")
plt.xlabel("U / t"); plt.ylabel("|m| per site"); plt.title("mean-field Néel order (U_c ≈ 2.23 t)");"""),
    ("code", """r = res[-4]
print(r)
r.model.plot_bands(["G", "K", "M", "G"], weights=r.model.spin_operator('z'), title=f"AFM bands, U = {Us[-4]:.2f}");"""),
    ("md", "## 2. Square lattice: Néel order for any U, gap = U|m|"),
    ("code", """sq = M.square().make_spinful().supercell([[1, 1], [1, -1]])
for U in (1.0, 2.0, 4.0):
    r = MF.hubbard(sq, U, filling=0.5, init={0: (0, 0, 1), 1: (0, 0, -1)}, nk=60, kT=0.002, maxiter=600)
    print(f"U={U}: m={r.moments[0, 2]:+.4f}   gap={hf.band_gap(r.model, 2, 60):.4f}   U|m|={U*abs(r.moments[0, 2]):.4f}")"""),
    ("md", "## 3. Ferromagnetism near a van Hove singularity (Stoner)"),
    ("code", """sq1 = M.square(t2=-0.3).make_spinful()
for U in (2.0, 4.0, 6.0):
    r = MF.hubbard(sq1, U, filling=0.3, init="ferro", nk=60, kT=0.005, maxiter=600)
    print(f"U={U}: m_z = {r.moments[0, 2]:.4f}  converged {r.converged}")"""),
    ("md", "## 4. Floquet: light-induced Chern insulator in graphene"),
    ("code", """g0 = M.honeycomb()
K = g0.special_points["K"]
for A0 in (0.2, 0.4, 0.6):
    d = F.circular(A0, 10.0)
    print(f"A0={A0}: gap(K) eff={np.diff(F.EffectiveModel(g0, d).energies(K))[0]:.5f}  "
          f"exact={np.diff(F.FloquetModel(g0, d).quasienergies(K)[0])[0]:.5f}  "
          f"C={hf.chern_number(F.EffectiveModel(g0, d), 1, 40):+.0f}")
print("opposite helicity: C =", round(hf.chern_number(F.EffectiveModel(g0, F.circular(0.4, 10.0, -1)), 1, 40)))
print("linear polarisation: C =", round(hf.chern_number(F.EffectiveModel(g0, F.linear(0.4, 10.0)), 1, 40)))"""),
    ("code", """d = F.circular(0.6, 5.0)
Fl = F.FloquetModel(g0, d, n_max=4)
p = hf.kpath(g0, ["G", "K", "M", "G"], 200)
q = np.array([Fl.quasienergies(k)[0] for k in p.k])
Ee = F.EffectiveModel(g0, d).energies(p.k)
plt.plot(p.x, q, "k", label="exact quasienergies"); plt.plot(p.x, Ee, "r--", lw=0.8)
plt.xticks(p.ticks, p.labels); plt.ylim(-3.2, 3.2); plt.title("Floquet bands (ω = 5t): replicas fold back into the zone");"""),
    ("md", "## 5. Floquet Hall conductivity (effective model)"),
    ("code", """Eff = F.EffectiveModel(g0, F.circular(1.0, 8.0))      # strong drive: gap ~ 0.1 t
print("gap at K:", np.diff(Eff.energies(K))[0], "  C =", round(hf.chern_number(Eff, 1, 60)))
print(hf.anomalous_hall(Eff, 0.0, 300, n_jobs=-1))"""),
    ("md", """## Exercises
1. Compute the mean-field phase diagram of the half-filled kagome Hubbard model (try ferro and 120° initial guesses on a supercell).
2. Find where the high-frequency expansion breaks down by comparing effective and exact gaps versus ω.
3. Drive the Kane-Mele model with circular light: can you obtain a spin-polarised quantum anomalous Hall phase?"""),
] + quiz_cells(16, "Mean field and Floquet")


# =============================================================================
NOTEBOOKS["17_step_by_step_build_a_new_hamiltonian"] = [
    ("md", r"""# 17 · Step by step: build a new Hamiltonian from a paper and simulate it

We rebuild, from scratch, the **Lieb-lattice topological insulator of Weeks & Franz**
(PRB 82, 085310 (2010)) and take it through every stage of a real study:

1. Write down the Hamiltonian.
2. Define the lattice and orbitals.
3. Check the geometry (neighbour shells).
4. Add hoppings, including complex spin-orbit phases.
5. Check Hermiticity and symmetries.
6. Bands and DOS.
7. Topology (Z2), compared with the paper.
8. Edge states (bulk-boundary correspondence).
9. Quantum transport through a device with disorder.
10. Parameter scans and a phase diagram (in parallel).
11. Package it: register the model and write a test.

$$H=-t\sum_{\langle ij\rangle\sigma}c^\dagger_{i\sigma}c_{j\sigma}
+i\lambda\sum_{\langle\langle ij\rangle\rangle}\nu_{ij}\,c^\dagger_i\sigma_z c_j,
\qquad \nu_{ij}=\pm1 \text{ (hopping turns left/right through the corner site)}$$"""),
    ("code", SETUP + "\nfrom hopforge import transport as tr\nfrom hopforge.models._util import chirality"),
    ("md", "## Step 1-2: lattice and orbitals\nThe Lieb lattice has three sites per square cell: a corner (Cu) and two edge centres (Ox, Oy). Positions are fractional."),
    ("code", """lat = hf.Lattice([[1, 0], [0, 1]],
                 [("Cu", [0.0, 0.0]), ("Ox", [0.5, 0.0]), ("Oy", [0.0, 0.5])])
model = hf.Model(lat, spinful=True, name="Weeks-Franz Lieb TI")
print(model)"""),
    ("md", "## Step 3: inspect the geometry before adding anything\nWhich bonds are 1st, 2nd and 3rd neighbours? The SOC term lives on the 2nd-neighbour Ox–Oy bonds."),
    ("code", """bare = hf.Model(lat)            # spinless copy for drawing
hf.plot_lattice(bare, shells=3);
for n, sh in enumerate(hf.plotting.neighbor_shells(bare, 3), 1):
    pairs = {(bare.lat.names[i], bare.lat.names[j]) for i, j, R, d in sh["bonds"]}
    print(f"shell {n}: d = {sh['distance']:.3f}  pairs = {sorted(pairs)}")"""),
    ("md", "## Step 4: hoppings\nNN hopping on shell 1. For the SOC we loop over the 2nd-shell Ox–Oy bonds and compute the chirality ν from the path through the shared Cu neighbour. The Hermitian conjugate is added automatically."),
    ("code", """t, lam = 1.0, 0.2
model.add_neighbors(-t, shell=1)
nso = 0
for i, j, R, d in model.neighbors(2):
    if {model.lat.names[i], model.lat.names[j]} == {"Ox", "Oy"}:
        nu = chirality(model, i, j, R, nn_dist=0.5)
        model.add_hopping(1j * lam * nu * hf.pauli("z"), i, j, R)
        nso += 1
print("SOC bonds per cell:", nso)
model.special_points = {"G": np.zeros(2), "X": np.array([0.5, 0]), "M": np.array([0.5, 0.5])}
print(model.summary())"""),
    ("md", "## Step 5: sanity checks: Hermiticity, time reversal, S_z conservation"),
    ("code", """k = np.random.rand(5, 2)
H = model.hamiltonian(k)
print("Hermitian:", np.allclose(H, H.conj().transpose(0, 2, 1)))
# time reversal: Theta = i s_y K  ->  H(-k) = T H(k)* T^-1
T = np.kron(np.eye(3), 1j * hf.pauli('y'))
print("TRS:", np.allclose(model.hamiltonian(-k), T @ H.conj() @ T.conj().T))
Sz = model.spin_operator('z')
print("[H, S_z] = 0:", np.allclose(H @ Sz - Sz @ H, 0))"""),
    ("md", "## Step 6: bands and DOS\nWithout SOC there is a flat band touching a Dirac cone at M. The SOC gaps it out."),
    ("code", """fig, ax = plt.subplots(1, 3, figsize=(15, 4))
m0 = hf.models.lieb().make_spinful(); m0.special_points = model.special_points
m0.plot_bands(["G", "X", "M", "G"], ax=ax[0], title="λ = 0: flat band + Dirac cone")
model.plot_bands(["G", "X", "M", "G"], ax=ax[1], title=f"λ = {lam}: gaps open")
hf.plot_dos(hf.dos(model, nk=150, sigma=0.03), ax=ax[2])
print("gaps:", hf.band_gap(model, 2, 60), hf.band_gap(model, 4, 60))"""),
    ("md", "## Step 7: topology, compared with the paper\nWeeks & Franz find a Z2 topological insulator at 1/3 and 2/3 filling."),
    ("code", """print("Z2 (1/3 filling) =", hf.z2_invariant(model, 2), "  Z2 (2/3 filling) =", hf.z2_invariant(model, 4))
ks, wcc = hf.wilson_loop(model, 2, nscan=61)
hf.plot_wcc(wcc, kperp=ks, title="Kramers partner switching (Z2 = 1)");"""),
    ("md", "## Step 8: edge states (bulk-boundary correspondence)"),
    ("code", """rib = model.ribbon(1, 30)
b = hf.bands(rib, ["-X", "G", "X"], n=300, vectors=True)
w = b.weights(rib.spin_operator('z') @ rib.projector(list(range(rib.norb - 6, rib.norb))))
fig, ax = plt.subplots(figsize=(5, 4))
for n in range(rib.nbands):
    ax.scatter(b.x, b.energies[:, n], c=w[:, n], s=2, cmap="coolwarm", vmin=-0.3, vmax=0.3)
ax.set_ylim(-2.5, 2.5); ax.set_xticks(b.path.ticks); ax.set_xticklabels(b.path.labels)
ax.set_title("helical edge states (colour: top-edge s_z)");"""),
    ("md", "## Step 9: quantum transport through a disordered device\nInside the gap the helical edges carry exactly 2e²/h and are immune to non-magnetic disorder."),
    ("code", """Ek = model.energies(hf.kmesh(model, 60))
E_in_gap = 0.5 * (Ek[:, 1].max() + Ek[:, 2].min())     # middle of the gap at 1/3 filling
print("probing at E =", round(E_in_gap, 3), " (gap edges", round(Ek[:, 1].max(), 3), round(Ek[:, 2].min(), 3), ")")
for W in (0.0, 1.0):
    dev = tr.System(model, lambda r: 0 <= r[0] < 20 and 0 <= r[1] < 15, onsite=tr.anderson(W, seed=2) if W else None)
    dev.attach_lead((-1, 0)); dev.attach_lead((1, 0))
    print(f"disorder W = {W}: T = {dev.transmission(1, 0, E_in_gap):.4f} (e²/h)")"""),
    ("md", "## Step 10: parameter scan and phase diagram\nAdd a potential ±Δ on Ox/Oy (it breaks C4, like uniaxial strain) that competes with λ, then map Z2 in parallel. (A Cu/O offset is *not* enough: it never closes the gap at 1/3 filling. Try it.)"),
    ("code", """def build(lam, delta):
    m = hf.Model(lat, spinful=True)
    m.add_neighbors(-1.0, 1)
    for i, j, R, d in m.neighbors(2):
        if {m.lat.names[i], m.lat.names[j]} == {"Ox", "Oy"}:
            m.add_hopping(1j * lam * chirality(m, i, j, R, 0.5) * hf.pauli("z"), i, j, R)
    m.add_onsite(delta, "Ox"); m.add_onsite(-delta, "Oy")
    return m
pd = hf.phases.topological_phase_diagram(build, "λ_SO", np.linspace(0.05, 0.4, 10), "Δ", np.linspace(0, 3.0, 13),
                                          occ=2, invariant="z2", nk=30, n_jobs=-1)
hf.phases.plot_phase_diagram(pd, contour="gap", title="Weeks-Franz Lieb model: Z2 at 1/3 filling");"""),
    ("md", """## Step 11: package your model for the community
```python
@hf.models.register_model("weeks_franz_lieb", 2, "Lieb-lattice Z2 TI, Weeks & Franz PRB 82, 085310 (2010)")
def weeks_franz_lieb(t=1.0, lam=0.2, delta=0.0):
    ...                       # the code of Steps 2-4
    return m

# tests/test_my_model.py
def test_weeks_franz_z2():
    assert hf.z2_invariant(hf.models.get("weeks_franz_lieb"), 2) == 1
```
Then open a pull request (see CONTRIBUTING.md). Every model enters the library with a test against a published result."""),
] + quiz_cells(17, "Building a Hamiltonian step by step")

NOTEBOOKS["18_step_by_step_orbitronics_bilayers_moire"] = [
    ("md", r"""# 18 · Step by step: a multi-orbital model, then orbital/spin Hall and torques in bilayers and moiré bilayers

1. Slater-Koster p-orbital layers: a heavy layer and a ferromagnet.
2. Atomic L·S spin-orbit coupling and exchange.
3. Orbital and spin Hall conductivities of each layer.
4. Stack them into a bilayer (heterostructure).
5. Layer-resolved Edelstein responses.
6. Spin-orbit, orbital and self-torques (Kubo-Středa: field-like and damping-like parts).
7. The same in a twisted (moiré) bilayer.

The torque decomposition follows Freimuth, Blügel & Mokrousov (PRB 2014) and Go et al. (orbital torque).
These are model calculations for building intuition, not material predictions."""),
    ("code", SETUP + "\nfrom hopforge import response as R, moire as MO\nfrom hopforge.models._util import LS_matrix"),
    ("md", "## Step 1-2: build the two layers with Slater-Koster parameters"),
    ("code", """def p_layer(pps, ppp, pps2, ppp2, lso=0.0, J=0.0, eps_z=0.0, name="layer"):
    lat = hf.Lattice([[1, 0], [0, 1]], [("X:px", [0, 0]), ("X:py", [0, 0]), ("X:pz", [0, 0])])
    m = hf.Model(lat, spinful=True, name=name)
    m.add_slater_koster({"pps": pps, "ppp": ppp}, 1)          # nearest neighbours
    m.add_slater_koster({"pps": pps2, "ppp": ppp2}, 2)        # next-nearest (px-py mixing)
    if lso: m.add_onsite_matrix(LS_matrix(1, lso), ["X:px", "X:py", "X:pz"])
    if J:   m.add_exchange(J, (0, 0, 1))
    m.add_onsite(eps_z, "X:pz")
    m.special_points = {"G": np.zeros(2), "X": np.array([0.5, 0]), "M": np.array([0.5, 0.5])}
    return m
heavy = p_layer(1.0, -0.3, 0.3, -0.1, lso=0.0, name="light-metal (no SOC)")
ferro = p_layer(0.8, -0.25, 0.2, -0.08, lso=0.05, J=0.6, name="ferromagnet")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
heavy.plot_bands(["G", "X", "M", "G"], ax=ax[0], weights=heavy.orbital_operator('z') @ heavy.orbital_operator('z'), cmap="viridis")
ferro.plot_bands(["G", "X", "M", "G"], ax=ax[1], weights=ferro.spin_operator('z'));"""),
    ("md", "## Step 3: orbital vs spin Hall of each layer\nThe orbital Hall effect needs no SOC (Go et al., PRL 121, 086602 (2018))."),
    ("code", """mus = np.linspace(-2.5, 2.5, 51)
ohc = hf.orbital_hall(heavy, mus, nk=80, eta=0.02, n_jobs=-1).sigma
shc = hf.spin_hall(heavy, mus, nk=80, eta=0.02, n_jobs=-1).sigma
plt.plot(mus, ohc, label="OHC (light metal, no SOC)"); plt.plot(mus, shc, label="SHC (light metal)")
plt.legend(); plt.xlabel("μ"); plt.ylabel("(ħ/e) e²/h");"""),
    ("md", "## Step 4: stack the layers into a bilayer\n`moire.bilayer` assembles any two commensurate layers, with Slater-Koster interlayer hopping between their orbitals."),
    ("code", """bi = MO.bilayer(ferro, heavy, d=1.0, interlayer={"pps": 0.4, "ppp": -0.1}, cutoff=1.6, name="FM / light-metal bilayer")
L = bi.info["layers"]
print(bi, " bottom (FM) orbitals:", L["bottom"], " top orbitals:", L["top"])
bi.special_points = ferro.special_points
bi.plot_bands(["G", "X", "M", "G"], weights=bi.projector(L["bottom"]), cmap="coolwarm", title="weight on the FM layer");"""),
    ("md", "## Step 5: layer-resolved Edelstein responses (E ∥ x)"),
    ("code", """mu = 0.3
for lay in ("bottom", "top"):
    for name, op in (("S_y", bi.spin_operator('y')), ("L_y", bi.orbital_operator('y'))):
        A = R.layer_operator(bi, L[lay], op)
        r = R.kubo_streda(bi, A, "x", mu=mu, Gamma=0.05, nk=80, n_jobs=-1)
        print(f"{lay:6s} <{name}>: Fermi surface {r['fermi_surface'][0]:+.4f}   Fermi sea {r['fermi_sea'][0]:+.4f}")"""),
    ("md", "## Step 6: torques on the ferromagnet: SOT, orbital torque and self-torque\nThe **self-torque** is the torque a current in a ferromagnet exerts on its own magnetisation. By symmetry it **vanishes in a centrosymmetric ferromagnet**, so we first check that. Then we break inversion inside the FM with an interface (orbital-Rashba) term $\\gamma\\,\\hat d$ that hybridises $p_z$ with $p_{x,y}$. Finally we compare with the torque on the same FM inside the bilayer, which adds the angular momentum injected from the other layer."),
    ("code", """Jex = 0.6
def with_orbital_rashba(m, gamma):
    m = m.copy()
    m.add_neighbors(lambda d: gamma * d[0], 1, ["X:pz"], ["X:px"])
    m.add_neighbors(lambda d: gamma * d[1], 1, ["X:pz"], ["X:py"])
    return m
cases = {"FM alone (centrosymmetric)": (ferro, None),
         "FM alone + interface term (self-torque)": (with_orbital_rashba(ferro, 0.3), None),
         "FM inside the bilayer": (bi, L["bottom"])}
for name, (sys_, orbs) in cases.items():
    T = R.torkance(sys_, (0, 0, 1), Jex, "x", mu=mu, Gamma=0.05, nk=80, orbitals=orbs, n_jobs=-1)
    print(f"{name:40s} FL(FS) {T['fermi_surface_FL'][0]:+.4f}  DL(FS) {T['fermi_surface_DL'][0]:+.4f}  "
          f"FL(sea) {T['fermi_sea_FL'][0]:+.4f}  DL(sea) {T['fermi_sea_DL'][0]:+.4f}")"""),
    ("md", "The centrosymmetric ferromagnet gives exactly zero, as symmetry requires. Breaking inversion produces a self-torque, and the bilayer adds torques driven by the angular momentum (spin and orbital) coming from the neighbouring layer."),
    ("code", """mus = np.linspace(-2.0, 2.0, 21)
T = R.torkance(bi, (0, 0, 1), Jex, "x", mu=mus, Gamma=0.05, nk=60, orbitals=L["bottom"], n_jobs=-1)
plt.plot(mus, T["fermi_surface_FL"], label="field-like (Fermi surface)")
plt.plot(mus, T["fermi_sea_DL"], label="damping-like (Fermi sea)")
plt.legend(); plt.xlabel("μ"); plt.ylabel("torkance (per cell, natural units)"); plt.title("torque on the FM layer of the bilayer");"""),
    ("md", "## Step 7: a twisted (moiré) bilayer\nThe same workflow runs on a commensurate twisted bilayer. `rotate_model` rotates the lattice, the p-orbital Slater-Koster blocks and the spin frame. Here two graphene layers at 13.2°, with an exchange field on the bottom layer and Rashba-like proximity SOC on the top."),
    ("code", """g = MO.graphene_layer().make_spinful()
gF = g.copy(); gF.add_exchange(0.1, (0, 0, 1))
gR = g.copy(); gR.add_rashba(0.05 * 2.7 / 2.7, 1)
theta, L1, L2 = MO.commensurate_hexagonal(2, 1)
b1, b2 = g.lat.vectors[1][:2], g.lat.vectors[1][:2] - g.lat.vectors[0][:2]
S = np.array([L1[0]*b1 + L1[1]*b2, L2[0]*b1 + L2[1]*b2])
tw = MO.bilayer(gF, MO.rotate_model(gR, theta), S, 3.35, MO.moon_koshino(), 6.0, name=f"twisted {np.degrees(theta):.2f} deg")
print(tw)
ahc = hf.anomalous_hall(tw, np.linspace(-0.6, 0.6, 25), nk=24, eta=0.01, n_jobs=-1)
she = hf.spin_hall(tw, np.linspace(-0.6, 0.6, 25), nk=24, eta=0.01, n_jobs=-1)
plt.plot(ahc.mu, ahc.sigma, label="AHC"); plt.plot(she.mu, she.sigma, label="SHC (s_z)")
plt.legend(); plt.xlabel("μ (eV)"); plt.title("moiré bilayer: Hall responses");"""),
    ("code", """T = R.torkance(tw, (0, 0, 1), 0.1, "x", mu=0.2, Gamma=0.02, nk=16, orbitals=tw.info["layers"]["bottom"], n_jobs=-1)
print("moiré bilayer torque on the magnetic layer: FL =", T["fermi_surface_FL"][0], " DL =", T["fermi_sea_DL"][0])"""),
    ("md", """## Exercises
1. Turn on SOC in the light-metal layer and separate the spin-torque and orbital-torque contributions.
2. Reverse the magnetisation and check which torque components are even or odd in m.
3. Scan the twist angle (m = 1, 2, 3) and plot the torkance versus θ."""),
] + quiz_cells(18, "Orbitronics in bilayers and moire")

NOTEBOOKS["19_step_by_step_interacting_hamiltonians_phase_diagrams"] = [
    ("md", r"""# 19 · Step by step: interacting Hamiltonians, doping and quantum phase diagrams

1. Write an interacting model (Hubbard) on a cluster cut from any lattice.
2. Solve it exactly (Lanczos ED) and check against an exact result.
3. Measure correlations, gaps and the spectral function.
4. Dope it: energies and correlations versus filling.
5. Mean-field phase diagram versus U and doping (PM / FM / Néel).
6. A quantum phase transition: the transverse-field Ising chain (order parameter, Binder cumulant, entanglement, central charge).
7. Bosons: Mott lobes of the Bose-Hubbard model."""),
    ("code", SETUP + "\nfrom hopforge import manybody as MB, phases as PH"),
    ("md", "## Step 1-2: Hubbard model on a 2×4 periodic cluster; check against the dimer formula first"),
    ("code", """E = MB.Hubbard(M.chain().finite(2), U=4.0).ground_state(1, 1)[0]
print("dimer:", E, " exact:", (4 - np.sqrt(16 + 16)) / 2)
cluster = M.square().supercell([2, 4])         # 8 sites, periodic
ed = MB.Hubbard(cluster, U=4.0, pbc=True)
print("Hilbert space at half filling:", ed.dimension(4, 4))
E0, psi = ed.ground_state(4, 4)
print("E0 =", E0)"""),
    ("md", "## Step 3: correlations, gaps, spectral function"),
    ("code", """C = np.array([[ed.spin_correlation(psi, 0, j, 4, 4) for j in range(8)]])
print("<S_0 . S_j> =", np.round(C, 3))
print("total spin S =", round(ed.total_spin(psi, 4, 4), 4), "  charge gap =", round(ed.charge_gap(4, 4), 4),
      "  spin gap =", round(ed.spin_gap(4, 4), 4))
w = np.linspace(-8, 8, 800)
A = -ed.greens_function(0, 4, 4, w, eta=0.15).imag / np.pi
plt.plot(w, A); plt.xlabel("ω"); plt.title("local spectral function: Hubbard bands at U = 4");"""),
    ("md", "## Step 4: doping the cluster"),
    ("code", """Us = [2.0, 4.0, 8.0]
for U in Us:
    ed = MB.Hubbard(cluster, U=U, pbc=True)
    rows = []
    for N in range(2, 9, 2):
        E, sec = ed.ground_state_all_sectors(N, n_jobs=-1)
        rows.append((N, E, sec))
    print(f"U={U}: " + "  ".join(f"N={N}: E={E:.3f} {sec}" for N, E, sec in rows))"""),
    ("md", "## Step 5: mean-field phase diagram versus U and doping\nCompeting PM / FM / Néel solutions are converged at every point, and the lowest energy wins (as in Hirsch 1985)."),
    ("code", """sq2 = M.square(t2=-0.2).make_spinful().supercell([[1, 1], [1, -1]])
pd = PH.hubbard_phase_diagram(sq2, Us=np.linspace(1, 10, 10), fillings=np.linspace(0.5, 0.3, 9), nk=20, n_jobs=-1)
PH.plot_phase_diagram(pd, contour="|m|", title="mean-field Hubbard (t' = -0.2): U vs doping");"""),
    ("md", "## Step 6: a quantum phase transition: the transverse-field Ising chain"),
    ("code", """scan = PH.quantum_critical_scan(lambda L, h: MB.PauliModel.transverse_field_ising(L, 1.0, h), "h / J",
                               np.linspace(0.5, 1.5, 21), Ls=(8, 10, 12), n_jobs=-1)
PH.plot_critical_scan(scan)
print("Binder crossing h_c ≈", round(PH.binder_crossing(scan, 0, -1), 3), "(exact: 1)")
psi = MB.PauliModel.transverse_field_ising(16, 1.0, 1.0).ground_state()[1]
print("central charge at h = J:", round(PH.central_charge(psi, 16)[0], 3), "(Ising CFT: 1/2)")"""),
    ("md", "## Step 7: Bose-Hubbard Mott lobes"),
    ("code", """pdb = PH.mott_lobes(np.linspace(0.005, 0.2, 40), np.linspace(0.0, 3.0, 60), n_jobs=-1)
ax = PH.plot_phase_diagram(pdb, title="Bose-Hubbard mean field")
for n in (1, 2, 3):
    mu = np.linspace(n - 1, n, 200)[1:-1]
    ax.plot(MB.mott_lobe_boundary(n, 1.0, mu), mu, "k--", lw=1)
print("lobe tip (U/zt)_c =", 1 / MB.mott_lobe_boundary(1), "(3 + 2√2)");"""),
    ("md", """## Exercises
1. Add a nearest-neighbour V to the cluster (`Hubbard(..., V=1.0)`) and look for charge order.
2. Build a two-orbital Kanamori cluster (`atoms=[[0, 1], ...]`) and find the high-spin/low-spin transition versus J.
3. Repeat Step 6 for the XXZ chain (`SpinModel.chain(L, Delta=...)`)."""),
] + quiz_cells(19, "Interacting Hamiltonians and phase diagrams")

NOTEBOOKS["20_reproduce_research_models_bernevig_vishwanath_sachdev"] = [
    ("md", r"""# 20 · Reproduce research Hamiltonians: Bernevig, Vishwanath and Sachdev

Each section builds a model, solves it, and compares with the published number.
`python -m hopforge.benchmarks` runs all of these checks."""),
    ("code", SETUP + "\nfrom hopforge import manybody as MB, moire as MO"),
    ("md", "## 1. Higher-order TI: Benalcazar-Bernevig-Hughes quadrupole (Science 2017)"),
    ("code", """b = M.bbh_quadrupole(gamma=0.5)
print("nested Wilson loop p_y^{ν_x} =", hf.nested_wilson_loop(b, 2, 0))
f = M.bbh_quadrupole(gamma=0.5, delta=1e-3).finite(16)
E, V = np.linalg.eigh(f.hamiltonian())
rho = (np.abs(V[:, :f.nbands // 2])**2).sum(1).reshape(-1, 4).sum(1) - 2
P = f.lat.positions[::4]
plt.scatter(P[:, 0], P[:, 1], c=rho, cmap="RdBu", s=40, vmin=-0.3, vmax=0.3); plt.colorbar(label="excess charge")
plt.title("corner charges ±e/2"); plt.gca().set_aspect("equal")
print("corner charge =", round(rho[(P[:, 0] > 7.5) & (P[:, 1] > 7.5)].sum(), 3))"""),
    ("md", "## 2. Fractional Chern insulator (Regnault & Bernevig, PRX 2011)"),
    ("code", """cb = M.checkerboard_chern()
cb.plot_bands(title="flat Chern band (C = 1)")
Es = MB.Fermions(cb.supercell([3, 4]), V=[1.0]).spectrum(4, k=8)
plt.figure(); plt.plot(Es - Es[0], "o"); plt.title("ν = 1/3 ED: threefold quasi-degenerate ground state"); plt.ylabel("E - E0");"""),
    ("md", "## 3. Twisted bilayer graphene: chiral magic angle (Tarnopolsky, Kruchkov & Vishwanath, PRL 2019)"),
    ("code", """al = np.linspace(0.05, 0.75, 36)
plt.plot(al, MO.velocity_curve(al, 0.0, n_jobs=-1), label="chiral (w0 = 0)")
plt.plot(al, MO.velocity_curve(al, 0.8, n_jobs=-1), label="w0/w1 = 0.8")
plt.plot(al, (1 - 3*al**2)/(1 + 6*al**2), "k:", label="BM perturbative")
plt.axhline(0, c="gray"); plt.legend(); plt.xlabel("α = w1 / v kθ"); plt.ylabel("v*/v")
print("first magic α (chiral):", MO.magic_angles(0.0, n=29)[0], "-> θ ≈", round(MO.alpha_to_theta(0.586), 3), "deg")"""),
    ("code", """x, E, ticks, labels = MO.BistritzerMacDonald(0.586, 0.0).bands(n=80, nbands=8, n_jobs=-1)
plt.plot(x, E, "k"); plt.xticks(ticks, labels); plt.ylim(-2, 2); plt.ylabel("E / v kθ")
plt.title("flat bands at the magic angle");"""),
    ("md", "## 4. Sachdev-Ye-Kitaev model: ground-state energy and random-matrix level statistics"),
    ("code", """Ns = [14, 16, 18, 20]
E = [np.mean([MB.SYK(N, seed=s).ground_energy() / N for s in range(6)]) for N in Ns]
plt.plot(1 / np.array(Ns), E, "o-"); plt.xlabel("1/N"); plt.ylabel("E0 / N")
c = np.polyfit(1 / np.array(Ns), E, 2); print("extrapolated E0/N =", round(c[-1], 4), "(large-N: -0.0406)")
for N in (16, 18, 20):
    print(f"N = {N} (N mod 8 = {N % 8}): <r> = {MB.syk_level_statistics(N, 10):.3f}")"""),
    ("md", "## 5. Quantum criticality (Sachdev, Quantum Phase Transitions): Ising chain against the exact solution"),
    ("code", """for h in (0.5, 1.0, 2.0):
    E = MB.PauliModel.transverse_field_ising(12, 1.0, h).ground_state()[0]
    print(f"h = {h}: ED {E:.10f}   Pfeuty exact {MB.tfim_exact_energy(12, 1.0, h):.10f}")"""),
    ("md", "## 6. Kitaev honeycomb spin model (exact ground state on a small cluster)"),
    ("code", """cl = M.honeycomb().supercell([2, 2])
kh = MB.PauliModel.kitaev_honeycomb(cl, 1.0, 1.0, 1.0)
E0 = kh.ground_state()[0]
print("8-site Kitaev cluster: E0 / site =", E0 / cl.norb)"""),
] + quiz_cells(20, "Research models")


def build(execute=False):
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                    for kind, src in cells]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = OUT / f"{name}.ipynb"
        if execute:
            from nbconvert.preprocessors import ExecutePreprocessor
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(OUT)}})
        nbf.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    build(execute="--execute" in sys.argv)
