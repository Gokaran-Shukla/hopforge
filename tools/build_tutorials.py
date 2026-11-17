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
