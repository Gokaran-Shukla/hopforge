# hopforge

**Forge any tight-binding Hamiltonian, from 1D chains to 3D crystals.**

hopforge is a general, easy-to-use Python engine for tight-binding models. You can define
any lattice, any set of orbitals, spin, spin-orbit coupling, magnetism and Slater-Koster
s/p/d bonding in a few lines, or discretise a k·p Hamiltonian. On top of that it computes:

* band structures, DOS, edge and surface states;
* topological invariants (Berry phase, Chern, Z2, Wilson loops, winding);
* intrinsic Kubo Hall responses (anomalous, spin and orbital Hall);
* **Kwant-style quantum transport**: scattering regions of any shape, semi-infinite leads,
  Landauer-Büttiker multi-terminal conductance, quantum Hall / QSH devices, conservation
  laws, Andreev reflection, LDOS, scattering wavefunctions and bond currents;
* the **kernel polynomial method** (DOS, LDOS, Kubo-Bastin conductivity) for systems with
  millions of orbitals.

It ships **29 published models** and **11 tutorial notebooks, each followed by a
20-question quiz** (220 questions).

![neighbour shells](docs/img/lattice_shells.png)

## Install

```bash
git clone https://github.com/Gokaran-Shukla/hopforge
cd hopforge
pip install -e ".[notebooks,test]"
pytest            # 33 physics regression tests
```

## Quick start

```python
import numpy as np
import hopforge as hf

# any model from scratch
lat = hf.Lattice([[1, 0], [0.5, np.sqrt(3) / 2]], [("A", [1/3, 1/3]), ("B", [2/3, 2/3])])
g = hf.Model(lat)
g.add_neighbors(-1.0, shell=1)          # every NN bond (h.c. added automatically)
g.add_onsite(0.2, "A").add_onsite(-0.2, "B")
hf.plot_lattice(g)                      # 1st / 2nd / 3rd neighbour shells

# or one of the 28 published models
h = hf.models.haldane(t2=0.1, phi=np.pi / 2, M=0.1)
h.plot_bands(["G", "K", "M", "G"])
hf.chern_number(h, occ=1)               # 1.0
hf.anomalous_hall(h, mu=0.0)            # sigma_xy = -1 e^2/h
hf.z2_invariant(hf.models.kane_mele(), occ=2)     # 1
hf.z2_3d(hf.models.diamond_fkm(), occ=2)          # (1, 1, 1, 1)
hf.spin_hall(hf.models.kane_mele(), mu=0.0)       # -1 (hbar/e) e^2/h
hf.orbital_hall(hf.models.p_orbital_square())     # finite OHC without SOC

# geometry
rib = h.ribbon(1, 30)                   # 1D ribbon (edges), also .slab / .finite / .supercell
```

![gallery](docs/img/gallery.png)

### Quantum transport and KPM

```python
from hopforge import transport as tr, kpm

# a Hall bar cut out of any model, with a magnetic field
B = 2 * np.pi * 0.05
hb = tr.System(hf.models.square(), lambda r: 0 <= r[0] < 60 and 0 <= r[1] < 30,
               peierls=tr.landau_gauge(B, "x"))
hb.attach_lead((-1, 0)); hb.attach_lead((1, 0))
for x0 in (12, 40):
    hb.attach_lead((0, 1),  shape=lambda r, x0=x0: x0 <= r[0] < x0 + 8, peierls=None)
    hb.attach_lead((0, -1), shape=lambda r, x0=x0: x0 <= r[0] < x0 + 8, peierls=None)
S = hb.smatrix(-3.4)
S.four_terminal_resistance(0, 1, 2, 3)   # R_xy = 1 h/e^2 (lead order: 0,1 current; 2,3 top/bottom)
S.transmission(1, 0)                     # Landauer transmission
bonds, J = S.current(0)                  # bond currents (sum over a cut = T)

# Andreev reflection with a superconducting BdG lead
N, Sc = hf.models.bdg(hf.models.square(), mu=-2, delta=0), hf.models.bdg(hf.models.square(), mu=-2, delta=0.1)
ns = tr.System(N, lambda r: 0 <= r[0] < 6 and 0 <= r[1] < 10)
ns.attach_lead((-1, 0)); ns.attach_lead((1, 0), model=Sc)
tr.andreev_conductance(ns, 0.0, 0, dict(orbitals=["A_e"]), dict(orbitals=["A_h"]))   # = 2N

# KPM on 80,000 orbitals, and Kubo-Bastin Hall conductivity of a disordered Chern insulator
E, rho = kpm.dos(hf.models.honeycomb().supercell([200, 200]).hamiltonian_sparse())
m = hf.add_disorder(hf.models.haldane(t2=0.15).supercell([50, 50]), 1.5)
kpm.conductivity(m.hamiltonian_sparse(), m.velocity_sparse("x"), m.velocity_sparse("y"),
                 m.lat.cell_volume, [0.0])   # ~ -1 e^2/h

# continuum k.p -> tight binding
ti = hf.discretize("M*tau_z - B*(k_x**2+k_y**2+k_z**2)*tau_z"
                   " + A*(k_x*tau_x*sigma_x + k_y*tau_x*sigma_y + k_z*tau_x*sigma_z)",
                   "xyz", grid=1.0, params=dict(M=1, B=0.5, A=1))
hf.z2_3d(ti, 2)                          # (1, 0, 0, 0)
```

![transport](docs/img/transport.png)

## What it can do

| Area | Features |
|---|---|
| **Building models** | Any dimension (0D–3D, also 2D embedded in 3D), any orbitals, automatic neighbour shells, scalar / spin-matrix / Pauli-vector amplitudes, block (k·p-style) matrices, exchange & Zeeman, Rashba, Kane-Mele-type SOC, on-site L·S, Slater-Koster s/p/d tables, Peierls phases, Wannier90 `_hr.dat` export |
| **Geometry** | Supercells (any integer matrix), ribbons, slabs, finite flakes, vacancies, Anderson disorder, magnetic supercells |
| **Spectra** | k-paths from special points, band plots coloured by any operator (spin, L, orbital or edge weight), DOS and projected DOS, Fermi level, gaps |
| **Topology** | Berry phase, Zak phase, polarisation, Berry curvature (Kubo) and maps, Chern number (Fukui-Hatsugai-Suzuki), Wilson loops / hybrid WCC, Z2 (largest-gap method), 3D (ν0;ν1ν2ν3), Fu-Kane parity criterion, 1D winding numbers |
| **Transport** | Intrinsic AHC, SHC and OHC (any components), μ scans with one diagonalisation, broadening, temperature, units e²/h (2D) and S/cm (3D) |
| **Quantum transport** | Scattering regions of any shape (flood fill), semi-infinite leads along any lattice direction with automatic interface filling, exact lead modes and self-energies, Caroli/Fisher-Lee transmission, multi-terminal conductance matrix and four-terminal resistances, magnetic fields (Landau gauge, field-free probes), spin / electron-hole resolved transmissions, BdG builder and Andreev (BTK) conductance, LDOS, scattering wavefunctions, charge and spin bond currents, Anderson disorder |
| **Large-scale (KPM)** | Sparse Hamiltonians and velocities, KPM DOS / operator-projected DOS / LDOS with Jackson or Lorentz kernels, Kubo-Bastin conductivity (Hall and longitudinal) with symmetry-projected noise reduction |
| **Continuum** | `hf.discretize` turns k·p Hamiltonians (with σ/τ Pauli algebra) into tight-binding models |
| **Visualisation** | Lattice with 1st/2nd/3rd neighbour shells (1D arcs, 2D, 3D), highlight mode with coordination circles, bands, DOS, WCC flow, k-space maps |
| **Learning** | 11 executed tutorial notebooks, 220 quiz questions (clickable in Jupyter) |

## Model library

```python
hf.models.list_models()
m = hf.models.get("bhz", M=1.0, B=1.0)
```

Any model can be turned into a superconductor with `hf.models.bdg(model, mu, delta)`.

| 1D | 2D | 3D |
|---|---|---|
| chain, SSH, Rice-Mele, Kitaev chain, Creutz ladder | square, triangular, honeycomb/graphene, Haldane, Kane-Mele, QWZ, BHZ, kagome (+SOC), Lieb, checkerboard, dice/α-T3, AB bilayer graphene, Rashba (ferro)magnet, d-wave altermagnet, Hofstadter, p-orbital square (orbital Hall) | simple cubic, bcc, fcc, Fu-Kane-Mele diamond TI, Wilson-Dirac (Bi2Se3-type) TI, Weyl semimetal, pyrochlore |

Every model is an ordinary `Model`, so you can add terms, cut ribbons or build supercells from it.

## Tutorials and quizzes

| Notebook | Topic |
|---|---|
| [01](tutorials/01_lattices_and_models.ipynb) | Lattices, orbitals, hoppings, neighbour shells, spin, SOC, Slater-Koster |
| [02](tutorials/02_bands_and_dos.ipynb) | k-paths, band structures, DOS/PDOS, Fermi level, altermagnet spin splitting |
| [03](tutorials/03_supercells_ribbons_edges.ipynb) | Supercells, zigzag/armchair ribbons, chiral & helical edge states, flakes, Hofstadter butterfly |
| [04](tutorials/04_model_zoo.ipynb) | Tour of all 28 models |
| [05](tutorials/05_berry_phase_and_chern.ipynb) | Zak phase, polarisation, Thouless pump, Berry curvature, Chern numbers, Haldane phase diagram |
| [06](tutorials/06_wilson_loops_and_z2.ipynb) | Wilson loops, WCC, 2D and 3D Z2, Fu-Kane parity, TI surface states |
| [07](tutorials/07_kubo_hall_transport.ipynb) | Anomalous, spin and orbital Hall conductivities, Weyl AHC in S/cm |
| [08](tutorials/08_scattering_and_leads.ipynb) | Scattering regions, leads, conductance steps, QPC, unitarity, Anderson localisation |
| [09](tutorials/09_multiterminal_quantum_hall_qsh.ipynb) | Hall bars, quantum Hall plateaus, edge currents, QSH spin-resolved transport, TRS breaking |
| [10](tutorials/10_superconductors_ldos_wavefunctions.ipynb) | BdG, Andreev/BTK conductance, Majorana zero-bias peak, LDOS, wavefunctions, current maps |
| [11](tutorials/11_kpm_and_continuum.ipynb) | KPM DOS/LDOS on 10^5 orbitals, Kubo-Bastin Hall of disordered Chern insulators, k·p discretisation |

Each notebook ends with a 20-question quiz:

```python
quiz = hf.Quiz(5)
quiz.interactive()          # clickable in Jupyter
quiz.check({1: "b", 2: "d"})  # or grade text answers, with explanations
```

The notebooks are generated by `tools/build_tutorials.py` (`--execute` re-runs them).

## Conventions

* k-points are in reduced coordinates (units of reciprocal vectors) unless `cartesian=True`.
* Bloch Hamiltonian in the atomic gauge, `H_ij(k) = Σ_R t_ij(R) e^{ik·(R + r_j − r_i)}`. `gauge="periodic"` drops orbital positions.
* Spinful basis is orbital-major: index = 2·orbital + spin.
* ħ = 1, spin s = σ/2. Lengths in Å for S/cm conversion.
* Berry curvature follows Xiao, Chang & Niu (RMP 2010), and Hall conductivities follow the Wannier90 sign convention, so σ_xy = −C e²/h in 2D.

## Validation

`tests/test_physics.py` checks the package against known results: analytic dispersions,
flat bands, a finite-difference velocity operator, the Slater-Koster tables, supercell
band folding, Chern numbers of Haldane, QWZ and Hofstadter (1, −2, 1), SSH/Kitaev winding
numbers, Z2 for Kane-Mele and BHZ, (1;000) for the Wilson-Dirac TI, (1;111) and (0;111)
for Fu-Kane-Mele, the Weyl AHC, quantised AHC and SHC, and a finite OHC.
`tests/test_transport.py` checks the analytic chain surface Green's function, exact
conductance steps, unitarity, reciprocity, current conservation through every cut,
quantum Hall R_xy = h/(νe²) with R_xx = 0, QSH 2e²/h (spin-resolved, under disorder),
Andreev doubling, LDOS sum rules, exact KPM moments, KPM Kubo-Bastin against k-space Kubo,
and continuum discretisation (convergence, Chern number, 3D Z2).

hopforge's transport layer follows the design of [Kwant](https://kwant-project.org)
(Groth et al., New J. Phys. 16, 063065 (2014)) but is an independent implementation.
If you need Kwant's mature feature set, such as symbolic builders with arbitrary site
families, MUMPS-accelerated solvers or `qsymm`, use Kwant itself.

## License

MIT © Gokaran Shukla
