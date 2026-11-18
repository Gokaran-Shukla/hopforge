# hopforge API reference

_Generated from docstrings by `tools/make_api.py`._


## `hopforge`

hopforge: a universal tight-binding engine - from 1D chains to 3D crystals:
bands, topology, Kubo/Boltzmann/nonlinear response, Kwant-style quantum
transport, KPM, surface Green's functions, Wannier90 import, mean-field
Hubbard and Floquet - with node-level parallelism and published benchmarks.

* **`Lattice(vectors, orbitals, coords='fractional')`**: A Bravais lattice plus a list of orbitals.
* **`Model(lattice: 'Lattice', spinful: 'bool' = False, name: 'str' = 'model')`**: A tight-binding model on a :class:`Lattice`.
* **`Quiz(module)`**: 
* **`add_disorder(model: 'Model', W: 'float', seed=None) -> 'Model'`**: Anderson disorder: random on-site energies uniform in [-W/2, W/2].
* **`anomalous_hall(model, mu=0.0, nk=100, component='xy', **kw)`**: Anomalous Hall conductivity (Berry curvature of occupied states).
* **`band_gap(model, n_occ, nk=60)`**: Indirect gap between band n_occ-1 and n_occ on a uniform mesh.
* **`bands(model, path=None, n=200, vectors=False)`**: Band structure along a path. Returns a :class:`Bands` object.
* **`berry_curvature(model, k, occ=None, mu=None, component='xy', eta=0.0, per_band=False, n_jobs=None)`**: Berry curvature Omega^{ab} at k-points (Kubo formula).
* **`berry_curvature_map(model, occ=None, nk=60, plane=(0, 1), fixed=None, mu=None, component='xy', n_jobs=None)`**: Berry curvature on an nk x nk grid of the reduced plane; returns (k1, k2, Omega) ready for plotting.plot_kmap.
* **`berry_phase(model, occ, kstart=None, direction=0, nk=100, gauge='atomic')`**: Berry phase (sum over occupied bands) along the closed loop ``kstart -> kstart + b_direction``. Returns a value in (-pi, pi].
* **`chern_number(model, occ, nk=60, plane=(0, 1), fixed=None, n_jobs=None)`**: Chern number of the occupied bands by the Fukui-Hatsugai-Suzuki lattice method (exactly integer for a fine enough mesh).
* **`cut(model: 'Model', direction: 'int', n: 'int', periodic: 'bool' = False) -> 'Model'`**: Make the model finite along lattice direction ``direction`` with ``n`` unit cells (open boundaries). Returns a model with one fewer periodic dimension: a ribbon from a 2D model, a slab from a 3D model, a chain segment (0D) from a 1D model.
* **`discretize(expr, coords='xy', grid=1.0, params=None, name=None)`**: Turn a k.p Hamiltonian (string or sympy Matrix) into a hopforge Model on a square/cubic grid of spacing ``grid``. Returns a periodic Model whose orbitals ``o0, o1, ...`` are the components of the matrix.
* **`dos(model, nk=60, energies=None, sigma=0.05, projections=None, ne=600, n_jobs=None)`**: Gaussian-broadened density of states per unit cell (states / energy).
* **`fermi_level(model, n_electrons, nk=40)`**: Chemical potential (zero temperature) for ``n_electrons`` per unit cell.
* **`finite(model: 'Model', n) -> 'Model'`**: Finite (0D) flake with ``n`` cells along every periodic direction (``n`` may be an int or a list).
* **`fu_kane_parity(model, occ, parity_op, trim=None)`**: Z2 from parity eigenvalues at TRIM (Fu & Kane 2007) for models with inversion symmetry. ``parity_op`` is the inversion matrix acting on the basis (in the periodic gauge). Returns (z2, products) where products holds delta_i for each TRIM. Kramers pairs are counted once.
* **`hall_conductivity(model, mu=0.0, nk=100, operator=None, component='xy', eta=0.0, kT=0.0, op_name=None, length_unit_angstrom=True, n_jobs=None)`**: Intrinsic Hall-type conductivity on a uniform k mesh.
* **`kmesh(model, n, shift=0.0, endpoint=False)`**: Uniform Monkhorst-Pack-like mesh in reduced coordinates.
* **`kpath(model, path=None, n=200)`**: Piecewise-linear path through special points.
* **`nested_wilson_loop(model, occ, direction=0, nk=60, nperp=60, branch=1)`**: Nested Wilson loop for multipole insulators (Benalcazar, Bernevig & Hughes, Science 357, 61 (2017)). The Wilson loop along ``direction`` gives Wannier bands nu(k_perp); the Berry phase of the Wannier sector with nu > 0 (``branch=+1``) along k_perp is the Wannier-sector polarisation p^{nu}, which is 1/2 in the quadrupole phase. Returns p in [0, 1).
* **`orbital_hall(model, mu=0.0, nk=100, L='z', component='xy', **kw)`**: Intrinsic orbital Hall conductivity with the atomic L operator (atom-centred approximation), built from the orbital kinds (p, d).
* **`pauli(which)`**: Return a Pauli matrix: ``pauli(0..3)`` or ``pauli('x'|'y'|'z'|'0')``.
* **`plot_bands(model, path=None, n=200, ax=None, color='k', weights=None, cmap='coolwarm', ylim=None, lw=1.2, title=None, fermi=None, **kw)`**: Plot the band structure. ``weights`` may be an operator (matrix): the bands are then coloured by its expectation value (e.g. spin_operator('z')).
* **`plot_dos(d, ax=None, label='total', **kw)`**: 
* **`plot_kmap(values, extent=(0, 1, 0, 1), ax=None, cmap='RdBu_r', title='', symmetric=True)`**: Colour map of a quantity on a 2D k grid (e.g. Berry curvature).
* **`plot_lattice(model, shells=3, cells=None, ax=None, bonds='shells', annotate=True, show_cell=True, highlight='auto', projection=None, s=None, legend=True, title=None, circles=True)`**: Draw the crystal with its 1st, 2nd, 3rd, ... nearest-neighbour bonds.
* **`plot_wcc(wcc, ax=None, kperp=None, title='Wannier charge centres')`**: Plot hybrid Wannier charge centres (output of topology.wilson_loop).
* **`polarization(model, occ=None, nk=200)`**: Electronic polarisation (in units of e * lattice constant, mod 1) of a 1D model = -Zak / 2 pi  (electron charge -e).
* **`remove_orbitals(model: 'Model', orbitals) -> 'Model'`**: Remove orbitals (vacancies). Bonds touching them are deleted.
* **`slater_koster(kind1: 'str', kind2: 'str', d, params: 'dict') -> 'float'`**: Single Slater-Koster matrix element <kind1, 0 | H | kind2, d>.
* **`spin_block(t0=0.0, tx=0.0, ty=0.0, tz=0.0)`**: 2x2 spin matrix ``t0*1 + tx*sx + ty*sy + tz*sz``.
* **`spin_hall(model, mu=0.0, nk=100, spin='z', component='xy', **kw)`**: Intrinsic spin Hall conductivity sigma^{s_spin}_{component}.
* **`supercell(model: 'Model', P) -> 'Model'`**: Supercell with new lattice vectors ``A' = P @ A``.
* **`wilson_loop(model, occ, direction=0, scan=None, nk=100, nscan=51, fixed=None, scan_range=(0.0, 1.0), n_jobs=None)`**: Hybrid Wannier charge centres (WCC) along ``direction`` as a function of the momentum along ``scan``.
* **`winding_number(model, chiral, nk=400)`**: Winding number of a 1D chiral-symmetric model.
* **`z2_3d(model, occ, nk=60, nscan=61, n_jobs=None)`**: Strong and weak Z2 indices (nu0; nu1 nu2 nu3) of a 3D TRS model from the six TRS-invariant planes k_i = 0 and k_i = 1/2.
* **`z2_from_wcc(wcc)`**: Z2 index from WCC over half the BZ (k_scan in [0, 0.5]) using the largest-gap method of Soluyanov & Vanderbilt, PRB 83, 235401 (2011).
* **`z2_invariant(model, occ, direction=0, nk=100, nscan=101, fixed=None, scan=None, n_jobs=None)`**: Z2 invariant of a time-reversal-invariant 2D model (or of one TRS-invariant plane of a 3D model, chosen with ``fixed``).
* **`zak_phase(model, occ=None, nk=200, gauge='atomic')`**: Zak phase of a 1D model (occupied bands = lower half by default).

## `hopforge.models`

Library of published tight-binding models. See :func:`list_models`.

* **`LS_matrix(l, lam)`**: lambda L.S in the real-orbital (x) spin basis, orbital-major.
* **`altermagnet_dwave(t=1.0, t1=0.6, t2=0.2, J=0.5, direction=(0, 0, 1), lso=0.0)`**: Minimal d-wave altermagnet (two sublattices with opposite moments and anisotropic NNN hopping, cf. Smejkal, Sinova & Jungwirth PRX 12, 031042 (2022)). Spin splitting vanishes along G-M and is maximal along G-X / G-Y. Optional Rashba ``lso``.
* **`bbh_quadrupole(gamma=0.5, lam=1.0, delta=0.0)`**: Benalcazar-Bernevig-Hughes quadrupole insulator (Science 357, 61 (2017)): H = (g + l cos kx) G4 + l sin kx G3 + (g + l cos ky) G2 + l sin ky G1 + d G0. Quadrupole phase (q_xy = 1/2, corner charges +-e/2) for |g/l| < 1. Four orbitals per cell (pi flux per plaquette); delta breaks chiral symmetry.
* **`bcc(t=1.0)`**: Body-centred cubic lattice (conventional a = 1).
* **`bdg(model, mu=0.0, delta=0.0)`**: Bogoliubov-de Gennes doubling of any model: electron block H - mu, hole block -(H - mu)^*, on-site pairing delta between e and h of every orbital (the spin-degenerate s-wave toy model of Kwant's superconductor tutorial; for spinful models use delta * i sigma_y explicitly).
* **`bhz(A=1.0, B=1.0, C=0.0, D=0.0, M=1.0)`**: Bernevig-Hughes-Zhang lattice model of HgTe quantum wells (Science 314, 1757 (2006)). Orbitals E (s-like) and H (p-like), spinful. Quantum spin Hall for 0 < M/B < 8 (M/B != 4).
* **`bilayer_graphene(t=1.0, tperp=0.13, U=0.0, c=1.36)`**: AB (Bernal) bilayer graphene; B1-A2 dimer coupling ``tperp`` and layer bias U (opens a gap). Length unit = graphene lattice constant.
* **`cart_points(model, pts_2pi_over_a, path=None)`**: Special points given in Cartesian units of 2 pi / a -> reduced.
* **`chain(t=1.0, eps=0.0)`**: Monatomic chain: E(k) = eps - 2t cos k.
* **`checkerboard(t=1.0, tp=1.0)`**: Checkerboard (planar pyrochlore) lattice: NN t and crossed-plaquette NNN tp (A along x, B along y). Flat band at E = 2t when tp = t.
* **`checkerboard_chern(t=1.0, tp=0.3, tpp=-0.2, phi=0.7853981633974483)`**: Flat-band Chern insulator on the checkerboard lattice (Sun, Gu, Katsura & Das Sarma, PRL 106, 236803 (2011); parameters of Sheng, Gu, Sun & Sheng, Nat. Commun. 2, 389 (2011): t' = 0.3, t'' = -0.2, phi = pi/4, gap/bandwidth ~ 27); the host of fractional Chern insulators (Regnault & Bernevig, PRX 1, 021014 (2011)). NN -t e^{i phi s_x s_y}, NNN -/+t' (staggered), diagonal -t''.
* **`chirality(model, i, j, R, nn_dist, tol=0.0001, search=2)`**: nu_ij = sign((d_ik x d_kj)_z) for the shortest two-step path i -> k -> j through a nearest neighbour k (Haldane / Kane-Mele sign).
* **`community_benchmarks()`**: Benchmark entries of community-registered models (callables only), in the format of hopforge.benchmarks.REGISTRY.
* **`creutz_ladder(t=1.0, theta=1.5707963267948966, mperp=0.0)`**: Creutz ladder (PRL 83, 2636 (1999)): flat bands at +-2t for theta = pi/2, mperp = 0 (Aharonov-Bohm caging).
* **`cubic(t=1.0)`**: Simple cubic lattice: E = -2t (cos kx + cos ky + cos kz).
* **`diamond_fkm(t=1.0, dt=0.4, lso=0.125)`**: Fu-Kane-Mele diamond-lattice topological insulator (PRL 98, 106803 (2007)). NN hopping t (t + dt on the [111] bond), NNN SOC i (8 lso / a^2) s . (d1 x d2). Strong TI (1;111) for dt > 0.
* **`dice(t=1.0, alpha=1.0)`**: alpha-T3 / dice lattice (Raoux et al., PRL 112, 026402 (2014)): hub site H coupled to rim A with t and to rim B with alpha*t. Flat band at E = 0.
* **`fcc(t=1.0)`**: Face-centred cubic lattice (conventional a = 1).
* **`get(name, **params)`**: 
* **`graphene(t=1.0, delta=0.0, t2=0.0)`**: Graphene / honeycomb lattice (Wallace 1947). ``delta`` = +-delta staggered (hBN-like) potential, ``t2`` = NNN hopping. Dirac points at K, K'.
* **`haldane(t=1.0, t2=0.1, phi=1.5707963267948966, M=0.0)`**: Haldane model (PRL 61, 2015 (1988)): Chern insulator with C = +-1 for |M| < 3 sqrt(3) |t2 sin phi|.
* **`hofstadter(p=1, q=3, t=1.0)`**: Harper-Hofstadter model (PRB 14, 2239 (1976)): square lattice with flux p/q per plaquette (Landau gauge, q x 1 magnetic supercell).
* **`honeycomb(t=1.0, delta=0.0, t2=0.0)`**: Graphene / honeycomb lattice (Wallace 1947). ``delta`` = +-delta staggered (hBN-like) potential, ``t2`` = NNN hopping. Dirac points at K, K'.
* **`info(name)`**: Registry entry: description, reference, benchmark, author, source.
* **`kagome(t=1.0, lso=0.0)`**: Kagome lattice: flat band at E = 2t. With ``lso`` the spinful Guo-Franz intrinsic SOC i lso nu sz on NN bonds (PRB 80, 113102 (2009)).
* **`kane_mele(t=1.0, lso=0.06, lv=0.0, lr=0.0)`**: Kane-Mele model (PRL 95, 226801 & 146802 (2005)). Quantum spin Hall (Z2 = 1) for lv < 3 sqrt(3) lso when lr = 0.
* **`kitaev_chain(mu=0.5, t=1.0, delta=1.0)`**: Kitaev p-wave chain in Bogoliubov-de Gennes form (Phys.-Usp. 44, 131 (2001)). Basis (e, h). Topological for |mu| < 2|t|; chiral operator tau_x.
* **`lieb(t=1.0, delta=0.0)`**: Lieb lattice (CuO2-plane-like): flat band at E = 0 touching a Dirac cone at M. ``delta`` shifts the corner site.
* **`list_models(*a, **kw)`**: 
* **`load_plugins(verbose=False)`**: Import models published by other packages under the ``hopforge.models`` entry-point group.
* **`p_ip_superconductor(t=1.0, mu=-1.0, delta=0.5)`**: Spinless chiral p+ip superconductor on the square lattice (Read & Green, PRB 61, 10267 (2000); Bernevig & Hughes, ch. 16), BdG basis (e, h): xi = -2t (cos kx + cos ky) - mu,  Delta_k = delta (sin kx + i sin ky). Weak pairing |mu| < 4t: C = +-1 (sign flips at mu = 0); |mu| > 4t: C = 0.
* **`p_orbital_square(pps=1.0, ppp=-0.3, pps2=0.3, ppp2=-0.1, lso=0.0, J=0.0)`**: Square lattice of px, py, pz orbitals with Slater-Koster hopping: the standard toy model of the orbital Hall effect (Bhowal & Vignale PRB 103, 195309 (2021); Go et al. PRL 121, 086602 (2018)). Optional on-site lso L.S (spinful) and exchange J along z. NNN bonds (pps2, ppp2) are what hybridise px and py and give a finite orbital Hall conductivity.
* **`pyrochlore(t=1.0)`**: Pyrochlore lattice (corner-sharing tetrahedra): two flat bands at E = 2t.
* **`qwz(m_=1.0, t=1.0)`**: Qi-Wu-Zhang Chern insulator (PRB 74, 085308 (2006)): H = t sin kx sx + t sin ky sy + (m + t cos kx + t cos ky) sz. C = +-1 for 0 < |m| < 2t.
* **`rashba_square(t=1.0, alpha=0.3, J=0.0, direction=(0, 0, 1))`**: Square lattice with Rashba SOC and optional exchange J (2D Rashba ferromagnet: anomalous Hall, Edelstein, spin-orbit torque physics).
* **`register_model(name, dim, description='', reference=None, benchmark=None, author=None, tags=(), replace=False)`**: Decorator adding a model constructor to the registry.
* **`rice_mele(v=0.5, w=1.0, delta=0.2)`**: Rice-Mele chain (PRL 49, 1455 (1982)): SSH plus staggered potential.
* **`set_points(model, pts, path=None)`**: 
* **`spin_diag(h_up, h_dn)`**: Orbital-major matrix with spin-diagonal blocks h_up, h_dn.
* **`square(t=1.0, t2=0.0, eps=0.0)`**: Square lattice with NN t and NNN t2.
* **`ssh(v=0.5, w=1.0)`**: Su-Schrieffer-Heeger chain (PRL 42, 1698 (1979)). Intra-cell hopping v, inter-cell w; topological (winding 1) for |w| > |v|. A at 0, B at 1/2.
* **`triangular(t=1.0)`**: Triangular lattice.
* **`validate(name, run_benchmark=True, nk_test=5, seed=0)`**: Structural checks (builds, Hermitian H(k), dimension, reproducible) and the model's benchmark. Returns a :class:`Validation`.
* **`validate_all(run_benchmark=True, n_jobs=None, verbose=True)`**: Validate every registered model (parallel over models).
* **`weyl_minimal(t=1.0, m_=2.0)`**: Minimal two-band Weyl semimetal with two Weyl nodes at k = (0, 0, +-arccos(m - 2)) (1 < m < 3): H = t[sin kx sx + sin ky sy + (m - cos kx - cos ky - cos kz) sz].
* **`weyl_type2(t=1.0, m_=2.0, tilt=1.5)`**: Weyl semimetal with a tilt along z (type I for tilt < 1, type II with electron and hole pockets touching at the node for tilt > 1; Soluyanov, Gresch, Wang, Wu, Troyer, Dai & Bernevig, Nature 527, 495 (2015)): H = weyl_minimal + t * tilt * cos(kz) * 1.
* **`wilson_dirac_ti(A=1.0, B=1.0, M=1.0)`**: Four-band cubic-lattice 3D topological insulator (Bi2Se3-type Wilson-Dirac model, Zhang et al. Nat. Phys. 5, 438 (2009) regularised): H = A sum_i sin k_i tau_x s_i + (M - 2B sum_i (1 - cos k_i)) tau_z. Strong TI (1;000) for 0 < M/B < 4.

## `hopforge.transport`

Quantum transport (Kwant-style): scattering regions of any shape,
semi-infinite leads, S-matrix / Landauer-Buettiker conductance, value functions
with parameters, conservation laws, Andreev reflection, transmission eigenvalues
and shot noise, LDOS, scattering wavefunctions, bond currents, arbitrary
magnetic fields, and parallel sweeps over the cores of a node.

* **`Lead(system, vector, model, cell0, cell1, values, region_values, name)`**: A semi-infinite lead. Cell 0 touches the scattering region; cell j+1 is cell j translated by ``vector``. Matrices are rebuilt for every parameter set (``matrices(params)``).
* **`SMatrix(system, energy, params=None, eta=1e-10, lead_eta=1e-10, potential=None, lead_shifts=None)`**: Scattering quantities at one energy. Create with ``system.smatrix(E, params)``.
* **`System(model: 'Model', shape, start=None, onsite=None, peierls=None, hopping=None, max_sites=2000000)`**: Finite scattering region built from a template model and a shape.
* **`anderson(W, seed=None)`**: On-site function for Anderson disorder of strength W (uniform in [-W/2, W/2]); reproducible per site position and seed.
* **`andreev_conductance(system, energies, normal_lead=0, electron=None, hole=None, params=None, n_jobs=None)`**: NS-junction conductance (BdG), BTK form G = N_e - R_ee + R_he (e^2/h); parallel over energies. ``electron``/``hole`` are block specs such as dict(orbitals=['A_e']) / dict(orbitals=['A_h']).
* **`bias_potential(system, bias, axis=0, profile='linear')`**: Potential energy per basis state for a bias V between lead 0 (left, at +V/2 ... electron energy -eV/2) and lead 1: a linear drop along ``axis`` across the scattering region (non-self-consistent), or 'flat' (no drop).
* **`conductance(system, energies, to_lead=1, from_lead=0, params=None, n_jobs=None, **kw)`**: T(E) between two leads, parallel over energies.
* **`current(system, bias, mu=0.0, kT=0.0259, n_energy=201, params=None, axis=0, profile='linear', from_lead=0, to_lead=1)`**: Landauer current I(V) = (e/h) int dE T(E, V) [f_L(E) - f_R(E)] in amperes (energies in eV). Lead ``from_lead`` sits at mu + V/2, ``to_lead`` at mu - V/2. The transmission is recomputed at the bias (linear potential drop).
* **`disorder_average(make_system, seeds, energies, to_lead=1, from_lead=0, n_jobs=None, log=False)`**: Mean and std of T(E) (or ln T) over disorder realisations, in parallel. ``make_system(seed)`` returns a System with leads attached.
* **`field_gauge(B, lead_axis='x', origin=0.0, nquad=8)`**: Peierls phases for an arbitrary 2D field profile ``B(x, y, **params)`` (phase per unit area), like Kwant's ``magnetic_gauge`` for planar systems.
* **`iv_curve(system, biases, mu=0.0, kT=0.0259, n_energy=201, params=None, axis=0, profile='linear', n_jobs=None)`**: I(V) for many biases (parallel over biases). Returns amperes.
* **`landau_gauge(B, lead_axis='x')`**: Peierls phase for a uniform field B (phase per unit area, i.e. B = 2 pi * flux / (flux quantum * area)) in a Landau gauge that is translation invariant along ``lead_axis`` (``'x'``: A = (-B y, 0)).
* **`plot_current(sys, bonds, J, ax=None, cmap='inferno', title='current', density=None, arrows=True, s=None)`**: Bond currents as coloured lines plus site-averaged current arrows.
* **`plot_map(sys, values, ax=None, cmap='viridis', s=None, title='', colorbar=True, vmin=None, vmax=None)`**: Colour every site by a value (e.g. LDOS or density from a lead).
* **`plot_system(sys, ax=None, lead_cells=2, s=None, bonds=True, title=None)`**: Draw the scattering region (grey) and the first cells of every lead.
* **`smatrix(system, energy, params=None, **kw)`**: Solve the scattering problem at one energy.
* **`spectrum(system, params_list=None, k=10, sigma=0.0, n_jobs=None, dense=None)`**: Eigenvalues of the closed scattering region (leads ignored) for each parameter set (like kwant.plotter.spectrum). Dense for <= 2000 states, otherwise the ``k`` eigenvalues closest to ``sigma``. Parallel.
* **`sweep(system, func, energies=None, params_list=None, n_jobs=None)`**: Parallel sweep of ``func(smatrix)`` over energies and/or parameter sets. Returns a flat list (one parameter set) or a list per parameter set.

## `hopforge.kpm`

Kernel polynomial method (KPM) for large sparse Hamiltonians.

* **`conductivity(H, vx, vy, area, energies, num_moments=200, num_vectors=10, seed=0, kernel='jackson', kT=0.0, bounds=None, n_eps=4000, hall=None, n_jobs=None)`**: Kubo-Bastin conductivity sigma_xy(mu) by KPM (Garcia et al. 2015).
* **`dos(H, energies=None, num_moments=300, num_vectors=10, seed=0, kernel='jackson', operator=None, bounds=None, n_jobs=None)`**: KPM density of states (states per energy, integrates to the matrix dimension). ``operator`` gives e.g. spin- or orbital-projected DOS. Random vectors are distributed over ``n_jobs`` worker processes.
* **`jackson_kernel(N)`**: 
* **`ldos(H, sites, energies, num_moments=400, kernel='jackson', ns=1, bounds=None)`**: KPM local DOS on the given basis indices (exact unit vectors, no stochastic error). With ``ns`` > 1, ``sites`` are site indices and the internal (spin) components are summed. Returns array (len(sites), nE).
* **`lorentz_kernel(N, lam=4.0)`**: 
* **`moments(H, num_moments=200, vectors=None, num_vectors=10, operator=None, seed=0, bounds=None)`**: Chebyshev moments mu_n = <v| O T_n(H~) |v> (averaged over vectors). The recursion runs on the active back-end (GPU with hf.backend.use('cupy')).
* **`spectral_bounds(H, margin=0.01)`**: Estimate (Emin, Emax) with a sparse eigensolver (Gershgorin fallback).

## `hopforge.surface`

Surface (and bulk) spectral functions of semi-infinite crystals, in the
spirit of WannierTools' surface-state module.

* **`fermi_arc_map(model, direction, energy=0.0, nk=81, side='top', eta=0.002, operator=None, krange=(-0.5, 0.5), n_jobs=None)`**: Surface spectral weight at fixed energy on a 2D grid of the surface BZ (3D models): Fermi arcs, surface Fermi surfaces, spin textures. Returns (K1, K2, A) with reduced in-plane coordinates.
* **`plot_spectral(path, energies, A, ax=None, cmap='inferno', log=True, title='surface spectral function')`**: 
* **`pmap(f, items, n_jobs=None, threads_per_job=1, chunksize=1, ordered=True)`**: Parallel ``[f(x) for x in items]`` over the cores of one node.
* **`spectral(model, direction, kpath=None, energies=None, n=150, side='top', eta=0.001, operator=None, bulk=False, n_jobs=None)`**: Surface spectral function along a k-path in the surface Brillouin zone.
* **`spectral_at(model, kpar, energies, direction, side='top', eta=0.001, operator=None, bulk=False)`**: Surface (or bulk) spectral function at one k_par for many energies.

## `hopforge.response`

Linear and nonlinear response beyond the Hall conductivity.

* **`BoltzmannResult(mu: 'np.ndarray', sigma_over_tau: 'np.ndarray', seebeck: 'np.ndarray', kappa_over_tau: 'np.ndarray', lorenz: 'np.ndarray', kT: 'float', dim: 'int') -> None`**: BoltzmannResult(mu: 'np.ndarray', sigma_over_tau: 'np.ndarray', seebeck: 'np.ndarray', kappa_over_tau: 'np.ndarray', lorenz: 'np.ndarray', kT: 'float', dim: 'int')
* **`berry_curvature_dipole(model, mu=0.0, kT=0.01, nk=200, component='xy', n_jobs=None)`**: Berry curvature dipole (Sodemann & Fu, PRL 115, 216806 (2015)):
* **`boltzmann(model, mu, kT=0.025, nk=80, n_jobs=None)`**: Constant-relaxation-time Boltzmann transport.
* **`chunks_for(n_items, n_jobs, per_job=4)`**: Number of chunks: a few per worker for load balancing.
* **`drude_weight(model, mu, kT=0.01, nk=80, n_jobs=None)`**: Intraband (Drude) weight sigma/tau at chemical potential mu (natural units).
* **`edelstein(model, operator, field='x', mu=0.0, kT=0.01, tau=1.0, nk=100, eta=0.0001, part='both', n_jobs=None)`**: Current-induced polarisation of ``operator`` (spin s_a, orbital L_a, ...) per unit electric field along ``field``, per unit cell.
* **`kubo_streda(model, operator, field='x', mu=0.0, Gamma=0.01, nk=100, n_jobs=None, fermi_sea=True)`**: Linear response delta<A> per unit field of a (Hermitian) operator A (spin, orbital, torque, spin/orbital current ...), per unit cell, with a constant band broadening Gamma (e = hbar = 1):
* **`layer_operator(model, orbitals, base)`**: Restrict an operator (e.g. model.spin_operator('y') or model.orbital_operator('y')) to a set of orbitals (a layer / atom type): P A P with P the orbital projector.
* **`optical_conductivity(model, omegas, mu=0.0, kT=0.0, nk=100, eta=0.02, component='xx', drude_tau=None, n_jobs=None)`**: Kubo-Greenwood optical conductivity sigma_ab(omega) (complex, units e^2/hbar * length^(2-d); 2D: divide by 2 pi for e^2/h):
* **`orbital_magnetization(model, mu=0.0, kT=0.0, nk=100, component='xy', n_jobs=None)`**: Modern theory of orbital magnetisation (Xiao 2005, Thonhauser 2005, Ceresoli 2006) per unit cell:     M = (1/N_k) sum_nk f_n [ m_n + (mu - E_n) Omega_n ] (units e/hbar * energy * length^2; x MU_B_PER_UNIT -> mu_B per cell). In a Chern-insulator gap dM/dmu = C * A_cell / (2 pi) (Streda).
* **`orbital_moment(model, k, component='xy', eta=0.0)`**: Band-resolved orbital moment m_n^{ab}(k) = -Im sum_{m!=n} v^a_nm v^b_mn / (E_n - E_m) (units e/hbar * energy * length^2; multiply by MU_B_PER_UNIT for Bohr magnetons with eV and Angstrom). For two bands m_n = (E_n - E_m) Omega_n / 2. Returns (E, Omega, m) arrays (nk, nb).
* **`split(array, n_parts)`**: Split an array (e.g. a k-mesh) into ~equal contiguous chunks.
* **`torkance(model, magnetization, Jex, field='x', mu=0.0, Gamma=0.01, nk=100, orbitals=None, n_jobs=None)`**: Spin-orbit torque per unit field on the magnetisation m, from the current-induced spin density on the magnetic ``orbitals`` (default: all):

## `hopforge.meanfield`

Self-consistent mean-field (Hartree-Fock) Hubbard model on any spinful
hopforge model: collinear and noncollinear magnetism.

* **`MeanFieldResult(model: 'object', rho: 'np.ndarray', moments: 'np.ndarray', occupations: 'np.ndarray', mu: 'float', energy: 'float', converged: 'bool', iterations: 'int', history: 'list') -> None`**: MeanFieldResult(model: 'object', rho: 'np.ndarray', moments: 'np.ndarray', occupations: 'np.ndarray', mu: 'float', energy: 'float', converged: 'bool', iterations: 'int', history: 'list')
* **`hubbard(model, U, filling=None, n_electrons=None, init=None, nk=24, kT=0.01, mixing=0.5, tol=1e-06, maxiter=300, orbitals=None, seed=0, n_jobs=None, verbose=False)`**: Self-consistent Hubbard mean field.
* **`phase_diagram(model, Us, **kw)`**: Run :func:`hubbard` for a list of U (warm start from the previous solution). Returns list of results.

## `hopforge.floquet`

Floquet engineering of periodically driven tight-binding models.

* **`Drive(A: 'callable', omega: 'float', name: 'str' = 'drive') -> None`**: Drive(A: 'callable', omega: 'float', name: 'str' = 'drive')
* **`EffectiveModel(model, drive, n_max=3, n_time=128)`**: High-frequency Floquet Hamiltonian (first order in 1/omega): H_eff = H_0 + sum_{m>0} [H_m, H_-m] / (m omega).
* **`FloquetModel(model, drive, n_max=4, n_time=128)`**: Exact Floquet quasienergies in a truncated Sambe space with harmonics |m| <= n_max. ``hamiltonian(k)`` returns the Sambe matrix; ``quasienergies(k)`` returns the nbands quasienergies of the central replica (largest weight in m = 0).
* **`circular(A0, omega, handedness=1, D=2)`**: A(t) = A0 (cos wt, h sin wt).
* **`harmonics(model, drive, n_max=4, n_time=128)`**: Fourier components H_m (m = -n_max..n_max) as hopforge Models.
* **`linear(A0, omega, direction=(1, 0), D=2)`**: A(t) = A0 cos(wt) e.

## `hopforge.io`

Import / export: Wannier90 (hr.dat, tb.dat, centres.xyz, .win), VASP POSCAR.

* **`read_centres(filename, num_wann=None)`**: Wannier centres from a ``_centres.xyz`` file (Cartesian, Angstrom).
* **`read_poscar(filename)`**: Read a VASP POSCAR/CONTCAR. Returns (lattice 3x3, species list, Cartesian positions (n, 3)).
* **`read_wannier90(hr_file, lattice=None, positions=None, spinful=False, spin_order='interleaved', names=None, tol=1e-08, dim=3, name=None)`**: Build a Model from ``seedname_hr.dat``.
* **`read_wannier90_tb(tb_file, spinful=False, spin_order='interleaved', names=None, tol=1e-08, dim=3, use_centres=True)`**: Build a Model from ``seedname_tb.dat`` (lattice, H(R) and position matrix; Wannier centres are taken from the diagonal of r(R = 0)).
* **`read_win_cell(filename)`**: Read ``unit_cell_cart`` from a Wannier90 .win file (Angstrom).
* **`write_poscar(filename, lattice, species, positions, comment='hopforge')`**: Write a POSCAR (Cartesian coordinates).

## `hopforge.parallel`

Shared-memory parallelism over the cores of one node.

* **`available_cores() -> 'int'`**: 
* **`resolve_jobs(n_jobs) -> 'int'`**: 
* **`set_default_jobs(n: 'int')`**: Set the default number of worker processes (``-1`` = all cores).

## `hopforge.benchmarks`

Benchmarks against published results.

* **`Result(name: 'str', quantity: 'str', published: 'object', value: 'object', reference: 'str', passed: 'bool', note: 'str' = '', seconds: 'float' = 0.0) -> None`**: Result(name: 'str', quantity: 'str', published: 'object', value: 'object', reference: 'str', passed: 'bool', note: 'str' = '', seconds: 'float' = 0.0)
* **`benchmark(name, quantity, reference)`**: 
* **`markdown(results)`**: 
* **`run(n_jobs=None, names=None, verbose=True, write=None)`**: Run all benchmarks (parallel over benchmarks). Returns a list of Result.
* **`silicon_vogl(a=5.431)`**: Bulk Si in the sp3s* model of Vogl, Hjalmarson & Dow (1983), built with hopforge's Slater-Koster engine (parameters in eV, lattice in Angstrom).
* **`table(results)`**: 
