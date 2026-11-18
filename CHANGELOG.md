# Changelog

## 0.8.0 (roadmap release)
* `hf.dmrg`: matrix-product states and two-site DMRG; generic MPO builder with Jordan-Wigner strings; Heisenberg (S = 1/2, 1), AKLT, TFIM, Hubbard, t-V, `from_model`; excited states; correlations, string order, entanglement, central charges.
* `transport.negf`: finite-bias NEGF with contour-integrated density, self-consistent Hartree potential, parallel I-V, dI/dV, bond currents at bias.
* `hf.backend`: CuPy GPU back-end for k-mesh diagonalisation, Kubo kernels and KPM.
* Community model registry: reference + benchmark for every model, `validate`/`validate_all`, entry-point plugins, `community/` folder.
* 8 new benchmarks (77 total), tutorials 22-24 with quizzes.

## 0.7.0
* `hf.macdonald`: models of A. H. MacDonald's group. TMD heterobilayer moiré Hubbard model (Wu et al. PRL 2018), twisted TMD homobilayer topological bands (Wu et al. PRL 2019), generic plane-wave continuum engine with Kubo Chern numbers and tight-binding export, ABC N-layer graphene, graphene QAH, Rashba 2DEG universal spin Hall.
* 10 new published-result benchmarks (69 total); tutorial 21 and quiz 21.
* Beamer overview slides in `docs/slides/`.

## 0.6.0
* Tutorials 17-20 (step-by-step model building, orbitronics/moiré bilayers, interacting phase diagrams, research models) with quizzes 17-20.
* Fix: `moire.bilayer` now keeps one label per atom, so orbital angular momentum operators work on bilayers.
* Strongly correlated systems: ED for Hubbard / extended Hubbard / Kanamori, spinless fermions (FCI), spin-1/2 XXZ / J1-J2, general Pauli-string models (TFIM, Kitaev honeycomb), SYK, Bose-Hubbard mean field; Lanczos Green functions.
* Quantum phases: Hubbard phase diagrams vs U and doping, critical scans (Binder, fidelity, entanglement, central charge), Mott lobes, topological phase maps.
* Bernevig/Vishwanath models: BBH quadrupole + nested Wilson loops, p+ip, type-II Weyl, flat Chern band, Bistritzer-MacDonald / chiral TBG magic angles.
* Moire: real-space twisted and stacked bilayers (orbital + spin frame rotation), Moon-Koshino couplings.
* Kubo-Streda responses, spin-orbit / orbital / self torques (torkance).
* 59 published-result benchmarks.

## 0.5.0
* Benchmark suite: 37 published results (`python -m hopforge.benchmarks`), run in CI.
* Vogl sp3s* silicon via the Slater-Koster engine (`hf.benchmarks.silicon_vogl`).
* API reference (`docs/API.md`).

## 0.4.0
* Wannier90 (hr.dat, tb.dat, centres, .win) and POSCAR I/O.
* Surface Green's functions: surface/bulk spectral functions, spin-resolved, Fermi-arc maps.
* Response: Boltzmann (sigma, Seebeck, kappa, Lorenz), Edelstein (extrinsic + intrinsic,
  spin and orbital), Berry curvature dipole, orbital moment, modern orbital magnetisation,
  Kubo-Greenwood optical conductivity.
* Self-consistent mean-field Hubbard (collinear / noncollinear), Floquet (exact Sambe space
  and high-frequency effective models).
* `register_model` decorator, `Model.make_spinful`.
* Tutorials 12-16 with quizzes; CI, contributing guide, citation file.

## 0.3.0
* Node-level parallelism (`n_jobs`) everywhere; Kwant-style value functions with parameters;
  arbitrary magnetic-field gauges; transmission eigenvalues, shot noise; sweeps; spectra.

## 0.2.0
* Kwant-style transport (systems, leads, Landauer-Buettiker, Andreev, LDOS, currents),
  KPM (DOS, LDOS, Kubo-Bastin), continuum discretiser, BdG builder; tutorials 08-11.

## 0.1.0
* Core engine, 28 models, bands/DOS, topology, Kubo AHC/SHC/OHC, tutorials 01-07.
