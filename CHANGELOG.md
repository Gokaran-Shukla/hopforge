# Changelog

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
