"""hopforge: forge any tight-binding Hamiltonian, from 1D chains to 3D crystals.

Quick start
-----------
>>> import hopforge as hf
>>> m = hf.models.haldane(t=1, t2=0.1, phi=0.5 * 3.14159, M=0.0)
>>> hf.chern_number(m, occ=1)
1.0
>>> m.plot_bands(["G", "K", "M", "G"])
"""
from . import operators
from .operators import pauli, spin_block, slater_koster
from .model import Lattice, Model
from .geometry import supercell, cut, finite, remove_orbitals, add_disorder
from .solvers import kmesh, kpath, bands, dos, fermi_level, band_gap
from .topology import (berry_phase, zak_phase, polarization, wilson_loop, chern_number,
                       berry_curvature, berry_curvature_map, z2_invariant, z2_3d,
                       z2_from_wcc, winding_number, fu_kane_parity)
from .kubo import hall_conductivity, anomalous_hall, spin_hall, orbital_hall
from . import models, plotting, transport, kpm
from .continuum import discretize
from .plotting import plot_bands, plot_dos, plot_lattice, plot_wcc, plot_kmap
from .quiz import Quiz

__version__ = "0.2.0"

__all__ = [n for n in dir() if not n.startswith("_")]
