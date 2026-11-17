"""Disorder-averaged conductance and Fano factor of a graphene ribbon device,
parallel over the cores of one node (set HOPFORGE_NJOBS or pass n_jobs)."""
import time

import numpy as np

import hopforge as hf
from hopforge import transport as tr

N_JOBS = hf.parallel.resolve_jobs(None)
print(f"using {N_JOBS} worker processes")


def make(seed, W=2.0):
    s = tr.System(hf.models.honeycomb(), lambda r: 0 <= r[0] < 60 and 0 <= r[1] < 25,
                  onsite=tr.anderson(W, seed=seed))
    s.attach_lead((-1, 0))
    s.attach_lead((1, 0))
    return s


energies = np.linspace(0.05, 1.0, 20)
t0 = time.time()
mean, std = tr.disorder_average(make, range(4 * N_JOBS), energies, n_jobs=N_JOBS)
print(f"{4 * N_JOBS} realisations x {len(energies)} energies in {time.time() - t0:.1f} s")
for E, m, s in zip(energies, mean, std):
    print(f"E = {E:5.2f}   <T> = {m:7.3f} +- {s:6.3f}")
np.savetxt("conductance_vs_E.dat", np.column_stack([energies, mean, std]),
           header="E  <T>  std(T)")
