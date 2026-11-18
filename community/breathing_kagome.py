"""Example community contribution: the breathing kagome lattice.

Up triangles have hopping ta, down triangles tb (Ezawa, PRL 120, 026801 (2018):
a higher-order topological insulator in part of its phase diagram). The flat
band of the kagome lattice survives the breathing distortion at E = ta + tb,
which is the benchmark registered with the model.

Use it with
    import hopforge as hf
    import community.breathing_kagome            # registers the model
    hf.models.validate("breathing_kagome")
"""
import numpy as np

import hopforge as hf
from hopforge.models._util import HEX, HEX_POINTS, set_points


def _benchmark():
    m = breathing_kagome(ta=1.0, tb=0.5)
    E = m.energies(hf.kmesh(m, 24))
    flat = E[:, 2]
    return 1.5, round(float(flat.mean()), 10), bool(np.ptp(flat) < 1e-10 and abs(flat.mean() - 1.5) < 1e-10)


@hf.models.register_model("breathing_kagome", dim=2,
                          description="breathing kagome lattice (up/down triangle hoppings ta, tb)",
                          reference="Ezawa, PRL 120, 026801 (2018)",
                          benchmark=_benchmark, author="hopforge example", tags=("kagome", "flat band", "HOTI"),
                          replace=True)
def breathing_kagome(ta=1.0, tb=0.5):
    lat = hf.Lattice(HEX, [("A", [0, 0]), ("B", [0.5, 0]), ("C", [0, 0.5])])
    m = hf.Model(lat, name="breathing kagome")
    for i, j, R, d in m.neighbors(1):
        up = not np.any(R)                      # bonds inside the cell form the up triangles
        m.add_hopping(-(ta if up else tb), i, j, R)
    return set_points(m, HEX_POINTS, ["G", "K", "M", "G"])
