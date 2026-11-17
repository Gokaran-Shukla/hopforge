import numpy as np

from .. import operators as ops

S3 = np.sqrt(3.0)
HEX = [[1.0, 0.0], [0.5, S3 / 2]]
HEX_POINTS = {"G": [0, 0], "K": [2 / 3, 1 / 3], "K'": [1 / 3, 2 / 3], "M": [0.5, 0]}
SQ_POINTS = {"G": [0, 0], "X": [0.5, 0], "Y": [0, 0.5], "M": [0.5, 0.5]}


def set_points(model, pts, path=None):
    model.special_points = {k: np.asarray(v, float) for k, v in pts.items()}
    if path:
        model.info["default_path"] = list(path)
    return model


def cart_points(model, pts_2pi_over_a, path=None):
    """Special points given in Cartesian units of 2 pi / a -> reduced."""
    B = model.lat.reciprocal
    out = {}
    for k, v in pts_2pi_over_a.items():
        kc = 2 * np.pi * np.asarray(v, float)
        out[k] = np.linalg.lstsq(B.T, kc, rcond=None)[0]
    return set_points(model, out, path)


def chirality(model, i, j, R, nn_dist, tol=1e-4, search=2):
    """nu_ij = sign((d_ik x d_kj)_z) for the shortest two-step path i -> k -> j
    through a nearest neighbour k (Haldane / Kane-Mele sign)."""
    import itertools
    A, P = model.lat.vectors, model.lat.positions
    rj = P[j] + R @ A
    ri = P[i]
    for Rk in itertools.product(range(-search, search + 1), repeat=model.dim):
        rk0 = np.array(Rk) @ A
        for k in range(model.norb):
            rk = P[k] + rk0
            d1, d2 = rk - ri, rj - rk
            if abs(np.linalg.norm(d1) - nn_dist) < tol and abs(np.linalg.norm(d2) - nn_dist) < tol:
                c = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(c) > tol:
                    return float(np.sign(c))
    raise RuntimeError("no intermediate neighbour found")


def spin_diag(h_up, h_dn):
    """Orbital-major matrix with spin-diagonal blocks h_up, h_dn."""
    up = np.array([[1, 0], [0, 0]], complex)
    dn = np.array([[0, 0], [0, 1]], complex)
    return np.kron(h_up, up) + np.kron(h_dn, dn)


def LS_matrix(l, lam):
    """lambda L.S in the real-orbital (x) spin basis, orbital-major."""
    L = ops.angular_momentum_matrices(l)
    s = [ops.sx / 2, ops.sy / 2, ops.sz / 2]
    return lam * sum(np.kron(L[a], s[a]) for a in range(3))
