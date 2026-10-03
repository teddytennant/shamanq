"""KL-Shampoo fixed point, Eq. 9.

Three alternating passes, initialized at A = I. Solves use Cholesky, not an
explicit inverse. A relative jitter of CHOLESKY_JITTER times the mean
eigenvalue is added inside the solve only, so a singular sample does not
break the factorization. On well-conditioned data this jitter is far below
the fixed-point tolerance used in the tests, and the iteration approaches
the undamped fixed point of Eq. 9.

The paper writes the sum over calibration tokens, not the mean, because
H_cal is a sum. After each pass A is rescaled so tr(A) = n and G is scaled
by the inverse factor, which preserves the Kronecker product and the fixed
point of Eq. 9.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Relative to the mean eigenvalue, inside the Cholesky factor only.
CHOLESKY_JITTER = 1e-12
DEFAULT_PASSES = 3


def _quadratic_form_inv(S, vecs, jitter):
    """Row-wise v^T S^{-1} v via Cholesky. vecs is (N, d)."""
    S = 0.5 * (S + S.T)
    d = S.shape[0]
    mean_ev = jnp.trace(S) / d
    eye = jnp.eye(d, dtype=jnp.float64)
    S_d = S + (jitter * mean_ev) * eye
    L = jnp.linalg.cholesky(S_d)
    solved = jax.scipy.linalg.solve_triangular(L, vecs.T, lower=True)
    return jnp.sum(solved * solved, axis=0)


def _update_G(A, xs, deltas, jitter):
    """G = (1/n) sum_j (x_j^T A^{-1} x_j) delta_j delta_j^T."""
    n = xs.shape[1]
    weights = _quadratic_form_inv(A, xs, jitter)
    G = (deltas.T * weights) @ deltas / n
    return 0.5 * (G + G.T)


def _update_A(G, xs, deltas, jitter):
    """A = (1/m) sum_j (delta_j^T G^{-1} delta_j) x_j x_j^T."""
    m = deltas.shape[1]
    weights = _quadratic_form_inv(G, deltas, jitter)
    A = (xs.T * weights) @ xs / m
    return 0.5 * (A + A.T)


def _rescale_trace(A, G, n):
    """Enforce tr(A) = n and scale G by the inverse factor."""
    c = n / jnp.trace(A)
    return c * A, G / c


def kl_shampoo_fit(xs, deltas, n_passes=DEFAULT_PASSES, jitter=CHOLESKY_JITTER):
    """Alternating KL-Shampoo fit. xs is (N, n), deltas is (N, m).

    Returns (A, G) with tr(A) = n. Initialize A = I, then alternate the two
    updates of Eq. 9 for n_passes passes.
    """
    xs = jnp.asarray(xs, dtype=jnp.float64)
    deltas = jnp.asarray(deltas, dtype=jnp.float64)
    n = xs.shape[1]
    m = deltas.shape[1]
    A = jnp.eye(n, dtype=jnp.float64)
    G = jnp.eye(m, dtype=jnp.float64)
    for _ in range(n_passes):
        G = _update_G(A, xs, deltas, jitter)
        A = _update_A(G, xs, deltas, jitter)
        A, G = _rescale_trace(A, G, n)
    return A, G


def fixed_point_residual(A, G, xs, deltas, jitter=CHOLESKY_JITTER):
    """Relative residual of one more Eq. 9 pass, after the trace rescale.

    Returns (residual_A, residual_G). On well-conditioned synthetic data a
    few passes drive both below 1e-2, and usually much lower. The 1e-2
    figure is the loose tolerance used by the tests.
    """
    xs = jnp.asarray(xs, dtype=jnp.float64)
    deltas = jnp.asarray(deltas, dtype=jnp.float64)
    A = jnp.asarray(A, dtype=jnp.float64)
    G = jnp.asarray(G, dtype=jnp.float64)
    n = xs.shape[1]
    G2 = _update_G(A, xs, deltas, jitter)
    A2 = _update_A(G2, xs, deltas, jitter)
    A2, G2 = _rescale_trace(A2, G2, n)
    res_A = jnp.linalg.norm(A2 - A) / jnp.linalg.norm(A)
    res_G = jnp.linalg.norm(G2 - G) / jnp.linalg.norm(G)
    return res_A, res_G


def fresh_input_measurement(G, xs, deltas, jitter=CHOLESKY_JITTER):
    """One A-update of Eq. 9 with G held fixed. Renormalize tr(A) = n.

    A caller can run this immediately before factorization to refresh the
    input curvature. This is the A half-step of a warm-started Eq. 9 pass,
    not the paper's Step 2-1 unweighted second moment (see
    activation_second_moment).
    """
    xs = jnp.asarray(xs, dtype=jnp.float64)
    deltas = jnp.asarray(deltas, dtype=jnp.float64)
    G = jnp.asarray(G, dtype=jnp.float64)
    n = xs.shape[1]
    A = _update_A(G, xs, deltas, jitter)
    A = A * (n / jnp.trace(A))
    return 0.5 * (A + A.T)


def activation_second_moment(xs):
    """Paper Step 2-1: A_fresh = (1/N) sum_t x_t x_t^T, no gradient weighting."""
    xs = jnp.asarray(xs, dtype=jnp.float64)
    n_tok = xs.shape[0]
    A = (xs.T @ xs) / n_tok
    return 0.5 * (A + A.T)
