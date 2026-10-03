"""SVID projection, Eq. 23 and Eq. 24.

Independent of A and G. The paper does not specify the power-iteration
count. This code uses SVID_POWER_ITERS = 32, which is exact in one or two
steps on a rank-1 absolute value and is fixed so the projection is
deterministic.

Zeros in the proxy map to +1. The top left and right singular vectors of
|P| are returned nonnegative. Power iteration starts from the all-ones
vector, which stays nonnegative on a nonnegative matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

SVID_POWER_ITERS = 32


def sign_pm1(P):
    """Sign with zeros mapped to +1 (Eq. 23)."""
    P = jnp.asarray(P, dtype=jnp.float64)
    s = jnp.sign(P)
    return jnp.where(s == 0.0, 1.0, s)


def _normalize(v):
    nrm = jnp.linalg.norm(v)
    fallback = jnp.zeros_like(v).at[0].set(1.0)
    scaled = v / jnp.maximum(nrm, 1e-30)
    return jnp.where(nrm > 1e-30, scaled, fallback)


def top_singular_triplet(abs_P, n_iter=SVID_POWER_ITERS):
    """Power iteration for the top singular triplet of a nonnegative matrix.

    Returns (a, sigma, b) with a, b nonnegative and unit length when the
    matrix is not the zero matrix. sigma is a^T abs_P b.
    """
    abs_P = jnp.asarray(abs_P, dtype=jnp.float64)
    m, r = abs_P.shape
    b = jnp.ones((r,), dtype=jnp.float64)
    b = _normalize(b)
    a = jnp.ones((m,), dtype=jnp.float64)
    for _ in range(n_iter):
        a = _normalize(abs_P @ b)
        b = _normalize(abs_P.T @ a)
    # Nonnegative convention. Starting from ones, this is a no-op on |P|.
    a = jnp.maximum(a, 0.0)
    b = jnp.maximum(b, 0.0)
    a = _normalize(a)
    b = _normalize(b)
    sigma = jnp.dot(a, abs_P @ b)
    sigma = jnp.maximum(sigma, 0.0)
    return a, sigma, b


@dataclass
class SVIDFactors:
    Z: jax.Array
    S: jax.Array
    p: jax.Array
    q: jax.Array
    a: jax.Array
    b: jax.Array
    sigma: jax.Array


def svid(P, n_iter=SVID_POWER_ITERS):
    """Project P onto the SVID family, Eq. 24.

    S = sign(P), (a, sigma, b) = top singular triplet of |P|,
    p = sqrt(sigma) a, q = sqrt(sigma) b,
    SVID(P) = (sigma a b^T) odot S = Diag(p) S Diag(q).
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    S = sign_pm1(P)
    a, sigma, b = top_singular_triplet(jnp.abs(P), n_iter=n_iter)
    root = jnp.sqrt(sigma)
    p = root * a
    q = root * b
    Z = (p[:, None] * S) * q[None, :]
    return SVIDFactors(Z=Z, S=S, p=p, q=q, a=a, b=b, sigma=sigma)
