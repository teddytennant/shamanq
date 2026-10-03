"""Closed-form Mahalanobis Sylvester solve, Eq. 16.

G X R + alpha X = B, with G = Q diag(c) Q^T and R = S diag(nu) S^T,
has X = Q Y S^T and Y_ij = (Q^T B S)_ij / (c_i * nu_j + alpha).

Column-major vectorization turns this into (R ⊗ G + alpha I) vec(X) = vec(B)
when R is symmetric. The Kronecker matrix is never formed here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def sylvester_solve(G, R, B, alpha, left_eigh=None):
    """Solve G X R + alpha X = B for SPD (or SPSD) G and R, alpha > 0.

    left_eigh, if given, is (eigenvalues, eigenvectors) of G, so the layer
    curvature is not redecomposed on every ADMM step.
    """
    G = jnp.asarray(G, dtype=jnp.float64)
    R = jnp.asarray(R, dtype=jnp.float64)
    B = jnp.asarray(B, dtype=jnp.float64)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    if left_eigh is None:
        c, Q = jnp.linalg.eigh(0.5 * (G + G.T))
    else:
        c, Q = left_eigh
        c = jnp.asarray(c, dtype=jnp.float64)
        Q = jnp.asarray(Q, dtype=jnp.float64)
    nu, S = jnp.linalg.eigh(0.5 * (R + R.T))
    rhs = Q.T @ B @ S
    denom = c[:, None] * nu[None, :] + alpha
    Y = rhs / denom
    return Q @ Y @ S.T


def kron_u_solve(V, C_u, G, A, M, rho, lam, G_eigh=None):
    """KRON_U_SOLVE. Eq. 15 and Eq. 20.

    alpha = rho + lam, R^V = V^T A V, B^U = M V + rho C_u,
    G U R^V + alpha U = B^U.

    When G = I and A = I and M = W, this is the Euclidean right-sided
    normal equation (Eq. 5): U (V^T V + alpha I) = W V + rho C_u.
    lam must be positive so alpha > 0 even if rho is zero.
    """
    alpha = rho + lam
    R_V = V.T @ A @ V
    B_U = M @ V + rho * C_u
    return sylvester_solve(G, R_V, B_U, alpha, left_eigh=G_eigh)


def kron_v_solve(U, C_v, G, A, M, rho, lam, A_eigh=None):
    """KRON_V_SOLVE. Eq. 21 and Eq. 22.

    R^U = U^T G U, B^V = M^T U + rho C_v,
    A V R^U + alpha V = B^V.
    """
    alpha = rho + lam
    R_U = U.T @ G @ U
    B_V = M.T @ U + rho * C_v
    return sylvester_solve(A, R_U, B_V, alpha, left_eigh=A_eigh)
