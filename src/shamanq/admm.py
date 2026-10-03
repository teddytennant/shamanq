"""Kron-LB-ADMM for one transformed layer. Algorithm 2, Appendix C.

The paper says rho_k grows linearly over K iterations and does not give the
endpoints. This code uses

    rho_k = rho_0 + (rho_1 - rho_0) * k / (K - 1),  k = 0, ..., K - 1

with rho_0 = 0.01, rho_1 = 50, and ridge lambda = 1e-3. lambda > 0 keeps
every Sylvester denominator positive. The larger endpoint is big enough
that the SVID constraint is felt on synthetic layers whose transported
weights are not O(1). The paper's run uses K = 400; the default here is
K = 80, which is enough for the synthetic tests.

U_0, V_0 are a truncated SVD of the transformed weight, split by the square
root of the singular values so that U_0 V_0^T is the rank-r SVD. NanoQuant's
exact initialization is not in this paper. Scaled duals start at 0, which
the paper also does not specify.

Scale extraction is a simple stand-in for NanoQuant's magnitude balance
(its Eq. 7, not in this paper) and the row-wise mean absolute values of
§3.1. Column norms of the final proxies are split geometrically between the
two factors, signs are read off the balanced proxies, and scales are the
row-wise mean absolute values. Eq. 25 returns the scales to original
coordinates: s1 = d_out^{-1} odot s1_tilde, s2 = d_in^{-1} odot s2_tilde.

KD distillation (TUNE_FP, TUNE_LATENT, TUNE_SCALES) is not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from shamanq.kl_shampoo import fresh_input_measurement
from shamanq.metric import (
    DEFAULT_EPS,
    DEFAULT_GAMMA,
    DEFAULT_P,
    MetricState,
    mahalanobis_sq,
    prepare_metric_target,
    shrink,
    transform_weight,
)
from shamanq.svid import SVID_POWER_ITERS, sign_pm1, svid
from shamanq.sylvester import kron_u_solve, kron_v_solve

jax.config.update("jax_enable_x64", True)

DEFAULT_RHO0 = 1e-2
DEFAULT_RHO1 = 50.0
DEFAULT_LAMBDA = 1e-3
DEFAULT_K = 80


def truncated_svd_factors(W, rank):
    """U_0, V_0 from a truncated SVD of W. Not NanoQuant's unpublished init."""
    W = jnp.asarray(W, dtype=jnp.float64)
    m, n = W.shape
    if rank < 1 or rank > min(m, n):
        raise ValueError(f"rank {rank} is outside 1..{min(m, n)}")
    u, s, vh = jnp.linalg.svd(W, full_matrices=False)
    scale = jnp.sqrt(jnp.maximum(s[:rank], 0.0))
    U0 = u[:, :rank] * scale
    V0 = vh[:rank, :].T * scale
    return U0, V0


def _rho(k, K, rho0, rho1):
    if K <= 1:
        return rho1
    return rho0 + (rho1 - rho0) * (k / (K - 1))


def _finalize(P_u, P_v, d_out, d_in):
    """Column-norm split, then row-wise mean absolute scales, then Eq. 25."""
    nu = jnp.linalg.norm(P_u, axis=0)
    nv = jnp.linalg.norm(P_v, axis=0)
    nu_s = jnp.maximum(nu, 1e-12)
    nv_s = jnp.maximum(nv, 1e-12)
    P_u_b = P_u * jnp.sqrt(nv_s / nu_s)
    P_v_b = P_v * jnp.sqrt(nu_s / nv_s)
    S_u = sign_pm1(P_u_b)
    S_v = sign_pm1(P_v_b)
    s1_tilde = jnp.mean(jnp.abs(P_u_b), axis=1)
    s2_tilde = jnp.mean(jnp.abs(P_v_b), axis=1)
    s1 = s1_tilde / d_out
    s2 = s2_tilde / d_in
    W_hat = (s1[:, None] * S_u) @ (S_v.T * s2[None, :])
    return S_u, S_v, s1, s2, s1_tilde, s2_tilde, W_hat


@dataclass
class Factorization:
    S_u: jax.Array
    S_v: jax.Array
    s1: jax.Array
    s2: jax.Array
    s1_tilde: jax.Array
    s2_tilde: jax.Array
    W_hat: jax.Array
    U: jax.Array
    V: jax.Array
    P_u: jax.Array
    P_v: jax.Array


def kron_lb_admm(
    state,
    rank,
    K=DEFAULT_K,
    rho0=DEFAULT_RHO0,
    rho1=DEFAULT_RHO1,
    lam=DEFAULT_LAMBDA,
    n_svid=SVID_POWER_ITERS,
):
    """Algorithm 2 on one prepared layer.

    state is a MetricState (tempered transported metric, cached M and
    eigensystems). Returns the deployment factorization in original
    coordinates.
    """
    if lam <= 0.0:
        raise ValueError("lambda must be positive so every denominator is positive")
    if rho0 < 0.0 or rho1 < 0.0:
        raise ValueError("rho endpoints must be nonnegative")
    U, V = truncated_svd_factors(state.W_tilde, rank)
    Z_u = svid(U, n_iter=n_svid).Z
    Z_v = svid(V, n_iter=n_svid).Z
    Lam_u = jnp.zeros_like(U)
    Lam_v = jnp.zeros_like(V)
    G_eigh = (state.G_evals, state.G_evecs)
    A_eigh = (state.A_evals, state.A_evecs)
    for k in range(K):
        rho = _rho(k, K, rho0, rho1)
        C_u = Z_u - Lam_u
        C_v = Z_v - Lam_v
        U = kron_u_solve(
            V, C_u, state.G, state.A, state.M, rho, lam, G_eigh=G_eigh
        )
        V = kron_v_solve(
            U, C_v, state.G, state.A, state.M, rho, lam, A_eigh=A_eigh
        )
        Z_u = svid(U + Lam_u, n_iter=n_svid).Z
        Z_v = svid(V + Lam_v, n_iter=n_svid).Z
        Lam_u = Lam_u + U - Z_u
        Lam_v = Lam_v + V - Z_v
    P_u = U + Lam_u
    P_v = V + Lam_v
    S_u, S_v, s1, s2, s1_tilde, s2_tilde, W_hat = _finalize(
        P_u, P_v, state.d_out, state.d_in
    )
    return Factorization(
        S_u=S_u,
        S_v=S_v,
        s1=s1,
        s2=s2,
        s1_tilde=s1_tilde,
        s2_tilde=s2_tilde,
        W_hat=W_hat,
        U=U,
        V=V,
        P_u=P_u,
        P_v=P_v,
    )


def transformed_reconstruction(state, fact):
    """Deployment reconstruction in the transported coordinates."""
    return transform_weight(fact.W_hat, state.d_out, state.d_in)


def tempered_error(state, fact):
    """Mahalanobis error of the deployment factorization in the tempered metric."""
    E = state.W_tilde - transformed_reconstruction(state, fact)
    return mahalanobis_sq(E, state.G, state.A)


def zero_error(state):
    """Mahalanobis error of the zero reconstruction, same metric."""
    return mahalanobis_sq(state.W_tilde, state.G, state.A)


def identity_metric_state(state):
    """Same transformed weight and diagonal maps, with A = I and G = I.

    This is NanoQuant's geometry in the transported coordinates (§4.3).
    M = W_tilde. Evaluation of both runs should still use the curved metric.
    """
    m, n = state.W_tilde.shape
    eye_m = jnp.eye(m, dtype=jnp.float64)
    eye_n = jnp.eye(n, dtype=jnp.float64)
    return MetricState(
        W_tilde=state.W_tilde,
        A=eye_n,
        G=eye_m,
        M=state.W_tilde,
        d_in=state.d_in,
        d_out=state.d_out,
        A_evals=jnp.ones((n,), dtype=jnp.float64),
        A_evecs=eye_n,
        G_evals=jnp.ones((m,), dtype=jnp.float64),
        G_evecs=eye_m,
        A_transported=state.A_transported,
        G_transported=state.G_transported,
    )


def error_in_metric(state_curved, fact):
    """Deployment error of `fact` measured in `state_curved`'s tempered metric.

    The factorization may have been produced under a different metric (for
    example A = I, G = I) as long as it shares the diagonal maps, so that
    the transformed reconstruction lives in the same coordinates.
    """
    E = state_curved.W_tilde - transformed_reconstruction(state_curved, fact)
    return mahalanobis_sq(E, state_curved.G, state_curved.A)


def factorize_layer(
    W,
    A,
    G,
    rank,
    *,
    refresh_pairs=None,
    gamma=DEFAULT_GAMMA,
    eps=DEFAULT_EPS,
    p=DEFAULT_P,
    K=DEFAULT_K,
    rho0=DEFAULT_RHO0,
    rho1=DEFAULT_RHO1,
    lam=DEFAULT_LAMBDA,
    n_svid=SVID_POWER_ITERS,
):
    """Shrink, optionally refresh A, prepare the metric, and run Kron-LB-ADMM.

    Pass raw curvature statistics. If refresh_pairs = (xs, deltas) is set,
    A is replaced by one Eq. 9 A-update with G held fixed and tr(A) = n,
    immediately before shrinkage. Both statistics are then shrunk (Eq. 10)
    and transported. Shrinkage preserves the trace, so tr(A) = n still holds
    after the refresh. Returns (Factorization, MetricState).
    """
    W = jnp.asarray(W, dtype=jnp.float64)
    A = jnp.asarray(A, dtype=jnp.float64)
    G = jnp.asarray(G, dtype=jnp.float64)
    if refresh_pairs is not None:
        xs, deltas = refresh_pairs
        A = fresh_input_measurement(G, xs, deltas)
    A = shrink(A, gamma)
    G = shrink(G, gamma)
    state = prepare_metric_target(W, A, G, eps=eps, p=p)
    fact = kron_lb_admm(
        state, rank, K=K, rho0=rho0, rho1=rho1, lam=lam, n_svid=n_svid
    )
    return fact, state
