"""Mahalanobis geometry, shrinkage, transport, and spectral tempering.

Equations follow arXiv 2609.38521, §3.2 and §4.3 (Eq. 6, Eq. 10, Eq. 12, Eq. 13, Eq. 18, Eq. 19).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Paper values. gamma = 0.2, p = 1/2.
DEFAULT_GAMMA = 0.2
DEFAULT_P = 0.5
# Floor on the diagonal maps so both preconditioners stay invertible (Eq. 18).
DEFAULT_EPS = 1e-8


def mahalanobis_sq(E, G, A):
    """Squared Mahalanobis norm tr(E^T G E A).

    This is the trace form of Eq. 6. It does not form the mn x mn Kronecker
    matrix A ⊗ G. G is m x m, A is n x n, E is m x n.
    """
    E = jnp.asarray(E, dtype=jnp.float64)
    G = jnp.asarray(G, dtype=jnp.float64)
    A = jnp.asarray(A, dtype=jnp.float64)
    return jnp.trace(E.T @ G @ E @ A)


def shrink(S, gamma=DEFAULT_GAMMA):
    """Scale-matched shrinkage, Eq. 10.

    S_gamma = (1 - gamma) S + gamma (tr(S) / d) I.
    Trace and mean eigenvalue are unchanged.
    """
    S = jnp.asarray(S, dtype=jnp.float64)
    d = S.shape[0]
    mean_ev = jnp.trace(S) / d
    eye = jnp.eye(d, dtype=jnp.float64)
    return (1.0 - gamma) * S + gamma * mean_ev * eye


def diagonal_maps(A_gamma, G_gamma, eps=DEFAULT_EPS):
    """Square roots of the shrunken diagonals, floored at eps.

    Prose of §4.3: the diagonal preconditioners are the square roots of the
    diagonals of the shrunken statistics, floored at eps > 0.
    d_in = max(sqrt(diag(A_gamma)), eps), likewise d_out.
    """
    A_gamma = jnp.asarray(A_gamma, dtype=jnp.float64)
    G_gamma = jnp.asarray(G_gamma, dtype=jnp.float64)
    d_in = jnp.maximum(jnp.sqrt(jnp.maximum(jnp.diag(A_gamma), 0.0)), eps)
    d_out = jnp.maximum(jnp.sqrt(jnp.maximum(jnp.diag(G_gamma), 0.0)), eps)
    return d_in, d_out


def transport_statistics(A_gamma, G_gamma, d_in, d_out):
    """Congruence transport, Eq. 12. Does not refit A or G.

    A = D_in^{-1} A_gamma D_in^{-1}, G = D_out^{-1} G_gamma D_out^{-1}.
    Where the floor does not bind, the transported diagonals are 1.
    """
    A_gamma = jnp.asarray(A_gamma, dtype=jnp.float64)
    G_gamma = jnp.asarray(G_gamma, dtype=jnp.float64)
    d_in = jnp.asarray(d_in, dtype=jnp.float64)
    d_out = jnp.asarray(d_out, dtype=jnp.float64)
    A = A_gamma / d_in[:, None] / d_in[None, :]
    G = G_gamma / d_out[:, None] / d_out[None, :]
    A = 0.5 * (A + A.T)
    G = 0.5 * (G + G.T)
    return A, G


def temper(S, p=DEFAULT_P):
    """Trace-preserving spectral power, Eq. 13.

    If S = Q diag(lambda) Q^T, the tempered matrix is
    Q diag(lambda^p * mean(lambda) / mean(lambda^p)) Q^T.
    The mean eigenvalue (and the trace) is unchanged. The identity is
    unchanged. The log-condition-number is multiplied by p.

    After transport the statistics are unit-diagonal, so the mean eigenvalue
    is already 1 and stays 1. The formula here is the paper's trace-preserving
    power, which agrees with lambda^p / mean(lambda^p) exactly when the mean
    eigenvalue is 1.

    Returns (S_tempered, eigenvalues, eigenvectors). Eigenvectors are those of
    the input, which are also the eigenvectors of the tempered matrix.
    """
    S = jnp.asarray(S, dtype=jnp.float64)
    S = 0.5 * (S + S.T)
    evals, evecs = jnp.linalg.eigh(S)
    evals = jnp.maximum(evals, 0.0)
    powered = jnp.power(evals, p)
    mean_l = jnp.mean(evals)
    mean_p = jnp.mean(powered)
    scale = jnp.where(mean_p > 0.0, mean_l / mean_p, 1.0)
    new_evals = powered * scale
    S_t = (evecs * new_evals) @ evecs.T
    S_t = 0.5 * (S_t + S_t.T)
    return S_t, new_evals, evecs


def transform_weight(W, d_out, d_in):
    """W_tilde = D_out W D_in (Eq. 19)."""
    W = jnp.asarray(W, dtype=jnp.float64)
    d_out = jnp.asarray(d_out, dtype=jnp.float64)
    d_in = jnp.asarray(d_in, dtype=jnp.float64)
    return d_out[:, None] * W * d_in[None, :]


@dataclass
class MetricState:
    """Prepared layer state from Algorithm 1."""

    W_tilde: jax.Array
    A: jax.Array
    G: jax.Array
    M: jax.Array
    d_in: jax.Array
    d_out: jax.Array
    A_evals: jax.Array
    A_evecs: jax.Array
    G_evals: jax.Array
    G_evecs: jax.Array
    A_transported: jax.Array
    G_transported: jax.Array


def prepare_metric_target(W, A_gamma, G_gamma, eps=DEFAULT_EPS, p=DEFAULT_P):
    """Algorithm 1. A_gamma and G_gamma are the shrunken statistics.

    Transport, then temper. Do not refit A, G after transport.
    M = G_tempered W_tilde A_tempered is cached with the tempered eigensystems.
    """
    W = jnp.asarray(W, dtype=jnp.float64)
    A_gamma = jnp.asarray(A_gamma, dtype=jnp.float64)
    G_gamma = jnp.asarray(G_gamma, dtype=jnp.float64)
    d_in, d_out = diagonal_maps(A_gamma, G_gamma, eps=eps)
    A_t, G_t = transport_statistics(A_gamma, G_gamma, d_in, d_out)
    W_tilde = transform_weight(W, d_out, d_in)
    A, A_evals, A_evecs = temper(A_t, p=p)
    G, G_evals, G_evecs = temper(G_t, p=p)
    M = G @ W_tilde @ A
    return MetricState(
        W_tilde=W_tilde,
        A=A,
        G=G,
        M=M,
        d_in=d_in,
        d_out=d_out,
        A_evals=A_evals,
        A_evecs=A_evecs,
        G_evals=G_evals,
        G_evecs=G_evecs,
        A_transported=A_t,
        G_transported=G_t,
    )
