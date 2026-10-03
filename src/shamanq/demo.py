"""Synthetic one-layer ShamAN-Q demo. CPU, no data download.

    python -m shamanq.demo
"""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)

from shamanq.admm import (
    error_in_metric,
    factorize_layer,
    identity_metric_state,
    kron_lb_admm,
    tempered_error,
    zero_error,
)
from shamanq.kl_shampoo import fixed_point_residual, kl_shampoo_fit
from shamanq.rank_alloc import LayerSpec, depth_prior_allocate


def _spd(key, evals):
    evals = jnp.asarray(evals, dtype=jnp.float64)
    d = evals.shape[0]
    Q, _ = jnp.linalg.qr(random.normal(key, (d, d), dtype=jnp.float64))
    return (Q * evals) @ Q.T


def main():
    key = random.PRNGKey(0)
    m, n, r = 12, 10, 4
    kA, kG, kX, kD, kW = random.split(key, 5)
    evA = jnp.linspace(0.6, 2.4, n)
    evG = jnp.linspace(0.5, 2.2, m)
    A_true = _spd(kA, evA * (n / jnp.sum(evA)))
    G_true = _spd(kG, evG)
    n_tok = 256
    xs = random.normal(kX, (n_tok, n), dtype=jnp.float64) @ jnp.linalg.cholesky(A_true).T
    deltas = random.normal(kD, (n_tok, m), dtype=jnp.float64) @ jnp.linalg.cholesky(G_true).T
    A_fit, G_fit = kl_shampoo_fit(xs, deltas, n_passes=3)
    res_A, res_G = fixed_point_residual(A_fit, G_fit, xs, deltas)

    W = random.normal(kW, (m, n), dtype=jnp.float64)
    fact, state = factorize_layer(W, A_fit, G_fit, r, refresh_pairs=(xs, deltas), K=40)
    err = tempered_error(state, fact)
    zerr = zero_error(state)
    fact_id = kron_lb_admm(identity_metric_state(state), r, K=40)
    err_id = error_in_metric(state, fact_id)

    layers = [LayerSpec(256, 256, block=b, type="o") for b in (1, 2, 3, 4)]
    alloc = depth_prior_allocate(layers, beta=1.0)

    print("ShamAN-Q synthetic layer")
    print(f"shape m={m} n={n} rank={r}")
    print(
        f"kl_shampoo passes=3 tr_A={float(jnp.trace(A_fit)):.6f} "
        f"residual_A={float(res_A):.6e} residual_G={float(res_G):.6e}"
    )
    print(
        f"mahalanobis_error={float(err):.6f} zero_error={float(zerr):.6f} "
        f"identity_metric_error={float(err_id):.6f}"
    )
    print(
        f"factors S_u={tuple(fact.S_u.shape)} S_v={tuple(fact.S_v.shape)} "
        f"s1={tuple(fact.s1.shape)} s2={tuple(fact.s2.shape)}"
    )
    print(
        f"depth_prior first_rank={alloc.ranks[0]} last_rank={alloc.ranks[-1]} "
        f"uniform_bits={alloc.n_bits_uniform} alloc_bits={alloc.n_bits}"
    )


if __name__ == "__main__":
    main()
