"""Identities for the ShamAN-Q linear-algebra core. CPU, deterministic, no network."""

import math
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import inspect

import jax
import jax.numpy as jnp
import pytest
from jax import random

jax.config.update("jax_enable_x64", True)

from shamanq.admm import (
    DEFAULT_K,
    DEFAULT_LAMBDA,
    DEFAULT_RHO0,
    DEFAULT_RHO1,
    error_in_metric,
    identity_metric_state,
    kron_lb_admm,
    tempered_error,
    zero_error,
)
from shamanq.kl_shampoo import (
    CHOLESKY_JITTER,
    fixed_point_residual,
    fresh_input_measurement,
    kl_shampoo_fit,
)
from shamanq.metric import (
    mahalanobis_sq,
    prepare_metric_target,
    shrink,
    temper,
    transport_statistics,
    diagonal_maps,
)
from shamanq.rank_alloc import (
    LayerSpec,
    depth_prior_allocate,
    depth_weight,
    ideal_rank,
    n_bits,
    uniform_rank,
)
from shamanq.svid import sign_pm1, svid
from shamanq.sylvester import kron_u_solve, sylvester_solve


def _spd(key, d, trace, cond=2.5):
    kQ, = random.split(key, 1)
    Q, _ = jnp.linalg.qr(random.normal(key, (d, d), dtype=jnp.float64))
    ev = jnp.linspace(1.0, cond, d)
    ev = ev * (trace / jnp.sum(ev))
    return (Q * ev) @ Q.T, Q, ev


def test_mahalanobis_matches_kronecker_quadratic_form():
    key = random.PRNGKey(0)
    kE, kG, kA = random.split(key, 3)
    m, n = 3, 2
    E = random.normal(kE, (m, n), dtype=jnp.float64)
    Graw = random.normal(kG, (m, m), dtype=jnp.float64)
    Araw = random.normal(kA, (n, n), dtype=jnp.float64)
    G = Graw @ Graw.T + 0.3 * jnp.eye(m)
    A = Araw @ Araw.T + 0.3 * jnp.eye(n)
    trace_form = mahalanobis_sq(E, G, A)
    # Explicit (A ⊗ G) quadratic form, column-major vec. Tiny on purpose.
    vec = E.T.reshape(-1)
    kron = jnp.kron(A, G)
    explicit = vec @ kron @ vec
    assert kron.shape == (m * n, m * n)
    assert jnp.allclose(trace_form, explicit, rtol=1e-12, atol=1e-12)
    src = inspect.getsource(mahalanobis_sq)
    assert "kron" not in src


def test_kl_shampoo_fixed_point_and_trace():
    # Well-conditioned Kronecker-Gaussian pairs. Loose residual tolerance 1e-2
    # is documented in kl_shampoo.fixed_point_residual. The Cholesky jitter is
    # 1e-12 relative and does not set the tolerance.
    key = random.PRNGKey(1)
    m, n, N = 4, 5, 2000
    A_true, _, _ = _spd(random.fold_in(key, 1), n, trace=n, cond=2.5)
    G_true, _, _ = _spd(random.fold_in(key, 2), m, trace=float(m), cond=2.5)
    xs = random.normal(random.fold_in(key, 3), (N, n), dtype=jnp.float64)
    deltas = random.normal(random.fold_in(key, 4), (N, m), dtype=jnp.float64)
    xs = xs @ jnp.linalg.cholesky(A_true).T
    deltas = deltas @ jnp.linalg.cholesky(G_true).T

    A0 = jnp.eye(n, dtype=jnp.float64)
    G0 = jnp.eye(m, dtype=jnp.float64)
    res0_A, res0_G = fixed_point_residual(A0, G0, xs, deltas)
    res0 = float(res0_A + res0_G)

    residuals = []
    A, G = A0, G0
    for passes in range(1, 5):
        A, G = kl_shampoo_fit(xs, deltas, n_passes=passes)
        res_A, res_G = fixed_point_residual(A, G, xs, deltas)
        residuals.append(float(res_A + res_G))
        assert jnp.allclose(jnp.trace(A), n, rtol=1e-10, atol=1e-10)

    assert residuals[0] < res0
    assert residuals[-1] < residuals[0]
    assert residuals[-1] < 1e-2
    assert CHOLESKY_JITTER <= 1e-10

    A_fresh = fresh_input_measurement(G, xs, deltas)
    assert jnp.allclose(jnp.trace(A_fresh), n, rtol=1e-10, atol=1e-10)
    assert A_fresh.shape == (n, n)


def test_shrinkage_preserves_trace_and_mean_eigenvalue():
    key = random.PRNGKey(2)
    raw = random.normal(key, (6, 6), dtype=jnp.float64)
    S = raw @ raw.T + 0.2 * jnp.eye(6)
    S_g = shrink(S, gamma=0.2)
    tr = jnp.trace(S)
    tr_g = jnp.trace(S_g)
    assert jnp.allclose(tr_g, tr, rtol=1e-12, atol=1e-12)
    mean = tr / S.shape[0]
    mean_g = jnp.mean(jnp.linalg.eigvalsh(S_g))
    assert jnp.allclose(mean_g, mean, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(jnp.mean(jnp.linalg.eigvalsh(S)), mean, rtol=1e-12, atol=1e-12)


def test_temper_identity_and_log_condition():
    eye = jnp.eye(5, dtype=jnp.float64)
    tempered, evals, _ = temper(eye, p=0.5)
    assert jnp.allclose(tempered, eye, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(evals, jnp.ones(5), rtol=1e-12, atol=1e-12)

    lam = jnp.array([1.0, 2.0, 4.0, 16.0], dtype=jnp.float64)
    S = jnp.diag(lam)
    p = 0.5
    S_p, _, _ = temper(S, p=p)
    out = jnp.linalg.eigvalsh(S_p)
    logcond_in = jnp.log(jnp.max(lam) / jnp.min(lam))
    logcond_out = jnp.log(jnp.max(out) / jnp.min(out))
    assert jnp.allclose(logcond_out, p * logcond_in, rtol=1e-12, atol=1e-12)
    # Eq. 13 is trace-preserving, so the mean eigenvalue is unchanged.
    assert jnp.allclose(jnp.trace(S_p), jnp.trace(S), rtol=1e-12, atol=1e-12)


def test_sylvester_matches_dense_vec_solve():
    key = random.PRNGKey(3)
    kG, kR, kB = random.split(key, 3)
    m, n = 4, 3
    Graw = random.normal(kG, (m, m), dtype=jnp.float64)
    Rraw = random.normal(kR, (n, n), dtype=jnp.float64)
    G = Graw @ Graw.T + 0.4 * jnp.eye(m)
    R = Rraw @ Rraw.T + 0.4 * jnp.eye(n)
    B = random.normal(kB, (m, n), dtype=jnp.float64)
    alpha = 0.7
    X = sylvester_solve(G, R, B, alpha)
    kron = jnp.kron(R, G) + alpha * jnp.eye(m * n)
    vec = jnp.linalg.solve(kron, B.T.reshape(-1))
    X_dense = vec.reshape((n, m)).T
    assert jnp.allclose(X, X_dense, rtol=1e-10, atol=1e-10)
    residual = G @ X @ R + alpha * X - B
    assert jnp.linalg.norm(residual) <= 1e-10 * jnp.linalg.norm(B)


def test_identity_metric_continuous_step_matches_normal_equation():
    key = random.PRNGKey(4)
    kW, kV, kC = random.split(key, 3)
    m, n, r = 8, 6, 2
    W = random.normal(kW, (m, n), dtype=jnp.float64)
    V = random.normal(kV, (n, r), dtype=jnp.float64)
    C = random.normal(kC, (m, r), dtype=jnp.float64)
    rho, lam = 0.4, 0.1
    alpha = rho + lam
    G = jnp.eye(m, dtype=jnp.float64)
    A = jnp.eye(n, dtype=jnp.float64)
    U = kron_u_solve(V, C, G, A, W, rho, lam)
    rhs = W @ V + rho * C
    U_dense = jnp.linalg.solve(V.T @ V + alpha * jnp.eye(r), rhs.T).T
    assert jnp.allclose(U, U_dense, rtol=1e-10, atol=1e-10)


def test_svid_signed_rank1_and_zeros():
    p = jnp.array([1.2, 0.4, 2.0, 0.7], dtype=jnp.float64)
    q = jnp.array([0.5, 1.3, 0.8], dtype=jnp.float64)
    S = jnp.array(
        [
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=jnp.float64,
    )
    P = (p[:, None] * q[None, :]) * S
    out = svid(P, n_iter=8)
    assert jnp.allclose(out.Z, P, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(out.S, S)
    assert jnp.all(out.p >= -1e-12)
    assert jnp.all(out.q >= -1e-12)
    assert jnp.allclose(jnp.linalg.norm(out.a), 1.0, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(jnp.linalg.norm(out.b), 1.0, rtol=1e-10, atol=1e-10)
    # Nonnegative factors reconstruct the absolute value.
    assert jnp.allclose(out.p[:, None] * out.q[None, :], jnp.abs(P), rtol=1e-10, atol=1e-10)

    P_zero = P.at[1, 2].set(0.0)
    assert float(sign_pm1(P_zero)[1, 2]) == 1.0
    assert float(sign_pm1(jnp.zeros((2, 2)))[0, 0]) == 1.0
    # A structural zero row still maps to +1, and the projection formula holds.
    p_row0 = p.at[2].set(0.0)
    P_row = (p_row0[:, None] * q[None, :]) * S
    out_row = svid(P_row, n_iter=8)
    assert jnp.allclose(out_row.S[2], jnp.ones(3))
    assert jnp.allclose(out_row.Z, (out_row.p[:, None] * out_row.S) * out_row.q[None, :])


def test_rank_allocator_bit_parity_and_depth_prior():
    # Paper Eq. 11: r* = beta * m * n / (m + n) - 16, then floor to 32.
    # m = n = 256, beta = 1 -> r* = 112, floored rank 96.
    assert uniform_rank(1.0, 256, 256) == 96
    assert uniform_rank(1.0, 256, 256) % 32 == 0
    assert uniform_rank(1.0, 256, 256) <= min(256, 256)
    assert abs(ideal_rank(1.0, 256, 256) - 112.0) < 1e-9
    # Ceiling: a large budget cannot exceed the largest multiple of 32 <= min(m, n).
    assert uniform_rank(10.0, 100, 80) == 64
    assert uniform_rank(10.0, 100, 80) <= min(100, 80)
    assert n_bits(32, 64, 64) == (32 + 16) * (64 + 64)

    # Last block over first, equal shape, same type. Ratio of depth weights is e^{0.6}.
    n_blocks = 4
    w_first = depth_weight(1, n_blocks, "o")
    w_last = depth_weight(n_blocks, n_blocks, "o")
    assert abs(w_last / w_first - math.exp(0.6)) < 1e-12

    layers = [LayerSpec(256, 256, block=b, type="o") for b in range(1, n_blocks + 1)]
    alloc = depth_prior_allocate(layers, beta=1.0, n_blocks=n_blocks)
    assert all(r % 32 == 0 for r in alloc.uniform_ranks)
    assert all(r % 32 == 0 for r in alloc.ranks)
    assert all(r <= 256 for r in alloc.ranks)
    assert alloc.n_bits == alloc.n_bits_uniform
    assert sum(n_bits(r, 256, 256) for r in alloc.ranks) == sum(
        n_bits(r, 256, 256) for r in alloc.uniform_ranks
    )
    assert alloc.ranks[-1] > alloc.ranks[0]
    param = sum(layer.m * layer.n for layer in layers)
    weighted = sum(b * layer.m * layer.n for b, layer in zip(alloc.beta_ell, layers))
    assert abs(weighted - 1.0 * param) < 1e-6 * param

    # Same depth, v weight 1.10 vs q weight 0.85. If the ±32 repair erases the
    # rank gap, the pre-rounding budgets still separate the types.
    pair = [
        LayerSpec(512, 512, block=1, type="q"),
        LayerSpec(512, 512, block=1, type="v"),
        LayerSpec(512, 512, block=2, type="o"),
    ]
    pair_alloc = depth_prior_allocate(pair, beta=1.0, n_blocks=2)
    assert pair_alloc.beta_ell[1] > pair_alloc.beta_ell[0]
    assert pair_alloc.n_bits == pair_alloc.n_bits_uniform
    if not (pair_alloc.ranks[1] > pair_alloc.ranks[0]):
        pytest.fail(
            "v rank was not above q rank after repair; "
            f"ranks={pair_alloc.ranks} budgets={pair_alloc.beta_ell}"
        )


def _curved_layer(key, m=8, n=6, r=2):
    kG, kA, kN = random.split(key, 3)
    evG = jnp.array([40.0, 24.0, 0.15, 0.12, 0.1, 0.08, 0.07, 0.05], dtype=jnp.float64)
    evA = jnp.array([30.0, 18.0, 0.12, 0.1, 0.08, 0.05], dtype=jnp.float64)
    QG, _ = jnp.linalg.qr(random.normal(kG, (m, m), dtype=jnp.float64))
    QA, _ = jnp.linalg.qr(random.normal(kA, (n, n), dtype=jnp.float64))
    G = (QG * evG) @ QG.T
    A = (QA * evA) @ QA.T
    signal = (QG[:, :r] * jnp.array([1.5, 1.0])) @ QA[:, :r].T
    noise_coeff = random.normal(kN, (m - r, n - r), dtype=jnp.float64)
    noise = 5.0 * (QG[:, r:] @ noise_coeff @ QA[:, r:].T)
    W = signal + noise
    return W, A, G


def test_kron_lb_admm_beats_zero_and_diagonal_metric():
    key = random.PRNGKey(7)
    m, n, r = 8, 6, 2
    W, A, G = _curved_layer(key, m=m, n=n, r=r)
    A_g = shrink(A, 0.2)
    G_g = shrink(G, 0.2)
    state = prepare_metric_target(W, A_g, G_g, eps=1e-8, p=0.5)
    # Floor does not bind on this layer: transported diagonals are 1.
    assert jnp.allclose(jnp.diag(state.A_transported), 1.0, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(jnp.diag(state.G_transported), 1.0, rtol=1e-10, atol=1e-10)

    fact = kron_lb_admm(
        state, r, K=DEFAULT_K, rho0=DEFAULT_RHO0, rho1=DEFAULT_RHO1, lam=DEFAULT_LAMBDA
    )
    assert fact.S_u.shape == (m, r)
    assert fact.S_v.shape == (n, r)
    assert fact.s1.shape == (m,)
    assert fact.s2.shape == (n,)
    assert jnp.all(jnp.isin(fact.S_u, jnp.array([-1.0, 1.0])))
    assert jnp.all(jnp.isin(fact.S_v, jnp.array([-1.0, 1.0])))
    err = tempered_error(state, fact)
    zerr = zero_error(state)
    assert jnp.isfinite(err)
    assert float(err) < float(zerr)

    id_state = identity_metric_state(state)
    fact_id = kron_lb_admm(
        id_state, r, K=DEFAULT_K, rho0=DEFAULT_RHO0, rho1=DEFAULT_RHO1, lam=DEFAULT_LAMBDA
    )
    err_id = error_in_metric(state, fact_id)
    assert jnp.isfinite(err_id)
    assert float(err) < float(err_id), (
        f"kronecker error {float(err)} was not below identity-metric error {float(err_id)}"
    )
