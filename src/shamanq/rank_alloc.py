"""Depth-prior rank redistribution at bit parity. §4.2 and Appendix B.

Uniform rank follows Eq. 11 of the paper, not a sign-only budget:

    r* = beta * m * n / (m + n) - 16

beta is the bits-per-weight budget. beta = 1 is the 1 bpw setting written
in Eq. 11. The -16 accounts for the two fp16 scale vectors in Eq. 2,
N_bits = (r + 16) (m + n). r* is then floored to a multiple of 32 and
capped by the largest multiple of 32 that does not exceed min(m, n).

The hand table sets beta_ell = beta * zeta * omega_ell, with
omega_ell = exp(kappa * (b / (B - 1) - 1/2)) * w_t(ell), kappa = 0.6,
and b in {1, ..., B}. zeta is chosen so that
sum_ell beta_ell * m_ell * n_ell = beta * sum_ell m_ell * n_ell.
Ranks come from Eq. 11, are floored to multiples of 32, then moved in
steps of 32, largest relative deviation from beta_ell first, until the
total bit count equals the uniform total.

The measured sensitivity allocator of Appendix B is not implemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

KAPPA = 0.6
RANK_STEP = 32
TYPE_WEIGHTS = {
    "q": 0.85,
    "k": 0.90,
    "v": 1.10,
    "o": 1.00,
    "gate": 1.00,
    "up": 1.05,
    "down": 1.15,
}


def n_bits(rank, m, n):
    """Eq. 2. Two sign matrices of rank r and two fp16 scale vectors."""
    return (int(rank) + 16) * (int(m) + int(n))


def ideal_rank(beta, m, n):
    """Real-valued Eq. 11, before the multiple-of-32 floor."""
    return beta * m * n / (m + n) - 16.0


def floor_rank(r_real, m, n, step=RANK_STEP):
    """Floor to a multiple of `step`, at least 0, at most min(m, n)."""
    cap = min(int(m), int(n))
    max_ok = (cap // step) * step
    if r_real <= 0.0 or max_ok <= 0:
        return 0
    stepped = math.floor(r_real / step + 1e-9) * step
    if stepped > max_ok:
        return max_ok
    if stepped < 0:
        return 0
    return int(stepped)


def uniform_rank(beta, m, n, step=RANK_STEP):
    """r_uni from Eq. 11, floored to a multiple of 32, ceiling min(m, n)."""
    return floor_rank(ideal_rank(beta, m, n), m, n, step=step)


def depth_weight(block, n_blocks, type_name, kappa=KAPPA):
    """omega_ell = exp(kappa * (b/(B-1) - 1/2)) * w_t(ell).

    block is 1-indexed, in 1..n_blocks. For a single block the depth factor
    is 1, because b/(B-1) is undefined. Same type, last block over first
    block, is a factor of exp(kappa) before the type weight.
    """
    if type_name not in TYPE_WEIGHTS:
        raise KeyError(f"unknown layer type {type_name!r}")
    w_t = TYPE_WEIGHTS[type_name]
    if n_blocks < 2:
        return w_t
    depth = math.exp(kappa * (block / (n_blocks - 1) - 0.5))
    return depth * w_t


@dataclass(frozen=True)
class LayerSpec:
    m: int
    n: int
    block: int
    type: str


@dataclass
class RankAllocation:
    ranks: list[int]
    uniform_ranks: list[int]
    beta_ell: list[float]
    omega: list[float]
    zeta: float
    ideal_ranks: list[float]
    n_bits_uniform: int
    n_bits: int


def _as_spec(layer):
    if isinstance(layer, LayerSpec):
        return layer
    m, n, block, type_name = layer
    return LayerSpec(int(m), int(n), int(block), str(type_name))


def _implied_beta(rank, m, n):
    return (rank + 16) * (m + n) / (m * n)


def _repair(ranks, layers, beta_ell, target, step):
    """Move ranks by ±step until the bit total matches `target`.

    Over budget: decrease the layer with the largest relative deviation
    (beta_implied - beta_ell) / beta_ell that can still drop a step.
    Under budget: increase the layer with the smallest relative deviation
    that can still take a step. Ties break toward the smaller index.
    """
    ranks = list(ranks)

    def total():
        return sum(n_bits(r, layer.m, layer.n) for r, layer in zip(ranks, layers))

    def rel(i):
        layer = layers[i]
        return (_implied_beta(ranks[i], layer.m, layer.n) - beta_ell[i]) / beta_ell[i]

    seen = set()
    limit = 4 * (sum(min(layer.m, layer.n) // step + 1 for layer in layers) + 1)
    for _ in range(max(limit, 1)):
        key = tuple(ranks)
        if key in seen:
            break
        seen.add(key)
        diff = total() - target
        if diff == 0:
            return ranks
        if diff > 0:
            cands = [i for i in range(len(ranks)) if ranks[i] >= step]
            if not cands:
                break
            i = max(cands, key=lambda i: (rel(i), -i))
            ranks[i] -= step
        else:
            cands = [
                i
                for i in range(len(ranks))
                if ranks[i] + step <= min(layers[i].m, layers[i].n)
            ]
            if not cands:
                break
            i = min(cands, key=lambda i: (rel(i), i))
            ranks[i] += step
    return ranks


def depth_prior_allocate(layers, beta=1.0, kappa=KAPPA, n_blocks=None, step=RANK_STEP):
    """Hand-table depth-prior allocation at bit parity with the uniform budget.

    layers is a sequence of LayerSpec or (m, n, block, type) with block
    1-indexed. n_blocks defaults to the largest block index present.
    """
    specs = [_as_spec(layer) for layer in layers]
    if not specs:
        raise ValueError("layers is empty")
    if n_blocks is None:
        n_blocks = max(layer.block for layer in specs)
    for layer in specs:
        if layer.block < 1 or layer.block > n_blocks:
            raise ValueError(f"block {layer.block} is outside 1..{n_blocks}")

    uniform = [uniform_rank(beta, layer.m, layer.n, step=step) for layer in specs]
    target = sum(n_bits(r, layer.m, layer.n) for r, layer in zip(uniform, specs))

    omegas = [
        depth_weight(layer.block, n_blocks, layer.type, kappa=kappa) for layer in specs
    ]
    param_sum = sum(layer.m * layer.n for layer in specs)
    weighted = sum(w * layer.m * layer.n for w, layer in zip(omegas, specs))
    zeta = param_sum / weighted
    beta_ell = [beta * zeta * w for w in omegas]
    ideals = [ideal_rank(b, layer.m, layer.n) for b, layer in zip(beta_ell, specs)]
    ranks = [
        floor_rank(r_real, layer.m, layer.n, step=step)
        for r_real, layer in zip(ideals, specs)
    ]
    ranks = _repair(ranks, specs, beta_ell, target, step)
    got = sum(n_bits(r, layer.m, layer.n) for r, layer in zip(ranks, specs))
    return RankAllocation(
        ranks=ranks,
        uniform_ranks=uniform,
        beta_ell=beta_ell,
        omega=omegas,
        zeta=zeta,
        ideal_ranks=ideals,
        n_bits_uniform=target,
        n_bits=got,
    )
