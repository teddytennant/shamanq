"""ShamAN-Q: Kronecker-curvature LB-ADMM sub-1-bit factorization.

Faithful linear-algebra core of arXiv 2609.38521. CPU only.
"""

import os

# Force CPU before JAX initializes. This package is linear algebra, not a train.
os.environ["JAX_PLATFORMS"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)

from shamanq.admm import (
    Factorization,
    factorize_layer,
    identity_metric_state,
    kron_lb_admm,
    tempered_error,
)
from shamanq.kl_shampoo import (
    activation_second_moment,
    fixed_point_residual,
    fresh_input_measurement,
    kl_shampoo_fit,
)
from shamanq.metric import (
    MetricState,
    diagonal_maps,
    mahalanobis_sq,
    prepare_metric_target,
    shrink,
    temper,
    transport_statistics,
)
from shamanq.rank_alloc import (
    TYPE_WEIGHTS,
    LayerSpec,
    RankAllocation,
    depth_prior_allocate,
    n_bits,
    uniform_rank,
)
from shamanq.svid import SVID_POWER_ITERS, sign_pm1, svid
from shamanq.sylvester import kron_u_solve, kron_v_solve, sylvester_solve

__all__ = [
    "TYPE_WEIGHTS",
    "SVID_POWER_ITERS",
    "Factorization",
    "LayerSpec",
    "MetricState",
    "RankAllocation",
    "activation_second_moment",
    "depth_prior_allocate",
    "diagonal_maps",
    "factorize_layer",
    "fixed_point_residual",
    "fresh_input_measurement",
    "identity_metric_state",
    "kl_shampoo_fit",
    "kron_lb_admm",
    "kron_u_solve",
    "kron_v_solve",
    "mahalanobis_sq",
    "n_bits",
    "prepare_metric_target",
    "shrink",
    "sign_pm1",
    "svid",
    "sylvester_solve",
    "temper",
    "tempered_error",
    "transport_statistics",
    "uniform_rank",
]
