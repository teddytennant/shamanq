ShamAN-Q is a Kronecker-curvature LB-ADMM sub-1-bit factorization (arXiv 2609.38521).

Install from this directory with `pip install -e .`, then run `pytest`. With jax, jaxlib, and pytest already present, `pytest` from this directory is enough, because the test config adds `src` to the path.

Synthetic layer demo, CPU, under a minute:

    python -m shamanq.demo

If the package is not installed, use `PYTHONPATH=src python -m shamanq.demo`.

This package does not quantize Qwen3, does not download WikiText-2, and does not report perplexity. It does not reproduce the published 22.96, 16.72, or 13.80 figures. KD distillation is not implemented (TUNE_FP, TUNE_LATENT, TUNE_SCALES). The measured sensitivity allocator is not implemented. The paper does not specify the linear rho endpoints or the NanoQuant initialization. Here rho grows linearly from 0.01 to 50, the ridge is 1e-3, and U_0, V_0 come from a truncated SVD of the transformed weight. The paper uses K = 400; the synthetic default is K = 80. SVID uses 32 power iterations. KL-Shampoo Cholesky solves add a relative jitter of 1e-12 inside the solve only. Uniform rank follows Eq. 11 of the paper, r = beta * m * n / (m + n) - 16, with beta the bits-per-weight budget (beta = 1 is the 1 bpw setting), then floored to a multiple of 32. The wired pre-factorization refresh is one A-update of Eq. 9 with G held fixed. The paper's Step 2-1 fresh statistic is the unweighted activation second moment, which is also provided.
