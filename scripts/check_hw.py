"""Verify the Mac Studio stack: CPU threads, PyTorch MPS, MLX, LightGBM. Prints rough throughput."""
import os, time, platform
import numpy as np
import lightgbm as lgb  # must import before torch: both ship libomp, torch-first segfaults

def bench(fn, n=5):
    fn(); t = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t) / n

print(f"python {platform.python_version()}  cpu threads {os.cpu_count()}")

N = 8192
flops = 2 * N**3

import torch
assert torch.backends.mps.is_available(), "MPS not available"
a = torch.randn(N, N, device="mps"); b = torch.randn(N, N, device="mps")
def t_mm(): (a @ b); torch.mps.synchronize()
print(f"torch {torch.__version__} MPS fp32 matmul: {flops / bench(t_mm) / 1e12:.1f} TFLOPS")

import mlx.core as mx
x = mx.random.normal((N, N)); y = mx.random.normal((N, N))
def m_mm(): mx.eval(x @ y)
print(f"mlx {mx.__version__} GPU fp32 matmul: {flops / bench(m_mm) / 1e12:.1f} TFLOPS")
xh, yh = x.astype(mx.float16), y.astype(mx.float16)
def m_mmh(): mx.eval(xh @ yh)
print(f"mlx GPU fp16 matmul: {flops / bench(m_mmh) / 1e12:.1f} TFLOPS")

rng = np.random.default_rng(0)
X = rng.standard_normal((500_000, 50)); yv = X[:, 0] * 0.1 + rng.standard_normal(500_000)
t = time.perf_counter()
lgb.train({"objective": "regression", "num_threads": os.cpu_count(), "verbose": -1}, lgb.Dataset(X, yv), 200)
print(f"lightgbm {lgb.__version__} 500k x 50, 200 trees on {os.cpu_count()} threads: {time.perf_counter() - t:.1f}s")
print("ALL CHECKS PASSED")
