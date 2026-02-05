#!/usr/bin/env bash
# Quick TPU sanity checks — verifies matmul and SDPA+GQA work on TPU.
# Run from repo root: bash tpu_tests/sanity.sh
set -e

echo "=== TPU Sanity Check ==="

python3 -c "
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(f'Device: {device}')

# 1. Basic matmul
a = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
b = a @ a.T
xm.mark_step()
print(f'[PASS] Matmul: {b.shape} on {b.device}')

# 2. SDPA + GQA (matches DFlash attention shapes for Qwen3-4B)
bsz, q_len, kv_len, head_dim = 1, 16, 64, 128
query = torch.randn(bsz, 32, q_len, head_dim, dtype=torch.bfloat16, device=device)
key   = torch.randn(bsz, 8,  kv_len, head_dim, dtype=torch.bfloat16, device=device)
value = torch.randn(bsz, 8,  kv_len, head_dim, dtype=torch.bfloat16, device=device)

key   = key.repeat_interleave(4, dim=1)
value = value.repeat_interleave(4, dim=1)

out = torch.nn.functional.scaled_dot_product_attention(
    query, key, value, is_causal=False
)
xm.mark_step()
print(f'[PASS] SDPA+GQA: q={query.shape} -> out={out.shape}')

print('=== All sanity checks passed ===')
"
