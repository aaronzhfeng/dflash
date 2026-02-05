"""TPU-compatible attention backend for DFlash.

Uses PyTorch's scaled_dot_product_attention (SDPA) which is XLA-compatible,
replacing flash-attention-2 which is CUDA-only.
"""

from typing import Optional, Tuple

import torch


def tpu_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """SDPA-based attention for TPU via PyTorch/XLA.

    Handles GQA by repeating KV heads to match query heads,
    then delegates to F.scaled_dot_product_attention.
    """
    # GQA: expand key/value heads to match query heads.
    # Use repeat_interleave instead of transformers' repeat_kv because
    # repeat_kv uses expand()+reshape() which produces stride-0 views
    # that XLA cannot lower to HLO correctly.
    n_heads_q = query.size(1)
    n_heads_k = key.size(1)
    if n_heads_k != n_heads_q:
        n_rep = n_heads_q // n_heads_k
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout if module.training else 0.0,
        is_causal=False,
        scale=scaling,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
