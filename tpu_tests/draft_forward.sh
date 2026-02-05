#!/usr/bin/env bash
# Test: one draft model forward pass on TPU with real weights.
# Verifies the full pipeline (load models → prefill → draft forward)
# WITHOUT the spec_generate loop, so only 2 XLA compilations needed.
# Expected time: ~10-15 min (first run), ~1 min (cached).
# Run from repo root: bash tpu_tests/draft_forward.sh
set -e

cd "$(dirname "$0")/.."

echo "=== DFlash Draft Forward Test (TPU) ==="

python3 -c "
import time
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.dflash import DFlashDraftModel
from model.utils import extract_context_feature, sample

device = xm.xla_device()
print(f'Device: {device}')

# --- Load models ---
print('Loading draft model...')
draft = DFlashDraftModel.from_pretrained(
    'z-lab/Qwen3-4B-DFlash-b16',
    attn_implementation='sdpa',
    dtype=torch.bfloat16,
).to(device).eval()

if not hasattr(draft.config, 'dflash_config') or draft.config.dflash_config is None:
    draft.config.dflash_config = {}
draft.config.dflash_config['attn_implementation'] = 'tpu'
block_size = draft.block_size
print(f'Draft model loaded. block_size={block_size}')

print('Loading target model...')
target = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-4B',
    dtype=torch.bfloat16,
    attn_implementation='sdpa',
).to(device).eval()
print('Target model loaded.')

# --- Tokenizer & input ---
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
tokenizer.add_special_tokens({'mask_token': '<|MASK|>'})

prompt = 'How many positive whole-number divisors does 196 have?'
messages = [{'role': 'user', 'content': prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
input_ids = tokenizer([text], return_tensors='pt').input_ids.to(device)
num_input = input_ids.shape[1]
print(f'Input tokens: {num_input}')

# --- Step 1: Prefill (target model forward) ---
print('Step 1: Prefill...')
t0 = time.time()

max_length = num_input + 64
position_ids = torch.arange(max_length + block_size, device=device).unsqueeze(0)

output = target(
    input_ids,
    position_ids=position_ids[:, :num_input],
    use_cache=False,
    output_hidden_states=True,
)
xm.mark_step()

first_token = sample(output.logits[:, -1:, :], temperature=0.0)
target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)
xm.mark_step()

print(f'  Prefill done in {time.time() - t0:.1f}s')
print(f'  First token: {tokenizer.decode(first_token[0])}')
print(f'  target_hidden shape: {target_hidden.shape}')

# --- Step 2: One draft forward ---
print('Step 2: Draft forward...')
t0 = time.time()

# Create noise input (mask tokens for the block)
mask_ids = torch.full((1, block_size), tokenizer.mask_token_id, dtype=torch.long, device=device)
mask_ids[:, 0] = first_token[:, 0]
noise_embedding = target.model.embed_tokens(mask_ids)

# Position IDs must cover context + block (not just block) because
# RoPE is applied to the full KV sequence: k = cat([k_ctx, k_noise]).
# cos/sin need to have entries for all 26+16=42 positions.
# apply_rotary_pos_emb slices cos[-q_len:] for q, uses full cos for k.
all_position_ids = position_ids[:, :num_input + block_size]

draft_output = draft(
    target_hidden=target_hidden,
    noise_embedding=noise_embedding,
    position_ids=all_position_ids,
    use_cache=False,
    is_causal=False,
)
draft_logits = target.lm_head(draft_output[:, -block_size+1:, :])
draft_tokens = sample(draft_logits, temperature=0.0)
xm.mark_step()

print(f'  Draft forward done in {time.time() - t0:.1f}s')
print(f'  draft_output shape: {draft_output.shape}')
print(f'  draft_logits shape: {draft_logits.shape}')
print(f'  Draft tokens: {tokenizer.decode(draft_tokens[0])}')

print('=== PASSED: Draft forward on TPU works ===')
"
