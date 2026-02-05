#!/usr/bin/env bash
# End-to-end test: DFlash spec_generate on TPU using local fork.
# Loads draft + target models and runs speculative decoding.
# Run from repo root: bash tpu_tests/e2e_generate.sh
#
# NOTE: First run is slow due to XLA graph compilation.
#       Each decode iteration may trigger recompilation when cache shapes
#       change (DynamicCache grows/crops). This is a known XLA limitation.
#       A future optimization is to replace DynamicCache with StaticCache.
#
# Estimated first-run time: ~20-40 min (max_new_tokens=32)
set -e

cd "$(dirname "$0")/.."

MAX_NEW_TOKENS="${1:-32}"
echo "=== DFlash TPU End-to-End Test (max_new_tokens=${MAX_NEW_TOKENS}) ==="

python3 -c "
import time
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.dflash import DFlashDraftModel

device = xm.xla_device()
print(f'Device: {device}')

# --- Draft model (our local fork) ---
print('Loading draft model...')
draft = DFlashDraftModel.from_pretrained(
    'z-lab/Qwen3-4B-DFlash-b16',
    attn_implementation='sdpa',
    dtype=torch.bfloat16,
).to(device).eval()

if not hasattr(draft.config, 'dflash_config') or draft.config.dflash_config is None:
    draft.config.dflash_config = {}
draft.config.dflash_config['attn_implementation'] = 'tpu'
print(f'Draft model loaded. block_size={draft.block_size}')

# --- Target model ---
print('Loading target model...')
target = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-4B',
    dtype=torch.bfloat16,
    attn_implementation='sdpa',
).to(device).eval()
print('Target model loaded.')

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
tokenizer.add_special_tokens({'mask_token': '<|MASK|>'})

# --- Input ---
prompt = 'How many positive whole-number divisors does 196 have?'
messages = [{'role': 'user', 'content': prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
model_inputs = tokenizer([text], return_tensors='pt').to(device)
print(f'Input tokens: {model_inputs[\"input_ids\"].shape}')

# --- Generate ---
max_new_tokens = ${MAX_NEW_TOKENS}
print(f'Running spec_generate (max_new_tokens={max_new_tokens})...')
t0 = time.time()

generate_ids = draft.spec_generate(
    input_ids=model_inputs['input_ids'],
    max_new_tokens=max_new_tokens,
    temperature=0.0,
    target=target,
    mask_token_id=tokenizer.mask_token_id,
    stop_token_ids=[tokenizer.eos_token_id],
)
xm.mark_step()

elapsed = time.time() - t0
output = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
num_generated = generate_ids.shape[1] - model_inputs['input_ids'].shape[1]

print(f'=== OUTPUT ({num_generated} tokens in {elapsed:.1f}s) ===')
print(output)
print('=== DONE ===')
"
