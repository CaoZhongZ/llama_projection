import numpy as np

# hardware
from hw_config import *

# model
Hidden = 4096
AttHeads = 32
Headdim = Hidden / AttHeads
KVHidden = Headdim * 8
Interm = 14336
Vocab = 128256
Block = 32
Causal = True

# inputs
N = 32
Prompt = 1024
Gen = 1024

# 10us host submission
dummy_static =10/1000/1000

def dummy_time(tensor_size):
    dummy_static =10/1000/1000
    dummy_ops=10
    return max(tensor_size / B, dummy_static) * dummy_ops

# FF mlp + query + key + value + output in essential
C = Hidden * (3 * Interm + Hidden * 2 + KVHidden * 2)
# last gemm
logitw_size = Hidden * Vocab

prompt_kv = 2 * N * Prompt * KVHidden * kv_dtype
time_rope = prompt_kv / B
time_rsmnorm = 3 * prompt_kv / B

sdpa = 2 * N * Prompt * Prompt * Headdim * AttHeads
if not Causal:
    sdpa *= 2

time_sdpa = sdpa / T

# prefill roughly equal to promp compute time in each block
time_prefill_1block = 2 * Prompt * N * C / T + time_rope + time_sdpa + time_rsmnorm
time_prefill_model = time_prefill_1block * Block

# Generation in 1 transformer block, kv-history and model
# weights read or compute when batch size N is large
# The max() is an over simplification which could turn into
# other sophisticated functions
size_kv = 2 * N * (Prompt + Gen/2) * KVHidden * kv_dtype
t_size_kv = 2 * N * (Prompt + Gen) * KVHidden * kv_dtype * Block
weight_sz = C * w_dtype
gen_ops = 2 * N * C

time_gen_1block = size_kv / B + max(weight_sz/B, gen_ops/T)
time_gen_model = time_gen_1block * Block

# Last gemm for the logits, non-trivial if large vocabulary
time_next = max(logitw_size*fw_dtype/B, 2*N*logitw_size/(T/(fw_dtype/w_dtype)))

# total time prefill, logit gemm and generation of each token
time = time_prefill_model + time_next + (time_gen_model + time_next) * Gen

# Output with confidence info. Gen/Prefill ratio gives intuition how each portion affect final t/s
# Compute/Memory ratio show whether should downgrade memory efficiency
print("KV-cache size", t_size_kv/1024/1024, "MB")
print("Time = ", time)
print("T/s = ", N * (Gen + Prompt)/time)
print("Prefill time", time_prefill_model + time_next)
print("Next time", time_gen_model + time_next)
print("Generation time", (time_gen_model + time_next) * Gen)
print("Gen:Prefill ratio", (time_gen_model + time_next) * Gen / (time_prefill_model + time_next))
print("Compute/Memory ratio", (gen_ops/T)/(weight_sz/B))
