import numpy as np
from statistics import show

# hardware
from xe3p_fp4 import *

# model
from deepseek_v3 import *

# inputs
from batch_config import *

# 10us host submission
dummy_static =10/1000/1000

def dummy_time(tensor_size):
    dummy_static =10/1000/1000
    dummy_ops=10
    return max(tensor_size / B, dummy_static) * dummy_ops

q_a_proj = Hidden * q_lora_rank
q_b_proj = q_lora_rank * num_heads * (qk_nope_head_dim + qk_rope_head_dim)
kv_a_proj = Hidden * (kv_lora_rank + qk_rope_head_dim)
kv_b_proj = kv_lora_rank * num_heads * (qk_nope_head_dim + v_head_dim)
o_proj = num_heads * v_head_dim * Hidden
mla_gemm_weights = q_a_proj + q_b_proj + kv_a_proj + kv_b_proj + o_proj

q_head_dim = qk_nope_head_dim + qk_rope_head_dim
sdpa= 2 * N * Prompt * Prompt * q_head_dim * num_heads
if not Causal:
    sdpa *= 2

q_b_fold = q_lora_rank * num_heads * (kv_lora_rank + qk_rope_head_dim)
o_proj_fold = kv_lora_rank * num_heads * Hidden
reduced_gemm_weights = q_a_proj + q_b_fold + kv_a_proj + o_proj_fold

size_kv = [N * (Prompt + i) * (kv_lora_rank + qk_rope_head_dim) * kv_dtype for i in range(Gen)]
t_size_kv = N * (Prompt + Gen) * (kv_lora_rank + qk_rope_head_dim) * kv_dtype * Block

# Deepseek MLA modeling
def project_mla_prefill(N, Prompt):
    mla_gemm_read = mla_gemm_weights / B
    mla_gemm_compute = 2 * N * Prompt * mla_gemm_weights / T
    mla_io = N * Prompt * (q_lora_rank + kv_lora_rank + 2 * qk_rope_head_dim)
    mla_io_read = mla_io / B if mla_io > L3_size else mla_io / 4 / B
    time_sdpa = sdpa / T
    return max(mla_gemm_read, mla_gemm_compute) + mla_io_read + time_sdpa

def project_mla_gen(N, kv_size):
    mla_gemm_read = mla_gemm_weights / B
    mla_gemm_compute = 2 * N * 1 * mla_gemm_weights / T
    mla_io = N * (q_lora_rank + kv_lora_rank + 2 * qk_rope_head_dim)
    mla_io_read = mla_io / B if mla_io > L3_size else mla_io / 4 / B
    return max(mla_gemm_read, mla_gemm_compute) + mla_io_read + kv_size / B

# Fold will make things worse
def project_mla_folded_prefill(N, Prompt):
    mla_gemm_read = reduced_gemm_weights / B
    mla_gemm_compute = 2 * N * Prompt * reduced_gemm_weights / T
    mla_io = N * Prompt * (q_lora_rank + kv_lora_rank + 2 * qk_rope_head_dim)
    mla_io_read = mla_io / B if mla_io > L3_size else mla_io / 4 / B
    time_sdpa = sdpa / T
    return max(mla_gemm_read, mla_gemm_compute) + mla_io_read + time_sdpa

def project_mla_fold_gen(N, kv_size):
    mla_gemm_read = reduced_gemm_weights / B
    mla_gemm_compute = 2 * N * reduced_gemm_weights / T
    mla_io = N * Prompt * (q_lora_rank + kv_lora_rank + 2 * qk_rope_head_dim)
    mla_io_read = mla_io / B if mla_io > L3_size else mla_io / 4 / B
    kv_read = kv_size / B
    return max(mla_gemm_read, mla_gemm_compute) + kv_read

ffn_laynorm_prefill = 2 * N * Prompt * Hidden
FFN = 3 * Hidden * Interm

# MoE modeling
ep_rank = 0
n_group = 8

# What is the chance a token activate new set of expert
activation_rate = 0.6

FFN_normal = 3 * Hidden * Interm
FFN_expert = 3 * Hidden * moe_interm
FFN_shared = 3 * Hidden * moe_interm * n_shared_experts

# modeling gate fp32!!!
Gate = Hidden * n_routed_experts

def project_ffn(N, Prompt):
    ffn_gemm_read = FFN_normal / B
    ffn_gemm_compute = 2 * N * Prompt * FFN_normal / T
    return max(ffn_gemm_read / B, ffn_gemm_compute / T)

# Modeling worst case
def project_ffn_moe(N, Prompt):
    act_max = min(N * Prompt * num_experts_per_tok, n_routed_experts)
    tok_per_expert = N * Prompt * num_experts_per_tok / act_max
    ffn_gemm_read = FFN_expert * act_max
    ffn_gemm_compute = 2 * tok_per_expert * ffn_gemm_read
    ffn_gemm_shared = 2 * N * Prompt * FFN_shared
    return max(ffn_gemm_read / B, ffn_gemm_compute / T) + max(ffn_gemm_shared / T, FFN_shared/B)

def project_ffn_ratio(N, Prompt):
    act_max = min(N * Prompt * num_experts_per_tok, n_routed_experts)
    tok_per_expert = N * Prompt * num_experts_per_tok / act_max
    ffn_gemm_read = FFN_expert * act_max + FFN_shared
    ffn_gemm_compute = 2 * tok_per_expert * ffn_gemm_read
    return ffn_gemm_compute/ T / (ffn_gemm_read/ B)

weight_elem = mla_gemm_weights + 3 * Hidden * moe_interm * (n_routed_experts + n_shared_experts)
weight_sz = weight_elem * w_dtype
# last gemm
logitw_size = Hidden * Vocab

# prefill roughly equal to promp compute time in each block
time_prefill_1block = project_mla_prefill(N, Prompt) + project_ffn(N, Prompt)
time_prefill_1block_moe = project_mla_prefill(N, Prompt) + project_ffn_moe(N, Prompt)
time_prefill_model = time_prefill_1block * first_replace + time_prefill_1block_moe * (Block - first_replace)

# Generation in 1 transformer block, kv-history and model
# weights read or compute when batch size N is large
# The max() is an over simplification which could turn into
# other sophisticated functions

time_gen_1block = [each_kv / B + project_mla_gen(N, each_kv) + project_ffn(N, 1) for each_kv in size_kv]
time_gen_1block_moe = [each_kv / B + project_mla_gen(N, each_kv) + project_ffn_moe(N, 1) for each_kv in size_kv]

# Last gemm for the logits, non-trivial if large vocabulary
time_next = max(logitw_size*fw_dtype/B, 2*N*logitw_size/(T/(fw_dtype/w_dtype)))
time_gen_token = [ time_gen_1block[token] * first_replace + time_gen_1block_moe[token] * (Block - first_replace) + time_next for token in range(Gen) ]

# total time prefill, logit gemm and generation of each token
time = time_prefill_model + time_next + np.sum(time_gen_token)
model_sz = weight_sz * Block + logitw_size * fw_dtype * 2
total_sz = model_sz + t_size_kv

# Output with confidence info. Gen/Prefill ratio gives intuition how each portion affect final t/s
# Compute/Memory ratio show whether should downgrade memory efficiency
print("Memory size:", total_sz/1024/1024/1024, "GiB, KV-cache size", t_size_kv/1024/1024/1024, "GiB, Weights size", model_sz / 1024/1024/1024, "GiB")
print("Time = ", time)
print("T/s = ", N * Gen / time)
print("T/s(vllm) = ", N * (Gen + Prompt)/time)
print("Prefill time", time_prefill_model + time_next)
print("Next minmum: ", time_gen_token[0])
print("Next maximum: ", time_gen_token[-1])
show(time_gen_token)

print("Generation time", np.sum(time_gen_token))
print("Gen:Prefill ratio", np.sum(time_gen_token) / (time_prefill_model + time_next))
print("Compute/Memory ratio", project_ffn_ratio(N, 1))
