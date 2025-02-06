#llama3.3 70B-Instruct
#https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/config.json

import numpy as np


# hardware
from hw_config_n3p import *

G = 1000*1000*1000
GB = 1024*1024*1024
Tflops = 1000*1000*1000*1000

# model
Hidden = 8192 
AttHeads = 64 
Headdim = Hidden / AttHeads
KVHidden = Headdim * 8
Interm = 28672
Vocab = 128256
Block = 80
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

#calculate Weight (16bit?)
weight_layernorm = 2* Hidden
weight_embedding = Vocab * Hidden 
weight_mlp = 3* Hidden * Interm 
weight_kv = 2* Hidden * KVHidden 
weight_q = Hidden * Hidden 
weight_o = Hidden * Hidden 
weight_out = Vocab* Hidden

weight_total = weight_embedding  + Block * (weight_layernorm + weight_q + weight_kv + weight_o + weight_mlp) + weight_out

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
print("llama3 70B", "batchsize :", N, "ISL :", Prompt, "OSL :", Gen)
print("HW Info", "Compute Capability Tflops",round(T / T_eff / Tflops, 2),  "Bandwidth GB/sec", round(B/B_eff/G, 2), "Memory Capacity GB", round(M_size/G, 2))
print("HW Info", "Compute Capability Eff",T_eff,  "Bandwidth Eff", B_eff)
print("KV-cache size", round(t_size_kv/1024/1024/1024,2), "GB")
print("Weight size with 16bit", round(2* weight_total/1024/1024/1024,2), "GB")
print("Total GMEM requirments", round((t_size_kv + 2* weight_total)/1024/1024/1024,2), "GB")
print("Time = ", round(time, 2))
print("T/s = ", round(N * (Gen + Prompt)/time,2))
print("Prefill time", round(time_prefill_model + time_next, 2))
print("Next time", round(time_gen_model + time_next, 2))
print("Generation time", round((time_gen_model + time_next) * Gen, 2))
print("Gen:Prefill ratio", round((time_gen_model + time_next) * Gen / (time_prefill_model + time_next), 2))
print("Compute/Memory ratio", round((gen_ops/T)/(weight_sz/B),2))
