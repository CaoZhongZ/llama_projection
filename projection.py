import numpy as np

B = 864*1000*1000*1000
T = 733*1000*1000*1000*1000
N = 32

Hidden = 4096
AttHeads = 32
Headdim = Hidden / AttHeads
KVHidden = Headdim * 8
Interm = 14336
Vocab = 128256
Block = 32

C = Hidden * (3 * Interm + Hidden + KVHidden * 2)
FinalW_size = Hidden * Vocab

Prompt = 1024
Gen = 1024

kv_dtype = 1
w_dtype = 1
fw_dtype = 2

time_prefill_1block = 2 * Prompt * N * C / T
time_prefill_model = time_prefill_1block * Block

size_kv = 2 * N * (Prompt + Gen/2) * KVHidden * kv_dtype
gen_wread = C * w_dtype
gen_ops = 2 * N * C

time_gen_1block = size_kv / B + max(gen_wread/B, gen_ops/T)
time_gen_model = time_gen_1block * Block

time_next = max(FinalW_size*fw_dtype/B, 2*N*FinalW_size/T/2)

time = time_prefill_model + time_next + (time_gen_model + time_next) * Gen

print("Time = ", time)
print("T/s = ", N * (Gen + Prompt)/time)

