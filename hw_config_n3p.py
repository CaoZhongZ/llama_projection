B_eff = 0.9
# high and low bound of BW
B_H = 1380*1000*1000*1000 * B_eff
B_L = 920*1000*1000*1000 * B_eff
B = B_L

#memory capability (G)
M_size_L = 96*1000*1000*1000
M_size_H = 384*1000*1000*1000
M_size = M_size_L

T_eff = 1
# high and low bound of Tflops
T_L = 800*1000*1000*1000*1000 * T_eff
T_H = 1066*1000*1000*1000*1000 * T_eff
T = T_L

#
kv_dtype = 2
w_dtype  = 2
w_dtype_16bits = 2
kv_dtype_16bits = 2
w_dtype_8bits = 1
kv_dtype_4bits = 0.5
w_dtype_4bits = 0.5

# compute datatype
fw_dtype = 2
