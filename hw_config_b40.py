B_eff = 0.9
# high and low bound of BW
B_H = 1792*1000*1000*1000 * B_eff
B_L = 1792*1000*1000*1000 * B_eff

#memory capability (G)
M_size_L = 64*1000*1000*1000
M_size_H = 96*1000*1000*1000

T_eff = 1
# high and low bound of Tflops
T_H = 839*1000*1000*1000*1000 * T_eff
T_L = 839*1000*1000*1000*1000 * T_eff

#
w_dtype_16bits = 2
kv_dtype_16bits = 2
w_dtype_8bits = 1
kv_dtype_4bits = 0.5
w_dtype_4bits = 0.5

# compute datatype
fw_dtype = 2
