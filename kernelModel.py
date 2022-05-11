import math

# Titan V hardware characteristics
FP32_MAX = 13.8 * 10e12 # 13.8TFLOPs max

TOTAL_VRAM = 12884901888 # 12GB to bytes
TOTAL_L2 = 4718592 # 4.5MB
TOTAL_L1 = 98304 # 96KB

VRAM_BW = 701153411072 # 653GB/s
L2_BW = 1600 * (2**30) # 1600GB/s
L1_BW = 3

SM_COUNT = 80

MEM_CLK = 850000000 # estimated memory clock speed
GPU_CLK = 1200000000
GPU_CLK_BOOST = 1455000000 # boost clock speed

def calculate_compute_time():
    return 0

def calculate_memory_time():
    return 0

print('wtf')
