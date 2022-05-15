import math

# Titan V hardware characteristics
FP32_MAX = 13.8 * 10e12 # 13.8TFLOPs max

TOTAL_VRAM = 12884901888 # 12GB to bytes
TOTAL_L2 = 4718592 # 4.5MB
TOTAL_L1 = 98304 # 96KB

VRAM_BW = 701153411072 # 653GB/s
L2_BW = 2155 * (2**30) # 2155GB/s. Slightly inaccurate, using V100 value
L1_BW = 13800 * (2**30)

SM_COUNT = 80

MEM_CLK = 850000000 # estimated memory clock speed
GPU_CLK = 1200000000
GPU_CLK_BOOST = 1455000000 # boost clock speed

data_size = 4 # FP32, 32 bits, 8bits/byte

def calculate_compute_time():
    total_flops = 2*Ni*Nn*B # Add, multiply for problem size
    return total_flops/FP32_MAX

def calculate_memory_time():
    return 0

def set_size(N_i, N_n, batch):
    global Ni, Nn, B
    Ni = N_i
    Nn = N_n
    B = batch

def main():
    #Classifier Stuff
    if(class):
        #Parameter to be tweaked
        Ni = 2 ** 10    #Input size
        Nn = 2 ** 10    #Output size

        num_ops = Ni * Nn


if __name__ == "__main__":
    main()
