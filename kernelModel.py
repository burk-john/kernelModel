import math

# Titan V hardware characteristics
FP32_MAX = 13.8 * 10e12 # 13.8TFLOPs max
FLOPS_EFF = 0

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

L1_LATENCY = 28 # cycles, from microbenchmarking paper
L2_LATENCY = 193 # cycles
L1_LATENCY_TIME = L1_LATENCY/MEM_CLK
L2_LATENCY_TIME = L2_LATENCY/MEM_CLK

# assumes compute bound ONLY
def calculate_compute_time():
    total_flops = 2*Ni*Nn*B # Add, multiply for problem size
    return total_flops/FP32_MAX

#assume perfect reuse
def DRAM_move_time():
    #sqrt term accounts for when blocks are not full on threads, limiting their throughput
    total_bytes_move = data_size*( B*(Ni + Nn) + B*Ni*Nn)
    thread_overhead = B * .0044 / 256
    if(Nn < Blocks):
        return thread_overhead
    else:
        #there is a scaling factor for the DRAM that was found experimentally
        return thread_overhead + total_bytes_move/VRAM_BW * (15.3 - math.log(Nn / Blocks, 2) * 1.277)

#assume perfect reuse
def L2_move_time():
    total_bytes_move = data_size*( B*(Ni + Nn) + B*Ni*Nn)
    num_hits = total_bytes_move/TOTAL_L2
    latency_penalty = num_hits * L2_LATENCY_TIME
    return total_bytes_move/L2_BW + latency_penalty

#assume perfect reuse
def L1_move_time():
    if(Blocks < SM_COUNT):
        total_bytes_move = data_size*( B*(Ni + Nn) / Blocks + B*Ni*Nn )
        num_hits = total_bytes_move/TOTAL_L1
        latency_penalty = num_hits * L1_LATENCY_TIME
    else:
        total_bytes_move = data_size*( B*(Ni + Nn) / SM_COUNT * math.ceil(Blocks / SM_COUNT) + B*Ni*Nn)
        num_hits = total_bytes_move/TOTAL_L1
        latency_penalty = num_hits * L1_LATENCY_TIME
    return total_bytes_move/L1_BW

def effective_compute():
    op_intensity = 2*Ni*Nn*B/(data_size*( B*(Ni + Nn) + B*Ni*Nn))
    dram_compute = op_intensity*VRAM_BW
    l2_compute = op_intensity*L2_BW
    l1_compute = op_intensity*L1_BW
    FLOPS_EFF = min(FP32_MAX, dram_compute, l2_compute, l1_compute)
    return FLOPS_EFF

def set_size(N_i, N_n, batch, block_num):
    global Ni, Nn, B, Blocks
    Ni = N_i
    Nn = N_n
    B = batch
    Blocks = block_num

def main():
    #Parameters to be tweaked
    Ni = 512   #Input size
    Nn = 131072     #Output size
    B = 256
    Blocks = 2 ** 7 #2 ^ 7 = 128, optimal when testing large problem sizes
    op_intensity = 2*Ni*Nn*B/(data_size*( B*(Ni + Nn) + B*Ni*Nn))

    set_size(Ni, Nn, B, Blocks)
    effective_compute()
    times = [calculate_compute_time(), DRAM_move_time(), L2_move_time(), L1_move_time()]
    time_label = ["compute time", "DRAM bandwidth", "L2 bandwidth", "L1 bandwidth"]

    print("For the parameters: \nInput Size = %d \nOutput Size = %d \nBatch Size = %d \nBlock Number = %d" % (Ni, Nn, B, Blocks))
    print("Operational intensity (FLOPS/byte): " + str(op_intensity))
    print("Possible compute-bound throughput: ", str(FP32_MAX/(10e12)), "TFLOPS")
    print("Achieved throughput (memory or compute-bound): " + str(effective_compute()/10e12) + " TFLOPS")
    print("The execution time of the kernel and the limiting factor is:\n %f ms\t %s" % (max(times)*1000, time_label[times.index(max(times))]))

    print("\n")
    print("Estimated compute time (compute only): " + str(calculate_compute_time()*1000) + " ms")
    print("Estimated DRAM data transfer time: " + str(DRAM_move_time()*1000) + " ms")
    print("Estimated L2 cache data transfer time: " + str(L2_move_time()*1000) + " ms")
    print("Estimated L1 cache data transfer time: " + str(L1_move_time()*1000) + " ms")

if __name__ == "__main__":
    main()
