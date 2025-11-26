import matplotlib.pyplot as plt
import numpy as np

# --- DATA ENTRY ---
# Replace these lists with the actual output from your terminal runs
# Example data provided below based on typical A100 behaviors

# Sizes in Bytes (X-Axis)
sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 
         262144, 524288, 1048576, 2097152, 4194304, 8388608, 
         16777216, 33554432, 67108864, 134217728, 268435456]

# Bandwidth in GB/s (Y-Axis)
# 1. Your NCCL Naive Results (from previous step)
bw_nccl = [0.02, 0.04, 0.08, 0.16, 0.31, 0.58, 1.07, 2.12, 
           4.11, 7.18, 2.28, 4.47, 8.20, 10.91, 20.92, 
           32.52, 44.85, 47.13, 48.18]

# 2. Your CUDA IPC Results (Hypothetical - replace with your actual run)
# IPC usually has better startup latency but similar peak BW for naive kernels
bw_ipc =  [0.05, 0.10, 0.20, 0.40, 0.80, 1.50, 2.80, 5.00, 
           9.00, 15.00, 20.00, 30.00, 40.00, 45.00, 48.00, 
           49.00, 50.00, 50.50, 51.00] 

# --- PLOTTING ---
plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(sizes, bw_nccl, marker='o', label='NCCL P2P (Naive)', color='blue')
plt.plot(sizes, bw_ipc, marker='x', label='CUDA IPC (Async)', color='red', linestyle='--')

# Formatting
plt.xscale('log', base=2)
plt.xlabel('Message Size (Bytes)')
plt.ylabel('Bus Bandwidth (GB/s)')
plt.title('Ring AllReduce: NCCL P2P vs. CUDA IPC (A100)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# Annotation for the "Dip" if present
plt.annotate('Cache/TLB Thrashing?', xy=(1048576, 2.28), xytext=(200000, 10),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Save
plt.savefig('allreduce_comparison.png')
print("Plot saved as allreduce_comparison.png")