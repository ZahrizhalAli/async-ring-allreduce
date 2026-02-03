# AsyncAllReduce

AllReduce is a core communication primitive in distributed machine learn-
ing, underpinning both data parallelism and tensor parallelism by synchro-
nizing model state across GPUs. Conventional implementations built on
NCCL’s ncclSend() and ncclRecv() incur significant overhead from kernel
launches and synchronization protocols, limiting throughput, particularly
for small messages, and exposing communication latency for larger ones.
We present AsyncAllReduce, a new ring-based AllReduce algorithm that
replaces NCCL point-to-point operations with fully asynchronous cud-
aMemcpyAsync() transfers and pipelines communication with reduction
through mini-batching and double-buffered CUDA streams. This design
reduces per-step overhead and overlaps communication and computation,
improving effective bandwidth as input size grows. We implement Asyn-
cAllReduce and a baseline NCCL-style ring algorithm, and evaluate both
on a Perlmutter node with 2 and 4 A100 GPUs. AsyncAllReduce achieves
up to 31.60% higher throughput for large input sizes and exhibits favorable
scaling with the number of mini-batches, revealing interactions between
communication and computation costs not captured by classical α–β–γ
analysis. Profiling further confirms effective overlap of communication and
reduction. Our results demonstrate that asynchronous p2p pipelines can
substantially outperform traditional NCCL-style implementations for large
reductions and open new avenues for improved latencies in large-scale use,
such as during data parallelism.

## How to run

First-time:

```shell
conda create --prefix $PSCRATCH/project -c nvidia nccl
```

Every-time:

```shell
# Allocate GPUs with interractive session & navigate to directory
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account m4999_g
cd ~/CS5470/async-ring-allreduce/

conda activate $PSCRATCH/project

# Compile 
# NOTE: for the real benchmarking, compile with -DNDEBUG and -O2
nvcc -o benchmark \
    src/benchmark.cu src/utils.cu \
    src/nccl_ringreduce.cu src/naive_ringreduce.cu \
    src/pipelined_ringreduce_async.cu src/pipelined_ringreduce_nccl.cu \
    -I$PSCRATCH/project/include \
    -L$PSCRATCH/project/lib \
    -lnccl -lpthread

# Run Benchmark
LD_LIBRARY_PATH=$PSCRATCH/project/lib NCCL_DEBUG=WARN ./benchmark 4 output.csv
```

