# AsyncAllReduce

## Abstract 

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

## Implementation Details

In the classic AllReduce algorithm, a single CUDA stream and a single receive buffer are used to schedule the computation and communication serially. However, a single CUDA stream cannot run multiple operations asynchronously, thus being insufficient for our use case.

To overcome this limitation, our algorithm leverages double buffering of the CUDA streams and data buffers to overlap communication and computation. More specifically, we alternate the stream and the data buffer every step, reducing the data in one stream and buffer while communicating in the other. To prevent reading a buffer before the chunk is ready, each GPU records an event on the stream after it finishes the data transfer, and the receiving GPU waits for the event to be posted before processing it. To also avoid deadlocks arising from a GPU posting events on another GPU that is yet to be initialized, a shared mutex is used and the first thread to acquire the mutex is tasked with initializing the event and buffer addresses of all the ranks participating in the AllReduce. This ensures all the GPUs are ready before communication starts, substituting the functionality of the ncclGroup.

## Evaluation Methodology

To evaluate our algorithm, we implemented the classic Ring AllReduce algorithm using
ncclSend(), ncclRecv() and an element-wise addition kernel to use as our baseline. We use a single CUDA stream for communication and computation and leverage ncclGroups to avoid deadlocks.
We evaluated each implementation for input sizes spanning 1KB to 8GB, doubling the size
every iteration. For each iteration, we run the implementation once and check its output for correctness, then we run the implementation 200 times to warm up the GPU. Finally, we run the implementation another 200 times and record its average latency. We then divide the input size by the average latency to get the average throughput per GPU, which we plot against the input size to obtain our S curve.
All benchmarks were run on Perlmutter with 4x NVIDIA (40GB) A100s connected via 4th
generation NVLinks with a bandwidth of 25GB/s/direction (fully-connected topology). The
code was compiled with -DNDEBUG -O2 flags using NVCC.

