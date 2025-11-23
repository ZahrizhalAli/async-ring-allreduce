// naive_ringreduce.cu
// Implements naive ring all-reduce using RS + AG with ncclSend/ncclRecv.

#include <assert.h>
#include <stdio.h>

#include "interface.h"



// element-wise add kernel: dest[i + offset] += src[i]
__global__ void add_kernel(float* dest, const float* src, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dest[offset + idx] += src[idx];
}

// ring all-reduce using RS + AG
void ring_allreduce(
    float* inout_buf, int input_size, ncclComm_t comm, int rank, int n_ranks, cudaStream_t stream
) {
    // compute chunk size and allocate temporary receive buffer
    assert(input_size % n_ranks == 0);
    int chunk_size = input_size / n_ranks;
    float* temp_buf = nullptr;
    CUDA_CALL(cudaMalloc(&temp_buf, chunk_size * sizeof(float)));

    // --- REDUCE-SCATTER ---
    for (int step = 0; step < n_ranks - 1; step++) {
        int send_chunk = (n_ranks - 1 + rank - step) % n_ranks;
        int recv_chunk = (n_ranks - 2 + rank - step) % n_ranks;

        int next_rank = (rank + 1) % n_ranks;
        int prev_rank = (rank - 1 + n_ranks) % n_ranks;

        int send_off = send_chunk * chunk_size;
        int recv_off = recv_chunk * chunk_size;

        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclSend(inout_buf + send_off, chunk_size, ncclFloat, next_rank, comm, stream));
        NCCL_CALL(ncclRecv(temp_buf, chunk_size, ncclFloat, prev_rank, comm, stream));
        NCCL_CALL(ncclGroupEnd());

        CUDA_CALL(cudaStreamSynchronize(stream));

        // reduce
        const int threads = 256;
        int blocks = (chunk_size + threads - 1) / threads;
        add_kernel<<<blocks, threads, 0, stream>>>(inout_buf, temp_buf, recv_off, chunk_size);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaStreamSynchronize(stream));
    }

    // --- ALL-GATHER ---
    for (int step = 0; step < n_ranks - 1; step++) {
        int send_chunk = (n_ranks + rank - step) % n_ranks;
        int recv_chunk = (n_ranks + rank - step - 1) % n_ranks;

        int next_rank = (rank + 1) % n_ranks;
        int prev_rank = (rank - 1 + n_ranks) % n_ranks;

        int send_off = send_chunk * chunk_size;
        int recv_off = recv_chunk * chunk_size;

        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclSend(inout_buf + send_off, chunk_size, ncclFloat, next_rank, comm, stream));
        NCCL_CALL(ncclRecv(inout_buf + recv_off, chunk_size, ncclFloat, prev_rank, comm, stream));
        NCCL_CALL(ncclGroupEnd());

        CUDA_CALL(cudaStreamSynchronize(stream));
    }

    CUDA_CALL(cudaFree(temp_buf));
}



// interface function, runs for each rank
void* ring_naive(RunArgs* args) {
    int rank = args->rank;
    int dev = args->device;
    int n_ranks = args->n_ranks;
    int input_size = args->input_size;
    ncclComm_t comm = args->comm;


    // initialize CUDA stream
    CUDA_CALL(cudaSetDevice(dev));
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));


    // initialize input for this rank at d_buf
    float* d_buf = nullptr;
    CUDA_CALL(cudaMalloc(&d_buf, input_size * sizeof(float)));

    const int threads = 256;
    int blocks = (input_size + threads - 1) / threads;
    init_input_kernel<<<blocks, threads, 0, stream>>>(d_buf, rank, input_size);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaStreamSynchronize(stream));


    // call ring all-reduce (in-place)
    ring_allreduce(d_buf, input_size, comm, rank, n_ranks, stream);


    // copy back result to host and verify output, short circuit if incorrect
    float* h_res = (float*)malloc(input_size * sizeof(float));
    CUDA_CALL(cudaMemcpy(h_res, d_buf, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    *(args->correct) = check_correctness(h_res, rank, n_ranks, input_size, args->atol);
    free(h_res);

    if (!*(args->correct)) {
        CUDA_CALL(cudaFree(d_buf));
        CUDA_CALL(cudaStreamDestroy(stream));
        return nullptr;
    }


    // warmup
    for (int i = 0; i < args->n_warmup; i++)
        ring_allreduce(d_buf, input_size, comm, rank, n_ranks, stream);
    CUDA_CALL(cudaStreamSynchronize(stream));


    // benchmark
    double t0 = get_time();
    for (int i = 0; i < args->n_iters; i++)
        ring_allreduce(d_buf, input_size, comm, rank, n_ranks, stream);
    CUDA_CALL(cudaStreamSynchronize(stream));
    double t1 = get_time();
    if (rank == 0) *(args->avg_latency) = (t1 - t0) * 1e6 / (double)args->n_iters;  // µs per iter

    CUDA_CALL(cudaFree(d_buf));
    CUDA_CALL(cudaStreamDestroy(stream));
    return nullptr;
}
