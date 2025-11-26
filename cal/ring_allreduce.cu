#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>
#include <unistd.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// sum kernel
__global__ void add_vectors(float* dest, const float* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dest[idx] += src[idx];
    }
}


void ringAllReduce(float *sendbuff, float* recvbuff, size_t count, int rank, int nranks, ncclComm_t comm, cudaStream_t stream){
    
    // Copy input to output buffer
    CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // chunk size
    size_t chunk_count = count / nranks;
    size_t chunk_size_bytes = chunk_count * sizeof(float); //convert to bytes

    // allocate scratch buffer for incoming bandwth
    float* scratch_buff;
    CUDACHECK(cudaMalloc(&scratch_buff, chunk_size_bytes));

    // ring neighbors
    int send_to = (rank + 1) % nranks;
    // int send_to = (rank + 2) % nranks;
    int recv_from = (rank - 1 + nranks) % nranks;

    // scater reduce
    for (int i = 0 ; i < nranks-1; ++i){
      int send_chunk_idx = (rank - i + nranks) % nranks;
      int recv_chunk_idx = (rank - i - 1 + nranks) % nranks;

      // pointer
      float* send_ptr = recvbuff + (send_chunk_idx * chunk_count);
      float* dest_ptr = recvbuff + (recv_chunk_idx * chunk_count);

      // group the communication
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclSend(send_ptr, chunk_count, ncclFloat, send_to, comm, stream));
      NCCLCHECK(ncclRecv(scratch_buff, chunk_count, ncclFloat, recv_from, comm, stream));
      NCCLCHECK(ncclGroupEnd());

      // reduction kernel
      int threads = 512;
      int blocks = (chunk_count + threads - 1) / threads;
      add_vectors<<<blocks, threads, 0, stream>>>(dest_ptr, scratch_buff, chunk_count);
    }

    // all gather - no computation
    for (int i = 0 ; i < nranks - 1; ++i){
      int send_chunk_idx = (rank - i + 1 + nranks) % nranks;
      int recv_chunk_idx = (rank - i + nranks) % nranks;

      float* send_ptr = recvbuff + (send_chunk_idx * chunk_count);
      float* recv_ptr = recvbuff + (recv_chunk_idx * chunk_count);

      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclSend(send_ptr, chunk_count, ncclFloat, send_to, comm, stream));
       NCCLCHECK(ncclRecv(recv_ptr, chunk_count, ncclFloat, recv_from, comm, stream));
      NCCLCHECK(ncclGroupEnd());
    }

    // free
    CUDACHECK(cudaFree(scratch_buff));


}

int main(int args, char* argv[]){
  int mRank, nRanks;

  if (getenv("SLURM_PROCID")) {
        mRank = atoi(getenv("SLURM_PROCID"));
        nRanks = atoi(getenv("SLURM_NTASKS"));
    } else {
        printf("Please run with srun\n");
        return 1;
  }

  CUDACHECK(cudaSetDevice(mRank));

  ncclUniqueId id;
  if (mRank == 0) ncclGetUniqueId(&id);

  if (mRank == 0) {
        FILE* f = fopen("nccl_unique_id.bin", "wb");
        fwrite(&id, sizeof(id), 1, f);
        fclose(f);
  }

  sleep(3); 
  if (mRank != 0) {
        FILE* f = fopen("nccl_unique_id.bin", "rb");
        while(f==NULL) { f = fopen("nccl_unique_id.bin", "rb"); sleep(1); }
        fread(&id, sizeof(id), 1, f);
        fclose(f);

        
  }

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, mRank));

  // benchmark
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  size_t min_size = 1024;       // 1 KB
  size_t max_size = 256ULL * 1024 * 1024;

  float *d_send, *d_recv;

  if (mRank == 0) printf("%-15s %-15s\n", "Size(B)", "BusBW(GB/s)");

  for (size_t size = min_size; size <= max_size; size *= 2){
    size_t count = size / sizeof(float);

    if(count % nRanks != 0) count = (count + nRanks) - (count % nRanks);
    size_t adjusted_size = count * sizeof(float);

    CUDACHECK(cudaMalloc(&d_send, adjusted_size));
    CUDACHECK(cudaMalloc(&d_recv, adjusted_size));

    // run algo
    ringAllReduce(d_send, d_recv, count, mRank, nRanks, comm, stream);
    CUDACHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start, stream);
    int iter = 10;
    for (int i = 0; i < iter; i++){
      ringAllReduce(d_send, d_recv, count, mRank, nRanks, comm, stream);
    }

    cudaEventRecord(stop, stream);
    CUDACHECK(cudaStreamSynchronize(stream));

    float msec = 0;
    cudaEventElapsedTime(&msec, start, stop);

    if (mRank == 0){
      double avg_time_sec = (msec / 1000.0) / iter;
      double alg_bw = adjusted_size / avg_time_sec;
      double bus_bw = (2.0 * (nRanks - 1) / nRanks * adjusted_size) / avg_time_sec;
      printf("%-15lu %-15.2f\n", adjusted_size, bus_bw / 1e9);
    }

    CUDACHECK(cudaFree(d_send));
    CUDACHECK(cudaFree(d_recv));
  }

  ncclCommDestroy(comm);
  if (mRank == 0) printf("Done.\n");
  return 0;
}