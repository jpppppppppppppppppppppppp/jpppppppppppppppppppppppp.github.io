---
title: '[CMU15-779] Advanced Topics in Machine Learning Systems'
date: 2025-10-16 13:39:55
updated: 2025-10-19 0:51:43
home_cover: https://p.sda1.dev/28/9b814b6ee15ab26ef15fddece5ed8d41/cover.jpeg
post_cover: https://p.sda1.dev/28/70882c53fb39a654dcd771e0cce750c2/post.JPG
copyright_info: true
tags:
    - MLSys
categories:
    - Notes
mathjax: true
tikzjax: true
excerpt: 入门了解 ML System 的课程笔记.
code_block_shrink:  true
---

Homepage: <a href="https://www.cs.cmu.edu/~zhihaoj2/15-779/">Advanced Topics in Machine Learning Systems (LLM Edition)</a>

### Introduction

ML System 的目的是高效地在高并行, 异构, 快速变革的硬件上部署 ML 应用. 本课程将包括算法优化, 图优化, 分布式并行训练, ML 编译, 内存管理, GPU 编程等内容.

接下来简单介绍了各个板块.

图优化: 根据计算图和算子之间的数学性质, 进行算子的融合. Lecture 上举了两个例子, 一个是 Conv + BatchNorm, 另一个是 Conv + Conv. 后面的课程将会介绍如何自动化地做这些潜在的优化.

印象比较深的是介绍内存优化的部分, 一种是通过重计算来减少中间变量的存储, 一种是通过增加通信来零冗余分布式计算. 需要很仔细的 trade-off.

---

### GPU Architecture & CUDA Programming

CUDA 包含 grid, block, thread 三个层次的抽象, grid 包含 blocks, 每个 block 包含多个 threads. 在计算的时候如果遇到条件语句, 会采用 discard 的方法, 也就是两种情况都计算, 根据 mask 丢弃不需要的结果. 这种 divergent execution 是需要尽量避免的.

下面是内存模型, 首先是 Host(CPU) 和 CUDA Device 之间不能直接互相访问, 需要通过 cudaMemcpy 来传输数据. 在 Device 端, 共有三种不同的内存类型: 分别是 global memory(所有 thread 都可以读写), per-block shared memory(同一个 block 内的 threads 可以读写, 帮助 block 内的 threads 进行通信协作), per-thread private memory(每个 thread 私有的内存).

随后举了一个例子: 1D-Conv.

最简单的版本是每个 thread 计算 output 的一个元素.

```cpp
int N = 1024*1024;
cudaMalloc(&devInput,   sizeof(float)*(N+2));
cudaMalloc(&devOutput,  sizeof(float)*N);

conv1D<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(N, devInput, devOutput);

#define THREADS_PER_BLOCK 128
__global__ void conv1D(int N, float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i=0; i<3; i++)
        result += input[idx + i];
    output[idx] = result / 3.0f;
}
```

可以稍加改进, 因为一个 block 内的 threads 最多只会访问连续的 THREADS_PER_BLOCK+2 个元素, 可以利用 shared memory 来减少 global memory 的访问.

```cpp
#define THREADS_PER_BLOCK 128
__global__ void conv1D(int N, float *input, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shared[THREADS_PER_BLOCK + 2];
    shared[threadIdx.x] = input[idx];
    if (threadIdx.x < 2)
        shared[threadIdx.x + THREADS_PER_BLOCK] = input[idx + THREADS_PER_BLOCK];
    __syncthreads();    // barrier (all threads in block)

    float result = 0.0f;
    for (int i=0; i<3; i++)
        result += shared[threadIdx.x + i];
    output[idx] = result / 3.0f;
}
```

__syncthreads() 是 block 内的 同步原语, 还有原子操作也是同步原语.

CUDA 汇编, 目的是在不同的 GPU 上能运行相同的程序. 编译结果应该包含: 指令, 需要的资源, 包括每个 block 需要的 threads 数量, shared memory 大小, private memory 大小等. 一个假设是, block 之间的运算顺序不影响最终的结果, 编译好的程序会将不同的 blocks 根据 scheduling 分配到不同的核心上运行.

GPU 核心 (Streaming Multiprocessor, SM) 里布局如下: 最多有 64 个 warps (每个 warp 包含 32 个 threads), 包含四个选择器, 一共 96KB 的 shared memory, 以及大小为 256KB 最多 64 项的 warp context 上下文管理, 其实就是 register file. 每个时钟刻, 会至多选择 4 个 warps 来执行, 再至多选择 2 条指令来并行. 比如上文提到的 conv1D kernel, 每个 block 包含 128 个 threads, 也就是 4 个 warps, 需要 130*4=520bytes 的 shared memory.

从 GPU 设备的变革来看, SM 基本保持一致, 时钟周期变快, warps 的布局不变, shared memory 变大, 只有 SM 的数量变得更多. 但是最主要的算力提升来源是 tensor core 的引入. Tensor core 是专门处理 4\*4\*4 矩阵乘法的部件: D=A\*B+C.

Case Study 1: Matrix Multiplication

第一种最 na&iuml;ve 的实现是每个 thread 计算 output 的一个元素, 需要访问 $2N$ 个元素, 一共有 $N^2$ 个 threads, 访存量是 $2N^3$.

```cpp
int N = 1024;
dim3 threadsPerBlock(32, 32, 1);
dim3 numBlocks(N/32, N/32, 1);
matmul<<<numBlocks, threadsPerBlock>>>(A, B, C);
__global__ void mm(float A[N][N], float B[N][N], float C[N][N]) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    result = 0;
    for (int k = 0; k < N; ++k) {
        result += A[x][k] * B[k][y];
    }
    C[x][y] = result;
}
```

第一个优化是减少每个 thread 之间的重复计算, 让每个 thread 计算一个 $k\times k$ 大小的区域, 每个 thread 需要访存 $2Nk$, 一共只需要 $N^2/k^2$ 个 threads, 访存量是 $2N^3/k$.

```cpp
__global__ void mm(float A[N][N], float B[N][N], float C[N][N]) {
    int ybase = blockIdx.y * blockDim.y + threadIdx.y;
    int xbase = blockIdx.x * blockDim.x + threadIdx.x;
    float c[V][V] = {0};
    float a[V], b[V];
    for (int k = 0; k < N; ++k) {
        a[:] = A[xbase*V : xbase*V + V, k];
        b[:] = B[k, ybase*V : ybase*V + V];
        for (int y = 0; y < V; ++y) {
            for (int x = 0; x < V; ++x) {
                c[x][y] += a[x] * b[y];
                }
            }
        }
    C[xbase * V : xbase*V + V, ybase*V : ybase*V + V] = c[:];
}
```

第二个优化是利用 shared memory, 每个 block 内的 threads 共享同一块 shared memory, 假设是 $L\times L$ 大小, 这样搬运数据到 shared memory 只需啊哟 $2N^3/L$, shared memory 的访问次数是 $2N^3/k$.

```cpp
__global__ void mm(float A[N][N], float B[N][N], float C[N][N]) {
    __shared__ float sA[S][L], sB[S][L];
    float c[V][V] = {0};
    float a[V], b[V];
    int yblock = blockIdx.y;
    int xblock = blockIdx.x;
    for (int ko = 0; ko < N; ko += S) {
        __syncthreads();
        // needs to be implemented by thread cooperative fetching
        sA[:, :] = A[k : k + S, yblock * L : yblock * L + L];
        sB[:, :] = B[k : k + S, xblock * L : xblock * L + L];
        __syncthreads();
        for (int ki = 0; ki < S; ++ ki) {
            a[:] = sA[ki, threadIdx.y * V : threadIdx.y * V + V];
            b[:] = sA[ki, threadIdx.x * V : threadIdx.x * V + V];
            for (int y = 0; y < V; ++y) {
                for (int x = 0; x < V; ++x) {
                    c[y][x] += a[y] * b[x];
                }
            }
        }
    }
    int ybase = blockIdx.y * blockDim.y + threadIdx.y;
    int xbase = blockIdx.x * blockDim.x + threadIdx.x;
    C[ybase * V : ybase*V + V, xbase*V : xbase*V + V] = c[:];
}
```

其中可以更进一步使用 cooperative fetching, 让 block 内的 threads 分工合作搬运数据到 shared memory.

```cpp
sA[:, :] = A[k : k + S, yblock * L : yblock * L + L];
// cooperative fetching
int nthreads = blockDim.y * blockDim.x;
int tid = threadIdx.y * blockDim.x + threadIdx.x;
for(int j = 0; j < L * S / nthreads; ++j) {
    int y = (j * nthreads + tid) / L;
    int x = (j * nthreads + tid) % L;
    s[y, x] = A[k + y, yblock * L + x];
}
```

Cast Study 2: Parallel Reduction in CUDA

Map and reduce 是一个常见的计算模式, 这里以 sum 为例, 下面讨论如何在单个分片上实现 sum. 也顺便结合其他笔记, 学习一下如何使用 profiler, 弄到服务器 sudo 权限后, 就可以使用 Nsight Compute 来分析性能了. 实验设备是 NVIDIA A5000 GPU.

我们来看长度为 N 的数组求和, 最简单的实现是, 每个 thread 处理一个元素, 假设 BLOCK_SIZE=256, 那么需要 8 次迭代得到 256 个元素的和.

```cpp
#define N 32*1024*1024
#define BLOCK_SIZE 256

__global__ void reduce_v0(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

用 ncu 分析后, 可以发现它的带宽利用率如下: 计算带宽利用率为 68.37\%. 一个问题是 warp divergence 很严重, 一个 warp 内有 32 个 threads, 每个 threads 在 if 的条件判断上结果不一样, 导致一个 warp 内部效率不高, 而且用于条件的取模运算也比较耗时, 所以第一个优化方法是交错寻址, 目的是让相邻的 threads 的条件判断尽量相同, 且取代取模运算.

```cpp
__global__ void reduce_v1(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

profile 结果显示, 带宽利用率提到了 85.49\%. 下面一个优化和 shared memory 结构有关. shared memory 分为 32 个 banks, 每个 banks 可以并行访问, 同一个 warp 内的 threads 应该避免访问同一个 bank 的不同地址, 这会产生 bank conflict.

<center>
<img src="https://feldmann.nyc/posts/smem-microbenchmarks/no-conflicts.svg" />
</center>

例如上面这个例子中的第 0 个 warp 的第一次迭代时, thread_0 需要访问 sdata[0] 和 sdata[1] 并写入 sdata[0]. 而 thread_16 需要访问 sdata[32] 和 sdata[33] 并写入 sdata[32], 这产生了 conflict. 为了避免这个问题, 需要 sequential 的访存模式, 也就是说让不同的 threads 同时访问不同的 banks. 代码如下:

```cpp
#define N 32*1024*1024
#define BLOCK_SIZE 256

__global__ void reduce_v2(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

带宽利用率为 89.03\%. 下一个优化的观察是, 在 reduce 函数中, 一大半的 threads 都是空闲的, 比如第一次迭代时, 只有一半的 threads 在工作, 我们可以让每一个 threads 多干一些事情, 比如每个 block 管理 2*BLOCK_SIZE 个元素, 每个 thread 在第一次加载 shared memory 的时候就计算第一次迭代的结果, 这样子减半了 block 的数量.

```cpp
#define N 32*1024*1024
#define BLOCK_SIZE 256

__global__ void reduce_v3(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

这样子可以让时间几乎减半, 让访存带宽几乎翻倍, 而计算带宽几乎没变. 可以发现访存带宽为 300GB/s, 距离极限 700GB/s 还有一定距离, 下一个优化是使用循环展开, 突破指令瓶颈, 指令瓶颈不是指加载算术指令, 而是加载地址计算和循环的开销. 在最后的几次迭代中, 每一个 block 只有一个 warp 真正在干活, 而其他的 warps 仍然需要同步, 造成了浪费, 通过循环展开, 把 __syncthreads() 删掉, 只留下真正需要的数据操作:

```cpp
#define N 32*1024*1024
#define BLOCK_SIZE 256

__device__ void warpReduce(volatile float* cache, unsigned int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

__global__ void reduce_v4(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s >>= 1) {
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

注意这里函参使用了 **volatile** 关键字修饰 shared memory, 是为了避免编译器对它的寄存器优化. 比如一个地址被不同的线程使用, 新的值可能被缓存在寄存器中, 另一个线程读到的值就是错误的, 从而导致了错误的结果. 通过减少同步的开销, 访存带宽提升到了 530GB/s.

下面是利用 **__shfl_xor_sync** 原语来对一个 warp 内的 value 进行规约求和.

```cpp
#define N 32*1024*1024
#define BLOCK_SIZE 256
#define WARP_SIZE 32

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__global__ void reduce_v5(float* g_idata, float* g_odata, int n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * (blockDim.x * 2) + tid;
  constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // keep the data in register is enough for warp operaion.
  float sum = g_idata[idx] + g_idata[idx + blockDim.x];
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  // warp leaders store the data to shared memory.
  if (lane == 0) reduce_smem[warp] = sum;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) g_odata[blockIdx.x] = sum;
}
```

进行评估后, 访存带宽达到了 635GB/s, 在时间上相比最初的实现, 从 1.24ms 降到了 212us, 减少了 83\%.

---
