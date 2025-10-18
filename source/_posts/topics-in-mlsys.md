---
title: '[CMU15-779] Advanced Topics in Machine Learning Systems'
date: 2025-10-16 13:39:55
updated: 2025-10-16 13:39:55
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
code_block_shrink:  false
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

Map and reduce 是一个常见的计算模式, 这里以 sum 为例, 下面讨论如何在单个分片上实现 sum. 也顺便结合其他笔记, 学习一下如何使用 profiler, 弄到服务器 sudo 权限后, 就可以使用 Nsight Compute 来分析性能了.

我们来看长度为 N 的数组求和, 最简单的实现是, 每个 thread

