---
title: Benchmarking - results (2/2)

# Summary for listings and search engines
summary: In this post I will show the results of the benchmarks performed to compare the python packages NumPy, SciPy, QuTiP and TensorFlow when performing basic linear algebra operations.

# Link this post with a project
projects: []

# Date published
date: "2021-07-05"

# Date updated
lastmod: "2021-07-05"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: true

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'Image credit: [**MediaWiki**](https://www.mediawiki.org/wiki/Google_Summer_of_Code/2020)'
  focal_point: "Smart"
  placement: 3
  preview_only: false

authors:
- admin

tags:
- GSoC
- Python
- QuTiP

categories:
- GSoC
---

This is the continuation of the post [Benchmarking - motivation and
tools](http://localhost:1313/post/gsoc_2/) where we
argued why we chose `pytest-benchmark` for the benchmarks. Here we will present some of
the results obtained. In particular, we will compare how the python packages NumPy,
SciPy, QuTiP and TensorFlow perform for basic linear algebra operations. Hence, we will
have 6 different objects to represent data:
- NumPy: `ndarray`
- SciPy: `sparse.csr_matrix`
- QuTip 5.0 (still in development): `Dense` (should be compared to ndarray)
- QuTip 5.0 (still in development): `CSR` (should be compared to SciPy's `sparse.csr_matrix`)
- TensorFlow: `Tensor` (operates using a GPU and should be compared to
  `Dense` and `ndarray`).

The operations will be performed in two different matrix types, dense and sparse, with
variable input shape (NxN) with up to $N=2^{10}$ that would represent a simulation with 10
qubits. For the dense matrices we will use a random Hermitian matrix whereas for the
sparse case we will use a tridiagonal Hermitian matrix. The idea here is that we expect
TensorFlow to beat QuTiP's `Dense` and NumPy's `ndarray` only for sufficiently large N
as `Tensor` operations suffer from a memory transfer overhead. I include both dense and
sparse input matrices as both have applications in QuTiP (which is one of the reasons
why QuTiP 5.0 will include them) and I was curious to see how they compared with
`ndarray` and `Dense` in the sparse matrix case.

The hardware where these tests were run is:
- CPU: [Intel Core i7-6700](https://ark.intel.com/content/www/us/en/ark/products/88196/intel-core-i7-6700-processor-8m-cache-up-to-4-00-ghz.html)
- RAM: 16GB
- GPU: [GTX 970](https://www.amazon.com/STRIX-GTX970-DC2OC-4GD5-NVIDIA-GeForce-carte-graphique/dp/B00NFFAW50)
- Python 3.9.5
- Numpy 1.19.5
- TensorFlow 2.5.0
- SciPy 1.7.0
- QuTiP 5.0.0.dev (date: 06-07-2021)
- CUDA version: 11.3

## Element-wise addition.

The first operations that I will show is the addition operation.  This is an
element-wise operation that should be easily parallelized in a GPU. In the figure below
we plot the time it takes to perform 100 additions of matrices of shape $N\times N$. We
see that for small matrices, NumPy has the best performance. The apparent difference in
operation time for small N is mainly due to the instantiation of the class that contains
the array, which is slower for QuTiP than for NumPy. One of the reasons for this is that
`Qobj`, the object we operate with, is a wrapper around the class `Data` that can be
either `Dense` or `CSR` whereas the `ndarray` of NumPy is not wrapped by anything else.
However, it is quite surprising that the `CSR` case performs noticeably faster that then
SciPy's case.  This is due to safety checks that `sparse.csr_matrix` performs when
creating a new array. `CSR` does not include some of these checks as it has very tight
control of the input and output data. To understand the TensorFlow case, we need to
mention that the benchmarked operation includes the time to move data from the GPU to
the CPU. This is why we benchmarked 100 addition operations together instead of only one
addition operation as we will usually perform multiple operations before retrieving the
data from the GPU. We see that TensorFlow performs worse than the rest for small matrix
size but for approximately $N>2^{8}$, it performs faster as memory overhead is no longer
the limitation in the operation. 

{{< figure 
src="figures/add.svg" 
caption=`Figure 1: Benchmark for the addition of two random dense (left) and sparse (right)
matrices of size NxN. The matrices are represented using the python packages
in the legend (see text or details). The addition of the matrices is performed 100 times
to account for a fair comparison btween GPU and CPU as GPU operations for small matrices
are limited by memory transfer. Error bars represent one standard deviation.`
>}}


For the case where we operate with a sparse (tridiagonal) matrix, both SciPy's `CSR` and
specially QuTiP's `CSR` data representations show a faster operation time than the rest.
This is because the number of operations with an sparse representation of the data
scales with the number of non-zero elements, which for a tridiagonal matrix means that
the `CSR` representations require $\mathcal{O}(N)$ operations (compared to the
$\mathcal{O}(N^2)$ operations for the "dense" data representations employed in QuTiP's
`Dense`, TensorFlow and NumPy).

All this data suggests that there is some speed-up to benefit from when using the GPU
for linear algebra operations. However, it should be noted that element-wise operations
such addition represent the best case scenario. I will now show how the considered
Python packages perform when considering more complex operations such as matrix
multiplication, matrix exponentiation or eigenvalue solver. 

## Matrix multiplication

When we benchmark the matrix multiplication operation we obtain the results shown below.
This operation is performed 20 times which reduces the impact of the memory transfer
overhead in GPU and represents a more realistic use-case of the matrix multiplication.
Contrary to what happens in the previous case for matrix addition, TensorFlow does not perform faster for
$N>2^{8}$. This quite a surprising result as, although matrix multiplication is a more
involved operation than element-wise addition, I would naively expect to still be
relatively easily parallelizable. Furthermore, unlike element-wise addition, matrix
multiplication is a compute-bound operation[^1]. 

{{< figure src="figures/matmul.svg" 
caption=`Figure 2: Benchmark for the matrix multiplication of two random dense (left) and sparse (right)
matrices of size NxN. The matrices are represented using the python packages
in the legend (see text or details). The operation is performed 20 times
to account for a fair comparison btween GPU and CPU as GPU operations for small matrices
are limited by memory transfer. Error bars represent one standard deviation.`
>}}

There are several articles where they address this operation using  different hardware
[^2] [^3]. In those articles they find that indeed, GPU performs faster than CPU matrix
multiplication. There are currently two reasons I think this may not happen with my
hardware. One is that there may be bug in the code. You can find the
code employed for these results [here](https://github.com/qutip/qutip-tensorflow). The
other reason that I am considering is that the GPU I use is too old and does not include
the latest hardware improvements. In particular, it turns out that current RTX GPUs include [tensor](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/) 
cores which are supposed to bring a noticeable improvement in the matrix multiplication
performance.


## Matrix exponentiation and eigenvalue decomposition.

The last two operations that have been benchmarked are matrix exponentiation and
eigenvalue decomposition. These are run a single time (although they are repeated 5
times to obtain a statistical average) as they are already time consuming by themselves.
The results are shown below and they are similar to what I showed previously for matrix
multiplication. However, the results for matrix exponentiation are a little bit
surprising. It turns out that there is a range of matrix sizes for which TensorFlow is
_faster_. For any other value, greater or lower than this range, TensorFlow is slower. I
do not fully understand why this is the
case but I think is related with the CPU core utilization as the jump in performance for
the CPU implementations occurs when multiple cores are used instead of a single one.
 
{{< figure src="figures/expm.svg"
caption=`Figure 3: Benchmark for the matrix exponentiation of two random dense (left)
and sparse (right) matrices of size NxN. The matrices are represented using the python
packages in the legend (see text or details). The operation is performed only one time
to keep the benchmarks at reasonable durations. Error bars represent one standard deviation.`
>}}

{{< figure src="figures/eigenvalues.svg"
caption=`Figure 4: Benchmark for the calculations of the eigenvalues of dense (left) and
sparse (right) matrices of size NxN. The matrices are represented using the python
packages in the legend (see text or details). The operation is performed only one time
to keep the benchmarks at reasonable durations. Error bars represent one standard deviation.`
>}}

## Conclusions.
The results presented in this blog are somewhat discouraging although not disastrous.
They show that, with _my_ personal computer hardware it may not be straightforward to obtain a
significant speed-up in matrix operations. It is yet to be seen if using an RTX series
graphics card could provide a significant improvement due to its tensor cores.
Nevertheless, there are a few other features, such us auto differentiation and seamless
integration between TensorFlow and QuTiP that still motivate the development of
qutip-tensorflow. Furthermore, qutip-tensorflow would allow to use Google Colab's GPUs
which I found to have similar performance to my CPU operations using NumPy.
This is specially interesting for researches/students that only have
access to a laptop and still want to run computation heavy simulations in QuTiP.


[^1]: For both addition and matrix multiplication the memory requirements scale as
  $\mathcal{O}(N^2)$, with $N\times N$ the shape of the matrix. However, the number of basic
  (multiplication or addition of a complex number) operations scales a
  $\mathcal{O}(N^2)$ for addition and $\mathcal{O}(N^3)$ for matrix multiplication. This
  is why addition is considered memory-bound while matrix multiplication is considered
  compute-bound.

[^2]: The main page of [CuPy](https://cupy.dev/) shows a benchmark with the relative
  speedup of a GPU matrix multiplication vs a CPU matrix multiplication. The original
  blog post is
  [this](https://medium.com/rapids-ai/single-gpu-cupy-speedups-ea99cbbb0cbb) and
  unfortunately they do not provide the code for their benchmarks.
[^3]: Huang, Zhibin & Ma, Ning & Wang, Shaojun & Peng, Yu. (2019). GPU computing
  performance analysis on matrix multiplication. The Journal of Engineering. 2019.
  10.1049/joe.2018.9178. 
