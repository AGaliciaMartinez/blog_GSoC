---
title: A potential benefit to implement tensor networks to QuTiP.

# Summary for listings and search engines
summary: In this blog post I will show with an example the potential speed-ups that QuTiP can benefit from by exploiting the tensor structure of some linear algebra operations.

# Link this post with a project
projects: []

# Date published
date: "2021-08-19"

# Date updated
lastmod: "2021-08-19"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: true

authors:
- admin

tags:
- Tensor Network
- Python
- QuTiP

categories:
- GSoC
---

This is a collection of notes that I will use to create a draft for the Tensor
Network proposal in QuTiP. Here I showcase the potential benefits of adapting a
Tensor Network approach in QuTiP.

## Introduction
Tensors are algebraic objects that describe multi-linear relationships between
vector spaces. Which means that they represent linear functions that map
a set of vector spaces to another set of vector spaces. In quantum mechanics
these tensors can be used to represent the objects of our system: kets,
operators, density matrices, etc.

Interestingly, tensors have a neat graphical representation.
An example of such representation can be seen in the following figure:
{{< figure 
src="figures/tensor_network_example.svg" 
caption=`Figure 1: Graphical representation of a tensor netowork. Nodes
represent a tensor of order $k$ where $k$ is the number of legs. Legs represent the
indices of a tensor. We assume in that each index,$i$, follows $i \in \set{0,1}
$. a) 4-qubit state. b) 4 qubit operator.`
>}}

QuTiP can currently represent a `Qobj` with either a sparse matrix or a dense
matrix. What I will show in this example is that having only these two
representation methods will prove inefficient for certain operations.
Complementing these two representation methods with tensor networks can lead to
significant speed-ups in QuTiP. As example, lets consider the application of a
Hadamard gate to $n$ qubits,
$$
\left( \bigotimes^n H \right)| \psi \rangle ,
$$
where $|\psi\rangle$ is a random $n$-qubit sate and $H$ is the Hadamard matrix.
I will now show three different strategies that one can follow to represent
$\hat{O} = \bigotimes^n H$ and perform the desired operation.

## Method 1: dense representation of the operator $\hat{O}$.
We can obtain a dense representation of the operator by using the Kronecker
product. In QuTiP, this is achieve with the following code:
```python
n = 13
H = qutip.gates.hadamard_transform()  # One qubit hadamard
O = qutip.tensor([H]*n)
```
In this case, the memory requirements scale as $O(N^2)$, where $N=2^n$, as we
need $N^2$ complex numbers to represent $\hat{O}$. If we want to apply this
operation to a random sate, we do the full matrix multiplication:
```python
state = qutip.rand_ket(2**13, dims = [2]*n)
%%timeit # 125 ms
O*state
```
Notice that since we are doing a full matrix multiplication, the number of
floating point operations required are $O(N^2)$. Both the memory and the
operations scale quadratically with $N$. Note however that $N=2^n$ and hence
the scaling is quadratically exponential with the number of qubits. We will not
be able to get ride of the exponential scaling for the operations due to the
dense representation of the _state_, but we can improve the operation scaling
with the next method.

## Method 2: sparse representation of the operator $\hat{O}$.

A more efficient strategy one can follow in QuTiP 5 is the following. Instead
of representing the full matrix, we can represent the individual Hadamard
operations, $\hat{O}_j = \left(\bigotimes^j I \right)\otimes H
\otimes\left(\bigotimes^{n-j-1} I \right)$ using sparse matrices, where $I$ is
the identity operation. Notice that the full operator $\hat{O} = \prod_j
\hat{O}_j$ is not sparse, it has no single zero in it and hence we are forced
to represent it with a dense matrix. However, the operators $\hat{O}_j$ are extremely
sparse, with an sparsity equivalent to a tridiagonal matrix[^1].

The operation with the $\hat{O}_j$ operators would go as follows:

```python
H = qutip.hadamard_transform()
I = qutip.qeye(2)

# Representing O as a list of the operators Oj
O = [qutip.tensor([I]*j + [H] + [I]*(n-j)).to('csr') for j in range(n)]

# Operation
%%timeit # 1.93 ms
for j in range(n):
    state = O[j]*state
```

The reason for this method to be faster is that, by using sparse algebra
efficiently, we have improved the scaling of the operaitons.  We can estimate
the total number of operations by realizing that the operators $\hat{O}_j$ have
two non zero elements in each row. Hence, the total number of non-zero elements
is $O(N)$. The total number of operations is then $O(\log(N)N)$ as we perform
$\log(N) = n$ sparse-dense matrix multiplications. This is quite an improvement
compared to the previous case where for standard matrix vector multiplication we
required $O(N^2)$ operations.

I would like to point out that this method is already a big improvement both in
terms of operations and memory requirements. However, it is possible to perform
even better, specially in terms of memory requirements. With method 2 we still
require the memory to scale exponentialy while, intuitively, one can represent
the hadamard gate with _only four complex numbers_, independently of the number
of qubits, $n$.

One last thing to notice is that this method suffers from a huge overhead for
small systems. Indeed, for $n=4$ method 1 performs the matrix-vector
multiplication in 3 $\mu s$ whereas method 2 requires 300 $\mu s$.


## Method 3: tensor network based approach.
For this last method, we will take advantage of the tensor structure of the
operation. If we represent the state as $\psi_{i_1, j_1, k_1, l_1,...}$, with
$i_1,j_1,... \in \set{0,1}$ and the hadamard matrix as $H_{i_2}^{i_1}$, then, the operation
we are trying to perform consists on
$$
\sum_{i_1,j_1,k1,\cdots} H_{i_1}^{i_2} H_{j_1}^{j_2}H_{k_1}^{k_2} \cdots \psi^{i_1,j_1,k_1,\cdots}
$$
which is just a tensor contraction.

If we represent all these matrices as numpy arrays, we can use the
`np.tensordot` method in numpy to obtain a faster operation:
```python
H_np = qutip.hadamard_transform().full()
state_np = state.full().reshape([2]*n)

# 570 us with n=13, 50 us with n=4.
%% timeit 
for j in range(n):
    state = np.tensordot(H_np, state_np, axes=([0],[j]))
```
The complexity of this operation is $O(\log(N)N)$, same as for the sparse
case. This is because, once more, we perform $n=\log(N)$ operations
sequentially, each requiring $O(N)$ operations.


The Hadamard operation that we are using as example, has a nice graphical
representation using tensor networks as depicted in Fig. 2 a). I turns out
that, actually, all the three method presented here are tensor contractions
which have a neat graphical representation. For instance, the tensor
contraction corresponding to method 1 is depicted in Fig.  1b). There are
several other procedures to contract this tensor but, in general, finding the
optimal tensor contraction is a NP problem. Nevertheless, for this particular
yet general case, there is an optimal contraction method. This is depicted in
Fig.  1c) and consists and represents the same operation that the code above is
performing, i.e., we contract each of the Hadamard operations sequentially.

{{< figure 
src="figures/tensor_network_contraction_example.svg" 
caption=`Figure 2: a) operation thati is being analisez throught this
blog post. See first equation in the text. Green nodes represent the Hadamard
matrix and the blue blob represents the state of the system comprised by $n$
qubits. b) tensor network contraction following the method 1 discused in the
text. The operation consists on representing the hadamard matrices as a dense
(b.1))
matrix followed by the contraction of the resulting network (b.2)). c)
contraction sequence followed in the method 3 of the main text. It consists on
sequentially contracting each node until the whole tensor is contracted.`
>}}

This method has the nice properties of requiring memory that scales linearly
with the system size, and operations that scale similarly to the method 2, based
on sparse matrices. The reason that in practice method 3 is always faster than
the sparse method, is most likely due to a combination of lower memory
requirements, smaller overhead and more efficient implementation of low level
routines.

## Conclusions.

The results of the benchmarks for the problem presented in this post are
summarized in the following table:

| method      | $n=13$            | $n=4$             | Operation scaling | Memory scaling (operator) |
| ----------- | -----------       | -----             | ----              | ----                      |
| Dense (1)   | 125 $ms$   (x219) | 3 $\mu s$  (x1)   | $N^2$             | $N^2$                     |
| Sparse (2)  | 1.93 $ms$  (x3.3) | 300 $\mu s$ (100) | $N\log(N)$        | $N$                       |
| Tensor (3)  | 0.57 $ms$  (x1)   | 50 $\mu s$  (x16) | $N\log(N)$        | $\log(N)$                 |

Table: Benchmark results for the application of Hadamard matrices to a random
state as defined in the text. $n$ is the number of qubits employed in the
simulation and $N=2^n$.

We  see Dense is the fasts method for N=4 but the slowest for N=13, as it
has the worst scaling. For larger systems the Tensor method (3) is the most
efficient one both in terms of operation time and memory requirements. These
results show that representing the output of the `qutip.tensor` function as a
series of tensors can lead to significant speed-ups which I anticipate to be
most impactful for the qutip-qip package. Another example of potential usecases
comes with superoperators as they require an even larger matrix dimensions to
be represented. 

[^1]: The number of non-zero elements of $\bigotimes_{j=1}^n O_j$ is
$\prod_{j=1}^n k_j$ with $k_j$ being the non-zero elements of the matrix $O_j$.
This means that the total non-zero elements in a kroneker product is equal to
the product of non-zero elements of the individual matrices. 
