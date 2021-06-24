---
title: Benchmarking framework

# Summary for listings and search engines
summary: During this post I will show a benchmark framework I prepared for a future comparison between qutip-tensorflow and qutip. Given that qutip-tensorflow is yet to be writen, I will use this benchmarks to compare qutip with numpy, scipy and even TensorFlow working with a GPU.

# Link this post with a project
projects: []

# Date published
date: "2021-06-24"

# Date updated
lastmod: "2021-06-24"

# Is this an unpublished draft?
draft: true

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

categories:
- GSoC
---

In this post I will.... Furthermore, I will show some of the results ... But
first, let me argue why the benchmarks were thoutght to be necessary for this
project.

One of the main motivations for this project was to leverage the computational
power of a Graphics Processing Unit (GPU) to perform faster computations with
QuTiP. However, even though overall GPUs are rated with higher floating point
operations per second (FLOPS) than CPUs, it is not straightforward to use this
computational power in our advantage. In particular, GPUs require highly
parallelizable operations. Here we aim to give a set of benchmarks that will help
us understanding when and how we make use of this computational power. In
particular, we will show comparisons between the functions: `add` (element-wise
addition), `multiply`(element-wise multiplication),
`matmul`(matrix multiplication), `expm` (matrix exponentiation) and `eigvals`
(obtaining eigenvalues for a matrix). This are very common operations when
with QuTiP and for which an speed-up would be desirable.




# Why bechmarks. 
- Not sure what we will see. One of the motivations to have a tensorflow bavkend is the avitlity to speed up computations by using the GPU. Will we really achieve this.
- Use google colaab.
- We use complex number which are usually not as well suported as 

