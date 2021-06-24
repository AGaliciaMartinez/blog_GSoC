---
title: Benchmarking framework

# Summary for listings and search engines
summary: During this post I will show the benchmark framework I prepared for a future comparison between qutip-tensorflow and qutip. Given that qutip-tensorflow is yet to be writen, I will use this benchmarks to compare QuTiP with Numpy, Scipy and even TensorFlow working with a GPU.

# Link this post with a project
projects: []

# Date published
date: "2021-06-24"

# Date updated
lastmod: "2021-06-24"

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

categories:
- GSoC
---

In this post I will.... Furthermore, I will show some of the results ... But
first, let me argue why the benchmarks were thoutght to be necessary for this
project.

# Motivation and tools for benchmarking

One of the main motivations for this project was to leverage the computational
power of a Graphics Processing Unit (GPU) to perform faster computations with
QuTiP. However, even though overall GPUs are rated with higher floating point
operations per second (FLOPS) than CPUs, it is not straightforward to use this
computational power in our advantage. In particular, GPUs require highly
parallelizable operations. This is why during the first weeks of the project I
focused on preparing a set of benchmarks that will help us understanding when
and how to make use of the GPUs. I was specially interested to see if my own
hardware could benefit from using a GPU and how the hardware provided by colab
compares with it.

There are several approaches that can be followed to write benchmarks in python.
The simplest one would be to use the `time` module. This module is a great tool
for quick benchmarking of a function, but using it would require a lot of
boilerplate code for both parametrizing the benchmarks and saving the results.
This is why I decided to use more sophisticated tools such as `pytest-benchmark`
or `asv`.

## Pytest-benchmark

`pytest` is a popular python package for testing purposes for which
several plugins are available. One of these is `pytest-benchmark` which provides
benchmarking functionality. What I found most interesting about `pytest` is the
parametrization of a test (or a benchmark in `pytest-benchmark`) using
decorators which greatly simplifies this task. An example of this is:
```
import numpy as np
import pytest

@pytest.mark.parametrize("size", np.logspace(1, 3, 5, dtype=int).tolist())
def test_add(benchmark, size):
    # Create a random matrix
    a = np.random.random((size,size))

    # benchmark a+a
    benchmark(a.__add__, a)
```
In the above code we parametrize a benchmark for the addition of two NumPy
matrices of different sizes. 

## Airspeed velocity (`asv`)
`asv` defines itself as "a tool for benchmarking python package over its
lifetime". Indeed, it provides very useful functionality to test a package for
regression, such as running the benchmarks over a range of commits with a single
command (for example, `asv run master..mybranch ` would run the benchmarks for
the commits since branching off master). It also provides parametrization of the
benchmarks. An example of the above code in `asv` would be:
```
import numpy as np

class TimeLA:
    """
    Minimal linear algebra benchmark.
    """
    params = np.logspace(1, 3, 5, dtype=int).tolist() # Matrix sizes

    # Run this before benchmarking
    def setup(self, size):
	# Create a random matrix
        self.a = np.random.random((size,size))

    # benchmark a+a
    def time_add(self):
        _ = self.a + self.a
```

## Conclusions

As a side note, this tool is being used by the popular NumPy package to
keep track of the performance of their code over time.  However, for the
purposes of this project, we are not so much interested in keeping track of code
regression but rather comparing the performance between the different data
representations in QuTiP. I decided to use `pytest-benchmark` for being a plugin
for `pytest` that was going to use for testing.


I will now show some of the results obtained when comparing the following data
representations: `numpy`, `scipy`(sparse representation of a matrix), 

```
# Example python code
a = 5
print(a)
```
In particular, we will show comparisons between
the functions: `add` (element-wise addition), `multiply`(element-wise
multiplication), `matmul`(matrix multiplication), `expm` (matrix exponentiation)
and `eigvals` (obtaining eigenvalues for a matrix). These are very common
operations in QuTiP for which a speed-up would be desirable.
