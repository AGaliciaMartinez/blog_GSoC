---
title: Benchmarking - motivation and tools (1/2)

# Summary for listings and search engines
summary: During this post I will introduce the benchmark tools that I will be using for the project. 

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
- Python
- QuTiP

categories:
- GSoC
---

One of the main motivations for this project was to leverage the computational
power of a Graphics Processing Unit (GPU) to perform faster computations with
QuTiP. However, even though overall GPUs are rated with higher floating point
operations per second (FLOPS) than CPUs, it is not straightforward to use this
computational power for our advantage. In particular, GPUs require highly
parallelizable operations. This is why during the first weeks of the project I
focused on preparing a set of benchmarks that will help us understanding when
and how to make use of the GPUs. I was specially interested to see if my own
hardware could benefit from using a GPU and how the hardware provided by colab
compares with it. The final goal is to provide an easy function, `qutip_tensorflow.benchmarks()`
that allows the user to test qutip-tensorflow's performance in its own hardware.
In this post, I will explain which tools are available for benchmarking in
python and in the next post I will show some of the results obtained in the
benchmarks.

There are several approaches that can be followed to write benchmarks in python.
The simplest one would be to use the `time` module. This module is a great tool
for quick benchmarking of a function, but using it would require a lot of
boilerplate code for both parametrizing the benchmarks and saving the results.
Notice that I am mostly interested in comparing the performance of a function
for several matrix sizes and different data representations (for example,
NumPy's `ndarray`, QuTiP's new `Dense` representation or TensorFlow's
`Tensors`). Being able to seamlessly parametrize the benchmarks would
greatly simplify the writing process. This is why I decided to use more sophisticated tools such
as `pytest-benchmark` or `asv`. Both of these tools automatically store the
results in JSON format and have included an easy way to parametrize the
benchmarks, which I will explain a little bit more in detail now.

### [Pytest-benchmark](https://github.com/ionelmc/pytest-benchmark)

`pytest` is a popular python package for testing purposes for which
several plugins are available. One of these is `pytest-benchmark` which provides
benchmarking functionality. What I found most interesting about `pytest` is that 
test (or a benchmark in `pytest-benchmark`) can be parametrized using
decorators. An example of this is:
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
In the above code we parametrize the function to benchmark the addition of
two NumPy matrices as a function of the matrix size. This is achieved with the
decorator `@pytest.mark.parametrize("size", np.logspace(1, 3, 5,
dtype=int).tolist())`. 


### [Airspeed velocity](https://asv.readthedocs.io/en/stable/) (`asv`)
`asv` defines itself as "a tool for benchmarking python package over its
lifetime". Indeed, it includes very useful tools to test a package for
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
    	a = self.a
        _ = a + a
```
Another useful feature of `asv` is that it can easily generate an html page with
the benchmark results. You can see an example of such page 
[here](https://pv.github.io/numpy-bench/) for NumPy.

### Conclusions

Both `asv` and `pytest-benchmark` are great tools that would fit my requirements
to write benchmarks: they can be easily parametrized and store the results in
JSON format. `asv` has the advantage of providing extra tools that facilitates
benchmarking a code over its lifetime. However, I am mostly interested in on
comparing the performance of different functions for different input sizes.
Hence, I decided to use `pytest-benchmark` as I am already used to using `pytest`
for testing purposes. However, if the goal of the project had been to ensure no
regression occur in the future of this package, I would have chosen `asv` as the
tool for benchmarking.

In the next post I will show a comparison of the benchmark results. Hope to see
you there!

<!--I will now show some of the results obtained when comparing the following data-->
<!--representations: `numpy`, `scipy`(sparse representation of a matrix), -->

<!--```-->
<!--# Example python code-->
<!--a = 5-->
<!--print(a)-->
<!--```-->
<!--In particular, we will show comparisons between-->
<!--the functions: `add` (element-wise addition), `multiply`(element-wise-->
<!--multiplication), `matmul`(matrix multiplication), `expm` (matrix exponentiation)-->
<!--and `eigvals` (obtaining eigenvalues for a matrix). These are very common-->
<!--operations in QuTiP for which a speed-up would be desirable.-->
