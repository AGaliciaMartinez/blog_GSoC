---
title: TensorFlow Data Layer.

# Summary for listings and search engines
summary: In this post I will update the progress in creating a QuTiP data type that is backed by TensorFlow's Tensors.

# Link this post with a project
projects: []

# Date published
date: "2021-07-21"

# Date updated
lastmod: "2021-07-21"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: true

authors:
- admin

tags:
- GSoC
- Python
- QuTiP

categories:
- GSoC
---

In this post I will update the progress in creating a QuTiP data type that is
backed by TensorFlow's Tensors. There are three main motivations to have QuTiP
work with TensorFlow's Tensors:
- *Seamless integration between TensorFlow and QuTiP*. There is a research
  field that aims to harness machine learning advances in the context of
  quantum systems(see this[^1] for an example).  I believe qutip-tensorflow
  will be a very useful tool in this context.
- *Operate with a GPU*. Even people not familiar with TensorFlow will benefit
  from qutip-tensorflow as it will allow users to operate with the GPU. Take a
  look at may previous posts [here](https://agalicia.netlify.app/post/gsoc_3/)
  and [here](https://agalicia.netlify.app/post/gsoc_2/) for potential benefits
  and challenges in this regard. There is also another GSoC project,
  [qutip-cupy]() that will also provide this functionality.
- *Auto differentiation*. This will be the main topic of this blog post. 

In this post we will discuss what auto differentiation is and why it is such
and interesting feature to make it work with QuTiP. 


## Auto differentiation.
There are commonly three approaches that one can use to differentiate a
function. The first one consists on obtaining an analytical expression using
the derivatives rules we learn in high school, together with the chain rules.
However, this approach becomes less reliable when we deal with long and
complicated multivariate functions, as it is easier for humans to make an error
when calculating the derivative of a function. Furthermore, this is not an
flexible method as we have to manually obtain the derivative of each new
function. 

An alternative numeric approach consists on using the finite difference
methods. An example of these is the forward difference where the derivative of
a funtion $f(x)$ is approximated by:
$$
\frac{df(x)}{dx} = \frac{f(x+h) - f(x)}{h}
$$
for $h$ an small number. The smaller it is, the more precise this approximation
is expected to be. Although this method could be applied to arbitrary
functions, it does not provide an _exact_ value of the derivative. Furthermore,
it suffers from numerical rounding errors as it is necessary to subtract two
very similar floating point values.

The third method is auto differentiation. This method is an _exact_ method that
makes use of the fact that functions in computation are composed of simpler
operations (such us exponentiation or multiplication) for which an exact
derivative is known. This method first creates a graph with the operations that
compose a function and then uses the chain rule to obtain with back propagation
the derivative of a function. Hence, this method is both exact, and extensible
as it can be applied to arbitrary functions.

Obtaining the exact derivative of a function is key in minimization problems.
These are extensively used in TensorFlow in order to train models, where the
minimized function is the cost function. However, QuTiP could also potentially
benefit from auto differentiation. For example, QuTiP 4 has the module [Quantum
Optimal Control](https://qutip.org/docs/4.0.2/guide/guide-control.html) that
minimizes a cost function to achieve the desired dynamics with a
limited control of a system.

## Support for TensorFlow in QuTiP - progress.

To support auto differentiation in qutip-tensorflow we need to give support to the two main
classes of TensorFlow: `tensorflow.Tensor` and `tensorflow.Variable`. We do
this by creating a `Data` class in QuTiP, `TfTensor`, that at this moment wraps
a `Tensor`.  However, in the future we plan `TfTensor` to wrap both `Tensor`
and `Variable`.

QuTiP uses an instance of `Data` to represent a quantum object, `Qobj`. It also
uses a dispatcher system to support different types of data. This dispatcher
system is quite flexible and it is used to define how different `Data`
instances operate.
For instance, there may be cases where you want to operate with two `Qobj`that
are backed with different data types. An example of this is the matrix multiplication
of a Hamiltonian (usually represented with an sparse matrix) times a ket
(usually represented with a dense matrix). The dispatcher allows to
specify _specialisations_ that handle this operation with a particular input data
types, such as a dense matrix times a sparse matrix or a dense matrix times a
dense matrix.

As far as qutip-tensorflow is concerned, the dispatcher system allows us to
conveniently define QuTiP's operations using TensorFlow's `Tensor`. As an
example of how this works, we consider the matrix multiplication of two
`TfTensor` which store a `Tensor` in the `_tf` attribute.
```python
def matmul_tftensor(left, right, scale=1, out=None):
    """
    Perform the operation
        ``out := scale * (left @ right) + out``
    where `left`, `right` and `out` are matrices.  `scale` is a complex scalar,
    defaulting to 1. If `out` not given is assumed to be 0. `left` and right
    are instances of `TfTensor`.
    """
    result = tensorflow.matmul(scale * left._tf, right._tf)

    if out is None:
        return qutip_tensorflow.data.TfTensor(result)
    else:
        out._tf = result + out._tf
```
we then add this specialisation to the dispatcher:
```python
qutip.data.matmul.add_specialisations([(TfTensor, TfTensor, TfTensor, matmul_tftensor)])
```
With this, `Qobj` will now support matrix multiplication between two `TfTensor`
that wraps a `tensorflow.Tensor`.

Currently only addition and matrix multiplication have been added as
specialisations but, the rest are ready to be included (I just need to create
a PR).

## Auto differentiation in QuTiP - challenges.
To seamlessly support auto differentiation in qutip-tensorflow, we first need
to overcome a number of challenges:
- We need to allow `tensorflow.GradientTape` to accept a `Qobj`. I plan to do
  this by providing a `qutip_tensorflow.GradientTape` that wraps TensorFlow's
  version so that the `gradient` (and similar) methods accept as input a `Qobj`.

- We also want to allow operations of the form:
```python
a = tensorflow.Variable(1+1j)  # Complex _scalar_ variable.
qobj = qutip.Qobj(tensor)  # Qobj represented with a tensorflow.Tensor 
a*qobj  
```
At this moment this operation is not supported in QuTiP (see [this
issue](https://github.com/qutip/qutip/issues/1607)). One of my next tasks will
be to add this functionality to QuTiP in a sensible way.

- Allow `qutip.core.operators` functions to accept `Variable` as input. An example of
  this is the function `qutip.squeezing(a1, a2, z)` where `z` is a complex
  scalar. We would like to have this functions to accept arbitrary scalar-like
  objects such as `tensorflow.Variable`. This requires the previous point to
  work but there are a few other things to consider. For instance, one of the
  operations inside `qutip.squeezing` is `np.conj(z)` which will not work
  if `z` is a `tensorflow.Variable`. One approach could be to substitute `np`
  in `qutip.code.operators` with `tensorflow.experimental.numpy` when importing
  qutip-tensorflow. However, this seems prone to errors so I am still thinking
  for an alternative way. 







