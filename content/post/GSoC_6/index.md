---
title: GSoC 2021 final blog post

# Summary for listings and search engines
summary: In this blog post I will summarize my contributions during the GSoC 2021.

# Link this post with a project
projects: []

# Date published
date: "2021-08-18"

# Date updated
lastmod: "2021-08-18"

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

During last 10 weeks I have been participating in the Google Summer of Code
2021. This post will be a permanent link for the work done during this time. My
mentors have been Eric Giguère, Jake Lishman, Shahnawaz Ahmed and Simon Cross.
I would like to thank them all for their help during this project. I really
enjoyed working with you all and I always felt very welcome in QuTiP's
community.

My project has consisted on developing the python package qutip-tensorflow, a
plugin for QuTiP that provides support for linear algebra operations 
using TensorFlow as backend. QuTiP is a useful package when
it comes to working with quantum systems, and we wanted to provide extra
functionality by leveraging some of TensoFlow's strengths, namely GPU
computation and auto differentiation. The project has been possible thanks to
previous work on QuTiP to abstract the numerical operations into a dispatching
system. This allows having different data representations of the main class in
QuTiP, the `Qobj`. 

Although most of the work has been focused on creating the new qutip-tensorflow
package (which can be accessed
[here](https://github.com/qutip/qutip-tensorflow)), it has also required to
adapt some of QuTiP's functions to allow [auto
differentiation](https://www.tensorflow.org/guide/autodiff) to work. I will now
explain in a little more detail what my contributions have been.

## Contributions to qutip-tensorflow

- _The TfTensor class_ (PR [#8](https://github.com/qutip/qutip-tensorflow/pull/8), merged):
This PR includes the base class that wraps around a `tensorflow.Tensor` to be compatible
with QuTiP. It also includes the `to` and `create` specialisations for
`TfTensor`, allowing QuTiP's `Qobj` to represent its data with the `TfTensor`
class.

- _Specialisations_ (PR [#13](https://github.com/qutip/qutip-tensorflow/pull/13)-
  [#17](https://github.com/qutip/qutip-tensorflow/pull/17),
  [#20](https://github.com/qutip/qutip-tensorflow/pull/20),
  [#21](https://github.com/qutip/qutip-tensorflow/pull/21). All merged.):
These pull requests add most of the specialisations required to make the
TfTensor class work seamlessly in QuTiP. The specialisations are functions
that define a operation for a particular combination of inputs. When these
specialisations are registered in QuTiP, it allows the `Qobj` class backed
with a `TfTensor` to operate using TensorFlow in the backend. They act as a
translation from TensorFlow to QuTiP. 

- _Specialisations in review process_ (PR
  [#22](https://github.com/qutip/qutip-tensorflow/pull/22) and PR
  [#24](https://github.com/qutip/qutip-tensorflow/pull/24), in review).  These
  specialisations are still in review and include both the different
  normalization operations and reshape operations that
  QuTiP makes use of.

- _Example notebook_ (PR
  [#25](https://github.com/qutip/qutip-tensorflow/pull/25), in review).
  This PR includes an example showing a potential use case for qutip-tensoflow.
  It consists on the optimizaiton of a random unitary gate to perform a
  specific operation. It shows how to work with TensorFlow's optimizers and
  QuTiP's `Qobj`.

- _Benchmark code_ (PR [#4](https://github.com/qutip/qutip-tensorflow/pull/4)):
  This includes as set of benchmarks that help assessing for which system sizes
  can qutip-tensorflow's GPU operations provide a meaningful improvement.

- _README update_ (PR [#26](https://github.com/qutip/qutip-tensorflow/pull/26))
  This includes an update of the README page in GitHub with installation
  instructions and a description of qutip-tensorflow's features.

## Contributions to QuTiP
- _Extending expect to return different types_ (PR
  [#1636](https://github.com/qutip/qutip/pull/1636), in review process): Some
  of the functions in QuTiP assume that the specialisations return an instance
  of `number.Number`. However, for auto differentiation to work, it is
  necessary to return a `tensorflow.Tensor`. This pull  request
  addresses this issue in the `qutip.expect` function. It should be noted that
  there are still a few other places in the code where the same assumption is
  done and hence they should be updated.

- _Allowing arbitrary scalar multiplication times `Qobj`_ (PR
  [#1620](https://github.com/qutip/qutip/pull/1620), merged): This PR improves
  `__mul__` implementation to allow `Qobj` instances to be multiplied by
  `tf.Variables`. This is an extremely useful feature for auto-differentiation
  as most common use cases in optimization rely on scalar variables.

- _Extending specialisation test suite_ (PR
  [#1622](https://github.com/qutip/qutip/pull/1622),
  [#1626](https://github.com/qutip/qutip/pull/1626) and
  [#1630](https://github.com/qutip/qutip/pull/1630), merged. PR
  [#1635](https://github.com/qutip/qutip/pull/1635) and
  [#1637](https://github.com/qutip/qutip/pull/1637) in review):
  This PR extend the already existing test suite for specialisation functions.
  This test suite is also used in qutip-tensor flow and facilitates enormously
  the development of new data layers. Some of these PR also address a few bugs
  that were found during the development of these tests.

## What is next?
I believe qutip-tensorflow will become a useful tool for developers working in
the areas of quantum optimal control and machine learning for quantum computing.
However, there are still a few improvements that will make it more attractive:

- _Missing specialisations_. There are still some missing specialisations in
  qutip-tensorflow. This will not be a problem in most of the cases as, when an
  operation in QuTiP is required for which an specialisation does not exist,
  QuTiP will automatically convert a data type into another for which the
  specialisation is known. However, this means that you lose the auto
  differentiation feature with that operations. Most of the specialisations are
  already included but there are still a few missing (see issue
  [#28](https://github.com/qutip/qutip-tensorflow/issues/28)).

- _Support for_ `tensorflow.function`. TensorFlow provides this decorator for JIT
  compiling functions in Python. Extending this function in qutip-tensorflow to
  work seamlessly with `Qobj` will further extend qutip-tensorflow
  capabilities.

- _Support for_ `tensorflow.keras.Model` _and batched operations_. If
  qutip-tensorflow wants to stand out in the quantum machine learning field it
  will be necessary to support batched operations, as these improve the
  efficiency of common operations. Similarly, seamless integrations with Keras
  `Model` class will facilitate creating new machine learning models.

