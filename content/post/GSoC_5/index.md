---
title: qutip-tensorflow working example.

# Summary for listings and search engines
summary: In this post I show how qutip-tensorflow can benefit from TensorFlow's auto-differentiation features to optimize an operation in a qubit.

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

In this blog post I will show an example of use for qutip-tensorflow. We will
optimize an arbitrary single qubit gate to perform a full rotation from the state $|
0 \rangle$ to the state $| 1 \rangle$. Even though this is a very simple
example with analytical solution, it will show how qutip-tensorflow allows
seamless use of QuTiP's `Qobj` with TensorFlow's auto-differentiation features
through `GradientTape`.

Hence we start by importing the modules.
```python
import qutip as qt
import tensorflow as tf
import qutip_tensorflow
```

## Parametrized single qubit gate

We begin defining the parametrized unitary operation (single qubit gate) that
we aim to optimize.  Besides, we will do this using QuTiP's native functions.
The first challenge we encounter is that these functions return a `Qobj`
represented with the `Dense` data backend. This is one of the native data
backends of QuTiP 5.0, but we want the data to be backed by a `tf.Tensor` in
such a way that we can benefit from TensorFlow's `GradientTape`. For this
we employ qutip-tensorflow's `TfTensor` data backend, which is registered when
importing qutip-tensorflow. We then change the underlying data with the `to`
method. As an example, consider Pauli's X operator:
```python
sx = qt.sigmax()  # Qobj
sx.data # This is an instance of `Dense` class
sx = sx.to('tftensor')  # Transform data to TfTensor
sx.data  # This is an instance of the TfTensor class
```

When the data is represented as TfTensor, it is possible to access the
underlying `tf.Tensor` with the attribute `_tf`. However, this should not be
needed in most cases, which is way it is accessed as a private attribute.
```python
sx.data._tf  # This is an instance of tf.Tensor
```

An arbitrary single qubit gate can be composed with a consecutive set of  three
rotations in the Z-Y-Z directions. The parameters representing the rotation
angle of each unitary-rotation are $\alpha$, $\beta$ and $\gamma$.  Hence, the
single qubit gate we need can be written as:
```python
def single_qubit_gate(alpha, beta, gamma):
    # Pauli matrices backed with TfTensor
    sz = qt.sigmax().to('tftensor')
    sy = qt.sigmay().to('tftensor')
    
    # Tensorflow does not support automatic casting of real 
    # tf.Variable to complex se we do it manuall.
    alpha = tf.cast(alpha, dtype=tf.complex128)
    beta = tf.cast(beta, dtype=tf.complex128)
    gamma = tf.cast(gamma, dtype=tf.complex128)

    # Note that all this operations are backed with tf.Tensors
    # so they will support backpropagation if alpha, beta and/or gamma
    # are instaces of tf.Variable.
    rot = (1j*alpha/2*sz).expm(dtype='tftensor')
    rot *= (1j*beta/2*sy).expm(dtype='tftensor')
    rot *= (1j*gamma/2*sz).expm(dtype='tftensor')

    return rot
```

## Loss function and minimization
We now define the loss function that we will minimize. This will consist on
the evolution of an initial $|0\rangle$ state with the unitary, $U(\alpha,
\beta, \gamma)$, defined in
`single_qubit_gate`. We then compute the overlap of the evolved state with
the desired $|1\rangle$ state. This defines an output subjected to maximization.
We can minimize the output instead if we take the absolute negative value. Hence, the
loss function subject to minimization is:

$$ \text{loss}(\alpha, \beta, \gamma) = -| \langle 1 | U(\alpha, \beta, \gamma)| 0 \rangle |$$

Once more, with qutip-tensorflow imported, we can code this function using
native QuTiP methods:
```python
# Initial values
# Note that the variables are defined out of the scope of the funtion.
alpha = tf.Variable(0.01, dtype=tf.float64, name='alpha')
beta = tf.Variable(0.01, dtype=tf.float64, name='beta')
gamma = tf.Variable(0.01, dtype=tf.float64, name='gamma')

var_list = [alpha, beta, gamma]

# loss function to be optimized
def loss():
    # initial (|0>) and final (|1>) states   
    initial = qt.basis(2,0).to('tftensor')
    final = qt.basis(2,1).to('tftensor')
    
    # Unitary gate. Note that variables are defined out of the
    # function scope.
    U = single_qubit_gate(*var_list)

    # Overlap between the inital state and the desired state. 
    overlap = final.dag()*U*initial
    
    return - tf.abs(overlap)
```

Now we carry the minimization of the loss function with TensorFlow's native
optimizers:

```python
# Create an optimizer. We choose the SGD method.
opt = tf.keras.optimizers.SGD(learning_rate=1)

for epoch in range(10):
    opt.minimize(loss, var_list=var_list)  # One optimization step
    print(f"Epoch: {epoch+1:2f} | Cost: {loss():3f}")

print(f"Cost: {loss():3f}")  # -0.999992 
print(f"alpha: {alpha.numpy():3f}")  # alpha: 1.566813
print(f"beta: {beta.numpy():3f}")  # beta: 0.397178
print(f"gamma: {gamma.numpy():3f}")  # gamma: 1.566813
```

As we can see, we obtain a set of parameters, $\alpha$, $\beta$ and $\gamma$
that define, with a good approximation, a unitary gate that transforms the state
$|0\rangle$ to the state $| 1 \rangle$.

## Visualization
To finalise this blog post let me show another selling point for
qutip-tensorflow which once more relies in the seamless integration we aim to
provide between QuTiP and Tensorflow. QuTiP implements a few useful
visualization tools. Among these, the Bloch class facilitates plotting the
Bloch sphere for a state. For example, we can store the rotations obtained
after each epoch in the optimization process and visualize them using the Bloch
class[^1]:
```python
# Save initial rotation. Note that we store a Qobj backed with tf.Tensor
initial = qt.basis(2,0).to('tftensor')
state_hist = [single_qubit_gate(*var_list)*initial]

# Create an optimizer.
opt = tf.keras.optimizers.SGD(learning_rate=1)

for epoch in range(10):
    opt.minimize(loss, var_list=var_list)  # One optimization step
    print(f"Epoch: {epoch+1:2f} | Cost: {loss():3f}")
    
    # Keep track of the evolution after each epoch
    state_hist.append(single_qubit_gate(*var_list)*initial)
```

```python
# Code adapted from qgrad: 
from qutip import Bloch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt

plt.rcParams["animation.html"] = "jshtml"
fig = plt.figure()
ax = Axes3D(fig, azim=-40, elev=30)
sphere = Bloch(axes=ax)

def animate(i):
    sphere.clear()
    # Note that the input is a Qobj backed with TfTensor
    sphere.add_states(state_hist[i])
    sphere.make_sphere()
    return ax

def init():
    sphere.vector_color = ["b"]
    return ax

ani = animation.FuncAnimation(
    fig,
    animate,
    np.arange(len(state_hist)),
    init_func=init,
    repeat=True
)
ani
```
The code above will output the following animation:

<video controls src="bloch_animation.mp4"></video>

In this animation we see that after a few epochs, the Unitary operation is
optimized to perform the desired rotation: $|0\rangle \rightarrow
|1\rangle$ .

## Conclusions

In this blog post we have seen how QuTiP tensorflow allows us to use QuTiP's
native functions to optimize an operation on Quantum objects. In the future, we
aim to extend qutip-tensorflow to seamlessly work with `keras.layers` and
`keras.models`. This will facilitate writing Machine learning models using
QuTiP. Another future goal is to make qutip-tensorflow work with qutip-qip.
This will facilitate writing quantum models that rely on quantum circuits.


_Note: at the moment of writing this post some of the code snipets may not
fully work yet as there are a few PR waiting to be merged._

[^1]: The code for the animation was obtained from [here](https://github.com/qgrad/qgrad/blob/master/examples/Qubit_Rotation.py).
