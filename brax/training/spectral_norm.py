# from dm-haiku: https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/spectral_norm.py
"""Flax-style Normalization module."""

from typing import (Any, Callable, Optional, Tuple)

from jax import lax
from jax.nn import initializers
import jax.numpy as jnp

from flax import linen
from flax.linen.initializers import lecun_normal, zeros


default_kernel_init = lecun_normal()

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?

_no_init = lambda rng, shape: ()


def _l2_normalize(x, axis=None, eps=1e-12):
  """Normalizes along dimension `axis` using an L2 norm.
  This specialized function exists for numerical stability reasons.
  Args:
    x: An input ndarray.
    axis: Dimension along which to normalize, e.g. `1` to separately normalize
      vectors in a batch. Passing `None` views `t` as a flattened vector when
      calculating the norm (equivalent to Frobenius norm).
    eps: Epsilon to avoid dividing by zero.
  Returns:
    An array of the same shape as 'x' L2-normalized along 'axis'.
  """
  return x * lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


class SpectralNorm(linen.Module):
  """Normalizes an input by its first singular value.
  This module uses power iteration to calculate this value based on the
  input and an internal hidden state.
  """
  dtype: Any = jnp.float32
  eps: float = 1e-4
  n_steps: int = 1

  @linen.compact
  def __call__(self, w) -> jnp.ndarray:
    """Applies layer normalization on the input.
    Args:
      w: the inputs
    Returns:
      Normalized inputs (the same shape as inputs).
    """
    w = jnp.asarray(w, jnp.float32)

    update_stats = True
    error_on_non_matrix = False
    w_shape = w.shape

    # Handle scalars.
    if w.ndim <= 1:
      raise ValueError("Spectral normalization is not well defined for "
                      "scalar inputs.")
    # Handle higher-order tensors.
    elif w.ndim > 2:
      if error_on_non_matrix:
        raise ValueError(
            f"Input is {w.ndim}D but error_on_non_matrix is True")
      else:
        w = jnp.reshape(w, [-1, w.shape[-1]])

    key = self.make_rng('sing_vec')
    u0 = self.variable('sing_vec', 'u0', initializers.normal(stddev=1.), key, (1, w.shape[-1]))
    # u0_state = self.variable('sing_vec', 'u0', initializers.normal(stddev=1.), key, (1, w.shape[-1]))
    # u0 = u0_state.value

    # Power iteration for the weight's singular value.
    for _ in range(self.n_steps):
      # v0 = _l2_normalize(jnp.matmul(u0, w.transpose([1, 0])), eps=self.eps)
      # u0 = _l2_normalize(jnp.matmul(v0, w), eps=self.eps)
      v0 = _l2_normalize(jnp.matmul(u0.value, w.transpose([1, 0])), eps=self.eps)
      u0.value = _l2_normalize(jnp.matmul(v0, w), eps=self.eps)

    u0.value = lax.stop_gradient(u0.value)
    v0 = lax.stop_gradient(v0)

    sigma = jnp.matmul(jnp.matmul(v0, w), jnp.transpose(u0.value))[0, 0]

    w /= sigma
    w_bar = w.reshape(w_shape)

    # if update_stats:
      # u0_state.value = u0

    return w_bar.astype(self.dtype)


class SNDense(linen.Module):
  """A linear transformation applied over the last dimension of the input
  with spectral normalization.

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """
  features: int
  use_bias: bool = True
  dtype: Any = jnp.float32
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  eps: float = 1e-4
  n_steps: int = 1

  @linen.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = self.param('kernel',
                        self.kernel_init,
                        (inputs.shape[-1], self.features))
    kernel = jnp.asarray(kernel, self.dtype)

    # for spectral normalization
    kernel_shape = kernel.shape
    # Handle scalars.
    if kernel.ndim <= 1:
      raise ValueError("Spectral normalization is not well defined for "
                      "scalar inputs.")
    # Handle higher-order tensors.
    elif kernel.ndim > 2:
      kernel = jnp.reshape(kernel, [-1, kernel.shape[-1]])
    key = self.make_rng('sing_vec')
    u0_state = self.variable('sing_vec', 'u0', initializers.normal(stddev=1.), key, (1, kernel.shape[-1]))
    u0 = u0_state.value

    # Power iteration for the weight's singular value.
    for _ in range(self.n_steps):
      v0 = _l2_normalize(jnp.matmul(u0, kernel.transpose([1, 0])), eps=self.eps)
      u0 = _l2_normalize(jnp.matmul(v0, kernel), eps=self.eps)

    u0 = lax.stop_gradient(u0)
    v0 = lax.stop_gradient(v0)

    sigma = jnp.matmul(jnp.matmul(v0, kernel), jnp.transpose(u0))[0, 0]

    kernel /= sigma
    kernel = kernel.reshape(kernel_shape)

    u0_state.value = u0

    y = lax.dot_general(inputs, kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),
                        precision=self.precision)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y
