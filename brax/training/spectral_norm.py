# from dm-haiku: https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/spectral_norm.py
"""Flax-style Normalization module."""

from typing import (Any, Callable, Optional, Tuple)

from jax import lax
from jax.nn import initializers
import jax.numpy as jnp

from flax.linen.module import Module, compact, merge_param


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


class SpectralNorm(Module):
  """Normalizes an input by its first singular value.
  This module uses power iteration to calculate this value based on the
  input and an internal hidden state.
  """
  dtype: Any = jnp.float32
  eps: float = 1e-4
  n_steps: int = 1

  @compact
  def __call__(self, x) -> jnp.ndarray:
    """Applies layer normalization on the input.
    Args:
      x: the inputs
    Returns:
      Normalized inputs (the same shape as inputs).
    """
    x = jnp.asarray(x, jnp.float32)

    update_stats = True
    error_on_non_matrix = False
    x_shape = x.shape

    # Handle scalars.
    if x.ndim <= 1:
      raise ValueError("Spectral normalization is not well defined for "
                       "scalar or vector inputs.")
    # Handle higher-order tensors.
    elif x.ndim > 2:
      if error_on_non_matrix:
        raise ValueError(
            f"Input is {x.ndim}D but error_on_non_matrix is True")
      else:
        x = jnp.reshape(x, [-1, x.shape[-1]])

    key = self.make_rng('sing_vec')
    u0_state = self.variable('sing_vec', 'u0', initializers.normal(stddev=1.), key, (1, x.shape[-1]))
    u0 = u0_state.value

    # Power iteration for the weight's singular value.
    for _ in range(self.n_steps):
      v0 = _l2_normalize(jnp.matmul(u0, x.transpose([1, 0])), eps=self.eps)
      u0 = _l2_normalize(jnp.matmul(v0, x), eps=self.eps)

    u0 = lax.stop_gradient(u0)
    v0 = lax.stop_gradient(v0)

    sigma = jnp.matmul(jnp.matmul(v0, x), jnp.transpose(u0))[0, 0]

    x /= sigma
    x_bar = x.reshape(x_shape)

    if update_stats:
      u0_state.value = u0

    return x_bar.astype(self.dtype)
