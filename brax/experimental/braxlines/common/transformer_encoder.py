# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer Encoder"""
from typing import Any, Callable, Optional
from flax import linen
from flax.linen.initializers import lecun_normal, zeros
import jax
import jax.numpy as jnp


class TransformerEncoderLayer(linen.Module):
  """TransformerEncoderLayer module."""
  d_model: int
  num_heads: int
  dim_feedforward: int
  dropout_rate: float = 0.1
  dtype: Any = jnp.float32
  qkv_features: Optional[int] = None
  activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
  kernel_init: Callable[..., Any] = lecun_normal()
  bias_init: Callable[..., Any] = zeros
  deterministic: bool = False if dropout_rate > 0.0 else True

  @linen.compact
  def __call__(
      self,
      src: jnp.ndarray,
      src_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    src2 = linen.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_features,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=True,
        broadcast_dropout=False,
        dropout_rate=self.dropout_rate)(src, src, mask=src_mask)
    src = src + linen.Dropout(
        rate=self.dropout_rate,
        deterministic=self.deterministic)(src2)
    src = linen.LayerNorm(dtype=self.dtype)(src)
    src2 = linen.Dense(
        self.dim_feedforward,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(src)
    src2 = self.activation(src2)
    src2 = linen.Dropout(
        rate=self.dropout_rate,
        deterministic=self.deterministic)(src2)
    src2 = linen.Dense(
        self.d_model,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(src2)
    src = src + linen.Dropout(
        rate=self.dropout_rate,
        deterministic=self.deterministic)(src2)
    src = linen.LayerNorm(dtype=self.dtype)(src)
    return src


class TransformerEncoder(linen.Module):
  """TransformerEncoder module."""
  num_layers: int
  d_model: int
  num_heads: int
  dim_feedforward: int
  norm: Optional[Callable[..., Any]] = None
  dropout_rate: float = 0.1
  dtype: Any = jnp.float32
  qkv_features: Optional[int] = None
  activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
  kernel_init: Callable[..., Any] = lecun_normal()
  bias_init: Callable[..., Any] = zeros

  @linen.compact
  def __call__(
      self,
      src: jnp.ndarray,
      src_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
      output = src
      for _ in range(self.num_layers):
          output = TransformerEncoderLayer(
              d_model=self.d_model,
              num_heads=self.num_heads,
              dim_feedforward=self.dim_feedforward,
              dropout_rate=self.dropout_rate,
              dtype=self.dtype,
              qkv_features=self.qkv_features,
              activation=self.activation,
              kernel_init=self.kernel_init,
              bias_init=self.bias_init)(output, src_mask)
      if self.norm is not None:
          output = self.norm(dtype=self.dtype)(output)
      return output
