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

# python3
"""Network definitions."""

from typing import Any, Callable, Sequence, Tuple

import dataclasses
from flax import linen
import jax
import jax.numpy as jnp

from brax.training.spectral_norm import SNDense
from brax.experimental.braxlines.common.transformer_encoder import TransformerEncoder


@dataclasses.dataclass
class FeedForwardModel:
  init: Any
  apply: Any


class MLP(linen.Module):
  """MLP module."""
  layer_sizes: Sequence[int]
  activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
  kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias)(
              hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
    return hidden


class SNMLP(linen.Module):
  """MLP module with Spectral Normalization."""
  layer_sizes: Sequence[int]
  activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.relu
  kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = SNDense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias)(
              hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
    return hidden


class TransformerModel(linen.Module):
  """Transformer Policy/Critic"""
  num_layers: int
  d_model: int
  num_heads: int
  dim_feedforward: int
  output_size: int
  dropout_rate: float = 0.5
  transformer_norm: bool = False
  condition_decoder: bool = False

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    input_size = data.shape[-1]
    # encoder
    output = linen.Dense(
      self.d_model,
      kernel_init=jax.nn.initializers.uniform(scale=0.1),
      bias_init=linen.initializers.zeros)(
        data) * jnp.sqrt(input_size)
    output = TransformerEncoder(
      num_layers=self.num_layers,
      norm=linen.LayerNorm if self.transformer_norm else None,
      d_model=self.d_model,
      num_heads=self.num_heads,
      dim_feedforward=self.dim_feedforward,
      dropout_rate=self.dropout_rate)(output)
    if self.condition_decoder:
      output = jnp.concatenate([output, data], axis=-1)
    # decoder
    output = linen.Dense(
      self.output_size,
      kernel_init=jax.nn.initializers.uniform(scale=0.1),
      bias_init=linen.initializers.zeros)(data)
    return output


def make_model(layer_sizes: Sequence[int],
               obs_size: int,
               activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
               spectral_norm: bool = False,
               ) -> FeedForwardModel:
  """Creates a model.

  Args:
    layer_sizes: layers
    obs_size: size of an observation
    activation: activation
    spectral_norm: whether to use a spectral normalization (default: False).

  Returns:
    a model
  """
  dummy_obs = jnp.zeros((1, obs_size))
  if spectral_norm:
    module = SNMLP(layer_sizes=layer_sizes, activation=activation)
    model = FeedForwardModel(
        init=lambda rng1, rng2: module.init(
            {'params': rng1, 'sing_vec': rng2}, dummy_obs),
        apply=module.apply)
  else:
    module = MLP(layer_sizes=layer_sizes, activation=activation)
    model = FeedForwardModel(
        init=lambda rng: module.init(rng, dummy_obs), apply=module.apply)
  return model


def make_models(policy_params_size: int,
                obs_size: int) -> Tuple[FeedForwardModel, FeedForwardModel]:
  """Creates models for policy and value functions.

  Args:
    policy_params_size: number of params that a policy network should generate
    obs_size: size of an observation

  Returns:
    a model for policy and a model for value function
  """
  policy_model = make_model([32, 32, 32, 32, policy_params_size], obs_size)
  value_model = make_model([256, 256, 256, 256, 256, 1], obs_size)
  return policy_model, value_model


def make_transformer(obs_size: int,
                     output_size: int,
                     num_layers: int = 3,
                     d_model: int = 128,
                     num_heads: int = 2,
                     dim_feedforward: int = 256,
                     dropout_rate: float = 0.0,
                     transformer_norm: bool = True,
                     condition_decoder: bool = False) -> FeedForwardModel:
  """Creates a transformer model (https://arxiv.org/abs/2010.01856).

  Args:
    layer_sizes: layers
    obs_size: size of an observation
    output_size: size of an output (for policy)
    num_layers: number of layers in TransformerEncoder
    d_model: size of an input for TransformerEncoder
    num_heads: number of heads in the multiheadattention
    dim_feedforward: the dimension of the feedforward network model
    dropout_rate: the dropout value
    transformer_norm: whether to use a layer normalization
    condition_decoder: whether to concat the features of the joint

  Returns:
    a model
  """
  dummy_obs = jnp.zeros((1,) + obs_size)  # correct?
  module = TransformerModel(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dim_feedforward=dim_feedforward,
    output_size=output_size,
    dropout_rate=dropout_rate,
    transformer_norm=transformer_norm,
    condition_decoder=condition_decoder)
  model = FeedForwardModel(
        init=lambda rng: module.init(rng, dummy_obs), apply=module.apply)
  return model


def make_transformers(policy_params_size: int,
                      obs_size: int
                      ) -> Tuple[FeedForwardModel, FeedForwardModel]:
  """Creates transformer models for policy and value functions.

  Args:
    policy_params_size: number of params that a policy network should generate
    obs_size: size of an observation

  Returns:
    a model for policy and a model for value function
  """
  policy_model = make_transformer(
    obs_size=obs_size, output_size=policy_params_size)
  value_model = make_transformer(obs_size=obs_size, output_size=1)
  return policy_model, value_model
