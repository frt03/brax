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
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp

from brax.training.spectral_norm import SpectralNorm

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

class SNMLP:
  def __init__(self, mlp: Any):
    # wrapper for spectral normalization
    self.mlp = mlp
    self.layer_sizes: Sequence[int] = mlp.layer_sizes
    self.spectral_normalization = [
      SpectralNorm(dtype=jnp.float32, eps=1e-4, n_steps=1) for _ in range(len(self.layer_sizes))
    ]
    self.idx = 0

  def init(self, rng, dummy_x):
    rngs = jax.random.split(rng, len(self.layer_sizes)*2 + 1)
    params = self.mlp.init(rngs[0], dummy_x)
    sn_params = []
    unfreezed_params = unfreeze(params)['params']
    self.idx = 0
    for module_name, param_dict in unfreezed_params.items():
      for k, v in param_dict.items():
        if k == 'kernel':
          sn_params.append(
            self.spectral_normalization[self.idx].init(
              {'params': rngs[self.idx+1], 'sing_vec': rngs[self.idx+len(self.layer_sizes)]},
              v
            )
          )
          self.idx += 1
    self.idx = 0
    self.sn_params = sn_params
    return params

  def apply(self, params, x, rng):
    rngs = jax.random.split(rng, len(self.layer_sizes) + 1)
    new_params = {}
    unfreezed_params = unfreeze(params)['params']
    self.idx = 0
    for module_name, param_dict in unfreezed_params.items():
      new_params[module_name] = {}
      for k, v in param_dict.items():
        def apply_sn_to_kernel(k, v):
          if k == 'kernel':
            applied_v, self.sn_params[self.idx] = self.spectral_normalization[self.idx].apply(
              self.sn_params[self.idx], v,
              rngs={'sing_vec': rngs[self.idx+1]}, mutable=['sing_vec']
            )
            self.idx += 1
            return applied_v
          elif k == 'bias':
            return v
        new_params[module_name][k] = apply_sn_to_kernel(k, v)
    params = freeze({'params': new_params})
    self.idx = 0
    y = self.mlp.apply(params, x)
    return y

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

  Returns:
    a model
  """
  mlp = MLP(layer_sizes=layer_sizes, activation=activation)
  module = SNMLP(mlp=mlp) if spectral_norm else mlp
  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardModel(
        init=lambda rng: module.init(rng, dummy_obs), apply=module.apply)


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
