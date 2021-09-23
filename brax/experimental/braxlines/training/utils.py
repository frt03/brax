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

"""Training utilities."""
from brax.envs.env import Env
import functools
from typing import Any, Callable, Dict
from brax.experimental.braxlines.common.morphology import ModularWrapper
from brax.experimental.composer import composer
from jax import numpy as jnp


def zero_fn(params, obs: jnp.ndarray, rng: jnp.ndarray, action_size: int):
  """Output zero actions."""
  del params, rng
  return jnp.zeros(obs.shape[:-1] + (action_size,))

def create_modular(**kwargs) -> Env:
  """Creates an Env with from a brax system with modularized observations"""
  env = composer.create(**kwargs)
  return ModularWrapper(env)

def create_modular_fn(**kwargs) -> Callable[..., Env]:
  """Returns a function that when called, creates a modularized Env."""
  return functools.partial(create_modular, **kwargs)