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
"""Composer for multi-agent environments."""

import collections
from collections import OrderedDict as odict
import copy
import functools
import itertools
from typing import Dict, Any, Callable, Tuple, Optional, Union

import brax
from brax import envs
from brax.envs import Env
from brax.envs import State
from brax.envs import wrappers
from brax.experimental.braxlines.common import sim_utils
from brax.experimental.braxlines.envs import wrappers as braxlines_wrappers
from brax.experimental.composer import agent_utils
from brax.experimental.composer import component_editor
from brax.experimental.composer import composer_utils
from brax.experimental.composer import data_utils
from brax.experimental.composer import envs as composer_envs
from brax.experimental.composer import observers
from brax.experimental.composer import reward_functions
from brax.experimental.composer.components import load_component
from brax.experimental.composer.components import register_default_components
from brax.experimental.composer import ComponentEnv
from brax.experimental.composer import Composer
from jax import numpy as jnp

inspect_env = composer_envs.inspect_env
list_env = composer_envs.list_env
register_env = composer_envs.register_env
register_lib = composer_envs.register_lib

register_default_components()
composer_envs.register_default_libs()

MetaData = collections.namedtuple('MetaData', [
    'components',
    'edges',
    'global_options',
    'config_str',
    'config_json',
    'extra_observers',
    'reward_features',
    'reward_fns',
    'agent_groups',
])


class MAComponentEnv(ComponentEnv):
  """Make a brax Env from config/metadata for training and inference."""

  def step(self,
           state: State,
           action: jnp.ndarray,
           normalizer_params: Dict[str, jnp.ndarray] = None,
           extra_params: Dict[str, Dict[str, jnp.ndarray]] = None) -> State:
    """Run one timestep of the environment's dynamics."""
    del normalizer_params, extra_params
    qp, info = self.sys.step(state.qp, action)
    obs_dict, reward_features = self._get_obs(qp, info)
    obs = data_utils.concat_array(obs_dict, self.observer_shapes)
    reward_tuple_dict = odict([
        (k, fn(action, reward_features))
        for k, fn in self.composer.metadata.reward_fns.items()
    ])
    if self.is_multiagent:  # multi-agent
      reward, score, done = agent_utils.process_agent_rewards(
          self.metadata, reward_tuple_dict)
    else:
      reward, done, score = jnp.zeros((3,))
      for r, s, d in reward_tuple_dict.values():
        reward += r
        score += s
        done = jnp.logical_or(done, d)
    done = self.composer.term_fn(done, self.sys, qp, info)
    state.info['rewards'] = odict(
        [(k, v[0]) for k, v in reward_tuple_dict.items()] +
        [(k, v) for k, v in zip(self.agent_names, reward)])
    state.info['scores'] = odict(
        [(k, v[1]) for k, v in reward_tuple_dict.items()] +
        [(k, v) for k, v in zip(self.agent_names, score)])
    state.info['score'] = score
    return state.replace(
        qp=qp, obs=obs, reward=reward, done=done.astype(jnp.float32))

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jnp.ndarray:
    """Observe."""
    obs_dict, reward_features = self.composer.obs_fn(self.sys, qp, info)
    if self.observer_shapes is None:
      self.observer_shapes = data_utils.get_array_shapes(
          obs_dict, batch_shape=())
    return obs_dict, reward_features