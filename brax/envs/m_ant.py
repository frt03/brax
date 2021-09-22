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

"""Trains an ant to run in the +x direction."""

import jax
import brax
from brax.envs.ant import Ant
import jax.numpy as jnp
from brax.experimental.braxlines.common import sim_utils


class MorphAnt(Ant):
  """Trains an ant to run in the +x direction."""

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jnp.ndarray:
    bodies = dict(self.sys.body_idx)
    if 'Ground' in bodies:
      bodies.pop('Ground')
    indices = sim_utils.names2indices(self.sys.config, bodies, 'body')[0]

    obs_dict = {}
    for type_ in ('pos', 'rot', 'vel', 'ang'):
      for index, body in zip(indices, bodies):
        if not body in obs_dict:
          obs_dict[body] = []
        value = getattr(qp, type_)[index]
        obs_dict[body].append(value)

    obs = []
    for body in sorted(bodies):
      obs.append(jnp.concatenate(obs_dict[body]))

    return jnp.vstack(obs)

  @property
  def observation_size(self) -> int:
    """The size of the observation vector returned in step and reset."""
    rng = jax.random.PRNGKey(0)
    reset_state = self.reset(rng)
    return reset_state.obs.shape
