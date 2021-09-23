# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""temporal functions. Need to refactor a lot."""
import collections
from typing import Any, Dict, Optional
import jax
from jax import numpy as jnp

from brax.envs.env import Env, State
from brax.experimental.composer.composer import get_env_obs_dict_shape


LIMB_DICT = {
  'ant': {
    'bodies': [
      '$ Torso', 'Aux 1', '$ Body 4', 'Aux 2', '$ Body 7',
      'Aux 3', '$ Body 10', 'Aux 4', '$ Body 13'],
    'joints': {
        'Aux 1': [[-30.0, 30.0], '$ Torso_Aux 1'],
        '$ Body 4': [[30.0, 70.0], 'Aux 1_$ Body 4'],
        'Aux 2': [[-30.0, 30.0], '$ Torso_Aux 2'],
        '$ Body 7': [[-70.0, -30.0], 'Aux 2_$ Body 7'],
        'Aux 3': [[-30.0, 30.0], '$ Torso_Aux 3'],
        '$ Body 10': [[-70.0, -30.0], 'Aux 3_$ Body 10'],
        'Aux 4': [[-30.0, 30.0], '$ Torso_Aux 4'],
        '$ Body 13': [[30.0, 70.0], 'Aux 4_$ Body 13']}},
  'halfcheetah': {
      'bodies': ['torso', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot'],
      'joints': {
        'bthigh': [-29.793806076049805, 60.16056823730469],
        'bshin': [-44.97718811035156, 44.97718811035156],
        'bfoot': [-22.918312072753906, 44.97718811035156],
        'fthigh': [-57.295780181884766, 40.1070442199707],
        'fshin': [-68.75493621826172, 49.847328186035156],
        'ffoot': [-28.647890090942383, 28.647890090942383]}}
  }


def quat2expmap(quat: jnp.ndarray) -> jnp.ndarray:
  """Converts a quaternion to an exponential map
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
  Args:
    quat: 4-dim quaternion
  Returns:
    r: 3-dim exponential map
  Raises:
    ValueError if the l2 norm of the quaternion is not close to 1
  """
  if jnp.abs(jnp.linalg.norm(quat) - 1) > 1e-3:
    raise (ValueError, 'quat2expmap: input quaternion is not norm 1')

  sinhalftheta = jnp.linalg.norm(quat[1:])
  coshalftheta = quat[0]
  r0 = jnp.divide(quat[1:], (jnp.linalg.norm(quat[1:]) + jnp.finfo(jnp.float32).eps))
  theta = 2 * jnp.arctan2(sinhalftheta, coshalftheta)
  theta = jnp.mod(theta + 2 * jnp.pi, 2 * jnp.pi)
  if theta > jnp.pi:
    theta = 2 * jnp.pi - theta
    r0 = -r0
  r = r0 * theta
  return r


def get_local_limb_states(
    env: Env, state: State, env_name: str = 'halfcheetah') -> jnp.ndarray:
  """Reference:
    - https://github.com/yobibyte/amorpheus/blob/master/modular-rl/src/environments/ModularEnv.py
  Currently, only support ant, halfcheetah
  """
  if env_name == 'halfcheetah':
    def _get_obs_per_limb(b: str, idx: int):
      if b == 'torso':
        limb_type_vec = jnp.array((1, 0, 0, 0))
      elif 'thigh' in b:
        limb_type_vec = jnp.array((0, 1, 0, 0))
      elif 'leg' in b:
        limb_type_vec = jnp.array((0, 0, 1, 0))
      elif 'foot' in b:
        limb_type_vec = jnp.array((0, 0, 0, 1))
      else:
        limb_type_vec = jnp.array((0, 0, 0, 0))
      torso_x_pos = state.qp.pos[0][0]
      xpos = state.qp.pos[idx]
      jax.ops.index_update(xpos, 0, xpos[0] - torso_x_pos)
      quat = state.qp.rot[idx]
      expmap = quat2expmap(quat)
      obs = jnp.concatenate(
        [
          xpos,
          jnp.clip(state.qp.vel[idx], -10, 10),
          state.qp.ang[idx],
          expmap,
          limb_type_vec])
      # include current joint angle and joint range as input
      if b == 'torso':
        angle = 0.0
        joint_range = [0.0, 0.0]
      else:
        # some pre-processing to pull joint angles and velocities
        (joint_angle,), _ = env.sys.joint_revolute.angle_vel(state.qp)
        angle = joint_angle[idx-1]
        joint_range = LIMB_DICT[env_name]['joints'][b]
        # normalize
        angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
        joint_range[0] = (180.0 + joint_range[0]) / 360.0
        joint_range[1] = (180.0 + joint_range[1]) / 360.0
      obs = jnp.concatenate([obs, [angle], joint_range])
      return obs
  elif 'ant':  # TODO: ignore contact forces for now
    def _get_obs_per_limb(b: str, idx: int):
      if 'Torso' in b:
        limb_type_vec = jnp.array((1, 0, 0))
      elif 'Aux' in b:
        limb_type_vec = jnp.array((0, 1, 0))
      elif 'Body' in b:
        limb_type_vec = jnp.array((0, 0, 1))
      else:
        limb_type_vec = jnp.array((0, 0, 0))
      torso_x_pos = state.qp.pos[0][0]
      xpos = state.qp.pos[idx]
      jax.ops.index_update(xpos, 0, xpos[0] - torso_x_pos)
      quat = state.qp.rot[idx]
      expmap = quat2expmap(quat)
      obs = jnp.concatenate(
        [
          xpos,
          jnp.clip(state.qp.vel[idx], -10, 10),
          state.qp.ang[idx],
          expmap,
          limb_type_vec])
      # include current joint angle and joint range as input
      if b == '$ Torso':
        angle = 0.0
        joint_range = [0.0, 0.0]
      else:
        # some pre-processing to pull joint angles and velocities
        (joint_angle,), _ = env.sys.joint_revolute.angle_vel(state.qp)
        angle = joint_angle[idx-1]
        joint_range = LIMB_DICT[env_name]['joints'][b][0]
        # normalize
        angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
        joint_range[0] = (180.0 + joint_range[0]) / 360.0
        joint_range[1] = (180.0 + joint_range[1]) / 360.0
      obs = jnp.concatenate([obs, jnp.array([angle]), jnp.array(joint_range)])
      return obs

  local_limb_states = jnp.concatenate(
    [_get_obs_per_limb(b, idx) for idx, b in enumerate(LIMB_DICT[env_name]['bodies'])])
  return local_limb_states


def concat_obs(obs_dict: Dict[str, jnp.ndarray],
               observer_shapes: Dict[str, Dict[str, Any]]) -> jnp.ndarray:
  """Concatenate observation dictionary to a vector."""
  return jnp.concatenate([
      o.reshape(o.shape[:-1] + o.shape[:-len(s['shape'])] + s['shape'])
      for o, s in zip(obs_dict.values(), observer_shapes.values())
  ], axis=-1)


def split_obs(
    obs: jnp.ndarray,
    observer_shapes: Dict[str, Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
  """Split observation vector to a dictionary."""
  obs_leading_dims = obs.shape[:-1]
  return collections.OrderedDict({
    k: obs[..., v['start']:v['end']].reshape(
      obs_leading_dims + v['shape']) for (k, v) in observer_shapes.items()
  })


class ModularWrapper(Env):

  def __init__(self, env: Env):
    self._env = env
    self.sys = self._env.sys

  @property
  def observation_size(self) -> int:
    """The size of the observation vector returned in step and reset."""
    rng = jax.random.PRNGKey(0)
    reset_state = self.reset(rng)
    return reset_state.obs.shape[-2:] # avoid reporting batch dimensions

  @property
  def action_size(self) -> int:
      return 1

  def reset(self, rng: jnp.ndarray, z: jnp.ndarray = None) -> State:
    state = self._env.reset(rng)
    return state.replace(obs=self.from_vectorized(state.obs),)

  def from_vectorized(self, observation: jnp.array):
    # TODO: better way around this. There could be many wrappers
    obs_dict = split_obs(
      observation,
      get_env_obs_dict_shape(self._env.unwrapped.unwrapped)
    )
    modular_obs = []
    for key in sorted(obs_dict.keys()):
        modular_obs.append(jnp.expand_dims(obs_dict[key], obs_dict[key].ndim - 1))
    return jnp.concatenate(modular_obs, axis=obs_dict[key].ndim - 1)

  def to_vectorized(self, observation: jnp.array):
    observer_shapes = get_env_obs_dict_shape(self._env.unwrapped.unwrapped)
    obs_dict = collections.OrderedDict({})
    for index, key in enumerate(sorted(observer_shapes.keys())):
        obs_dict[key] = observation[..., index, :]
    return jnp.concatenate([value for value in obs_dict.values()], axis=-1)

  def step(
    self,
    state: State,
    action: jnp.ndarray,
    normalizer_params: Dict[str, jnp.ndarray] = None,
    extra_params: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None) -> State:
    state = self._env.step(
      state=state.replace(
        obs=self.to_vectorized(state.obs),
      ),
      action=action,
    )
    return state.replace(obs=self.from_vectorized(state.obs),)


class ConditionalModularWrapper(ModularWrapper):

  def reset(self, rng: jnp.ndarray, z: jnp.ndarray = None) -> State:
    state = self._env.reset(rng)
    return state.replace(obs=self.from_parametrized(state),)

  def from_vectorized(self, observation: jnp.array):
    observation, z = self._env.split(observation)
    # TODO: better way around this. There could be many wrappers
    obs_dict = split_obs(
      observation,
      get_env_obs_dict_shape(self._env.unwrapped.unwrapped)
    )
    modular_obs = []
    for key in sorted(obs_dict.keys()):
        o = jnp.concatenate([obs_dict[key], z], axis=-1)
        o = jnp.expand_dims(o, obs_dict[key].ndim - 1)
        modular_obs.append(o) # TODO: is this the correct way? temporary
    return jnp.concatenate(modular_obs, axis=obs_dict[key].ndim - 1)

  def to_vectorized(self, observation: jnp.array):
    observer_shapes = get_env_obs_dict_shape(self._env.unwrapped.unwrapped)
    obs_dict = collections.OrderedDict({})
    z = None
    for index, key in enumerate(sorted(observer_shapes.keys())):
        observation, z = self._env.split(observation[..., index, :])
        obs_dict[key] = observation
    flattened = jnp.concatenate([value for value in obs_dict.values()], axis=-1)
    return jnp.concatenate([flattened, z], axis=-1)

  def step(
    self,
    state: State,
    action: jnp.ndarray,
    normalizer_params: Dict[str, jnp.ndarray] = None,
    extra_params: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None) -> State:
    state = self._env.step(
      state=state.replace(
        obs=self.to_vectorized(state.obs),
      ),
      action=action,
      normalizer_params=normalizer_params,
      extra_params=extra_params
    )
    return state.replace(obs=self.from_vectorized(state.obs),)
