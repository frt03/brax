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
from jax import numpy as jnp
from brax.envs import env


LIMB_DICT = {
  'ant': {
    'bodies': [
      '$ Torso', 'Aux 1', '$ Body 4', 'Aux 2', '$ Body 7',
      'Aux 3', '$ Body 10', 'Aux 4', '$ Body 13'],
    'joints': {
        'Aux 1': [(-30.0, 30.0), '$ Torso_Aux 1'],
        '$ Body 4': [(30.0, 70.0), 'Aux 1_$ Body 4'],
        'Aux 2': [(-30.0, 30.0), '$ Torso_Aux 2'],
        '$ Body 7': [(-70.0, -30.0), 'Aux 2_$ Body 7'],
        'Aux 3': [(-30.0, 30.0), '$ Torso_Aux 3'],
        '$ Body 10': [(-70.0, -30.0), 'Aux 3_$ Body 10'],
        'Aux 4': [(-30.0, 30.0), '$ Torso_Aux 4'],
        '$ Body 13': [(30.0, 70.0), 'Aux 4_$ Body 13']}
      },
  'halfcheetah': {
      'bodies': ['torso', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot'],
      'joints': {
        'bthigh': (-29.793806076049805, 60.16056823730469),
        'bshin': (-44.97718811035156, 44.97718811035156),
        'bfoot': (-22.918312072753906, 44.97718811035156),
        'fthigh': (-57.295780181884766, 40.1070442199707),
        'fshin': (-68.75493621826172, 49.847328186035156),
        'ffoot': (-28.647890090942383, 28.647890090942383)}
      }
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
    env: env, state: env.State, env_name: str = 'halfcheetah') -> jnp.ndarray:
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
      xpos[0] -= torso_x_pos
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
      xpos[0] -= torso_x_pos
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
        joint_range = LIMB_DICT[env_name]['joints'][b][0]
        # normalize
        angle = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
        joint_range[0] = (180.0 + joint_range[0]) / 360.0
        joint_range[1] = (180.0 + joint_range[1]) / 360.0
      obs = jnp.concatenate([obs, [angle], joint_range])
      return obs

  local_limb_states = jnp.concatenate([_get_obs_per_limb(b) for b in LIMB_DICT[env_name]['bodies']])

  return local_limb_states
