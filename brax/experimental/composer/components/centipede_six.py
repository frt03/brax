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

"""Centipede with three bodies.
  from https://github.com/WilsonWangTHU/NerveNet/blob/master/environments/assets/CentipedeSix.xml
"""
import brax
from brax import jumpy as jp
from brax.envs import env


class CentipedeSix(env.Env):
  """Trains a centipede with six legs to run in the +x direction."""

  def __init__(self, **kwargs):
    super().__init__(_SYSTEM_CONFIG, **kwargs)

  def reset(self, rng: jp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jp.random_split(rng, 3)
    qpos = self.sys.default_angle() + jp.random_uniform(
        rng1, (self.sys.num_joint_dof,), -.1, .1)
    qvel = jp.random_uniform(rng2, (self.sys.num_joint_dof,), -.1 / 2, .1 / 2)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_ctrl_cost': zero,
        'reward_contact_cost': zero,
        'reward_forward': zero,
        'reward_survive': zero,
    }
    return env.State(qp, obs, reward, done, metrics)

  def step(self, state: env.State, action: jp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    qp, info = self.sys.step(state.qp, action)
    obs = self._get_obs(qp, info)

    x_before = state.qp.pos[0, 0]
    x_after = qp.pos[0, 0]
    forward_reward = (x_after - x_before) / self.sys.config.dt
    ctrl_cost = .5  * 4 / 6 * jp.sum(jp.square(action))
    contact_cost = (0.5 * 1e-3 * 4 / 6 *
                    jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1))))
    survive_reward = jp.float32(1)
    reward = forward_reward - ctrl_cost - contact_cost + survive_reward

    done = jp.where(qp.pos[0, 2] < 0.2, x=jp.float32(1), y=jp.float32(0))
    done = jp.where(qp.pos[0, 2] > 1.0, x=jp.float32(1), y=done)
    state.metrics.update(
        reward_ctrl_cost=ctrl_cost,
        reward_contact_cost=contact_cost,
        reward_forward=forward_reward,
        reward_survive=survive_reward)

    return state.replace(qp=qp, obs=obs, reward=reward, done=done)

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
    """Observe ant body position and velocities."""
    # some pre-processing to pull joint angles and velocities
    (joint_angle,), (joint_vel,) = self.sys.joints[0].angle_vel(qp)

    # qpos:
    # Z of the torso (1,)
    # orientation of the torso as quaternion (4,)
    # joint angles (8,)
    qpos = [qp.pos[0, 2:], qp.rot[0], joint_angle]

    # qvel:
    # velcotiy of the torso (3,)
    # angular velocity of the torso (3,)
    # joint angle velocities (8,)
    qvel = [qp.vel[0], qp.ang[0], joint_vel]

    # external contact forces:
    # delta velocity (3,), delta ang (3,) * 10 bodies in the system
    # Note that mujoco has 4 extra bodies tucked inside the Torso that Brax
    # ignores
    cfrc = [
        jp.clip(info.contact.vel, -1, 1),
        jp.clip(info.contact.ang, -1, 1)
    ]
    # flatten bottom dimension
    cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]

    return jp.concatenate(qpos + qvel + cfrc)

COLLIDES = (
    'torso_0',
    'torso_1',
    'torso_2',
    'legbody_0',
    'frontFoot_0',
    'legbody_1',
    'frontFoot_1',
    'legbody_2',
    'frontFoot_2',
    'legbody_3',
    'frontFoot_3',
    'legbody_4',
    'frontFoot_4',
    'legbody_5',
    'frontFoot_5')

ROOT = 'torso_0'

DEFAULT_OBSERVERS = ('root_z_joints', 'cfrc')

def term_fn(done, sys, qp: brax.QP, info: brax.Info, component,
            **unused_kwargs):
  """Termination."""
  done = common.height_term_fn(
      done,
      sys,
      qp,
      info,
      component,
      max_height=1.0,
      min_height=0.2,
      **unused_kwargs)
  done = common.upright_term_fn(done, sys, qp, info, component, **unused_kwargs)
  return done


_SYSTEM_CONFIG = """
bodies {
  name: "floor"
  colliders {
    plane {
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
  }
}
bodies {
  name: "torso_0"
  colliders {
    position {
    }
    sphere {
      radius: 0.25
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 6.544985
}
bodies {
  name: "legbody_0"
  colliders {
    position {
      y: -0.14
    }
    rotation {
      x: 90.0
      y: -0.0
    }
    capsule {
      radius: 0.08
      length: 0.44
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.19435987
}
bodies {
  name: "frontFoot_0"
  colliders {
    position {
      y: -0.3
    }
    rotation {
      x: 90.0
      y: -0.0
    }
    capsule {
      radius: 0.08
      length: 0.76
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.3552094
}
bodies {
  name: "legbody_1"
  colliders {
    position {
      y: 0.14
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.08
      length: 0.44
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.19435987
}
bodies {
  name: "frontFoot_1"
  colliders {
    position {
      y: 0.3
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.08
      length: 0.76
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.3552094
}
bodies {
  name: "torso_1"
  colliders {
    position {
    }
    sphere {
      radius: 0.25
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 6.544985
}
bodies {
  name: "legbody_2"
  colliders {
    position {
      y: -0.14
    }
    rotation {
      x: 90.0
      y: -0.0
    }
    capsule {
      radius: 0.08
      length: 0.44
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.19435987
}
bodies {
  name: "frontFoot_2"
  colliders {
    position {
      y: -0.3
    }
    rotation {
      x: 90.0
      y: -0.0
    }
    capsule {
      radius: 0.08
      length: 0.76
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.3552094
}
bodies {
  name: "legbody_3"
  colliders {
    position {
      y: 0.14
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.08
      length: 0.44
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.19435987
}
bodies {
  name: "frontFoot_3"
  colliders {
    position {
      y: 0.3
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.08
      length: 0.76
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.3552094
}
bodies {
  name: "torso_2"
  colliders {
    position {
    }
    sphere {
      radius: 0.25
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 6.544985
}
bodies {
  name: "legbody_4"
  colliders {
    position {
      y: -0.14
    }
    rotation {
      x: 90.0
      y: -0.0
    }
    capsule {
      radius: 0.08
      length: 0.44
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.19435987
}
bodies {
  name: "frontFoot_4"
  colliders {
    position {
      y: -0.3
    }
    rotation {
      x: 90.0
      y: -0.0
    }
    capsule {
      radius: 0.08
      length: 0.76
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.3552094
}
bodies {
  name: "legbody_5"
  colliders {
    position {
      y: 0.14
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.08
      length: 0.44
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.19435987
}
bodies {
  name: "frontFoot_5"
  colliders {
    position {
      y: 0.3
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.08
      length: 0.76
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.3552094
}
joints {
  name: "lefthip_0"
  stiffness: 5000.0
  parent: "torso_0"
  child: "legbody_0"
  parent_offset {
    y: -0.28
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -40.0
    max: 40.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "ankle_0"
  stiffness: 5000.0
  parent: "legbody_0"
  child: "frontFoot_0"
  parent_offset {
    y: -0.28
  }
  child_offset {
  }
  rotation {
    y: -0.0
  }
  angle_limit {
    min: 30.0
    max: 100.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "righthip_1"
  stiffness: 5000.0
  parent: "torso_0"
  child: "legbody_1"
  parent_offset {
    y: 0.28
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -40.0
    max: 40.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "ankle_1"
  stiffness: 5000.0
  parent: "legbody_1"
  child: "frontFoot_1"
  parent_offset {
    y: 0.28
  }
  child_offset {
  }
  rotation {
    y: -0.0
  }
  angle_limit {
    min: 30.0
    max: 100.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "body_1"
  stiffness: 5000.0
  parent: "torso_0"
  child: "torso_1"
  parent_offset {
    x: 0.25
  }
  child_offset {
    x: -0.25
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -20.0
    max: 20.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "bodyupdown_1"
  stiffness: 5000.0
  parent: "torso_0"
  child: "torso_1"
  parent_offset {
    x: 0.25
  }
  child_offset {
    x: -0.25
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angle_limit {
    min: -10.0
    max: 30.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "lefthip_2"
  stiffness: 5000.0
  parent: "torso_1"
  child: "legbody_2"
  parent_offset {
    y: -0.28
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -40.0
    max: 40.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "ankle_2"
  stiffness: 5000.0
  parent: "legbody_2"
  child: "frontFoot_2"
  parent_offset {
    y: -0.28
  }
  child_offset {
  }
  rotation {
    y: -0.0
  }
  angle_limit {
    min: 30.0
    max: 100.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "righthip_3"
  stiffness: 5000.0
  parent: "torso_1"
  child: "legbody_3"
  parent_offset {
    y: 0.28
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -40.0
    max: 40.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "ankle_3"
  stiffness: 5000.0
  parent: "legbody_3"
  child: "frontFoot_3"
  parent_offset {
    y: 0.28
  }
  child_offset {
  }
  rotation {
    y: -0.0
  }
  angle_limit {
    min: 30.0
    max: 100.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "body_2"
  stiffness: 5000.0
  parent: "torso_1"
  child: "torso_2"
  parent_offset {
    x: 0.25
  }
  child_offset {
    x: -0.25
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -20.0
    max: 20.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "bodyupdown_2"
  stiffness: 5000.0
  parent: "torso_1"
  child: "torso_2"
  parent_offset {
    x: 0.25
  }
  child_offset {
    x: -0.25
  }
  rotation {
    y: -0.0
    z: 90.0
  }
  angle_limit {
    min: -10.0
    max: 30.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "lefthip_4"
  stiffness: 5000.0
  parent: "torso_2"
  child: "legbody_4"
  parent_offset {
    y: -0.28
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -40.0
    max: 40.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "ankle_4"
  stiffness: 5000.0
  parent: "legbody_4"
  child: "frontFoot_4"
  parent_offset {
    y: -0.28
  }
  child_offset {
  }
  rotation {
    y: -0.0
  }
  angle_limit {
    min: 30.0
    max: 100.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "righthip_5"
  stiffness: 5000.0
  parent: "torso_2"
  child: "legbody_5"
  parent_offset {
    y: 0.28
  }
  child_offset {
  }
  rotation {
    y: -90.0
  }
  angle_limit {
    min: -40.0
    max: 40.0
  }
  reference_rotation {
    y: -0.0
  }
}
joints {
  name: "ankle_5"
  stiffness: 5000.0
  parent: "legbody_5"
  child: "frontFoot_5"
  parent_offset {
    y: 0.28
  }
  child_offset {
  }
  rotation {
    y: -0.0
  }
  angle_limit {
    min: 30.0
    max: 100.0
  }
  reference_rotation {
    y: -0.0
  }
}
actuators {
  name: "lefthip_0"
  joint: "lefthip_0"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "ankle_0"
  joint: "ankle_0"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "righthip_1"
  joint: "righthip_1"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "ankle_1"
  joint: "ankle_1"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "lefthip_2"
  joint: "lefthip_2"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "ankle_2"
  joint: "ankle_2"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "righthip_3"
  joint: "righthip_3"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "ankle_3"
  joint: "ankle_3"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "body_1"
  joint: "body_1"
  strength: 100.0
  angle {
  }
}
actuators {
  name: "bodyupdown_1"
  joint: "bodyupdown_1"
  strength: 100.0
  angle {
  }
}
actuators {
  name: "lefthip_4"
  joint: "lefthip_4"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "ankle_4"
  joint: "ankle_4"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "righthip_5"
  joint: "righthip_5"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "ankle_5"
  joint: "ankle_5"
  strength: 150.0
  angle {
  }
}
actuators {
  name: "body_2"
  joint: "body_2"
  strength: 100.0
  angle {
  }
}
actuators {
  name: "bodyupdown_2"
  joint: "bodyupdown_2"
  strength: 100.0
  angle {
  }
}
friction: 0.6
gravity {
  z: -9.81
}
velocity_damping: 1.0
angular_damping: -0.05
baumgarte_erp: 0.1
collide_include {
  first: "floor"
  second: "torso_0"
}
collide_include {
  first: "floor"
  second: "legbody_0"
}
collide_include {
  first: "floor"
  second: "frontFoot_0"
}
collide_include {
  first: "floor"
  second: "legbody_1"
}
collide_include {
  first: "floor"
  second: "frontFoot_1"
}
collide_include {
  first: "floor"
  second: "torso_1"
}
collide_include {
  first: "floor"
  second: "legbody_2"
}
collide_include {
  first: "floor"
  second: "frontFoot_2"
}
collide_include {
  first: "floor"
  second: "legbody_3"
}
collide_include {
  first: "floor"
  second: "frontFoot_3"
}
collide_include {
  first: "floor"
  second: "torso_2"
}
collide_include {
  first: "floor"
  second: "legbody_4"
}
collide_include {
  first: "floor"
  second: "frontFoot_4"
}
collide_include {
  first: "floor"
  second: "legbody_5"
}
collide_include {
  first: "floor"
  second: "frontFoot_5"
}
dt: 0.02
substeps: 4
"""


def get_specs():
  return dict(
      message_str=SYSTEM_CONFIG,
      collides=COLLIDES,
      root=ROOT,
      term_fn=term_fn,
      observers=DEFAULT_OBSERVERS)
