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
  from https://github.com/WilsonWangTHU/NerveNet/blob/master/environments/assets/CentipedeThree.xml
"""
import brax
from brax.experimental.composer.components import common

COLLIDES = ('torso_0', 'torso_1', 'legbody_0', 'frontFoot_0', 'legbody_1', 'frontFoot_1', 'legbody_2', 'frontFoot_2')

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


SYSTEM_CONFIG = """
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
friction: 0.6
gravity {
  z: -9.81
}
velocity_damping: 1.0
angular_damping: -0.05
baumgarte_erp: 0.1
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
