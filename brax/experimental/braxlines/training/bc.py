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

"""Behavioral Cloning (BC) training."""

import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
from brax import envs
from brax.experimental.braxlines.training import env
from brax.training import distribution
from brax.training import networks
from brax.training import normalization
from brax.training import pmap
from brax.training import ppo
from brax.training.types import Params, PRNGKey

import flax
import jax
import jax.numpy
import optax


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  params: Params
  key: PRNGKey
  normalizer_params: Params


def compute_ppo_loss(
    models: Dict[str, Params],
    data: ppo.StepData,
    udata: ppo.StepData,
    rng: PRNGKey,
    parametric_action_distribution: distribution.ParametricDistribution,
    policy_apply: Any,
    value_apply: Any,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    lambda_: float = 0.95,
    ppo_epsilon: float = 0.3,
    extra_loss_update_ratios: Optional[Dict[str, float]] = None,
    extra_loss_fns: Optional[Dict[str, Callable[[ppo.StepData],
                                                jnp.ndarray]]] = None):
  """"Computes PPO loss."""
  policy_params, value_params = models['policy'], models['value']
  extra_params = models.get('extra', {})
  policy_logits = policy_apply(policy_params, data.obs)
  baseline = value_apply(value_params, data.obs)
  baseline = jnp.squeeze(baseline, axis=-1)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = baseline[-1]
  baseline = baseline[:-1]

  # At this point, we have unroll length + 1 steps. The last step is only used
  # as bootstrap value, so it's removed.

  # already removed at data generation time
  # actions = actions[:-1]
  # logits = logits[:-1]

  rewards = data.rewards[1:] * reward_scaling
  truncation = data.truncation[1:]
  termination = data.dones[1:] * (1 - truncation)

  target_action_log_probs = parametric_action_distribution.log_prob(
      policy_logits, data.actions)
  behavior_action_log_probs = parametric_action_distribution.log_prob(
      data.logits, data.actions)

  vs, advantages = ppo.compute_gae(
      truncation=truncation,
      termination=termination,
      rewards=rewards,
      values=baseline,
      bootstrap_value=bootstrap_value,
      lambda_=lambda_,
      discount=discounting)
  rho_s = jnp.exp(target_action_log_probs - behavior_action_log_probs)

  surrogate_loss1 = rho_s * advantages
  surrogate_loss2 = jnp.clip(rho_s, 1 - ppo_epsilon,
                             1 + ppo_epsilon) * advantages

  policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

  # Value function loss
  v_error = vs - baseline
  value_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

  # Entropy reward
  entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
  entropy_loss = entropy_cost * -entropy

  total_loss = policy_loss + value_loss + entropy_loss

  # Additional losses
  extra_losses = {}
  if extra_loss_fns:
    for key, loss_fn in extra_loss_fns.items():
      loss, rng = loss_fn(data=data, udata=udata, rng=rng, params=extra_params)
      if extra_loss_update_ratios and key in extra_loss_update_ratios:
        # enable loss gradient p*100 percent of the time
        rng, key_update = jax.random.split(rng)
        p = extra_loss_update_ratios[key]
        b = jax.random.bernoulli(key_update, p=jnp.array(p))
        loss = jnp.where(b, loss, jax.lax.stop_gradient(loss))
      total_loss += loss
      extra_losses[key] = loss

  return total_loss, dict(
      extra_losses, **{
          'total_loss': total_loss,
          'policy_loss': policy_loss,
          'value_loss': value_loss,
          'entropy_loss': entropy_loss,
      })


def train(environment_fn: Callable[..., envs.Env],
          num_timesteps,
          episode_length: int,
          action_repeat: int = 1,
          num_envs: int = 1,
          max_devices_per_host: Optional[int] = None,
          num_eval_envs: int = 128,
          learning_rate=1e-4,
          entropy_cost=1e-4,
          discounting=0.9,
          seed=0,
          unroll_length=10,
          batch_size=32,
          num_minibatches=16,
          num_update_epochs=2,
          log_frequency=10,
          normalize_observations=False,
          reward_scaling=1.,
          progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
          parametric_action_distribution_fn: Optional[Callable[[
              int,
          ], distribution.ParametricDistribution]] = distribution
          .NormalTanhDistribution,
          make_models_fn: Optional[Callable[
              [int, int],
              Tuple[networks.FeedForwardModel]]] = networks.make_models,
          policy_params: Optional[Dict[str, jnp.ndarray]] = None,
          value_params: Optional[Dict[str, jnp.ndarray]] = None,
          extra_params: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
          extra_step_kwargs: bool = True,
          extra_loss_update_ratios: Optional[Dict[str, float]] = None,
          extra_loss_fns: Optional[Dict[str, Callable[[ppo.StepData],
                                                      jnp.ndarray]]] = None):
  """PPO training."""
  assert batch_size * num_minibatches % num_envs == 0
  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  
