# -*- coding: utf-8 -*-
"""amorpheus.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16708CGp_J71HRATiaOKqRlfyHSqVGN-A

# Training Mutual Information Maximization (MI-Max) RL algorithms in Brax

In [Brax Training](https://colab.research.google.com/github/google/brax/blob/main/notebooks/training.ipynb) we tried out [gym](https://gym.openai.com/)-like environments and PPO, SAC, evolutionary search, and trajectory optimization algorithms. We can build various RL algorithms on top of these ultra-fast implementations. This colab runs a family of [variational GCRL](https://arxiv.org/abs/2106.01404) algorithms or MI-maximization (MI-max) algorithms, which include [goal-conditioned RL](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.3077) and [DIAYN](https://arxiv.org/abs/1802.06070) as special cases. Let's try it out!

This provides a bare bone implementation based on minimal modifications to the
baseline [PPO](https://github.com/google/brax/blob/main/brax/training/ppo.py),
enabling training in a few minutes. More features, examples, and benchmarked results will be added.

```
# This is formatted as code
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/mimax.ipynb)
"""

#@title Colab setup and imports
#@markdown ## ⚠️ PLEASE NOTE:
#@markdown This colab runs best using a TPU runtime.  From the Colab menu, choose Runtime > Change Runtime Type, then select **'TPU'** in the dropdown.

from brax.envs.env import Wrapper
from datetime import datetime
import functools
import math
import os
import pprint

import jax
import jax.numpy as jnp
# from IPython.display import HTML, clear_output
import matplotlib.pyplot as plt

# try:
#   import brax
# except ImportError:
#   !pip install git+https://github.com/google/brax.git@main
#   clear_output()
#   import brax

# try:
#   import haiku as hk
# except ImportError:
#   !pip install git+https://github.com/deepmind/dm-haiku
#   clear_output()

import haiku as hk

from brax.io import file as io_file
from brax.io import html
from brax.experimental.composer import composer
from brax.experimental.composer import register_default_components
from brax.experimental.braxlines.common import evaluators
from brax.experimental.braxlines.common import logger_utils
from brax.experimental.braxlines.training import ppo
from brax.experimental.braxlines.vgcrl import evaluators as vgcrl_evaluators
from brax.experimental.braxlines.vgcrl import utils as vgcrl_utils
from brax.training.networks import make_transformers


register_default_components()

import tensorflow_probability as tfp

tfp = tfp.substrates.jax
tfd = tfp.distributions

if "COLAB_TPU_ADDR" in os.environ:
  from jax.tools import colab_tpu
  colab_tpu.setup_tpu()


#@title Define task and experiment parameters

#@markdown **Task Parameters**
#@markdown 
#@markdown As in [DIAYN](https://arxiv.org/abs/1802.06070)
#@markdown and [VGCRL](https://arxiv.org/abs/2106.01404),
#@markdown we assume some task knowledge about interesting dimensions
#@markdown of the environment `obs_indices` and their range `obs_scale`.
#@markdown This is also used for evaluation and visualization.
#@markdown
#@markdown When the **task parameters** are the same, the metrics computed by
#@markdown [vgcrl/evaluators.py](https://github.com/google/brax/blob/main/brax/experimental/braxlines/vgcrl/evaluators.py)
#@markdown are directly comparable across experiment runs with different
#@markdown **experiment parameters**. 
env_name = 'mod_ant'  # @param ['ant', 'humanoid', 'halfcheetah', 'uni_ant', 'bi_ant']
obs_indices = 'vel'  # @param ['vel', 'delta_pos', 'delta_vel']
obs_scale = 10.0 #@param{'type': 'number'}
obs_indices_str = obs_indices
obs_indices = dict(
    vel=dict(
      ant = (13,14),
      humanoid = (22, 23),
      halfcheetah = (11,),
      uni_ant = (('body_vel:torso_ant1', 0),('body_vel:torso_ant1', 1)),
      bi_ant = (('body_vel:torso_ant1', 0),('body_vel:torso_ant2', 0)),
      mod_ant = (15, 14),
    ),
    delta_pos=dict(
      bi_ant = (('delta_pos', 0),('delta_pos', 1)),
    ),
    delta_vel=dict(
      bi_ant = (('delta_vel', 0),('delta_vel', 1)),
    ),
)[obs_indices][env_name]

#@markdown **Experiment Parameters**
#@markdown See [vgcrl/utils.py](https://github.com/google/brax/blob/main/brax/experimental/braxlines/vgcrl/utils.py)
evaluate_mi = False # @param{'type': 'boolean'}
evaluate_lgr = False # @param{'type': 'boolean'}
algo_name = 'diayn'  # @param ['morph_diayn_full', 'gcrl', 'cdiayn', 'diayn', 'diayn_full', 'fixed_gcrl']
env_reward_multiplier =   0# @param{'type': 'number'}
obs_norm_reward_multiplier =   0# @param{'type': 'number'}
normalize_obs_for_disc = False  # @param {'type': 'boolean'}
normalize_obs_for_rl = True # @param {'type': 'boolean'}
seed =   0# @param {type: 'integer'}
diayn_num_skills = 8  # @param {type: 'integer'}
spectral_norm = True  # @param {'type': 'boolean'}
output_path = '' # @param {'type': 'string'}
task_name = "" # @param {'type': 'string'}
exp_name = '' # @param {'type': 'string'}
if output_path:
  output_path = output_path.format(
    date=datetime.now().strftime('%Y%m%d'))
  task_name = task_name or f'{env_name}_{obs_indices_str}_{obs_scale}'
  exp_name = exp_name or algo_name 
  output_path = f'{output_path}/{task_name}/{exp_name}'
print(f'output_path={output_path}')

# @title Initialize Brax environment
visualize = False # @param{'type': 'boolean'}

# Create baseline environment to get observation specs
base_env_fn = composer.create_fn(env_name=env_name)
base_env = base_env_fn()

# Create discriminator-parameterized environment
disc = vgcrl_utils.create_disc_fn(
  algo_name=algo_name,
  observation_size=base_env.observation_size,
  obs_indices=obs_indices,
  scale=obs_scale,
  diayn_num_skills=diayn_num_skills,
  spectral_norm=spectral_norm,
  env=base_env,
  normalize_obs=normalize_obs_for_disc
)()
extra_params = disc.init_model(
  rng=jax.random.PRNGKey(seed=seed)
)

# Create enviroment
env_fn = vgcrl_utils.create_modular_fn(
    env_name=env_name, 
    wrapper_params=dict(
      disc=disc, 
      env_reward_multiplier=env_reward_multiplier,
      obs_norm_reward_multiplier=obs_norm_reward_multiplier
    ),
)
# Create evaluation environment function
eval_env_fn = functools.partial(env_fn, auto_reset=False)

# Create training environment
core_env = env_fn()

# Make inference functions and goals for LGR metric

params, inference_fn = ppo.make_params_and_inference_fn(
    observation_size=core_env.observation_size, 
    action_size=core_env.action_size,
    normalize_observations=normalize_obs_for_rl,
    make_models_fn=make_transformers, 
    extra_params=extra_params
)
inference_fn = jax.jit(inference_fn)
goals = tfd.Uniform(
    low=-disc.obs_scale, 
    high=disc.obs_scale
).sample(
    seed=jax.random.PRNGKey(0), 
    sample_shape=(10,)
)


# Visualize
if visualize:
  env = env_fn()
  jit_env_reset = jax.jit(env.reset)
  state = jit_env_reset(rng=jax.random.PRNGKey(seed=seed))
  clear_output()  # clear out jax.lax warning before rendering
  HTML(html.render(env.sys, [state.qp]))

#@title Debugging inference with Transformer model
num_z = 5  # @param {type: 'integer'}
num_samples_per_z = 5  # @param {type: 'integer'}
time_subsampling = 10  # @param {type: 'integer'}
time_last_n = 500 # @param {type: 'integer'}
eval_seed = 0  # @param {type: 'integer'}

# with jax.disable_jit():
#   params, inference_fn = ppo.make_params_and_inference_fn(
#       observation_size=core_env.observation_size, 
#       action_size=core_env.action_size,
#       normalize_observations=normalize_obs_for_rl,
#       make_models_fn=make_transformer, 
#       extra_params=extra_params
#   )

#   vgcrl_evaluators.visualize_skills(
#     env_fn=eval_env_fn,
#     disc=disc,
#     inference_fn=inference_fn,
#     params=params,
#     output_path=output_path,
#     verbose=True,
#     num_z=num_z,
#     num_samples_per_z=num_samples_per_z,
#     time_subsampling=time_subsampling,
#     time_last_n=time_last_n,
#     save_video=True,
#     seed=eval_seed
# )

#@title Training
num_timesteps_multiplier =   6# @param {type: 'number'}
ncols = 5 # @param{type: 'integer'}

tab = logger_utils.Tabulator(output_path=f'{output_path}/training_curves.csv', append=False)

# We determined some reasonable hyperparameters offline and share them here.
n = num_timesteps_multiplier
if env_name == 'humanoid':
  train_fn = functools.partial(
    ppo.train,
    num_timesteps=int(50_000_000 * n),
    log_frequency=20,
    reward_scaling=0.1,
    episode_length=1000,
    normalize_observations=normalize_obs_for_rl,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=16,
    num_update_epochs=8,
    discounting=0.97,
    learning_rate=1e-4,
    entropy_cost=1e-3,
    num_envs=2048,
    batch_size=1024,
    make_models_fn=make_transformers,
) 
else:
  train_fn = functools.partial(
    ppo.train,
    num_timesteps=int(50_000_000 * n),
    log_frequency=20,
    reward_scaling=10,
    episode_length=1000,
    normalize_observations=normalize_obs_for_rl,
    action_repeat=1,
    unroll_length=5,
    num_minibatches=32,
    num_update_epochs=4,
    discounting=0.95,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=2048,
    batch_size=1024,
    make_models_fn=make_transformers,
)

times = [datetime.now()]
plotdata = {}
plotkeys = ['eval/episode_reward', 'losses/disc_loss', 'metrics/lgr',
            'metrics/entropy_all_', 'metrics/entropy_z_', 'metrics/mi_']

def plot(output_path:str =None, output_name:str = 'training_curves'):
  matched_keys = [key for key in sorted(plotdata.keys()) if any(plotkey in
                                                                key for plotkey in plotkeys)]
  num_figs = len(matched_keys)
  nrows = int(math.ceil(num_figs/ncols)) 
  fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3.5 * ncols, 3 * nrows))
  for i, key in enumerate(matched_keys):
    col, row = i % ncols, int(i/ncols)
    ax = axs
    if nrows > 1:
      ax = ax[row]
    if ncols > 1:
      ax = ax[col]
    ax.plot(plotdata[key]['x'], plotdata[key]['y'])
    ax.set(xlabel='# environment steps', ylabel=key)
    ax.set_xlim([0, train_fn.keywords['num_timesteps']])
  fig.tight_layout()
  if output_path:
    with io_file.File(f'{output_path}/{output_name}.png', 'wb') as f:
      plt.savefig(f)

def progress(num_steps, metrics, params):
  if evaluate_mi:
    mi_metrics = vgcrl_evaluators.estimate_empowerment_metric(
      env_fn=env_fn, disc=disc, inference_fn=inference_fn, params=params,
      # custom_obs_indices = list(range(core_env.observation_size))[:30],
      # custom_obs_scale = obs_scale,
    )
    metrics.update(mi_metrics)
  
  if evaluate_lgr:
    lgr_metrics = vgcrl_evaluators.estimate_latent_goal_reaching_metric( 
      params=params, env_fn=env_fn, disc=disc, inference_fn=inference_fn,
      goals=goals)
    metrics.update(lgr_metrics)
  
  times.append(datetime.now())
  for key, v in metrics.items():
    plotdata[key] = plotdata.get(key, dict(x=[], y=[]))
    plotdata[key]['x'] += [num_steps]
    plotdata[key]['y'] += [v]

  # the first step does not include losses
  if num_steps > 0:
    tab.add(num_steps=num_steps, **metrics)
    tab.dump()
  # clear_output(wait=True)
  plot()
  plt.pause(0.01)

plt.ion()
extra_loss_fns = dict(disc_loss=disc.disc_loss_fn) if extra_params else None
_, params, _ = train_fn(
    environment_fn=env_fn, progress_fn=progress, extra_params=extra_params,
    extra_loss_fns=extra_loss_fns,
)
# clear_output(wait=True)
plot(output_path=output_path)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

#@title Visualizing skills of the learned inference function in 2D plot
num_z = 5  # @param {type: 'integer'}
num_samples_per_z = 5  # @param {type: 'integer'}
time_subsampling = 10  # @param {type: 'integer'}
time_last_n = 500 # @param {type: 'integer'}
eval_seed = 0  # @param {type: 'integer'}

vgcrl_evaluators.visualize_skills(
    env_fn=eval_env_fn,
    disc=disc,
    inference_fn=inference_fn,
    params=params,
    output_path=output_path,
    verbose=True,
    num_z=num_z,
    num_samples_per_z=num_samples_per_z,
    time_subsampling=time_subsampling,
    time_last_n=time_last_n,
    save_video=True,
    seed=eval_seed
)
plt.show()

# @title Estimate [Latent Goal Reaching metric](https://arxiv.org/abs/2106.01404)
num_samples_per_z =   10# @param {type: 'integer'}
time_subsampling = 1  # @param {type: 'integer'}
time_last_n = 500 # @param {type: 'integer'}
eval_seed = 0  # @param {type: 'integer'}


metrics = vgcrl_evaluators.estimate_latent_goal_reaching_metric( 
    params=params,
    env_fn = eval_env_fn,
    disc=disc,
    inference_fn=inference_fn,
    goals=goals,
    num_samples_per_z=num_samples_per_z,
    time_subsampling=time_subsampling,
    time_last_n=time_last_n,
    seed=eval_seed,
)
pprint.pprint(metrics)

#@title Estimate empowerment metrics using 1D/2D binning
num_z =   10# @param {type: 'integer'}
num_samples_per_z =   10# @param {type: 'integer'}
time_subsampling = 1  # @param {type: 'integer'}
time_last_n = 500 # @param {type: 'integer'}
eval_seed = 0  # @param {type: 'integer'
num_1d_bins = 1000  # @param {type: 'integer'}
num_2d_bins =   30# @param {type: 'integer'}

metrics = vgcrl_evaluators.estimate_empowerment_metric(
    env_fn=eval_env_fn,
    disc=disc,
    inference_fn=inference_fn,
    params=params,
    num_z=num_z,
    num_samples_per_z=num_samples_per_z,
    time_subsampling=time_subsampling,
    time_last_n=time_last_n,
    num_1d_bins = num_1d_bins,
    num_2d_bins = num_2d_bins,
    verbose = True,
    seed=eval_seed)
pprint.pprint(metrics)

#@title Visualizing a trajectory of the learned inference function
#@markdown If `z_value` is `None`, sample `z`, else fix `z` to `z_value`.
z_value =   0# @param {'type': 'raw'}
eval_seed = 0  # @param {'type': 'integer'}

z = {
    'fixed_gcrl': jnp.ones(disc.z_size) * z_value,
    'gcrl': jnp.ones(disc.z_size) * z_value,
    'cdiayn': jnp.ones(disc.z_size) * z_value,
    'diayn': jax.nn.one_hot(jnp.array(int(z_value)), disc.z_size),
    'diayn_full': jax.nn.one_hot(jnp.array(int(z_value)), disc.z_size),
}[algo_name] if z_value is not None else None

env, states = evaluators.visualize_env(
    env_fn=eval_env_fn,
    inference_fn=inference_fn,
    params=params,
    batch_size=0,
    seed = eval_seed,
    reset_args = (z,),
    step_args = (params['normalizer'], params['extra']),
    output_path=output_path,
    output_name=f'video_z_{z_value}',
)
HTML(html.render(env.sys, [state.qp for state in states]))