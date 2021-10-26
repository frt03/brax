# BIG-Gym

A Large-Scale Benchmark and Study of Deep RL Environments for Control (BIG-Gym) is a *crowd-sourcing* challenge for benchmark designs, inspired by [BIG-Bench](https://github.com/google/BIG-bench) in NLP.
We expect that the toolkits provided by BIG-Gym to lead to an **MNIST-moment in Deep RL for continuous control**: the proliferation of environments and datasets with sufficient-enough complexity and fast experimentation turnaround that enables exploring more insightful and diverse ideas in RL.
We are organizing two tracks: **Open-Ended Creativity Track** and **Goal-Oriented Competition Track**.

The benchmark organizers can be contacted at \<e-mail>.

**Table of contents**
* [Getting Started](#getting-started)
* [Open-Ended Creativity Track](#open-ended-creativity-track)
* [Goal-Oriented Competition Track](#goal-oriented-competition-track)
* [Frequently Asked Questions](#frequently-asked-questions)


## Getting Started
We recommend you to use Brax on Google Colab with free TPU. To use Brax locally, see [README](https://github.com/google/brax#using-brax-locally) of Brax.

**Introduction to Brax**

<a href="https://colab.research.google.com/github/google/brax/blob/main/notebooks/basics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

BIG-Gym utilizes [Brax](https://github.com/google/brax), a purely pythonic simulator which trains environments 100-1000x faster than many prior alternatives.

(add the link to the colab)


**Composer Basics**

We provide [Composer](https://github.com/google/brax/tree/main/brax/experimental/composer), a programmatic API for generating continuous control environments. Composer allows programmatic procedural generation of parameterized environments.

(add the link to the colab)


**Building & Submitting to BIG-Gym**

We also provide a template colab to build & submit to BIG-Gym.

First, [fork the repository](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) in GitHub!
<a href="https://docs.github.com/en/github/getting-started-with-github/fork-a-repo">
<div style="text-align:center"><img src="https://docs.github.com/assets/images/help/repository/fork_button.jpg" alt="fork button" width="500"/></div>
</a>

Your fork will have its own location, which we will call `PATH_TO_YOUR_FORK`.
Next, [clone the forked repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) and create a branch for your new submission, which here we will call **my_new_submission**:

```bash
git clone $PATH_TO_YOUR_FORK
cd brax/experimental/biggym/envs
git checkout -b my_new_submission
```

Create a new directory named as your team name `<team_name>`:
```bash
mkdir `<team_name>`
```
You can add your new benchmark or Composer components this directory.

(add the link to the colab)


For submission, you need to make [git pull requests](https://github.com/google/brax/pulls). Commit and push your changes:
```bash
git add brax/experimental/biggym/envs/<team_name>
git commit -m "Added my_new_submission"
git push --set-upstream origin my_new_submission
```
Finally, [submit a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).
The last `git push` command prints a URL that can be copied into a browser to initiate such a pull request.
Alternatively, you can do so from the GitHub website.
<a href="https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request">
<div style="text-align:center"><img src="https://docs.github.com/assets/images/help/pull_requests/pull-request-start-review-button.png" alt="pull request button" width="500"/></div>
</a>


## Open-Ended Creativity Track
In this track, you can design environments to realize *creative* behaviors. You can see the evaluation criteria on the website.

### Submission Format

You need to submit (1) code and (2) supprimental videos.

**Code Submission**

Submit a Brax env compatible with [Brax RL](https://colab.research.google.com/github/google/brax/blob/main/notebooks/training.ipynb), [Braxlines MI-Max](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/mimax.ipynb), and/or [Braxlines D-Min](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/dmin.ipynb), through [git pull requests](https://github.com/google/brax/pulls)!

Submitted entries will reside under `brax/experimental/biggym/envs/<team_name>/` directories and usable in the Colab Examples below as:

```python
import brax.experimental.biggym as gym
gym.register('<team_name>')
env = gym.make('<team_name>', **your_env_parameters)
```


**Supplementary Submission**

Submit up to 10 sample videos of generated behaviors through google form! One of submitted videos will be posted on Twitter.

<img src="https://github.com/google/brax/raw/main/docs/img/braxlines/ant_smm.gif" width="300" height="214"/><img src="https://github.com/google/brax/raw/main/docs/img/braxlines/humanoid_smm.gif" width="300" height="214"/>

(add link to google form & twitter)


### Submission Examples and Colabs
- [Brax RL](https://colab.research.google.com/github/google/brax/blob/main/notebooks/training.ipynb)
- [Braxlines MI-Max](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/mimax.ipynb)
- [Braxlines D-Min](https://colab.research.google.com/github/google/brax/blob/main/notebooks/braxlines/dmin.ipynb)

(need to make example colabs?)


## Goal-Oriented Competition Track
In this track, you can design a new Composer component that satisfies constraints, and competes in any of Race, Sumo, or Push tasks.

### Submission Format
You need to submit (1) code and (2) supprimental videos.

**Code Submission**

Submit a new Composer component that satisfies constraints, and competes in any of:
- **Task 1:** Race
- **Task 2:** Sumo
- **Task 3:** Push

We force the component to follow:
- Weight constraints
- Volume constraints
- Actuator energy constraints

Submitted entries will reside under `brax/experimental/biggym/envs/<team_name>/components/ ` directories and usable in the Colab Examples below as:
```python
import brax.experimental.biggym as gym
import brax.experimental.composer.composer as composer
gym.register_component('<team_name>', '<component_name>')
env = composer.create(env_desc=dict(components=dict(agent1=dict(component='<team_name>_<component_name>'))))
```


**Supplementary Submission**

(add explanation)


### Submission Examples and Colabs

(add link)


## Frequently Asked Questions
### Can I submit to each track?
Yes! We wellcome for you to submit your codes & videos to both tracks.

### Can I make multiple submissions per track?
Yes! Multiple submissions are allowed, while we encourage merge them into single submission per track.

### Can I make a pull request or contribute to Brax?
Yes! If you feel generic features or documentations are missing Brax, Braxlines, or Composer, feel free to directly add [git pull requests](https://github.com/google/brax/pulls) or open [issues](https://github.com/google/brax/issues) on Brax github tagging #biggym.

(add more)
