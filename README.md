# Panda Pick-And-Place RL Project

This project trains a robot arm in PyBullet to push or place a small cube toward a target on a table.

The robot is a Franka Panda arm.
The learning algorithm is PPO from Stable-Baselines3.
The simulator is PyBullet.

## What The Project Does

The environment lives in `main.py`.

It creates:
- a table-like floor
- a Panda robot arm
- a small cube
- a target position

The agent sees numbers that describe the world.
Then it picks an action.
That action moves the robot hand a little bit and opens or closes the gripper.

The reward is designed to teach the agent in small steps:
- get the hand closer to the cube
- move the cube closer to the target
- get a bonus when the hand is very close
- get a big bonus when the cube reaches the goal

That matters because reinforcement learning works better when the agent gets clear feedback instead of only one final yes-or-no signal.

## Files

- `main.py`: the robot environment
- `train.py`: trains a PPO agent
- `eval.py`: runs a saved model and prints results
- `test_panda_env.py`: small tests for the environment
- `requirements.txt`: Python packages used by the project
- `panda_ppo_gripper.zip`: earlier trained model
- `panda_ppo_gripper_v2.zip`: improved trained model
- `smoke_model.zip`: tiny test model from a short training run

## How The Environment Works

The action has 8 numbers.

- The first 3 numbers move the robot hand in `x`, `y`, and `z`.
- The last number opens or closes the gripper.
- The middle 4 numbers are currently unused.

The observation has 16 numbers.

- 7 joint angles
- 3 cube position values
- 3 target position values
- 3 robot hand position values

In simple words:
the observation is the robot's "state of the world" in number form.

## Setup

This project was trained with a Conda environment named `bullet_env`.

```bash
conda activate bullet_env
```

That environment matters because `pybullet` is easier to run there than on newer Python versions.

## Install Packages

If you already have the Conda environment, the main packages are:

```bash
python -m pip install torch stable-baselines3 shimmy pytest
```

If you want to install the packages listed in `requirements.txt`, run:

```bash
python -m pip install -r requirements.txt
```

## Train The Agent

Basic training command:

```bash
MPLCONFIGDIR=/tmp/matplotlib python train.py --timesteps 200000 --save-path panda_ppo_gripper_v2
```

What this means:
- `MPLCONFIGDIR=/tmp/matplotlib` avoids a matplotlib cache warning
- `train.py` starts PPO training
- `--timesteps 200000` means how long the agent practices
- `--save-path panda_ppo_gripper_v2` is the output model name

Important idea:
more timesteps means more practice, but not guaranteed better results.
The right way to know is to train different checkpoints and compare them with evaluation.

## Test A Trained Agent

To evaluate the improved model:

```bash
MPLCONFIGDIR=/tmp/matplotlib python eval.py --model-path panda_ppo_gripper_v2 --episodes 20
```

What `eval.py` does:
- loads a trained model
- runs full episodes
- prints reward, steps, success, and final cube-to-target distance
- prints a final summary

## Current Result

The earlier model had a `0.00%` success rate in a short evaluation run.

The improved model `panda_ppo_gripper_v2` reached about:

- `15.00%` success rate over 20 evaluation episodes
- `0.1148` average final distance

This means the new setup is not solved yet, but it is clearly better than before.

## Run Tests

```bash
MPLCONFIGDIR=/tmp/matplotlib python -m pytest -q test_panda_env.py
```

At the time of writing, the environment tests passed with:

```text
6 passed, 2 skipped
```

## Good Next Steps

If you want better results, the most useful next things to try are:

1. Train longer, like `500000` or `1000000` timesteps.
2. Add curriculum learning so early episodes are easier.
3. Add a GUI mode so you can watch the robot and understand failure cases.
4. Save checkpoints during training and compare them with `eval.py`.

## Simple Mental Model

You can think about this project like teaching by trial and error:

- the environment is the world
- the observation is what the robot "sees"
- the action is what the robot "does"
- the reward is how the robot is graded
- PPO is the learning rule that slowly changes the policy

If the reward teaches the right little steps, the agent usually learns faster.
