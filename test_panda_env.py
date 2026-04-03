"""Tests for PandaEnv.step return types and observation dimension.

Further tests worth adding later (heavier / flaky in CI):
- Long rollout: cube lifted and placed without persistent penetration (needs success policy).
- Contact impulse: peak contact force stays below a threshold when closing gripper.
- Curriculum: randomize friction / mass and assert env still steps without NaNs.
"""

import numpy as np
import pytest

pytest.importorskip("pybullet")

import pybullet as p

from main import ACTION_DIM, OBS_DIM, PandaEnv


def test_step_returns_tuple_with_expected_types_and_obs_shape():
    env = PandaEnv()
    obs, _ = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    out = env.step(action)
    assert len(out) == 5

    observation, reward, terminated, truncated, info = out

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (OBS_DIM,)
    assert observation.dtype == np.float32

    assert isinstance(reward, float)

    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    assert isinstance(info, dict)


def test_truncated_after_max_steps():
    env = PandaEnv(max_episode_steps=3)
    env.reset(seed=0)
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    _, _, _, truncated, _ = env.step(a)
    assert not truncated
    env.step(a)
    _, _, _, truncated, _ = env.step(a)
    assert truncated


def test_action_wrong_shape_raises():
    env = PandaEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros(ACTION_DIM - 1, dtype=np.float32))


def test_reward_matches_negative_cube_to_target_distance():
    env = PandaEnv()
    env.reset(seed=2)
    obs, r, _, _, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    cube = obs[7:10]
    expected = float(-np.linalg.norm(cube - env.target_pos))
    assert abs(r - expected) < 1e-4


def test_gripper_fingers_reset_open():
    env = PandaEnv()
    env.reset(seed=0)
    if not env._finger_joints:
        pytest.skip("no finger joints in URDF")
    span = np.mean(
        [
            p.getJointState(env.robot_id, fi, physicsClientId=env.client)[0]
            for fi in env._finger_joints
        ]
    )
    assert span > 0.02


def test_gripper_action_open_close_changes_finger_positions():
    env = PandaEnv()
    env.reset(seed=0)
    if not env._finger_joints:
        pytest.skip("no finger joints in URDF")

    def mean_fingers():
        return float(
            np.mean(
                [
                    p.getJointState(env.robot_id, fi, physicsClientId=env.client)[0]
                    for fi in env._finger_joints
                ]
            )
        )

    open0 = mean_fingers()

    # Close gripper (action[7] = +1)
    a_close = np.zeros(ACTION_DIM, dtype=np.float32)
    a_close[7] = 1.0
    for _ in range(4):
        env.step(a_close)
    close = mean_fingers()

    # Re-open gripper (action[7] = -1)
    a_open = np.zeros(ACTION_DIM, dtype=np.float32)
    a_open[7] = -1.0
    for _ in range(4):
        env.step(a_open)
    open1 = mean_fingers()

    assert close < open0 - 0.005
    assert open1 > close + 0.005


def test_cube_linear_speed_bounded_after_small_action():
    env = PandaEnv()
    env.reset(seed=0)
    a = np.full(ACTION_DIM, 0.02, dtype=np.float32)
    env.step(a)
    lv, _ = p.getBaseVelocity(env.cube_id, physicsClientId=env.client)
    assert np.linalg.norm(lv) < 10.0
