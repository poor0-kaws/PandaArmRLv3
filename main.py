import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

ACTION_DIM = 8
OBS_DIM = 16


class PandaEnv(gym.Env):
    """Franka Panda pick-and-place (incremental build)."""

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 500):
        super().__init__()

        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            useFixedBase=True,
            physicsClientId=self.client,
        )

        # Workspace in world frame: cube/target stay within typical Panda reach (base at origin).
        self.xy_low = np.array([0.28, -0.35], dtype=np.float32)
        self.xy_high = np.array([0.72, 0.35], dtype=np.float32)
        self.z_table = 0.02

        cube_xy = (self.xy_low + self.xy_high) / 2.0
        cube_pos = np.r_[cube_xy, self.z_table].astype(np.float32)
        self.cube_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=cube_pos.tolist(),
            physicsClientId=self.client,
        )

        # Goal pose (updated later in reset); must exist for _get_obs.
        self.target_pos = np.r_[cube_xy, self.z_table].astype(np.float32)
        self._ee_link = 11
        self._arm_joints = self._discover_arm_joints()
        self._finger_joints = self._discover_finger_joints()
        self._home_q = np.array([0.0, -0.5, 0.5, -2.4, 0.0, 1.6, 0.0], dtype=np.float64)

        # 8-D action: 7 arm deltas + 1 gripper open/close command.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )

        # 16-D observation: q(7) | cube(3) | target(3) | ee(3)
        self.joint_low = np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            dtype=np.float32,
        )
        self.joint_high = np.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            dtype=np.float32,
        )
        w_low = np.r_[self.xy_low, 0.0].astype(np.float32)
        w_high = np.r_[self.xy_high, 0.6].astype(np.float32)
        obs_low = np.concatenate([self.joint_low, w_low, w_low, w_low])
        obs_high = np.concatenate([self.joint_high, w_high, w_high, w_high])
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

    def _discover_arm_joints(self):
        arm = []
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            if info[2] != p.JOINT_REVOLUTE:
                continue
            name = info[1].decode()
            if "panda_joint" in name and "fixed" not in name:
                arm.append(i)
        arm.sort(
            key=lambda j: p.getJointInfo(self.robot_id, j, physicsClientId=self.client)[1].decode()
        )
        return arm[:7]

    def _discover_finger_joints(self):
        out = []
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            if info[2] != p.JOINT_REVOLUTE:
                continue
            if "finger" in info[1].decode():
                out.append(i)
        return sorted(out)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        min_sep, min_base_xy = 0.08, 0.15
        for _ in range(256):
            c_xy = self.np_random.uniform(self.xy_low, self.xy_high)
            t_xy = self.np_random.uniform(self.xy_low, self.xy_high)
            if np.linalg.norm(c_xy - t_xy) < min_sep:
                continue
            if np.linalg.norm(c_xy) < min_base_xy or np.linalg.norm(t_xy) < min_base_xy:
                continue
            break
        else:
            c_xy = np.array([0.5, 0.0], dtype=np.float32)
            t_xy = np.array([0.5, 0.12], dtype=np.float32)

        self.cube_pos = np.r_[c_xy, self.z_table].astype(np.float32)
        self.target_pos = np.r_[t_xy, self.z_table].astype(np.float32)

        p.resetBasePositionAndOrientation(
            self.cube_id,
            self.cube_pos.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client,
        )

        for ji, ang in zip(self._arm_joints, self._home_q):
            p.resetJointState(self.robot_id, ji, float(ang), physicsClientId=self.client)
        for fi in self._finger_joints:
            p.resetJointState(self.robot_id, fi, 0.04, physicsClientId=self.client)

        p.stepSimulation(physicsClientId=self.client)
        return self._get_obs(), {}

    def _joint_positions(self):
        q = np.empty(7, dtype=np.float32)
        for k, ji in enumerate(self._arm_joints):
            q[k] = p.getJointState(self.robot_id, ji, physicsClientId=self.client)[0]
        return q

    def _ee_position(self):
        pos = p.getLinkState(self.robot_id, self._ee_link, physicsClientId=self.client)[0]
        return np.array(pos, dtype=np.float32)

    def _cube_position(self):
        return np.array(
            p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self.client)[0],
            dtype=np.float32,
        )

    def step(self, action):
        if np.asarray(action).shape != (ACTION_DIM,):
            raise ValueError(
                f"action must have shape ({ACTION_DIM},), got {np.asarray(action).shape}"
            )

        a = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        arm_action = a[:7]
        gripper_cmd = float(np.clip(a[7], -1.0, 1.0))  # -1=open, +1=closed
        q = self._joint_positions()
        ee = self._ee_position()
        cube = self._cube_position()
        d_ee_cube = float(np.linalg.norm(ee - cube))

        # Smaller joint deltas when the hand is close to the cube (softer contact, less "slapping").
        blend = (
            1.0
            if d_ee_cube >= 0.12
            else max(0.25, min(1.0, d_ee_cube / 0.12))
        )
        scale = 0.05 * blend

        for k, ji in enumerate(self._arm_joints):
            tgt_q = float(
                np.clip(
                    q[k] + scale * arm_action[k],
                    self.joint_low[k],
                    self.joint_high[k],
                )
            )
            p.setJointMotorControl2(
                self.robot_id,
                ji,
                p.POSITION_CONTROL,
                targetPosition=tgt_q,
                positionGain=0.18,
                velocityGain=0.95,
                force=80.0,
                physicsClientId=self.client,
            )

        # Gripper: learned open/close. -1=open (~0.04), +1=closed (~0.0).
        finger_target = float(0.02 * (1.0 - gripper_cmd))  # in [0.0, 0.04]
        for fi in self._finger_joints:
            p.setJointMotorControl2(
                self.robot_id,
                fi,
                p.POSITION_CONTROL,
                targetPosition=finger_target,
                positionGain=0.15,
                velocityGain=0.9,
                force=35.0,
                physicsClientId=self.client,
            )

        for _ in range(8):
            p.stepSimulation(physicsClientId=self.client)

        self._step_count += 1
        obs = self._get_obs()
        cube_n = obs[7:10]
        reward = float(-np.linalg.norm(cube_n - self.target_pos))

        terminated = bool(np.linalg.norm(cube_n - self.target_pos) < 0.05)
        truncated = bool(self._step_count >= self.max_episode_steps)

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """State -> 16-D vector: q(7) | cube_xyz(3) | target_xyz(3) | ee_xyz(3)."""
        q = self._joint_positions()

        cube = self._cube_position()
        ee = self._ee_position()
        tgt = self.target_pos.astype(np.float32, copy=False)

        obs = np.concatenate([q, cube, tgt, ee])
        assert obs.shape == (OBS_DIM,)
        return obs.astype(np.float32)
