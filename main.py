import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

ACTION_DIM = 8
OBS_DIM = 16
SUCCESS_DISTANCE = 0.05


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
        self._ee_orientation = p.getQuaternionFromEuler(
            [np.pi, 0.0, 0.0], physicsClientId=self.client
        )
        self._ee_low = np.array([0.30, -0.30, 0.03], dtype=np.float32)
        self._ee_high = np.array([0.72, 0.30, 0.30], dtype=np.float32)
        self._ee_target_pos = np.array([0.5, 0.0, 0.18], dtype=np.float32)
        self._prev_cube_to_target = 0.0
        self._prev_ee_to_cube = 0.0

        # 8-D action:
        # - first 3 numbers move the hand in x/y/z
        # - next 4 are reserved for future use
        # - last number opens/closes the gripper
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

    def _set_gripper(self, gripper_cmd):
        finger_target = float(0.02 * (1.0 - gripper_cmd))  # -1=open, +1=closed

        for fi in self._finger_joints:
            p.setJointMotorControl2(
                self.robot_id,
                fi,
                p.POSITION_CONTROL,
                targetPosition=finger_target,
                positionGain=0.2,
                velocityGain=0.95,
                force=60.0,
                physicsClientId=self.client,
            )

    def _move_ee_to(self, target_pos):
        target_pos = np.clip(target_pos, self._ee_low, self._ee_high).astype(np.float32)
        self._ee_target_pos = target_pos

        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            self._ee_link,
            targetPosition=target_pos.tolist(),
            targetOrientation=self._ee_orientation,
            physicsClientId=self.client,
        )

        for k, ji in enumerate(self._arm_joints):
            tgt_q = float(np.clip(joint_targets[k], self.joint_low[k], self.joint_high[k]))
            p.setJointMotorControl2(
                self.robot_id,
                ji,
                p.POSITION_CONTROL,
                targetPosition=tgt_q,
                positionGain=0.2,
                velocityGain=1.0,
                force=120.0,
                physicsClientId=self.client,
            )

    def _sample_cube_and_target_xy(self):
        for _ in range(256):
            cube_xy = self.np_random.uniform(self.xy_low, self.xy_high)
            angle = float(self.np_random.uniform(-np.pi, np.pi))
            distance = float(self.np_random.uniform(0.08, 0.18))
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            target_xy = cube_xy + distance * direction

            if np.any(target_xy < self.xy_low) or np.any(target_xy > self.xy_high):
                continue

            if np.linalg.norm(cube_xy) < 0.20:
                continue

            return cube_xy.astype(np.float32), target_xy.astype(np.float32)

        return (
            np.array([0.50, 0.00], dtype=np.float32),
            np.array([0.62, 0.00], dtype=np.float32),
        )

    def _reset_robot_pose(self, cube_xy, target_xy):
        direction = target_xy - cube_xy
        direction_norm = float(np.linalg.norm(direction))

        if direction_norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / direction_norm

        start_xy = cube_xy - 0.06 * direction
        start_pos = np.array([start_xy[0], start_xy[1], 0.12], dtype=np.float32)
        start_pos = np.clip(start_pos, self._ee_low, self._ee_high)

        self._move_ee_to(start_pos)
        self._set_gripper(-1.0)

        for _ in range(30):
            p.stepSimulation(physicsClientId=self.client)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        c_xy, t_xy = self._sample_cube_and_target_xy()

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

        self._reset_robot_pose(c_xy, t_xy)

        cube = self._cube_position()
        ee = self._ee_position()
        self._prev_cube_to_target = float(np.linalg.norm(cube - self.target_pos))
        self._prev_ee_to_cube = float(np.linalg.norm(ee - cube))

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
        delta_xyz = 0.03 * a[:3]
        gripper_cmd = float(np.clip(a[7], -1.0, 1.0))

        self._move_ee_to(self._ee_target_pos + delta_xyz)
        self._set_gripper(gripper_cmd)

        for _ in range(12):
            p.stepSimulation(physicsClientId=self.client)

        self._step_count += 1
        obs = self._get_obs()
        cube = obs[7:10]
        ee = obs[13:16]
        cube_to_target = float(np.linalg.norm(cube - self.target_pos))
        ee_to_cube = float(np.linalg.norm(ee - cube))

        cube_progress = self._prev_cube_to_target - cube_to_target
        ee_progress = self._prev_ee_to_cube - ee_to_cube
        contact_bonus = 0.25 if ee_to_cube < 0.06 else 0.0
        success_bonus = 25.0 if cube_to_target < SUCCESS_DISTANCE else 0.0

        reward = 0.0
        reward += -2.0 * cube_to_target
        reward += -0.5 * ee_to_cube
        reward += 12.0 * cube_progress
        reward += 2.0 * ee_progress
        reward += contact_bonus
        reward += success_bonus

        self._prev_cube_to_target = cube_to_target
        self._prev_ee_to_cube = ee_to_cube

        terminated = bool(cube_to_target < SUCCESS_DISTANCE)
        truncated = bool(self._step_count >= self.max_episode_steps)

        info = {
            "cube_to_target_distance": cube_to_target,
            "ee_to_cube_distance": ee_to_cube,
            "is_success": terminated,
        }

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        """State -> 16-D vector: q(7) | cube_xyz(3) | target_xyz(3) | ee_xyz(3)."""
        q = self._joint_positions()

        cube = self._cube_position()
        ee = self._ee_position()
        tgt = self.target_pos.astype(np.float32, copy=False)

        obs = np.concatenate([q, cube, tgt, ee])
        assert obs.shape == (OBS_DIM,)
        return obs.astype(np.float32)
