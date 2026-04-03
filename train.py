import argparse

import numpy as np

from main import PandaEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--save-path", type=str, default="panda_ppo_gripper")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env():
        env = PandaEnv(max_episode_steps=args.max_episode_steps)
        env.reset(seed=args.seed)
        return env

    vec_env = DummyVecEnv([make_env])

    # PPO works well for continuous actions (Box spaces).
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)
    print(f"Saved model to: {args.save_path}")


if __name__ == "__main__":
    main()
