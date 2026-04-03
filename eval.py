import argparse

import numpy as np
from stable_baselines3 import PPO

from main import PandaEnv


def run_one_episode(model, env, episode_seed):
    obs, _info = env.reset(seed=episode_seed)

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = env.step(action)

        total_reward += float(reward)
        step_count += 1

        if terminated or truncated:
            break

    cube_pos = obs[7:10]
    target_pos = obs[10:13]
    final_distance = float(np.linalg.norm(cube_pos - target_pos))
    success = bool(terminated)

    return {
        "total_reward": total_reward,
        "step_count": step_count,
        "success": success,
        "final_distance": final_distance,
        "timed_out": bool(truncated and not terminated),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="panda_ppo_gripper")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = PandaEnv(max_episode_steps=args.max_episode_steps)
    model = PPO.load(args.model_path)

    results = []

    for episode_index in range(args.episodes):
        episode_seed = args.seed + episode_index
        result = run_one_episode(model, env, episode_seed)
        results.append(result)

        print(
            f"Episode {episode_index + 1}: "
            f"reward={result['total_reward']:.3f}, "
            f"steps={result['step_count']}, "
            f"success={result['success']}, "
            f"final_distance={result['final_distance']:.4f}, "
            f"timed_out={result['timed_out']}"
        )

    avg_reward = float(np.mean([result["total_reward"] for result in results]))
    avg_steps = float(np.mean([result["step_count"] for result in results]))
    avg_final_distance = float(np.mean([result["final_distance"] for result in results]))
    success_rate = float(np.mean([result["success"] for result in results]))

    print()
    print("Summary")
    print(f"average_reward={avg_reward:.3f}")
    print(f"average_steps={avg_steps:.1f}")
    print(f"average_final_distance={avg_final_distance:.4f}")
    print(f"success_rate={success_rate:.2%}")


if __name__ == "__main__":
    main()
