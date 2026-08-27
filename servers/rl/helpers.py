"""
Reinforcement Learning Helpers MCP Server
Gym env listing, info, rollout templates, return computation, policy eval stubs.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from servers.common import create_server, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "rl-helpers",
    "Reinforcement learning utilities: list Gym environments, env info, "
    "rollout / return templates, and policy-eval scaffolding.",
)


@mcp.tool()
def list_gym_envs(limit: int = 20) -> str:
    """List available Gymnasium / classic Gym environment IDs (if installed)."""
    try:
        import gymnasium as gym
        envs = list(gym.envs.registry.keys())[:limit]
        return safe_json({"source": "gymnasium", "envs": envs, "count": len(envs)})
    except ImportError:
        try:
            import gym
            envs = list(gym.envs.registry.env_specs.keys())[:limit]
            return safe_json({"source": "gym", "envs": envs, "count": len(envs)})
        except ImportError:
            return safe_json({
                "error": "Neither gymnasium nor gym installed",
                "hint": "pip install gymnasium",
                "popular": ["CartPole-v1", "LunarLander-v2", "MountainCar-v0", "Pendulum-v1", "HalfCheetah-v4"],
            })


@mcp.tool()
def env_info(env_id: str = "CartPole-v1") -> str:
    """Return observation/action space info for an environment (requires gymnasium)."""
    try:
        import gymnasium as gym
        env = gym.make(env_id)
        info = {
            "env_id": env_id,
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
            "reward_range": getattr(env, "reward_range", None),
        }
        env.close()
        return safe_json(info)
    except Exception as e:
        return safe_json({"error": str(e), "env_id": env_id})


@mcp.tool()
def create_rollout_template(env_id: str = "CartPole-v1", max_steps: int = 500) -> str:
    """Return a ready-to-run random-policy rollout snippet."""
    code = f'''import gymnasium as gym
env = gym.make("{env_id}")
obs, info = env.reset()
total_reward = 0.0
for t in range({max_steps}):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break
env.close()
print("steps", t + 1, "return", total_reward)
'''
    return safe_json({"env_id": env_id, "code": code})


@mcp.tool()
def compute_returns(rewards: List[float], gamma: float = 0.99) -> str:
    """Compute discounted returns from a reward sequence (G_t)."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return safe_json({"returns": returns, "gamma": gamma, "total_undiscounted": sum(rewards)})


@mcp.tool()
def policy_eval_template(env_id: str = "CartPole-v1") -> str:
    """Scaffold for a simple Monte-Carlo policy evaluation loop."""
    code = f'''import gymnasium as gym
import numpy as np

env = gym.make("{env_id}")
n_episodes = 50
returns = []
for _ in range(n_episodes):
    obs, info = env.reset()
    ep_ret = 0.0
    done = False
    while not done:
        action = env.action_space.sample()  # replace with your policy
        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += reward
        done = terminated or truncated
    returns.append(ep_ret)
env.close()
print("mean return", np.mean(returns), "std", np.std(returns))
'''
    return safe_json({"env_id": env_id, "code": code})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
