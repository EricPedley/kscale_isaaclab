# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train an RL agent with PufferLib.

Reads agent configuration from INI files (puffer_cfg_entry_point in gym registry).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with PufferLib.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the agent config entry point key. Defaults to None, which uses "
        "--algorithm to select 'puffer_cfg_entry_point' or 'puffer_{algorithm}_cfg_entry_point'."
    ),
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO"],
    help="The RL algorithm to use.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="Number of policy update iterations.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import pufferlib
import pufferlib.pytorch
from pufferlib import pufferl

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "puffer_cfg_entry_point" if algorithm == "ppo" else f"puffer_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("puffer_")[-1].lower()


class IsaacLabPufferEnv(pufferlib.PufferEnv):
    """PufferLib-compatible wrapper around an Isaac Lab vectorized environment.

    Isaac Lab envs:
    - step(actions: torch.Tensor) -> (obs_dict, reward, terminated, truncated, info)
      where obs_dict["policy"] is shape (num_envs, obs_dim)
    - reset() -> (obs_dict, info)
    - Auto-reset episodes internally; no manual per-episode reset needed.

    PufferLib expects:
    - single_observation_space, single_action_space, num_agents set before super().__init__()
    - reset(seed) -> (obs_numpy, infos_list)
    - step(actions_numpy) -> (obs_numpy, rewards, terminals, truncations, infos_list)
    - PufferEnv.__init__ allocates self.observations, self.rewards, etc. as numpy buffers.
    """

    def __init__(self, env, device: str = "cuda"):
        self._env = env
        self._device = device
        self._isaac_env = env.unwrapped

        # Resolve spaces from Isaac Lab env
        obs_space = self._isaac_env.single_observation_space["policy"]
        try:
            act_space = self._isaac_env.single_action_space
        except AttributeError:
            act_space = self._isaac_env.action_space

        self.single_observation_space = obs_space
        self.single_action_space = act_space
        self.num_agents = self._isaac_env.num_envs
        self.num_envs = self._isaac_env.num_envs

        # PufferEnv.__init__ allocates self.observations, self.rewards, self.terminals, self.truncations
        super().__init__()

        self._initialized = False

    def reset(self, seed=None):
        obs_dict, info = self._env.reset()
        self._initialized = True
        obs = self._extract_obs(obs_dict)
        self.observations[:] = obs
        return self.observations, [info] * self.num_envs

    def step(self, actions):
        # PufferLib passes numpy actions; convert to torch tensor on sim device
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self._device, dtype=torch.float32)

        obs_dict, reward, terminated, truncated, info = self._env.step(actions)

        obs = self._extract_obs(obs_dict)
        self.observations = obs
        self.rewards = reward.flatten()
        self.terminals = terminated.flatten()
        self.truncations = truncated.flatten()

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            [info] * self.num_envs,
        )

    def _extract_obs(self, obs_dict):
        """Extract policy observations as a contiguous float32 numpy array."""
        if isinstance(obs_dict, dict):
            obs_tensor = obs_dict["policy"]
        else:
            obs_tensor = obs_dict
        return obs_tensor.cpu().numpy().astype(np.float32)

    def close(self):
        self._env.close()


class Policy(nn.Module):
    """MLP policy for continuous control, compatible with PufferLib's PPO trainer.

    forward_eval must return (distribution, value) where distribution is
    torch.distributions.Normal for continuous action spaces, so PufferLib's
    sample_logits() can compute log-probs and entropy correctly.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_layers: list):
        super().__init__()

        def make_mlp():
            layers = []
            in_dim = obs_dim
            for h in hidden_layers:
                layers.append(pufferlib.pytorch.layer_init(nn.Linear(in_dim, h)))
                layers.append(nn.ELU())
                in_dim = h
            return nn.Sequential(*layers), in_dim

        self.actor_net, actor_out_dim = make_mlp()
        self.critic_net, critic_out_dim = make_mlp()

        self.action_mean = pufferlib.pytorch.layer_init(nn.Linear(actor_out_dim, act_dim), std=0.01)
        self.action_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(critic_out_dim, 1), std=1.0)

    def forward_eval(self, observations, state=None):
        actor_hidden = self.actor_net(observations)
        mean = self.action_mean(actor_hidden)
        std = torch.exp(self.action_logstd.expand_as(mean))
        dist = torch.distributions.Normal(mean, std)
        critic_hidden = self.critic_net(observations)
        value = self.value_head(critic_hidden)
        return dist, value

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


def build_train_config(agent_cfg: dict, env: "IsaacLabPufferEnv", args_cli) -> dict:
    """Build a PuffeRL train config dict from INI agent_cfg and CLI overrides."""
    train_cfg = dict(agent_cfg.get("train", {}))

    train_cfg['env'] = 'isaaclab' # TODO: get name of task
    # PufferLib required flag
    train_cfg["use_rnn"] = False

    # Device
    if args_cli.device is not None:
        train_cfg["device"] = args_cli.device
    elif "device" not in train_cfg:
        train_cfg["device"] = "cuda"

    # Seed
    if args_cli.seed is not None:
        train_cfg["seed"] = args_cli.seed
    elif "seed" not in train_cfg:
        train_cfg["seed"] = 42

    # Total timesteps: override via max_iterations * rollout_steps * num_envs
    rollout_steps = train_cfg.get("rollout_steps", 16)
    mini_batches = train_cfg.get("mini_batches", 2)
    if args_cli.max_iterations is not None:
        train_cfg["total_timesteps"] = args_cli.max_iterations * rollout_steps * env.num_envs
    elif "total_timesteps" not in train_cfg:
        train_cfg["total_timesteps"] = 10_000_000

    # Batch / minibatch sizes
    if train_cfg.get("batch_size") in ("auto", None):
        train_cfg["batch_size"] = env.num_envs * rollout_steps
    if train_cfg.get("minibatch_size") in ("auto", None):
        train_cfg["minibatch_size"] = max(1, (env.num_envs * rollout_steps) // mini_batches)
    if train_cfg.get("bptt_horizon") in ("auto", None):
        train_cfg["bptt_horizon"] = rollout_steps

    # PuffeRL required keys with defaults if not in INI
    train_cfg.setdefault("max_minibatch_size", 32768)
    train_cfg.setdefault("torch_deterministic", True)
    train_cfg.setdefault("cpu_offload", False)
    train_cfg.setdefault("precision", "float32")
    train_cfg.setdefault("compile", False)
    train_cfg.setdefault("compile_mode", "max-autotune-no-cudagraphs")
    train_cfg.setdefault("compile_fullgraph", True)
    train_cfg.setdefault("anneal_lr", True)
    train_cfg.setdefault("prio_alpha", 0.0)
    train_cfg.setdefault("prio_beta0", 1.0)
    train_cfg.setdefault("vtrace_rho_clip", 1.0)
    train_cfg.setdefault("vtrace_c_clip", 1.0)
    train_cfg.setdefault("checkpoint_interval", 200)
    train_cfg.setdefault("data_dir", "experiments")
    train_cfg.setdefault("adam_beta1", 0.9)
    train_cfg.setdefault("adam_beta2", 0.999)
    train_cfg.setdefault("adam_eps", 1e-8)

    # Remove INI-only keys not expected by PuffeRL
    for key in ("rollout_steps", "mini_batches"):
        train_cfg.pop(key, None)

    return train_cfg


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with PufferLib PPO agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training is not supported with CPU. Use --device cuda.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # seed
    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.get("base", {}).get("seed", 42)
    if seed == -1:
        seed = random.randint(0, 10000)
    env_cfg.seed = seed

    # log directory
    exp_dir = agent_cfg.get("experiment", {}).get("directory", "puffer")
    log_root_path = os.path.abspath(os.path.join("logs", "puffer", exp_dir))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    exp_name_base = agent_cfg.get("experiment", {}).get("experiment_name", "")
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir_name}")
    if exp_name_base:
        log_dir_name += f"_{exp_name_base}"
    log_dir = os.path.join(log_root_path, log_dir_name)

    # dump env config
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    # checkpoint resume path
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    env_cfg.log_dir = log_dir

    # create Isaac Lab environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert MARL to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap Isaac Lab env as PufferLib-compatible env
    sim_device = env_cfg.sim.device
    puffer_env = IsaacLabPufferEnv(env, device=sim_device)

    # build PuffeRL training config from INI sections
    train_config = build_train_config(agent_cfg, puffer_env, args_cli)

    # build policy
    obs_dim = puffer_env.single_observation_space.shape[0]
    act_dim = puffer_env.single_action_space.shape[0]
    hidden_layers = agent_cfg.get("policy", {}).get("hidden_layers", [256, 128, 64])
    train_device = train_config["device"]
    policy = Policy(obs_dim, act_dim, hidden_layers).to(train_device)

    print(f"[INFO] Obs dim: {obs_dim}  |  Action dim: {act_dim}  |  Hidden: {hidden_layers}")
    print(f"[INFO] Num envs: {puffer_env.num_envs}")
    print(f"[INFO] Total timesteps: {train_config['total_timesteps']}")
    print(f"[INFO] Batch size: {train_config['batch_size']}  |  Minibatch: {train_config['minibatch_size']}")

    # load checkpoint if specified
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=train_device)
        state_dict = ckpt["policy_state_dict"] if "policy_state_dict" in ckpt else ckpt
        policy.load_state_dict(state_dict)

    start_time = time.time()

    # create PuffeRL trainer
    trainer = pufferl.PuffeRL(train_config, puffer_env, policy)

    write_interval = agent_cfg.get("experiment", {}).get("write_interval", 100)

    # training loop
    try:
        while trainer.global_step < train_config["total_timesteps"]:
            trainer.evaluate()
            logs = trainer.train()

            if logs and trainer.global_step % (train_config["batch_size"] * write_interval) < train_config["batch_size"]:
                print(f"[INFO] Step {trainer.global_step}/{train_config['total_timesteps']}")
                if "environment/episode_reward_mean" in logs:
                    print(f"  Mean episode reward: {logs['environment/episode_reward_mean']:.3f}")
    finally:
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")

        os.makedirs(log_dir, exist_ok=True)
        ckpt_path = os.path.join(log_dir, "model.pt")
        torch.save(
            {
                "policy_state_dict": policy.state_dict(),
                "global_step": trainer.global_step,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "hidden_layers": hidden_layers,
            },
            ckpt_path,
        )
        print(f"[INFO] Saved model checkpoint to: {ckpt_path}")

        trainer.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
