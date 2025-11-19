import logging
import os
import datetime

import hydra
import torch
import numpy as np
import wandb
from torch import vmap
from omegaconf import OmegaConf, DictConfig

from omni.isaac.kit import SimulationApp
from hcsp import CONFIG_PATH, init_simulation_app
from hcsp.utils.torchrl import SyncDataCollector, AgentSpec
from hcsp.utils.wandb import init_wandb
from hcsp.utils.psro.meta_solver import get_meta_solver
from hcsp.utils.psro.convergence import ConvergedIndicator
from hcsp.learning import PSROPolicy_coselfplay_phase_one, PSROPolicy

from setproctitle import setproctitle
from torchrl.envs.transforms import (
    TransformedEnv,
    InitTracker,
    Compose,
)

from tqdm import tqdm
from typing import Callable, Dict, Optional
import traceback
from tensordict import TensorDict

import matplotlib.pyplot as plt

import seaborn as sns

from typing import Union

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1


@torch.no_grad()
def evaluate(env: TransformedEnv, policy: PSROPolicy_coselfplay_phase_one):
    """
    Use the latest strategies of both teams to evaluate the policy
    """
    high_level_policy = policy.policy_high_level
    logging.info("***************************************")
    logging.info("Evaluating the latest policy v.s. the latest policy.")
    frames = []
    info={}

    def record_frame(*args, **kwargs):
        frame = env.base_env.render(mode="rgb_array")
        frames.append(frame)

    _should_render=env.base_env._should_render
    env.base_env.enable_render(True)
    env.reset()
    env.eval()
    high_level_policy.eval_payoff = True
    high_level_policy.set_latest_strategy()

    env.rollout(
        max_steps=env.base_env.max_episode_length,
        policy=lambda x: policy(x, deterministic=False),
        callback=Every(record_frame, 2),
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False,
    )

    if len(frames):
        # video_array = torch.stack(frames)
        video_array = np.stack(frames).transpose(0, 3, 1, 2)
        info["recording"] = wandb.Video(video_array, fps=0.5 / env.base_env.dt, format="mp4")
    
    env.base_env.enable_render(_should_render)
    env.reset()
    env.train()
    high_level_policy.eval_payoff = False
    frames.clear()

    return info


@torch.no_grad()
def evaluate_share_population(env: TransformedEnv, policy: PSROPolicy_coselfplay_phase_one):
    """
    Use the latest strategy and second latest strategy to evaluate the policy (Only used in share_population mode)
    """
    high_level_policy = policy.policy_high_level
    logging.info("***************************************")
    logging.info("Evaluating the latest policy (Team 0) v.s. the second latest policy (Team 1).")

    info={}
    frames = []

    def record_frame(*args, **kwargs):
        frame = env.base_env.render(mode="rgb_array")
        frames.append(frame)

    _should_render=env.base_env._should_render
    env.base_env.enable_render(True)
    env.reset()
    env.eval()
    high_level_policy.eval_payoff = True

    high_level_policy.population_0.set_latest_policy()
    high_level_policy.population_1.set_second_latest_policy()

    env.rollout(
        max_steps=env.base_env.max_episode_length,
        policy=lambda x: policy(x, deterministic=False),
        callback=Every(record_frame, 2),
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False,
    )

    if len(frames):
        # video_array = torch.stack(frames)
        video_array = np.stack(frames).transpose(0, 3, 1, 2)
        info["recording_team_0_latest_vs_team_1_second_latest"] = wandb.Video(
                video_array, fps=0.5 / env.base_env.dt, format="mp4"
            )
    
    logging.info("***************************************")
    logging.info("Evaluating the second latest policy (Team 0) v.s. the latest policy (Team 1).")
    frames.clear()
    frames = []
    env.reset()

    high_level_policy.population_0.set_second_latest_policy()
    high_level_policy.population_1.set_latest_policy()

    env.rollout(
        max_steps=env.base_env.max_episode_length,
        policy=lambda x: policy(x, deterministic=False),
        callback=Every(record_frame, 2),
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False,
    )

    if len(frames):
        # video_array = torch.stack(frames)
        video_array = np.stack(frames).transpose(0, 3, 1, 2)
        info["recording_team_0_second_latest_vs_team_1_latest"] = wandb.Video(
                video_array, fps=0.5 / env.base_env.dt, format="mp4"
            )
    
    env.base_env.enable_render(_should_render)
    env.reset()
    env.train()
    high_level_policy.eval_payoff = False
    frames.clear()

    return info


@torch.no_grad()
def evaluate_debug(env: TransformedEnv, policy: PSROPolicy_coselfplay_phase_one):
    """
    Team_current_idx: use the currently trained networks
    Team_the_other: use the latest strategy in the policy population
    """
    high_level_policy = policy.policy_high_level
    logging.info("***************************************")
    logging.info("Evaluating the policy being trained v.s. the latest policy.")
    frames = []
    info={}

    def record_frame(*args, **kwargs):
        frame = env.base_env.render(mode="rgb_array")
        frames.append(frame)

    _should_render=env.base_env._should_render
    env.base_env.enable_render(True)
    env.reset()
    env.eval()
    
    assert high_level_policy.eval_payoff == False
    assert high_level_policy.current_player_id == 0
    high_level_policy.set_latest_strategy()

    env.rollout(
        max_steps=env.base_env.max_episode_length,
        policy=lambda x: policy(x, deterministic=False),
        callback=Every(record_frame, 2),
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False,
    )

    if len(frames):
        # video_array = torch.stack(frames)
        video_array = np.stack(frames).transpose(0, 3, 1, 2)
        info["recording_debug"] = wandb.Video(video_array, fps=0.5 / env.base_env.dt, format="mp4")
    
    env.base_env.enable_render(_should_render)
    env.reset()
    env.train()
    # high_level_policy.eval_payoff = False
    frames.clear()

    return info


def calculate_jpc(payoffs: np.ndarray): 
    """
    Calculate JPC (Joint Payoff Convexity) of the game
    """
    assert len(payoffs.shape) == 2 and payoffs.shape[0] == payoffs.shape[1]
    n = payoffs.shape[0]
    assert n > 0
    d = np.trace(payoffs) / n
    o = (np.sum(payoffs) - n * d) / (n * (n - 1))
    r = (d - o) / d
    return r


def _get_psro_high_low_level_policy(cfg: DictConfig, env: TransformedEnv):
    algos = {
        "psro_coselfplay_phase_one": PSROPolicy_coselfplay_phase_one,
    }
    algo_name = cfg.algo.name.lower()
    if algo_name not in algos:
        raise RuntimeError(f"{algo_name} not supported.")
    
    Opp_SecPass_goto_agent_spec: AgentSpec = env.agent_spec["Opp_SecPass_goto"]
    Opp_SecPass_agent_spec: AgentSpec = env.agent_spec["Opp_SecPass"]
    Opp_SecPass_hover_agent_spec: AgentSpec = env.agent_spec["Opp_SecPass_hover"]

    Opp_Att_goto_agent_spec: AgentSpec = env.agent_spec["Opp_Att_goto"]
    Opp_Att_agent_spec: AgentSpec = env.agent_spec["Opp_Att"]
    Opp_Att_hover_agent_spec: AgentSpec = env.agent_spec["Opp_Att_hover"]

    FirstPass_goto_agent_spec: AgentSpec = env.agent_spec["FirstPass_goto"]
    FirstPass_agent_spec: AgentSpec = env.agent_spec["FirstPass"]
    FirstPass_hover_agent_spec: AgentSpec = env.agent_spec["FirstPass_hover"]
    FirstPass_serve_agent_spec: AgentSpec = env.agent_spec["FirstPass_serve"]
    FirstPass_serve_hover_agent_spec: AgentSpec = env.agent_spec["FirstPass_serve_hover"]
    FirstPass_receive_agent_spec: AgentSpec = env.agent_spec["FirstPass_receive"]
    FirstPass_receive_hover_agent_spec: AgentSpec = env.agent_spec["FirstPass_receive_hover"]

    SecPass_goto_agent_spec: AgentSpec = env.agent_spec["SecPass_goto"]
    SecPass_agent_spec: AgentSpec = env.agent_spec["SecPass"]
    SecPass_hover_agent_spec: AgentSpec = env.agent_spec["SecPass_hover"]

    Att_goto_agent_spec: AgentSpec = env.agent_spec["Att_goto"]
    Att_agent_spec: AgentSpec = env.agent_spec["Att"]
    Att_hover_agent_spec: AgentSpec = env.agent_spec["Att_hover"]

    Opp_FirstPass_goto_agent_spec: AgentSpec = env.agent_spec["Opp_FirstPass_goto"]
    Opp_FirstPass_agent_spec: AgentSpec = env.agent_spec["Opp_FirstPass"]
    Opp_FirstPass_hover_agent_spec: AgentSpec = env.agent_spec["Opp_FirstPass_hover"]
    Opp_FirstPass_serve_agent_spec: AgentSpec = env.agent_spec["Opp_FirstPass_serve"]
    Opp_FirstPass_serve_hover_agent_spec: AgentSpec = env.agent_spec["Opp_FirstPass_serve_hover"]
    Opp_FirstPass_receive_agent_spec: AgentSpec = env.agent_spec["Opp_FirstPass_receive"]
    Opp_FirstPass_receive_hover_agent_spec: AgentSpec = env.agent_spec["Opp_FirstPass_receive_hover"]

    high_level_agent_spec: AgentSpec = env.agent_spec["high_level"]

    agent_spec_dict = {}

    agent_spec_dict["Opp_SecPass_goto"] = Opp_SecPass_goto_agent_spec
    agent_spec_dict["Opp_SecPass"] = Opp_SecPass_agent_spec
    agent_spec_dict["Opp_SecPass_hover"] = Opp_SecPass_hover_agent_spec

    agent_spec_dict["Opp_Att_goto"] = Opp_Att_goto_agent_spec
    agent_spec_dict["Opp_Att"] = Opp_Att_agent_spec
    agent_spec_dict["Opp_Att_hover"] = Opp_Att_hover_agent_spec

    agent_spec_dict["FirstPass_goto"] = FirstPass_goto_agent_spec
    agent_spec_dict["FirstPass"] = FirstPass_agent_spec
    agent_spec_dict["FirstPass_hover"] = FirstPass_hover_agent_spec
    agent_spec_dict["FirstPass_serve"] = FirstPass_serve_agent_spec
    agent_spec_dict["FirstPass_serve_hover"] = FirstPass_serve_hover_agent_spec
    agent_spec_dict["FirstPass_receive"] = FirstPass_receive_agent_spec
    agent_spec_dict["FirstPass_receive_hover"] = FirstPass_receive_hover_agent_spec

    agent_spec_dict["SecPass_goto"] = SecPass_goto_agent_spec
    agent_spec_dict["SecPass"] = SecPass_agent_spec
    agent_spec_dict["SecPass_hover"] = SecPass_hover_agent_spec

    agent_spec_dict["Att_goto"] = Att_goto_agent_spec
    agent_spec_dict["Att"] = Att_agent_spec
    agent_spec_dict["Att_hover"] = Att_hover_agent_spec

    agent_spec_dict["Opp_FirstPass_goto"] = Opp_FirstPass_goto_agent_spec
    agent_spec_dict["Opp_FirstPass"] = Opp_FirstPass_agent_spec
    agent_spec_dict["Opp_FirstPass_hover"] = Opp_FirstPass_hover_agent_spec
    agent_spec_dict["Opp_FirstPass_serve"] = Opp_FirstPass_serve_agent_spec
    agent_spec_dict["Opp_FirstPass_serve_hover"] = Opp_FirstPass_serve_hover_agent_spec
    agent_spec_dict["Opp_FirstPass_receive"] = Opp_FirstPass_receive_agent_spec
    agent_spec_dict["Opp_FirstPass_receive_hover"] = Opp_FirstPass_receive_hover_agent_spec

    agent_spec_dict["high_level"] = high_level_agent_spec

    policy = algos[algo_name](cfg.algo, agent_spec_dict=agent_spec_dict, device="cuda", num_envs=cfg.env.num_envs)
    return policy


def _load_low_level_policy(cfg: DictConfig, policy: PSROPolicy_coselfplay_phase_one):

    if cfg.get("Opp_SecPass_goto_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_SecPass_goto_policy_checkpoint_path), player="Opp_SecPass_goto")
        print(f"Load Opp_SecPass_goto policy from {cfg.Opp_SecPass_goto_policy_checkpoint_path}")
    if cfg.get("Opp_SecPass_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_SecPass_policy_checkpoint_path), player="Opp_SecPass")
        print(f"Load Opp_SecPass policy from {cfg.Opp_SecPass_policy_checkpoint_path}")
    if cfg.get("Opp_SecPass_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_SecPass_hover_policy_checkpoint_path), player="Opp_SecPass_hover")
        print(f"Load Opp_SecPass_hover policy from {cfg.Opp_SecPass_hover_policy_checkpoint_path}")

    if cfg.get("Opp_Att_goto_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_Att_goto_policy_checkpoint_path), player="Opp_Att_goto")
        print(f"Load Opp_Att_goto policy from {cfg.Opp_Att_goto_policy_checkpoint_path}")
    if cfg.get("Opp_Att_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_Att_policy_checkpoint_path), player="Opp_Att")
        print(f"Load Opp_Att policy from {cfg.Opp_Att_policy_checkpoint_path}")
    if cfg.get("Opp_Att_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_Att_hover_policy_checkpoint_path), player="Opp_Att_hover")
        print(f"Load Opp_Att_hover policy from {cfg.Opp_Att_hover_policy_checkpoint_path}")
    
    if cfg.get("FirstPass_goto_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.FirstPass_goto_policy_checkpoint_path), player="FirstPass_goto")
        print(f"Load FirstPass_goto policy from {cfg.FirstPass_goto_policy_checkpoint_path}")
    if cfg.get("FirstPass_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.FirstPass_policy_checkpoint_path), player="FirstPass")
        print(f"Load FirstPass policy from {cfg.FirstPass_policy_checkpoint_path}")
    if cfg.get("FirstPass_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.FirstPass_hover_policy_checkpoint_path), player="FirstPass_hover")
        print(f"Load FirstPass_hover policy from {cfg.FirstPass_hover_policy_checkpoint_path}")
    if cfg.get("FirstPass_serve_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.FirstPass_serve_policy_checkpoint_path), player="FirstPass_serve")
        print(f"Load FirstPass_serve policy from {cfg.FirstPass_serve_policy_checkpoint_path}")
    if cfg.get("FirstPass_serve_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.FirstPass_serve_hover_policy_checkpoint_path), player="FirstPass_serve_hover")
        print(f"Load FirstPass_serve_hover policy from {cfg.FirstPass_serve_hover_policy_checkpoint_path}")
    if cfg.get("FirstPass_receive_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.FirstPass_receive_policy_checkpoint_path), player="FirstPass_receive")
        print(f"Load FirstPass_receive policy from {cfg.FirstPass_receive_policy_checkpoint_path}")
    if cfg.get("FirstPass_receive_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.FirstPass_receive_hover_policy_checkpoint_path), player="FirstPass_receive_hover")
        print(f"Load FirstPass_receive_hover policy from {cfg.FirstPass_receive_hover_policy_checkpoint_path}")
    
    if cfg.get("SecPass_goto_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.SecPass_goto_policy_checkpoint_path), player="SecPass_goto")
        print(f"Load SecPass_goto policy from {cfg.SecPass_goto_policy_checkpoint_path}")
    if cfg.get("SecPass_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.SecPass_policy_checkpoint_path), player="SecPass")
        print(f"Load SecPass policy from {cfg.SecPass_policy_checkpoint_path}")
    if cfg.get("SecPass_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.SecPass_hover_policy_checkpoint_path), player="SecPass_hover")
        print(f"Load SecPass_hover policy from {cfg.SecPass_hover_policy_checkpoint_path}")
    
    if cfg.get("Att_goto_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Att_goto_policy_checkpoint_path), player="Att_goto")
        print(f"Load Att_goto policy from {cfg.Att_goto_policy_checkpoint_path}")
    if cfg.get("Att_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Att_policy_checkpoint_path), player="Att")
        print(f"Load Att policy from {cfg.Att_policy_checkpoint_path}")
    if cfg.get("Att_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Att_hover_policy_checkpoint_path), player="Att_hover")
        print(f"Load Att_hover policy from {cfg.Att_hover_policy_checkpoint_path}")

    if cfg.get("Opp_FirstPass_goto_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_FirstPass_goto_policy_checkpoint_path), player="Opp_FirstPass_goto")
        print(f"Load Opp_FirstPass_goto policy from {cfg.Opp_FirstPass_goto_policy_checkpoint_path}")
    if cfg.get("Opp_FirstPass_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_FirstPass_policy_checkpoint_path), player="Opp_FirstPass")
        print(f"Load Opp_FirstPass policy from {cfg.Opp_FirstPass_policy_checkpoint_path}")
    if cfg.get("Opp_FirstPass_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_FirstPass_hover_policy_checkpoint_path), player="Opp_FirstPass_hover")
        print(f"Load Opp_FirstPass_hover policy from {cfg.Opp_FirstPass_hover_policy_checkpoint_path}")
    if cfg.get("Opp_FirstPass_serve_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_FirstPass_serve_policy_checkpoint_path), player="Opp_FirstPass_serve")
        print(f"Load Opp_FirstPass_serve policy from {cfg.Opp_FirstPass_serve_policy_checkpoint_path}")
    if cfg.get("Opp_FirstPass_serve_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_FirstPass_serve_hover_policy_checkpoint_path), player="Opp_FirstPass_serve_hover")
        print(f"Load Opp_FirstPass_serve_hover policy from {cfg.Opp_FirstPass_serve_hover_policy_checkpoint_path}")
    if cfg.get("Opp_FirstPass_receive_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_FirstPass_receive_policy_checkpoint_path), player="Opp_FirstPass_receive")
        print(f"Load Opp_FirstPass_receive policy from {cfg.Opp_FirstPass_receive_policy_checkpoint_path}")
    if cfg.get("Opp_FirstPass_receive_hover_policy_checkpoint_path") is not None:
        policy.load_state_dict(torch.load(cfg.Opp_FirstPass_receive_hover_policy_checkpoint_path), player="Opp_FirstPass_receive_hover")
        print(f"Load Opp_FirstPass_receive_hover policy from {cfg.Opp_FirstPass_receive_hover_policy_checkpoint_path}")


def _load_high_level_policy(cfg: DictConfig, policy: PSROPolicy_coselfplay_phase_one):
    """
    Init high-level policy with
        population,
        actor network,
        actor optimizer,
        value network,
        value normalizer
    """

    init_populations_from_checkpoint = (cfg.get("actor_0_checkpoint_path") and cfg.get("actor_1_checkpoint_path"))
    actor_0_directory_path = cfg.get("actor_0_checkpoints_directory_path")
    actor_1_directory_path = cfg.get("actor_1_checkpoints_directory_path")
    init_populations_from_checkpoint_directory = (actor_0_directory_path and actor_1_directory_path)
    if init_populations_from_checkpoint and init_populations_from_checkpoint_directory:
        raise ValueError("Cannot use both actor_checkpoint_path and actor_checkpoints_directory_path to init populations and actor network.")

    # init population, actor network and actor_optimizer [method 1]
    if init_populations_from_checkpoint:
        actor_dict_0 = torch.load(cfg.actor_0_checkpoint_path)
        actor_dict_1 = torch.load(cfg.actor_1_checkpoint_path)
        policy.policy_high_level.init_population_from_checkpoint(actor_dict_0, 0) # init population_0
        policy.policy_high_level.init_population_from_checkpoint(actor_dict_1, 1) # init population_1
        print(f"Init population_0 from checkpoint {cfg.actor_0_checkpoint_path}")
        print(f"Init population_1 from checkpoint {cfg.actor_1_checkpoint_path}")

        policy.policy_high_level.set_actor_params_with_latest_policy(both_populations=True) # init actor network and actor_optimizer

    # init population, actor network and actor_optimizer [method 2]
    if init_populations_from_checkpoint_directory:
        # init population_0
        file_names = os.listdir(actor_0_directory_path)
        file_names = sorted(file_names)
        file_paths = [
            os.path.join(actor_0_directory_path, f)
            for f in file_names
            if os.path.isfile(os.path.join(actor_0_directory_path, f))
        ]
        for file_path in file_paths:
            if file_path == file_paths[0]:
                actor_dict_0 = torch.load(file_path)
                policy.policy_high_level.init_population_from_checkpoint(actor_dict_0, 0)
            else:
                actor_dict_0 = torch.load(file_path)
                policy.policy_high_level.population_0.add_actor(actor_dict_0)
            logging.info(f"Init population_0 by appending actor_dict_0 from {file_path}")

        # init population_1
        file_names = os.listdir(actor_1_directory_path)
        file_names = sorted(file_names)
        file_paths = [
            os.path.join(actor_1_directory_path, f)
            for f in file_names
            if os.path.isfile(os.path.join(actor_1_directory_path, f))
        ]
        for file_path in file_paths:
            if file_path == file_paths[0]:
                actor_dict_1 = torch.load(file_path)
                policy.policy_high_level.init_population_from_checkpoint(actor_dict_1, 1)
            else:
                actor_dict_1 = torch.load(file_path)
                policy.policy_high_level.population_1.add_actor(actor_dict_1)
            logging.info(f"Init population_1 by appending actor_dict_1 from {file_path}")
        
        assert len(policy.policy_high_level.population_0) == len(policy.policy_high_level.population_1)

        policy.policy_high_level.set_actor_params_with_latest_policy(both_populations=True) # init actor network and actor_optimizer
    
    # init value function and value normalizer
    if cfg.get("high_level_policy_checkpoint_path"):
        
        state_dict = torch.load(cfg.high_level_policy_checkpoint_path)
        
        policy.policy_high_level.critic.load_state_dict(state_dict["critic"])
        policy.policy_high_level.value_normalizer.load_state_dict(state_dict["value_normalizer"])

        logging.info(f"Load critic from {cfg.high_level_policy_checkpoint_path}")
        logging.info(f"Load value_normalizer from {cfg.high_level_policy_checkpoint_path}")


def get_transforms(
    cfg: DictConfig,
    base_env,
    logger_func: Callable[[Dict], None] = None
):
    from hcsp.utils.torchrl.transforms import (
        LogOnEpisode,
    )

    stats_keys = [
        k
        for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    logger = LogOnEpisode(
        cfg.env.num_envs,
        in_keys=stats_keys,
        log_keys=stats_keys,
        logger_func=logger_func,
        process_func=None,
    )
    transforms = [InitTracker(), logger]

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is None:
        pass
    elif action_transform == "rate":
        from hcsp.controllers import RateController as _RateController
        from hcsp.utils.torchrl.transforms import RateController

        controller = _RateController(
            9.81, base_env.drone.params).to(base_env.device)
        transform = RateController(controller)
        transforms.append(transform)
    elif not action_transform.lower() == "none":
        raise NotImplementedError(
            f"Unknown action transform: {action_transform}")

    return transforms


@torch.no_grad()
def eval_win_rate(env: TransformedEnv, policy: PSROPolicy_coselfplay_phase_one):
    high_level_policy = policy.policy_high_level
    env.reset()
    env.eval()
    high_level_policy.eval_payoff = True

    td = env.rollout(
        max_steps=env.base_env.max_episode_length,
        policy=lambda x: policy(x, deterministic=False),
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=True,
    )

    env.reset()
    env.train()
    
    high_level_policy.eval_payoff = False

    done: torch.Tensor = td["stats", "done"].squeeze(-1).bool()  # (E, max_episode_len,)
    actor_0_wins: torch.Tensor = td["stats", "actor_0_wins"].squeeze(-1)  # (E,max_episode_len,)
    actor_1_wins: torch.Tensor = td["stats", "actor_1_wins"].squeeze(-1)  # (E,max_episode_len,)
    draws: torch.Tensor = td["stats", "draws"].squeeze(-1)  # (E,max_episode_len,)
    
    num_wins = actor_0_wins[done].sum().item()
    num_loses = actor_1_wins[done].sum().item()
    num_draws = draws[done].sum().item()

    if num_wins + num_loses == 0:
        return 0.5

    return num_wins / (num_wins + num_loses)


def get_new_payoffs(
    env: TransformedEnv,
    policy: PSROPolicy_coselfplay_phase_one,
    old_payoffs: Optional[np.ndarray],
):
    """
    compute missing payoff tensor entries via game simulations
    """
    high_level_policy = policy.policy_high_level
    assert len(high_level_policy.population_0) == len(high_level_policy.population_1)
    n = len(high_level_policy.population_0)        
    new_payoffs = np.zeros(shape=(n, n))

    if old_payoffs is not None:
        assert (
            len(old_payoffs.shape) == 2
            and old_payoffs.shape[0] == old_payoffs.shape[1]
            and old_payoffs.shape[0] + 1 == n
        )
        new_payoffs[:-1, :-1] = old_payoffs
    
    for i in range(n):
        high_level_policy.set_pure_strategy(idx_0=n-1, idx_1=i)
        wr = eval_win_rate(env=env, policy=policy)
        new_payoffs[-1, i] = wr - (1 - wr)

    for i in range(n-1):
        high_level_policy.set_pure_strategy(idx_0=i, idx_1=n-1)
        wr = eval_win_rate(env=env, policy=policy)
        new_payoffs[i, -1] = wr - (1 - wr)

    return new_payoffs

def payoffs_to_win_rate(payoffs: np.ndarray) -> np.ndarray:
    """
    Convert payoffs to win rate
    """
    assert len(payoffs.shape) == 2 and payoffs.shape[0] == payoffs.shape[1]
    win_rates = (payoffs + 1) / 2
    return win_rates

def log_heatmap(win_rate: np.ndarray):
    """
    Get heatmap of win rate
    """
    plt.figure(figsize=(12,10))
    if win_rate.shape[0] < 5:
        hm = sns.heatmap(win_rate, annot=True, fmt=".3f", center=0.5, cmap="coolwarm",
                    xticklabels=range(win_rate.shape[1]), yticklabels=range(win_rate.shape[0]))
    else:
        hm = sns.heatmap(win_rate, annot=False, center=0.5, cmap="coolwarm",
                    xticklabels=range(win_rate.shape[1]), yticklabels=range(win_rate.shape[0]))
    hm.xaxis.tick_top()
    plt.title("Team 0 Win Rate")
    plt.xlabel("Team 1 Strategy Population")
    plt.ylabel("Team 0 Strategy Population")
    plt.savefig("heatmap.png")
    plt.close()
    wandb.log({"heatmap": wandb.Image("heatmap.png")})
    return None

def get_emprical_win_rate(data: TensorDict, win_key: str, lose_key: str) -> float:
    
    win = data['stats', win_key]
    lose = data['stats', lose_key]
    done = data['stats', 'done'].bool()

    if done.sum().item() == 0:
        return None

    num_wins = win[done].sum().item()
    num_loses = lose[done].sum().item()

    if num_wins + num_loses == 0:
        return None

    return num_wins / (num_wins + num_loses)

def train(
    cfg: DictConfig, simulation_app: SimulationApp, env: TransformedEnv, wandb_run
):
    init_by_latest_strategy = cfg.get("init_by_latest_strategy", False)
    share_population = cfg.get("share_population", False)
    max_population_size = cfg.get("max_population_size", False)
    
    # initiate meta-solver
    meta_solver = get_meta_solver(cfg.get("solver_type").lower())

    # initiate PSRO policy
    policy: PSROPolicy_coselfplay_phase_one = _get_psro_high_low_level_policy(cfg, env)
    _load_low_level_policy(cfg, policy) # load low-level policy
    _load_high_level_policy(cfg, policy) # load high-level policy

    high_level_policy = policy.policy_high_level
    high_level_policy.current_player_id = cfg.get("first_id", 0) # 0 or 1

    # initiate converged indicator
    converged_indicator = ConvergedIndicator(mean_threshold=cfg.mean_threshold, 
                                             std_threshold=cfg.std_threshold,
                                             min_iter_steps=cfg.min_iter_steps,
                                             max_iter_steps=cfg.max_iter_steps,
                                             player_id=high_level_policy.current_player_id
                                             )

    # other initializations
    low_level_skill_eval_interval = cfg.get("low_level_skill_eval_interval")
    low_level_skill_save_interval = cfg.get("low_level_skill_save_interval")
    cnt_update: int = 0 # count of policy updates
    meta_policy_0, meta_policy_1 = np.array([1.0]), np.array([1.0])
    payoffs = get_new_payoffs(env=env, policy=policy, old_payoffs=None) # (a) Complete: compute missing payoff tensor entries via game simulations
    # payoffs = np.zeros(shape=(1, 1)) # (a) Complete: compute missing payoff tensor entries via game simulations
    win_rate = payoffs_to_win_rate(payoffs)
    log_heatmap(win_rate)
    
    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )
    pbar = tqdm(collector, total=total_frames // frames_per_batch)

    env.train()

    logging.info(f"Run URL:{wandb_run.get_url()}")
    high_level_frames = 0
    for i, data in enumerate(pbar):

        if max_population_size and payoffs.shape[0] >= max_population_size:
            logging.info(f"Population size reaches the maximum size {max_population_size}.")
            break

        info = {"env_frames": collector._frames, "rollout_fps": collector._fps, "high_level_frames": high_level_frames}
        
        if i == 0: # first evaluation            
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate(env, policy))
            env.train()

        result = policy.train_op(data.to_tensordict())

        # debug low-level skills fine-tuning
        if low_level_skill_eval_interval > 0 and i > 0 and i % low_level_skill_eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate_debug(env=env,policy=policy))
            env.train()

        if low_level_skill_save_interval > 0 and i > 0 and i % low_level_skill_save_interval == 0:
            ckpt_path = os.path.join(wandb_run.dir, f"checkpoint_{collector._frames}_SecPass.pt")
            logging.info(f"Save checkpoint to {str(ckpt_path)}")
            torch.save(policy.state_dict(player="SecPass"), ckpt_path)

        high_level_result = result["high_level"]
        if high_level_result is not None: # update policy once with the high_level_policy sample buffer
            high_level_frames += cfg.task.env.num_envs * cfg.algo.train_every
            info.update(high_level_result) # update policy of the current player
            info.update({"high_level_frames": high_level_frames})
            wandb_run.log(info)
            print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))
        
        if "FirstPass" in result:
            FirstPass_result = result["FirstPass"]
            info.update(FirstPass_result)

        if "FirstPass_hover" in result:
            FirstPass_hover_result = result["FirstPass_hover"]
            info.update(FirstPass_hover_result)

        if "SecPass" in result:
            SecPass_result = result["SecPass"]
            info.update(SecPass_result)

        if "SecPass_hover" in result:
            SecPass_hover_result = result["SecPass_hover"]
            info.update(SecPass_hover_result)

        if "Att" in result:
            Att_result = result["Att"]
            info.update(Att_result)

        if "Att_hover" in result:
            Att_hover_result = result["Att_hover"]
            info.update(Att_hover_result)

        empirical_win_rate = get_emprical_win_rate(data, win_key=f"actor_{high_level_policy.current_player_id}_wins", lose_key=f"actor_{1-high_level_policy.current_player_id}_wins")
        if empirical_win_rate is not None:
            converged_indicator.update(empirical_win_rate, high_level_frames)
        
        if converged_indicator.converged():
            
            high_level_policy.append_actor(high_level_policy.current_player_id, share_population=share_population) # (c) Expand: add new policies to the population (If share_population == false and cnt_update % 2 == 1, the new policy is only used in the evaluation)
            high_level_policy.switch_current_player_id() # switch current player id

            if init_by_latest_strategy:
                high_level_policy.set_actor_params_with_latest_policy()
            else:
                high_level_policy.set_actor_params_with_initial_random_policy()
            
            converged_indicator.reset(high_level_policy.current_player_id)
            cnt_update += 1
            
            if share_population: 
                payoffs = get_new_payoffs(env, policy, payoffs) # (a) Complete: compute missing payoff tensor entries via game simulations
                win_rate = payoffs_to_win_rate(payoffs)
                log_heatmap(win_rate)
                meta_policy_0, meta_policy_1 = meta_solver.solve([payoffs, -payoffs.T]) # (b) Solve: calculate meta-strategy via meta-solver

                logging.info(f"cnt_update:{cnt_update}")
                logging.info(f"payoffs:{payoffs}")
                logging.info(f"Meta-policy_0:{meta_policy_0}, Meta-policy_1:{meta_policy_1}")

            else: 
                raise NotImplementedError("Not implemented yet.")
                # if cnt_update % 2 == 0: 
                #     if len(high_level_policy.population_0) == 1 and len(high_level_policy.population_1) == 1: # initial policy is random policy and is replaced by trained policy
                #         payoffs = get_new_payoffs(env, policy, None) # (a) Complete: compute missing payoff tensor entries via game simulations
                #     else:
                #         payoffs = get_new_payoffs(env, policy, payoffs) # (a) Complete: compute missing payoff tensor entries via game simulations
                #     win_rate = payoffs_to_win_rate(payoffs)
                #     log_heatmap(win_rate)
                #     meta_policy_0, meta_policy_1 = meta_solver.solve([payoffs, -payoffs.T]) # (b) Solve: calculate meta-strategy via meta-solver
                    
                #     logging.info(f"cnt_update:{cnt_update}")
                #     logging.info(f"payoffs:{payoffs}")
                #     # logging.log(f"JPC:{calculate_jpc(payoffs)}")
                #     logging.info(f"Meta-policy_0:{meta_policy_0}, Meta-policy_1:{meta_policy_1}")

            info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
            logging.info(f"Eval at {collector._frames} steps.")
            if share_population and len(high_level_policy.population_0) > 1:
                info.update(evaluate_share_population(env, policy))
            info.update(evaluate(env, policy))
            wandb_run.log(info)

            env.reset()

            if max_population_size and payoffs.shape[0] >= max_population_size:
                logging.info(f"Population size reaches the maximum size {max_population_size}.")
                break

        high_level_policy.sample_pure_strategy(meta_policy_0, meta_policy_1)

        wandb_run.log(info)

        pbar.set_postfix(
            {
                "rollout_fps": collector._fps,
                "frames": collector._frames,
                "high_level_frames": high_level_frames,
            }
        )

    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate(env, policy))
    wandb_run.log(info)

    if hasattr(policy, "state_dict"):
        # high_level
        ckpt_path = os.path.join(wandb_run.dir, "policy_high_level_final.pt")
        logging.info(f"Save checkpoint to {str(ckpt_path)}")
        torch.save(policy.policy_high_level.state_dict(), ckpt_path) # save high-level policy

    if low_level_skill_save_interval > 0:
        ckpt_path = os.path.join(wandb_run.dir, f"checkpoint_final_SecPass.pt")
        logging.info(f"Save checkpoint to {str(ckpt_path)}")
        torch.save(policy.state_dict(player="SecPass"), ckpt_path)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_coselfplay_phase_one")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    
    run_name_suffix = cfg.get("run_name_suffix")
    if cfg.get("share_population", False):
        if run_name_suffix is None:
            run = init_wandb(cfg, cfg.solver_type, "share_population")
        else:
            run = init_wandb(cfg, cfg.solver_type, "share_population", run_name_suffix)
    else:
        if run_name_suffix is None:
            run = init_wandb(cfg, cfg.solver_type)
        else:
            run = init_wandb(cfg, cfg.solver_type, run_name_suffix)
    
    simulation_app = init_simulation_app(cfg)

    setproctitle(run.name)

    from hcsp.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    def log(info):
        if cfg.wandb.get("mode", "disabled") != "online":
            tmp = {k: v for k, v in info.items() if k in (
                "train/stats.actor_0_wins", "train/stats.actor_1_wins", "train/stats.draws")}
            print(OmegaConf.to_yaml(tmp))
        run.log(info)
    
    transforms = get_transforms(cfg, base_env, log)

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    train(cfg=cfg, simulation_app=simulation_app, env=env, wandb_run=run)

    wandb.save(os.path.join(run.dir, "checkpoint*"))
    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
