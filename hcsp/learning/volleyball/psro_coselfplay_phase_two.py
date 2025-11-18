from numbers import Number
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from torch import vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.utils import expand_right
from tensordict.nn import make_functional, TensorDictModule, TensorDictParams, TensorDictModuleBase
from torch.optim import lr_scheduler

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    MultiDiscreteTensorSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec as UnboundedTensorSpec,
)

from hcsp.utils.torchrl.env import AgentSpec
from hcsp.utils.tensordict import print_td_shape

from ..utils import valuenorm
from ..utils.gae import compute_gae

LR_SCHEDULER = lr_scheduler._LRScheduler
from torchrl.modules import TanhNormal, IndependentNormal

from hcsp.learning import MAPPOPolicy, PSROPolicy, PSROPolicy2, PSROPolicy3, MAPPOPolicy_SecPass_new


class PSROPolicy_coselfplay_phase_two(object):

    def __init__(self, cfg, agent_spec_dict: Dict[str, AgentSpec], device, num_envs) -> None:
        super().__init__()

        Opp_SecPass_goto_agent_spec = agent_spec_dict["Opp_SecPass_goto"]
        Opp_SecPass_agent_spec = agent_spec_dict["Opp_SecPass"]
        Opp_SecPass_hover_agent_spec = agent_spec_dict["Opp_SecPass_hover"]

        Opp_Att_goto_agent_spec = agent_spec_dict["Opp_Att_goto"]
        Opp_Att_agent_spec = agent_spec_dict["Opp_Att"]
        Opp_Att_hover_agent_spec = agent_spec_dict["Opp_Att_hover"]

        FirstPass_goto_agent_spec = agent_spec_dict["FirstPass_goto"]
        FirstPass_agent_spec = agent_spec_dict["FirstPass"]
        FirstPass_hover_agent_spec = agent_spec_dict["FirstPass_hover"]
        FirstPass_serve_agent_spec = agent_spec_dict["FirstPass_serve"]
        FirstPass_serve_hover_agent_spec = agent_spec_dict["FirstPass_serve_hover"]
        FirstPass_receive_agent_spec = agent_spec_dict["FirstPass_receive"]
        FirstPass_receive_hover_agent_spec = agent_spec_dict["FirstPass_receive_hover"]

        SecPass_goto_agent_spec = agent_spec_dict["SecPass_goto"]
        SecPass_agent_spec = agent_spec_dict["SecPass"]
        SecPass_hover_agent_spec = agent_spec_dict["SecPass_hover"]

        SecPass_new_agent_spec = agent_spec_dict["SecPass_new"]

        Att_goto_agent_spec = agent_spec_dict["Att_goto"]
        Att_agent_spec = agent_spec_dict["Att"]
        Att_hover_agent_spec = agent_spec_dict["Att_hover"]

        Opp_FirstPass_goto_agent_spec = agent_spec_dict["Opp_FirstPass_goto"]
        Opp_FirstPass_agent_spec = agent_spec_dict["Opp_FirstPass"]
        Opp_FirstPass_hover_agent_spec = agent_spec_dict["Opp_FirstPass_hover"]
        Opp_FirstPass_serve_agent_spec = agent_spec_dict["Opp_FirstPass_serve"]
        Opp_FirstPass_serve_hover_agent_spec = agent_spec_dict["Opp_FirstPass_serve_hover"]
        Opp_FirstPass_receive_agent_spec = agent_spec_dict["Opp_FirstPass_receive"]
        Opp_FirstPass_receive_hover_agent_spec = agent_spec_dict["Opp_FirstPass_receive_hover"]

        high_level_old_agent_spec = agent_spec_dict["high_level_old"]
        high_level_agent_spec = agent_spec_dict["high_level"]

        cfg.agent_name = "Opp_SecPass_goto"
        self.policy_Opp_SecPass_goto = MAPPOPolicy(cfg=cfg, agent_spec=Opp_SecPass_goto_agent_spec, device=device)
        cfg.agent_name = "Opp_SecPass"
        self.policy_Opp_SecPass = MAPPOPolicy(cfg=cfg, agent_spec=Opp_SecPass_agent_spec, device=device)
        cfg.agent_name = "Opp_SecPass_hover"
        self.policy_Opp_SecPass_hover = MAPPOPolicy(cfg=cfg, agent_spec=Opp_SecPass_hover_agent_spec, device=device)

        cfg.agent_name = "Opp_Att_goto"
        self.policy_Opp_Att_goto = MAPPOPolicy(cfg=cfg, agent_spec=Opp_Att_goto_agent_spec, device=device)
        cfg.agent_name = "Opp_Att"
        self.policy_Opp_Att = MAPPOPolicy(cfg=cfg, agent_spec=Opp_Att_agent_spec, device=device)
        cfg.agent_name = "Opp_Att_hover"
        self.policy_Opp_Att_hover = MAPPOPolicy(cfg=cfg, agent_spec=Opp_Att_hover_agent_spec, device=device)
        
        cfg.agent_name = "FirstPass_goto"
        self.policy_FirstPass_goto = MAPPOPolicy(cfg=cfg, agent_spec=FirstPass_goto_agent_spec, device=device)
        cfg.agent_name = "FirstPass"
        self.policy_FirstPass = MAPPOPolicy(cfg=cfg, agent_spec=FirstPass_agent_spec, device=device)
        cfg.agent_name = "FirstPass_hover"
        self.policy_FirstPass_hover = MAPPOPolicy(cfg=cfg, agent_spec=FirstPass_hover_agent_spec, device=device)
        cfg.agent_name = "FirstPass_serve"
        self.policy_FirstPass_serve = MAPPOPolicy(cfg=cfg, agent_spec=FirstPass_serve_agent_spec, device=device)
        cfg.agent_name = "FirstPass_serve_hover"
        self.policy_FirstPass_serve_hover = MAPPOPolicy(cfg=cfg, agent_spec=FirstPass_serve_hover_agent_spec, device=device)
        cfg.agent_name = "FirstPass_receive"
        self.policy_FirstPass_receive = MAPPOPolicy(cfg=cfg, agent_spec=FirstPass_receive_agent_spec, device=device)
        cfg.agent_name = "FirstPass_receive_hover"
        self.policy_FirstPass_receive_hover = MAPPOPolicy(cfg=cfg, agent_spec=FirstPass_receive_hover_agent_spec, device=device)

        cfg.agent_name = "SecPass_goto"
        self.policy_SecPass_goto = MAPPOPolicy(cfg=cfg, agent_spec=SecPass_goto_agent_spec, device=device)
        cfg.agent_name = "SecPass"
        self.policy_SecPass = MAPPOPolicy(cfg=cfg, agent_spec=SecPass_agent_spec, device=device)
        cfg.agent_name = "SecPass_hover"
        self.policy_SecPass_hover = MAPPOPolicy(cfg=cfg, agent_spec=SecPass_hover_agent_spec, device=device)

        cfg.agent_name = "SecPass_new"
        self.policy_SecPass_new = MAPPOPolicy_SecPass_new(cfg=cfg, agent_spec=SecPass_new_agent_spec, device=device)

        cfg.agent_name = "Att_goto"
        self.policy_Att_goto = MAPPOPolicy(cfg=cfg, agent_spec=Att_goto_agent_spec, device=device)
        cfg.agent_name = "Att"
        self.policy_Att = MAPPOPolicy(cfg=cfg, agent_spec=Att_agent_spec, device=device)
        cfg.agent_name = "Att_hover"
        self.policy_Att_hover = MAPPOPolicy(cfg=cfg, agent_spec=Att_hover_agent_spec, device=device)
        
        cfg.agent_name = "Opp_FirstPass_goto"
        self.policy_Opp_FirstPass_goto = MAPPOPolicy(cfg=cfg, agent_spec=Opp_FirstPass_goto_agent_spec, device=device)
        cfg.agent_name = "Opp_FirstPass"
        self.policy_Opp_FirstPass = MAPPOPolicy(cfg=cfg, agent_spec=Opp_FirstPass_agent_spec, device=device)
        cfg.agent_name = "Opp_FirstPass_hover"
        self.policy_Opp_FirstPass_hover = MAPPOPolicy(cfg=cfg, agent_spec=Opp_FirstPass_hover_agent_spec, device=device)
        cfg.agent_name = "Opp_FirstPass_serve"
        self.policy_Opp_FirstPass_serve = MAPPOPolicy(cfg=cfg, agent_spec=Opp_FirstPass_serve_agent_spec, device=device)
        cfg.agent_name = "Opp_FirstPass_serve_hover"
        self.policy_Opp_FirstPass_serve_hover = MAPPOPolicy(cfg=cfg, agent_spec=Opp_FirstPass_serve_hover_agent_spec, device=device)
        cfg.agent_name = "Opp_FirstPass_receive"
        self.policy_Opp_FirstPass_receive = MAPPOPolicy(cfg=cfg, agent_spec=Opp_FirstPass_receive_agent_spec, device=device)
        cfg.agent_name = "Opp_FirstPass_receive_hover"
        self.policy_Opp_FirstPass_receive_hover = MAPPOPolicy(cfg=cfg, agent_spec=Opp_FirstPass_receive_hover_agent_spec, device=device)

        cfg.agent_name = "high_level_old"
        self.policy_high_level_old = PSROPolicy3(cfg=cfg, agent_spec=high_level_old_agent_spec, device=device)

        cfg.agent_name = "high_level"
        self.policy_high_level = PSROPolicy2(cfg=cfg, agent_spec=high_level_agent_spec, device=device)

        self.high_level_keys = {}
        self.high_level_keys[0] = ('agents', 'high_level_observation')
        self.high_level_keys[1] = ('agents', 'high_level_action')
        self.high_level_keys[2] = 'high_level_state_value'
        self.high_level_keys[3] = 'high_level.action_logp'
        self.high_level_keys[4] = 'high_level.action_entropy'
        
        self.high_level_keys[5] = ('next', ('agents', 'high_level_observation'))
        self.high_level_keys[6] = ('next', ('agents', 'high_level_reward'))
        self.high_level_keys[7] = ('next', 'done')

        assert len(self.high_level_keys) == 8

        self.num_envs = num_envs
        self.train_every = cfg.train_every
        self.buffer_max = self.train_every

        self.buffer = {}
        self.buffer[0] = torch.zeros((self.num_envs, self.train_every, 2, 100), device=device)
        self.buffer[1] = torch.zeros((self.num_envs, self.train_every, 2, 3), dtype=torch.int32, device=device)
        self.buffer[2] = torch.zeros((self.num_envs, self.train_every, 2, 1), device=device)
        self.buffer[3] = torch.zeros((self.num_envs, self.train_every, 2, 1), device=device)
        self.buffer[4] = torch.zeros((self.num_envs, self.train_every, 2, 1), device=device)
        
        self.buffer[5] = torch.zeros((self.num_envs, self.train_every, 2, 100), device=device) # next_obs
        self.buffer[6] = torch.zeros((self.num_envs, self.train_every, 2, 1), device=device) # reward
        self.buffer[7] = torch.zeros((self.num_envs, self.train_every, 1), dtype=torch.bool, device=device) # done
        
        self.temp_reward = torch.zeros((self.num_envs, 2, 1), device=device)
        self.temp_done = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=device)
        self.buffer_sample_count = torch.zeros((self.num_envs), dtype=torch.long, device=device)
        self.is_before_first_action = torch.ones((self.num_envs), dtype=torch.bool, device=device)

    def load_state_dict(self, state_dict, player: str):
        if player == "Opp_SecPass_goto":
            self.policy_Opp_SecPass_goto.load_state_dict(state_dict)
        elif player == "Opp_SecPass":
            self.policy_Opp_SecPass.load_state_dict(state_dict)
        elif player == "Opp_SecPass_hover":
            self.policy_Opp_SecPass_hover.load_state_dict(state_dict)

        elif player == "Opp_Att_goto":
            self.policy_Opp_Att_goto.load_state_dict(state_dict)
        elif player == "Opp_Att":
            self.policy_Opp_Att.load_state_dict(state_dict)
        elif player == "Opp_Att_hover":
            self.policy_Opp_Att_hover.load_state_dict(state_dict)
        
        elif player == "FirstPass_goto":
            self.policy_FirstPass_goto.load_state_dict(state_dict)
        elif player == "FirstPass":
            self.policy_FirstPass.load_state_dict(state_dict)
        elif player == "FirstPass_hover":
            self.policy_FirstPass_hover.load_state_dict(state_dict)
        elif player == "FirstPass_serve":
            self.policy_FirstPass_serve.load_state_dict(state_dict)
        elif player == "FirstPass_serve_hover":
            self.policy_FirstPass_serve_hover.load_state_dict(state_dict)
        elif player == "FirstPass_receive":
            self.policy_FirstPass_receive.load_state_dict(state_dict)
        elif player == "FirstPass_receive_hover":
            self.policy_FirstPass_receive_hover.load_state_dict(state_dict)

        elif player == "SecPass_goto":
            self.policy_SecPass_goto.load_state_dict(state_dict)
        elif player == "SecPass":
            self.policy_SecPass.load_state_dict(state_dict)
        elif player == "SecPass_hover":
            self.policy_SecPass_hover.load_state_dict(state_dict)
        
        elif player == "SecPass_new":
            self.policy_SecPass_new.load_state_dict(state_dict)

        elif player == "Att_goto":
            self.policy_Att_goto.load_state_dict(state_dict)
        elif player == "Att":
            self.policy_Att.load_state_dict(state_dict)
        elif player == "Att_hover":
            self.policy_Att_hover.load_state_dict(state_dict)

        elif player == "Opp_FirstPass_goto":
            self.policy_Opp_FirstPass_goto.load_state_dict(state_dict)
        elif player == "Opp_FirstPass":
            self.policy_Opp_FirstPass.load_state_dict(state_dict)
        elif player == "Opp_FirstPass_hover":
            self.policy_Opp_FirstPass_hover.load_state_dict(state_dict)
        elif player == "Opp_FirstPass_serve":
            self.policy_Opp_FirstPass_serve.load_state_dict(state_dict)
        elif player == "Opp_FirstPass_serve_hover":
            self.policy_Opp_FirstPass_serve_hover.load_state_dict(state_dict)
        elif player == "Opp_FirstPass_receive":
            self.policy_Opp_FirstPass_receive.load_state_dict(state_dict)
        elif player == "Opp_FirstPass_receive_hover":
            self.policy_Opp_FirstPass_receive_hover.load_state_dict(state_dict)

        # elif player == "high_level":
        #     self.policy_high_level.load_state_dict(state_dict)
        
        else:
            raise ValueError("player not found")
        
    def state_dict(self, player: str):
        if player == "Opp_SecPass_goto":
            return self.policy_Opp_SecPass_goto.state_dict()        
        elif player == "Opp_SecPass":
            return self.policy_Opp_SecPass.state_dict()
        elif player == "Opp_SecPass_hover":
            return self.policy_Opp_SecPass_hover.state_dict()

        elif player == "Opp_Att_goto":
            return self.policy_Opp_Att_goto.state_dict()
        elif player == "Opp_Att":
            return self.policy_Opp_Att.state_dict()
        elif player == "Opp_Att_hover":
            return self.policy_Opp_Att_hover.state_dict()

        elif player == "FirstPass_goto":
            return self.policy_FirstPass_goto.state_dict()
        elif player == "FirstPass":
            return self.policy_FirstPass.state_dict()
        elif player == "FirstPass_hover":
            return self.policy_FirstPass_hover.state_dict()
        elif player == "FirstPass_serve":
            return self.policy_FirstPass_serve.state_dict()
        elif player == "FirstPass_serve_hover":
            return self.policy_FirstPass_serve_hover.state_dict()
        elif player == "FirstPass_receive":
            return self.policy_FirstPass_receive.state_dict()
        elif player == "FirstPass_receive_hover":
            return self.policy_FirstPass_receive_hover.state_dict()
        
        elif player == "SecPass_goto":
            return self.policy_SecPass_goto.state_dict()
        elif player == "SecPass":
            return self.policy_SecPass.state_dict()
        elif player == "SecPass_hover":
            return self.policy_SecPass_hover.state_dict()
        
        elif player == "SecPass_new":
            return self.policy_SecPass_new.state_dict()

        elif player == "Att_goto":
            return self.policy_Att_goto.state_dict()
        elif player == "Att":
            return self.policy_Att.state_dict()
        elif player == "Att_hover":
            return self.policy_Att_hover.state_dict()
        
        elif player == "Opp_FirstPass_goto":
            return self.policy_Opp_FirstPass_goto.state_dict()
        elif player == "Opp_FirstPass":
            return self.policy_Opp_FirstPass.state_dict()
        elif player == "Opp_FirstPass_hover":
            return self.policy_Opp_FirstPass_hover.state_dict()
        elif player == "Opp_FirstPass_serve":
            return self.policy_Opp_FirstPass_serve.state_dict()
        elif player == "Opp_FirstPass_serve_hover":
            return self.policy_Opp_FirstPass_serve_hover.state_dict()
        elif player == "Opp_FirstPass_receive":
            return self.policy_Opp_FirstPass_receive.state_dict()
        elif player == "Opp_FirstPass_receive_hover":
            return self.policy_Opp_FirstPass_receive_hover.state_dict()
        
        elif player == "high_level_old":
            return self.policy_high_level_old.state_dict()
        elif player == "high_level":
            return self.policy_high_level.state_dict()
        
        else:
            raise ValueError("player not found")

    def train_op(self, tensordict: TensorDict):
        '''
        train with a high-level policy sample buffer
        '''
        tensordict_new = tensordict.select(*self.policy_high_level.train_in_keys)

        switch_turn = tensordict[("info", "switch_turn")].squeeze(-1) # [E, train_every]
        done = tensordict_new[("next", "done")] # [E, train_every, 1]
        reward = tensordict_new[("next", ("agents", "high_level_reward"))] # [E, train_every, 2, 1]

        for t in range(self.train_every):
            cur_reward = reward[:, t, ...] # [E, 2, 1]
            cur_done = done[:, t, ...] # [E, 1]
            cur_event = switch_turn[:, t] # [E]
            
            should_compute_temp = (~ self.is_before_first_action).nonzero(as_tuple=True)[0] # before the first action, no need to compute temp_reward and temp_done
            if should_compute_temp.numel() > 0:
                self.temp_reward[should_compute_temp] += cur_reward[should_compute_temp]
                self.temp_done[should_compute_temp] |= cur_done[should_compute_temp]
            
            triggered_envs = cur_event.nonzero(as_tuple=True)[0] # [E']
            if triggered_envs.numel() > 0:
                self.is_before_first_action[triggered_envs] = False # after the first action, set is_first_action to False
                for env_idx in triggered_envs:
                    temp_count = self.buffer_sample_count[env_idx].item()
                    if temp_count == 0:
                        for i in range(5):
                            self.buffer[i][env_idx, temp_count] = tensordict_new[self.high_level_keys[i]][env_idx, t]
                    elif temp_count > 0 and temp_count < self.buffer_max:
                        for i in range(5):
                            self.buffer[i][env_idx, temp_count] = tensordict_new[self.high_level_keys[i]][env_idx, t]

                        self.buffer[5][env_idx, temp_count - 1] = tensordict_new[self.high_level_keys[0]][env_idx, t] # next_obs should also be at the event-triggered time step.
                        self.buffer[6][env_idx, temp_count - 1] = self.temp_reward[env_idx] # next_reward
                        self.temp_reward[env_idx] = 0
                        self.buffer[7][env_idx, temp_count - 1] = self.temp_done[env_idx] # next_done
                        if self.temp_done[env_idx].item():
                            self.temp_done[env_idx] = False
                    elif temp_count == self.buffer_max:
                        self.buffer[5][env_idx, temp_count - 1] = tensordict_new[self.high_level_keys[0]][env_idx, t] # next_obs should also be at the event-triggered time step.
                        self.buffer[6][env_idx, temp_count - 1] = self.temp_reward[env_idx] # next_reward
                        self.temp_reward[env_idx] = 0
                        self.buffer[7][env_idx, temp_count - 1] = self.temp_done[env_idx] # next_done
                        if self.temp_done[env_idx].item():
                            self.temp_done[env_idx] = False
                    self.buffer_sample_count[env_idx] += 1
            
            stop_compute_temp = cur_done.nonzero(as_tuple=True)[0]
            if stop_compute_temp.numel() > 0:
                self.is_before_first_action[stop_compute_temp] = True

        if (self.buffer_sample_count >= self.buffer_max + 1).all().item():
            sample_td = TensorDict({})
            for i in range(8):
                sample_td[self.high_level_keys[i]] = self.buffer[i]
            sample_td.batch_size = [self.num_envs, self.train_every]

            result = self.policy_high_level.train_op(sample_td)

            self.buffer_sample_count.zero_()
            self.temp_done.zero_()
            self.temp_reward.zero_()
            for i in range(8):
                self.buffer[i].zero_()

            return result
        return None
        
    def __call__(self, tensordict: TensorDict, deterministic: bool = False):
        
        self.policy_high_level_old(tensordict, deterministic)
        self.policy_high_level(tensordict, deterministic)

        if deterministic == True: # only log state value of the central env when evaluating
            self._log_high_level_state_value(tensordict) 
        # self._recompute_obs(tensordict)

        self.policy_Opp_SecPass_goto(tensordict, deterministic)
        self.policy_Opp_SecPass(tensordict, deterministic)
        self.policy_Opp_SecPass_hover(tensordict, deterministic)

        self.policy_Opp_Att_goto(tensordict, deterministic)
        self.policy_Opp_Att(tensordict, deterministic)
        self.policy_Opp_Att_hover(tensordict, deterministic)

        self.policy_FirstPass_goto(tensordict, deterministic)
        self.policy_FirstPass(tensordict, deterministic)
        self.policy_FirstPass_hover(tensordict, deterministic)
        self.policy_FirstPass_serve(tensordict, deterministic)
        self.policy_FirstPass_serve_hover(tensordict, deterministic)
        self.policy_FirstPass_receive(tensordict, deterministic)
        self.policy_FirstPass_receive_hover(tensordict, deterministic)

        self.policy_SecPass_goto(tensordict, deterministic)
        self.policy_SecPass(tensordict, deterministic)
        self.policy_SecPass_hover(tensordict, deterministic)

        self.policy_SecPass_new(tensordict, deterministic)

        self.policy_Att_goto(tensordict, deterministic)
        self.policy_Att(tensordict, deterministic)
        self.policy_Att_hover(tensordict, deterministic)

        self.policy_Opp_FirstPass_goto(tensordict, deterministic)
        self.policy_Opp_FirstPass(tensordict, deterministic)
        self.policy_Opp_FirstPass_hover(tensordict, deterministic)
        self.policy_Opp_FirstPass_serve(tensordict, deterministic)
        self.policy_Opp_FirstPass_serve_hover(tensordict, deterministic)
        self.policy_Opp_FirstPass_receive(tensordict, deterministic)
        self.policy_Opp_FirstPass_receive_hover(tensordict, deterministic)

        return tensordict

    def _log_high_level_state_value(self, tensordict: TensorDict):
        '''
        Log the state value of the central environment at the time step when the
        central environment switches turn.
        '''
        central_env_idx = tensordict[("info", "central_env_idx")][0].item()
        high_level_action = tensordict[("agents", "high_level_action")]
        switch_turn = tensordict[("info", "switch_turn")]
        high_level_state_value = tensordict["high_level_state_value"]

        if switch_turn[central_env_idx].item():
            logging.info(f"high_level_action: {high_level_action[central_env_idx].tolist()}, high_level_state_value: {high_level_state_value[central_env_idx].tolist()}")
    
    # def _recompute_obs(self, tensordict: TensorDict):
    #     '''
    #     Recompute the agents' observations only at the time step when a drone 
    #     hits the ball. This ensures consistency between the high-level skill action
    #     and the corresponding low-level skill observation.
    #     '''

    #     high_level_action = tensordict[("agents", "high_level_action")] # (E, 2, 2)
    #     switch_turn = tensordict[("info", "switch_turn")] # (E, 1)
    #     turn = tensordict[("info", "turn")] # (E, 1)
        
    #     Att_hit = tensordict[("info", "Att_hit")].squeeze(-1) # (E,)
    #     Opp_Att_hit = tensordict[("info", "Opp_Att_hit")].squeeze(-1) # (E,)
    #     SecPass_hit = tensordict[("info", "SecPass_hit")].squeeze(-1) # (E,)
    #     Opp_SecPass_hit = tensordict[("info", "Opp_SecPass_hit")].squeeze(-1) # (E,)

    #     FirstPass_goto_obs = tensordict[("agents", "FirstPass_goto_observation")] # (E, 2, 26)
    #     FirstPass_obs = tensordict[("agents", "FirstPass_observation")] # (E, 2, 36)
    #     Opp_FirstPass_goto_obs = tensordict[("agents", "Opp_FirstPass_goto_observation")] # (E, 2, 26)
    #     Opp_FirstPass_obs = tensordict[("agents", "Opp_FirstPass_observation")] # (E, 2, 36)
    #     Att_obs = tensordict[("agents", "Att_observation")] # (E, 2, 36)
    #     Opp_Att_obs = tensordict[("agents", "Opp_Att_observation")] # (E, 2, 36)
        
    #     Opp_FirstPass_goto_left_rpos = tensordict[("info", "Opp_FirstPass_goto_left_rpos")]
    #     Opp_FirstPass_goto_middle_rpos = tensordict[("info", "Opp_FirstPass_goto_middle_rpos")]
    #     Opp_FirstPass_goto_right_rpos = tensordict[("info", "Opp_FirstPass_goto_right_rpos")]
    #     FirstPass_goto_left_rpos = tensordict[("info", "FirstPass_goto_left_rpos")]
    #     FirstPass_goto_middle_rpos = tensordict[("info", "FirstPass_goto_middle_rpos")]
    #     FirstPass_goto_right_rpos = tensordict[("info", "FirstPass_goto_right_rpos")]

    #     # used for both Att and Opp_Att, FirstPass and Opp_FirstPass
    #     Att_attacking_target_left = tensordict[("info", "Att_attacking_target_left")]
    #     Att_attacking_target_right = tensordict[("info", "Att_attacking_target_right")]
    #     FirstPass_obs_left_attacking_target = tensordict[("info", "FirstPass_obs_left_attacking_target")]
    #     FirstPass_obs_right_attacking_target = tensordict[("info", "FirstPass_obs_right_attacking_target")]
        
    #     if SecPass_hit.any().item():
    #         SecPass_hit_idx = torch.nonzero(SecPass_hit, as_tuple=True)[0]

    #         # Att
    #         near_side_high_level_action = high_level_action[SecPass_hit_idx, 0, :] # (E', 2)
    #         attack_left_or_right = (near_side_high_level_action[:, 1] == 1) # (E',) 0: left, 1: right
    #         Att_obs[SecPass_hit_idx, :, -2:] = torch.where(
    #             attack_left_or_right.unsqueeze(-1).unsqueeze(-1),
    #             Att_attacking_target_right[SecPass_hit_idx],
    #             Att_attacking_target_left[SecPass_hit_idx]
    #         )
    #         tensordict[("agents", "Att_observation")] = Att_obs

    #         # Opp_FirstPass_goto
    #         far_side_high_level_action = high_level_action[SecPass_hit_idx, 1, :] # (E', 2)
    #         Opp_FirstPass_goto_left_or_right = (far_side_high_level_action[:, 0] == 1) # (E',) 0: left, 1: right
    #         Opp_FirstPass_goto_obs[SecPass_hit_idx, :, 0:3] = torch.where(
    #             Opp_FirstPass_goto_left_or_right.unsqueeze(-1).unsqueeze(-1),
    #             Opp_FirstPass_goto_right_rpos[SecPass_hit_idx],
    #             Opp_FirstPass_goto_left_rpos[SecPass_hit_idx]
    #         )
    #         tensordict[("agents", "Opp_FirstPass_goto_observation")] = Opp_FirstPass_goto_obs
    #         # Opp_FirstPass
    #         Opp_FirstPass_obs[SecPass_hit_idx, :, -2:] = torch.where(
    #             Opp_FirstPass_goto_left_or_right.unsqueeze(-1).unsqueeze(-1),
    #             FirstPass_obs_right_attacking_target[SecPass_hit_idx],
    #             FirstPass_obs_left_attacking_target[SecPass_hit_idx]
    #         )
    #         tensordict[("agents", "Opp_FirstPass_observation")] = Opp_FirstPass_obs

    #     if Opp_SecPass_hit.any().item():
    #         Opp_SecPass_hit_idx = torch.nonzero(Opp_SecPass_hit, as_tuple=True)[0]

    #         # Opp_Att
    #         far_side_high_level_action = high_level_action[Opp_SecPass_hit_idx, 1, :] # (E', 2)
    #         attack_left_or_right = (far_side_high_level_action[:, 1] == 1) # (E',) 0: left, 1: right
    #         Opp_Att_obs[Opp_SecPass_hit_idx, :, -2:] = torch.where(
    #             attack_left_or_right.unsqueeze(-1).unsqueeze(-1),
    #             Att_attacking_target_right[Opp_SecPass_hit_idx],
    #             Att_attacking_target_left[Opp_SecPass_hit_idx]
    #         )
    #         tensordict[("agents", "Opp_Att_observation")] = Opp_Att_obs

    #         # FirstPass_goto
    #         near_side_high_level_action = high_level_action[Opp_SecPass_hit_idx, 0, :]
    #         FirstPass_goto_left_or_right = (near_side_high_level_action[:, 0] == 2) # (E',) 0: left, 1: right
    #         FirstPass_goto_obs[Opp_SecPass_hit_idx, :, 0:3] = torch.where(
    #             FirstPass_goto_left_or_right.unsqueeze(-1).unsqueeze(-1),
    #             FirstPass_goto_right_rpos[Opp_SecPass_hit_idx],
    #             FirstPass_goto_left_rpos[Opp_SecPass_hit_idx]
    #         )
    #         tensordict[("agents", "FirstPass_goto_observation")] = FirstPass_goto_obs
    #         # FirstPass
    #         FirstPass_obs[Opp_SecPass_hit_idx, :, -2:] = torch.where(
    #             FirstPass_goto_left_or_right.unsqueeze(-1).unsqueeze(-1),
    #             FirstPass_obs_right_attacking_target[Opp_SecPass_hit_idx],
    #             FirstPass_obs_left_attacking_target[Opp_SecPass_hit_idx]
    #         )
    #         tensordict[("agents", "FirstPass_observation")] = FirstPass_obs

