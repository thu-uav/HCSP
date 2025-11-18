from numbers import Number
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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

from omni_drones.utils.torchrl.env import AgentSpec
from omni_drones.utils.tensordict import print_td_shape

from ..utils import valuenorm
from ..utils.gae import compute_gae

LR_SCHEDULER = lr_scheduler._LRScheduler
from torchrl.modules import TanhNormal, IndependentNormal

from omni_drones.learning import MAPPOPolicy
from omni_drones.learning import MAPPOPolicy_mask

class MAPPOPolicy_Receive_hover(object):

    def __init__(self, cfg, agent_spec_dict: Dict[str, AgentSpec], device="cuda") -> None:
        super().__init__()

        Opp_Server_agent_spec = agent_spec_dict["Opp_Server"]
        Opp_Server_hover_agent_spec = agent_spec_dict["Opp_Server_hover"]

        FirstPass_goto_agent_spec = agent_spec_dict["FirstPass_goto"]
        FirstPass_agent_spec = agent_spec_dict["FirstPass"]
        FirstPass_hover_agent_spec = agent_spec_dict["FirstPass_hover"]

        cfg.agent_name = "Opp_Server"
        self.policy_Opp_Server = MAPPOPolicy(cfg=cfg, agent_spec=Opp_Server_agent_spec, device=device)
        cfg.agent_name = "Opp_Server_hover"
        self.policy_Opp_Server_hover = MAPPOPolicy(cfg=cfg, agent_spec=Opp_Server_hover_agent_spec, device=device)
        
        cfg.agent_name = "FirstPass_goto"
        self.policy_FirstPass_goto = MAPPOPolicy(cfg=cfg, agent_spec=FirstPass_goto_agent_spec, device=device)
        cfg.agent_name = "FirstPass"
        self.policy_FirstPass = MAPPOPolicy(cfg=cfg, agent_spec=FirstPass_agent_spec, device=device)
        cfg.agent_name = "FirstPass_hover"
        self.policy_FirstPass_hover = MAPPOPolicy_mask(cfg=cfg, agent_spec=FirstPass_hover_agent_spec, device=device, mask_name=('stats', 'FirstPass_hit'))

    def load_state_dict(self, state_dict, player: str):
        if player == "Opp_Server":
            self.policy_Opp_Server.load_state_dict(state_dict)
        elif player == "Opp_Server_hover":
            self.policy_Opp_Server_hover.load_state_dict(state_dict)
        
        elif player == "FirstPass_goto":
            self.policy_FirstPass_goto.load_state_dict(state_dict)
        elif player == "FirstPass":
            self.policy_FirstPass.load_state_dict(state_dict)
        elif player == "FirstPass_hover":
            self.policy_FirstPass_hover.load_state_dict(state_dict)
        else:
            raise ValueError("player should be 'Opp_Server', 'Opp_Server_hover', 'FirstPass_goto' or 'FirstPass' or 'FirstPass_hover'")
        
    def state_dict(self, player: str):
        if player == "Opp_Server":
            return self.policy_Opp_Server.state_dict()
        elif player == "Opp_Server_hover":
            return self.policy_Opp_Server_hover.state_dict()

        elif player == "FirstPass_goto":
            return self.policy_FirstPass_goto.state_dict()
        elif player == "FirstPass":
            return self.policy_FirstPass.state_dict()
        elif player == "FirstPass_hover":
            return self.policy_FirstPass_hover.state_dict()
        else:
            raise ValueError("player should be 'Opp_Server', 'Opp_Server_hover', 'FirstPass_goto' or 'FirstPass' or 'FirstPass_hover'")

    def train_op(self, tensordict: TensorDict):
        return self.policy_FirstPass_hover.train_op(tensordict)

    def __call__(self, tensordict: TensorDict, deterministic: bool = False):
        self.policy_Opp_Server(tensordict, deterministic)
        self.policy_Opp_Server_hover(tensordict, deterministic)

        self.policy_FirstPass_goto(tensordict, deterministic)
        self.policy_FirstPass(tensordict, deterministic)
        self.policy_FirstPass_hover(tensordict, deterministic)
        return tensordict