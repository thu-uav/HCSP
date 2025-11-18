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

class MAPPOPolicy_Serve_hover(object):
    """
    More specifically, PPO Policy for PSRO
    only tested in a two-agent task without RNN and actor sharing
    """

    def __init__(self, cfg, agent_spec_dict: Dict[str, AgentSpec], device="cuda") -> None:
        super().__init__()

        Server_agent_spec = agent_spec_dict["Server"]
        Server_hover_agent_spec = agent_spec_dict["Server_hover"]

        cfg.agent_name = "Server"
        self.policy_Server = MAPPOPolicy(cfg=cfg, agent_spec=Server_agent_spec, device=device)
        cfg.agent_name = "Server_hover"
        self.policy_Server_hover = MAPPOPolicy_mask(cfg=cfg, agent_spec=Server_hover_agent_spec, device=device, mask_name=('stats', 'Server_hit'))

    def load_state_dict(self, state_dict, player: str):
        if player == "Server":
            self.policy_Server.load_state_dict(state_dict)
        elif player == "Server_hover":
            self.policy_Server_hover.load_state_dict(state_dict)
        else:
            raise ValueError("player should be 'Server' or 'Server_hover'")
        
    def state_dict(self, player: str):
        if player == "Server":
            return self.policy_Server.state_dict()
        elif player == "Server_hover":
            return self.policy_Server_hover.state_dict()
        else:
            raise ValueError("player should be 'Server' or 'Server_hover'")

    def train_op(self, tensordict: TensorDict):
        return self.policy_Server_hover.train_op(tensordict)

    def __call__(self, tensordict: TensorDict, deterministic: bool = False):
        self.policy_Server(tensordict, deterministic)
        self.policy_Server_hover(tensordict, deterministic)
        return tensordict