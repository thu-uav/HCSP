import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictParams
from typing import Union, Callable, List, Optional
from torch import vmap
import os
import numpy as np
import contextlib
import copy
from torch import nn

class uniform_policy(nn.Module):
    def __init__(self):
        super(uniform_policy, self).__init__()
    
    def forward(self, tensordict: TensorDict):
        observation: torch.Tensor = tensordict['agents', 'observation'] # [E, 1, 33]
        
        action_dim = 4
        action_shape = observation.shape[:-1] + (action_dim,)
        
        action = 2 * torch.rand(action_shape, device=observation.device) - 1  # [E, 1, 4]

        action_log_prob = torch.log(torch.ones(size=action_shape, device=observation.device) / 2)  # [E, 1, 4]
        action_log_prob = torch.sum(action_log_prob, dim=-1, keepdim=True)  # [E, 1, 1]

        action_entropy = torch.ones(size=action_shape, device=observation.device) * torch.log(torch.tensor(2.0, device=observation.device))  # [E, 1, 4]
        action_entropy = torch.sum(action_entropy, dim=-1, keepdim=True)  # [E, 1, 1]

        tensordict.set(("agents", "action"), action)
        tensordict.set("drone.action_logp", action_log_prob)
        tensordict.set("drone.action_entropy", action_entropy)

        return tensordict
    
class high_level_uniform_policy(nn.Module):
    def __init__(self):
        super(high_level_uniform_policy, self).__init__()
    
    def forward(self, tensordict: TensorDict):
        observation: torch.Tensor = tensordict['agents', 'high_level_observation'] # [E, 1, 33]
        
        action_dim = 3
        action_shape = observation.shape[:-1] + (action_dim,)

        num_FirstPass_skills = 5
        num_SecPass_skills = 3
        num_Att_skills = 4
        
        action_FirstPass = torch.randint(0, num_FirstPass_skills, size=observation.shape[:-1], device=observation.device)  # [E, 1]
        action_SecPass = torch.randint(0, num_SecPass_skills, size=observation.shape[:-1], device=observation.device)  # [E, 1]
        action_Att = torch.randint(0, num_Att_skills, size=observation.shape[:-1], device=observation.device)  # [E, 1]
        action = action.stack([action_FirstPass, action_SecPass, action_Att], dim=-1)  # [E, 1, 3]

        assert action.shape == action_shape


        logp_FirstPass = torch.log(torch.ones(size=action_FirstPass.shape, device=observation.device) / num_FirstPass_skills)  # [E, 1]
        logp_SecPass = torch.log(torch.ones(size=action_SecPass.shape, device=observation.device) / num_SecPass_skills)  # [E, 1]
        logp_Att = torch.log(torch.ones(size=action_Att.shape, device=observation.device) / num_Att_skills)  # [E, 1]
        action_log_prob = torch.stack([logp_FirstPass, logp_SecPass, logp_Att], dim=-1)  # [E, 1, 3]
        action_log_prob = torch.sum(action_log_prob, dim=-1, keepdim=True)  # [E, 1, 1]


        action_entropy_FirstPass = torch.ones(size=action_FirstPass.shape, device=observation.device) * torch.log(torch.tensor(num_FirstPass_skills, device=observation.device))  # [E, 1]
        action_entropy_SecPass = torch.ones(size=action_SecPass.shape, device=observation.device) * torch.log(torch.tensor(num_SecPass_skills, device=observation.device))  # [E, 1]
        action_entropy_Att = torch.ones(size=action_Att.shape, device=observation.device) * torch.log(torch.tensor(num_Att_skills, device=observation.device))  # [E, 1]
        action_entropy = torch.stack([action_entropy_FirstPass, action_entropy_SecPass, action_entropy_Att], dim=-1)  # [E, 1, 3]
        action_entropy = torch.sum(action_entropy, dim=-1, keepdim=True)  # [E, 1, 1]

        tensordict.set(("agents", "high_level_action"), action)
        tensordict.set("high_level.action_logp", action_log_prob)
        tensordict.set("high_level.action_entropy", action_entropy)

        return tensordict
    

class high_level_fixed_policy(nn.Module):
    """
    always goto (stay still)
    """
    def __init__(self):
        super(high_level_fixed_policy, self).__init__()
    
    def forward(self, tensordict: TensorDict):
        observation: torch.Tensor = tensordict['agents', 'high_level_observation'] # [E, 1, 33]
        
        action_dim = 3
        action_shape = observation.shape[:-1] + (action_dim,)

        FirstPass_goto_idx = 1 # goto middle
        SecPass_goto_idx = 0
        Att_goto_idx = 0
        
        action = torch.zeros(action_shape, device=observation.device, dtype=torch.long)
        action[..., 0] = FirstPass_goto_idx
        action[..., 1] = SecPass_goto_idx
        action[..., 2] = Att_goto_idx

        action_log_prob = torch.zeros_like(action.float())  # [E, 1, 2]
        action_log_prob = torch.sum(action_log_prob, dim=-1, keepdim=True)  # [E, 1, 1]

        action_entropy = torch.zeros_like(action.float())  # [E, 1, 2]
        action_entropy = torch.sum(action_entropy, dim=-1, keepdim=True)  # [E, 1, 1]

        tensordict.set(("agents", "high_level_action"), action)
        tensordict.set("high_level.action_logp", action_log_prob)
        tensordict.set("high_level.action_entropy", action_entropy)

        return tensordict
    

class high_level_fixed_policy_2(nn.Module):
    """
    FirstPass & Att always goto (stay still).
    SecPass utilizes the SecPass skill.
    """
    def __init__(self):
        super(high_level_fixed_policy_2, self).__init__()
    
    def forward(self, tensordict: TensorDict):
        observation: torch.Tensor = tensordict['agents', 'high_level_observation'] # [E, 1, 33]
        
        action_dim = 3
        action_shape = observation.shape[:-1] + (action_dim,)

        FirstPass_goto_idx = 1 # goto middle
        SecPass_idx = 1
        Att_goto_idx = 0
        
        action = torch.zeros(action_shape, device=observation.device, dtype=torch.long)  # [E, 1, 3] always left
        action[..., 0] = FirstPass_goto_idx
        action[..., 1] = SecPass_idx
        action[..., 2] = Att_goto_idx

        action_log_prob = torch.zeros_like(action.float())  # [E, 1, 2]
        action_log_prob = torch.sum(action_log_prob, dim=-1, keepdim=True)  # [E, 1, 1]

        action_entropy = torch.zeros_like(action.float())  # [E, 1, 2]
        action_entropy = torch.sum(action_entropy, dim=-1, keepdim=True)  # [E, 1, 1]

        tensordict.set(("agents", "high_level_action"), action)
        tensordict.set("high_level.action_logp", action_log_prob)
        tensordict.set("high_level.action_entropy", action_entropy)

        return tensordict

_policy_t = Callable[[TensorDict], TensorDict]


class Population:
    def __init__(
        self,
        dir: str,
        module: TensorDictModule, # by reference
        initial_policy: Union[uniform_policy, high_level_uniform_policy, high_level_fixed_policy, high_level_fixed_policy_2, dict],
        device = "cuda",
    ):
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)
        self._module_idx = -1 # The index of the last policy
        self._module = module # Actor module: assume all modules are homogeneous
        self._current_module_idx = -1 # The index of the current used policy
        self.device = device

        self.policy_sets: List[Union[_policy_t, int]] = [] 
        # The list of policies. For instance, [policy_function_0, 1, 2, ...]
        # Case 1: The policy is a function: policy_function itself is stored in the list
        # Case 2: The policy is a TensorDictParams: index of the policy is stored in the list

        self._func = None # current used policy
        self._params = None # current used policy params

        # initial_policy
        if callable(initial_policy): # initial_policy is a function
            self.policy_sets.append(initial_policy) 
            self._module_idx += 1
        elif isinstance(initial_policy, dict): # initial_policy is a actor_dict
            self.add_actor(initial_policy)
        else:
            raise ValueError("Invalid initial_policy")

        self.sample(meta_policy=np.array([1.0]))

    def __len__(self) -> int:
        return len(self.policy_sets)

    def add_actor(self, actor_dict: dict):
        # if len(self.policy_sets) == 1 and callable(self.policy_sets[0]): # the first policy is a random policy, replace it with the new policy
        #     self._module_idx = 0
        #     torch.save(actor_dict, os.path.join(self.dir, f"{self._module_idx}.pt"))
        #     self.policy_sets = [self._module_idx]
        #     self._current_module_idx = -1 # the first random policy is replaced and abandoned

        self._module_idx += 1
        torch.save(actor_dict, os.path.join(self.dir, f"{self._module_idx}.pt"))
        self.policy_sets.append(self._module_idx)
        
        self.set_latest_policy() # set the new policy as the current policy

    def _set_policy(self, index: int):
        if self._current_module_idx == index:
            return
        
        if not isinstance(self.policy_sets[index], int):
            self._func = self.policy_sets[index]
        else:
            assert self._module is not None
            checkpoint = torch.load(os.path.join(self.dir, f"{self.policy_sets[index]}.pt"))
            self._params = checkpoint["actor_params"].detach()
            self._func = lambda tensordict:vmap(
                self._module, in_dims=(1, 0), out_dims=1, randomness="error"
            )(tensordict, self._params, deterministic=True)
        
        self._current_module_idx = index
    
    def set_latest_policy(self):
        self._set_policy(self._module_idx)

    def set_second_latest_policy(self):
        self._set_policy(self._module_idx - 1)

    def set_behavioural_strategy(self, index: int):
        self._set_policy(index)

    def sample(self, meta_policy: np.array):
        # import pdb; pdb.set_trace()
        if len(meta_policy) == len(self.policy_sets):
            self._set_policy(np.random.choice(len(self.policy_sets), p=meta_policy))
        elif len(meta_policy) == len(self.policy_sets) - 1: # the population of one player is updated with a new policy while the population of the other players remains the same
            prob = np.append(meta_policy, 0.0)
            self._set_policy(np.random.choice(len(self.policy_sets), p=prob))
        else:
            raise ValueError("Invalid meta_policy")

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        tensordict = tensordict.to(self.device)
        return self._func(tensordict)
    
    def _get_policy_checkpoint(self, index: int) -> dict:
        if not isinstance(self.policy_sets[index], int):
            raise ValueError("The policy params are not saved in the population")
        return torch.load(os.path.join(self.dir, f"{self.policy_sets[index]}.pt"))
    
    def get_latest_policy_checkpoint(self) -> dict:
        return self._get_policy_checkpoint(self._module_idx)