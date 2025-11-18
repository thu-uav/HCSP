from .mappo import MAPPOPolicy, MAPPOPolicy_SecPass_new
from .ppo import *
# from .test_single import Policy
# from .mappo_formation import PPOFormation as Policy
from ._ppo import PPOPolicy as Policy

from .psro import PSROPolicy, PSROPolicy2, PSROPolicy3

from .mappo_mask import MAPPOPolicy_mask
from .mappo_disc_action import MAPPOPolicy_disc_action
from .mappo_mask_kl import MAPPOPolicy_mask_KL

from .volleyball import (
    MAPPOPolicy_Attack,
    MAPPOPolicy_Attack_hover,
    MAPPOPolicy_Pass,
    MAPPOPolicy_Pass_hover,
    MAPPOPolicy_Receive,
    MAPPOPolicy_Receive_hover,
    MAPPOPolicy_Serve_hover,
    MAPPOPolicy_Set_hover,
    PSROPolicy_high_level,
    PSROPolicy_coselfplay_phase_one,
    PSROPolicy_coselfplay_phase_two
)

