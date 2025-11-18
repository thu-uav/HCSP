from typing import Optional
import functorch

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quaternion_to_euler
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.objects as objects
import omni.isaac.core.materials as materials
import torch
import torch.distributions as D
import torch.nn.functional as NNF

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec, MultiDiscreteTensorSpec
)
from pxr import UsdShade, PhysxSchema
import logging
from carb import Float3
from omni.isaac.debug_draw import _debug_draw
from omni_drones.utils.volleyball.common import (
    rectangular_cuboid_edges,_carb_float3_add, 
    draw_court, 
    calculate_ball_pass_the_net,
    calculate_ball_hit_the_net,
    calculate_ball_in_side,
    calculate_drone_hit_the_net,
    calculate_drone_pass_the_net,
    turn_to_obs, 
    target_to_obs, 
    attacking_target_to_obs, 
    quaternion_multiply, 
    transfer_root_state_to_the_other_side,
    quat_rotate,
    ball_side_to_obs,
    serve_or_rally_to_obs,
    minor_turn_to_obs,
    determine_game_result_3v3,
)
from omni_drones.utils.torchrl.transforms import append_to_h5
from omni.isaac.orbit.sensors import ContactSensorCfg, ContactSensor
from omegaconf import DictConfig
from typing import Tuple, List
from abc import abstractmethod
import os
import pandas as pd

class Coselfplay_Phase_two(IsaacEnv):
    """
    Opponent team:
        Opp_FirstPass drone [5]: first use FirstPass policy, then use FirstPass_hover policy.
        Opp_SecPass drone [0] : first use SecPass policy, then use SecPass_hover policy.
        Opp_Att drone [1]: first use goto policy, then use Att policy, then use Att_hover policy.
    Our team:
        FirstPass drone [2]: first use goto policy, then use FirstPass policy, then use FirstPass_hover.
        SecPass drone [3]: first use goto policy, then use SecPass policy, then use SecPass_hover.
        Att drone [4]: first use goto policy, then use Att policy, then use Att_hover.
    """

    def __init__(self, cfg, headless):

        self.L: float = cfg.task.court.L
        self.W: float = cfg.task.court.W
        self.H_NET: float = cfg.task.court.H_NET # height of the net
        self.W_NET: float = cfg.task.court.W_NET # not width of the net, but the width of the net's frame

        self.ball_mass: float = cfg.task.ball_mass
        self.ball_radius: float = cfg.task.ball_radius
        
        super().__init__(cfg, headless)

        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )
        print(f"Central env position:{self.central_env_pos}")

        self.drone.initialize()

        self.ball = RigidPrimView(
            "/World/envs/env_*/ball",
            reset_xform_properties=False,
            track_contact_forces=False, # set false because we use contact sensor to get contact forces
            shape=(-1, 1)
        )
        self.ball.initialize()

        self.ball_pos_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.ball_pos_dist.low, device=self.device), 
            high=torch.tensor(cfg.task.initial.ball_pos_dist.high, device=self.device)
        )
        self.ball_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.ball_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.ball_vel_dist.high, device=self.device)
        )

        self.serve_pos_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.serve_pos_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.serve_pos_dist.high, device=self.device)
        )
        self.receive_pos_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.receive_pos_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.receive_pos_dist.high, device=self.device)
        )

        self.Opp_FirstPass_rpy_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_FirstPass_rpy_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_FirstPass_rpy_dist.high, device=self.device)
        )
        self.Opp_FirstPass_lin_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_FirstPass_lin_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_FirstPass_lin_vel_dist.high, device=self.device)
        )
        self.Opp_FirstPass_ang_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_FirstPass_ang_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_FirstPass_ang_vel_dist.high, device=self.device)
        )

        self.Opp_SecPass_pos_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_SecPass_pos_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_SecPass_pos_dist.high, device=self.device)
        )
        self.Opp_SecPass_rpy_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_SecPass_rpy_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_SecPass_rpy_dist.high, device=self.device)
        )
        self.Opp_SecPass_lin_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_SecPass_lin_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_SecPass_lin_vel_dist.high, device=self.device)
        )
        self.Opp_SecPass_ang_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_SecPass_ang_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_SecPass_ang_vel_dist.high, device=self.device)
        )

        self.Opp_Att_pos_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_Att_pos_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_Att_pos_dist.high, device=self.device)
        )
        self.Opp_Att_rpy_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_Att_rpy_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_Att_rpy_dist.high, device=self.device)
        )
        self.Opp_Att_lin_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_Att_lin_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_Att_lin_vel_dist.high, device=self.device)
        )
        self.Opp_Att_ang_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_Att_ang_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_Att_ang_vel_dist.high, device=self.device)
        )
        
        self.FirstPass_rpy_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.FirstPass_rpy_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.FirstPass_rpy_dist.high, device=self.device)
        )
        self.FirstPass_lin_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.FirstPass_lin_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.FirstPass_lin_vel_dist.high, device=self.device)
        )
        self.FirstPass_ang_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.FirstPass_ang_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.FirstPass_ang_vel_dist.high, device=self.device)
        )

        self.SecPass_pos_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.SecPass_pos_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.SecPass_pos_dist.high, device=self.device)
        )
        self.SecPass_rpy_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.SecPass_rpy_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.SecPass_rpy_dist.high, device=self.device)
        )
        self.SecPass_lin_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.SecPass_lin_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.SecPass_lin_vel_dist.high, device=self.device)
        )
        self.SecPass_ang_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.SecPass_ang_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.SecPass_ang_vel_dist.high, device=self.device)
        )

        self.Att_pos_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Att_pos_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Att_pos_dist.high, device=self.device)
        )
        self.Att_rpy_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Att_rpy_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Att_rpy_dist.high, device=self.device)
        )
        self.Att_lin_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Att_lin_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Att_lin_vel_dist.high, device=self.device)
        )
        self.Att_ang_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Att_ang_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Att_ang_vel_dist.high, device=self.device)
        )
        
        self.ball_traj_vis = []
        self.draw = _debug_draw.acquire_debug_draw_interface()

        self.random_turn = cfg.task.random_turn
        self.serve_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # 0: near side team serve; 1: far side team serve
        self.is_rally = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # 0: serve; 1: rally
        self.serve_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: serve/goto; 1: serve_hover/receive; 2: serve_hover/receive_hover

        self.switch_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.ball_init_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.ball_last_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.ball_already_hit_the_ground = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        self.ball_first_hit_the_ground = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)

        hover_target_rpy = torch.zeros((1, 1, 3), device=self.device) # (1, 1, 3)
        hover_target_rot = euler_to_quaternion(hover_target_rpy) # (1, 1, 4)
        self.hover_target_heading = quat_axis(hover_target_rot.squeeze(1), 0).unsqueeze(1).expand(self.num_envs, 1, 3) # (E, 1, 3)

        self.Opp_FirstPass_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: not my turn; 1: my turn;
        self.Opp_FirstPass_already_hit = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        self.Opp_FirstPass_last_hit_t = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.Opp_FirstPass_goto_left_or_right = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # 0: left; 1: right
        self.Opp_FirstPass_goto_pos_left_before_hit = torch.tensor(cfg.task.get("Opp_FirstPass_goto_pos_before_hit"), device=self.device).expand(self.num_envs, 1, 3)
        self.Opp_FirstPass_goto_pos_right_before_hit = self.Opp_FirstPass_goto_pos_left_before_hit.clone()
        self.Opp_FirstPass_goto_pos_right_before_hit[..., 1] = - self.Opp_FirstPass_goto_pos_right_before_hit[..., 1]
        self.Opp_FirstPass_goto_pos_before_hit = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.Opp_FirstPass_hover_pos_after_hit = torch.tensor(cfg.task.get("Opp_FirstPass_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)

        self.Opp_SecPass_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: not my turn; 1: my turn;
        self.Opp_SecPass_already_hit = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        self.Opp_SecPass_last_hit_t = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.Opp_SecPass_goto_pos_before_hit = torch.tensor(cfg.task.get("Opp_SecPass_goto_pos_before_hit"), device=self.device).expand(self.num_envs, 1, 3)
        self.Opp_SecPass_hover_pos_after_hit = torch.tensor(cfg.task.get("Opp_SecPass_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)
        
        self.Opp_Att_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: not my turn; 1: my turn;
        self.Opp_Att_attacking_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # 0: left; 1: right
        self.Opp_Att_already_hit = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        self.Opp_Att_last_hit_t = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.Opp_Att_goto_pos_before_hit = torch.tensor(cfg.task.get("Opp_Att_goto_pos_before_hit"), device=self.device).expand(self.num_envs, 1, 3)
        self.Opp_Att_hover_pos_after_hit = torch.tensor(cfg.task.get("Opp_Att_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)

        self.FirstPass_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: not my turn; 1: my turn;
        self.FirstPass_already_hit = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        self.FirstPass_last_hit_t = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.FirstPass_goto_left_or_right = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # 0: left; 1: right
        self.FirstPass_goto_pos_left_before_hit = torch.tensor(cfg.task.get("FirstPass_goto_pos_before_hit"), device=self.device).expand(self.num_envs, 1, 3)
        self.FirstPass_goto_pos_right_before_hit = self.FirstPass_goto_pos_left_before_hit.clone()
        self.FirstPass_goto_pos_right_before_hit[..., 1] = - self.FirstPass_goto_pos_right_before_hit[..., 1]
        self.FirstPass_goto_pos_before_hit = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.FirstPass_hover_pos_after_hit = torch.tensor(cfg.task.get("FirstPass_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)

        self.SecPass_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: not my turn; 1: my turn;
        self.SecPass_already_hit = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        self.SecPass_last_hit_t = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.SecPass_goto_pos_before_hit = torch.tensor(cfg.task.get("SecPass_goto_pos_before_hit"), device=self.device).expand(self.num_envs, 1, 3)
        self.SecPass_hover_pos_after_hit = torch.tensor(cfg.task.get("SecPass_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)

        self.Att_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: not my turn; 1: my turn;
        self.Att_attacking_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # 0: left; 1: right
        self.Att_already_hit = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)
        self.Att_last_hit_t = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.Att_goto_pos_before_hit = torch.tensor(cfg.task.get("Att_goto_pos_before_hit"), device=self.device).expand(self.num_envs, 1, 3)
        self.Att_hover_pos_after_hit = torch.tensor(cfg.task.get("Att_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)

        self.racket_hit_ball = torch.zeros((self.num_envs, self.drone.n), device=self.device, dtype=torch.bool)
        self.racket_near_ball = torch.zeros((cfg.task.env.num_envs, self.drone.n), device=self.device, dtype=torch.bool)
        self.drone_near_ball = torch.zeros((cfg.task.env.num_envs, self.drone.n), device=self.device, dtype=torch.bool)

        attacking_target_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        attacking_target_right = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.FirstPass_obs_left_attacking_target = attacking_target_to_obs(attacking_target_right, Att_or_FirstPass=False) # both used by FirstPass and Opp_FirstPass
        self.FirstPass_obs_right_attacking_target = attacking_target_to_obs(attacking_target_left, Att_or_FirstPass=False) # both used by FirstPass and Opp_FirstPass
        self.FirstPass_obs_attacking_target = torch.zeros_like(self.FirstPass_obs_left_attacking_target)
        self.Opp_FirstPass_obs_attacking_target = torch.zeros_like(self.FirstPass_obs_left_attacking_target)

        self.ball_pass_net = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.high_level_obs_already_hit_in_one_turn = torch.zeros((self.num_envs, 2, 6), device=self.device, dtype=torch.bool)
        self.ball_side = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # 0: near_side; 1: far_side
        self.last_hit_side = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # for game result 0: near_side; 1: far_side 

        # serve
        self.FirstPass_serve_ball_rpos = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.Opp_FirstPass_serve_ball_rpos = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.serve_ball_anchor = torch.zeros((self.num_envs, 1, 3), device=self.device) 

        # receive
        self.Opp_FirstPass_goto_receive_pos = torch.tensor(cfg.task.get("Opp_FirstPass_goto_receive_pos"), device=self.device).expand(self.num_envs, 1, 3)
        self.FirstPass_goto_receive_pos = torch.tensor(cfg.task.get("FirstPass_goto_receive_pos"), device=self.device).expand(self.num_envs, 1, 3)

        self.drone_idx_dict = {
            "Opp_FirstPass": 5,
            "Opp_SecPass": 0,
            "Opp_Att": 1,
            "FirstPass": 2,
            "SecPass": 3,
            "Att": 4
        }
        self.near_side_drone_idx = [2, 3, 4]
        self.far_side_drone_idx = [0, 1, 5]

        self.not_reset_keys_in_stats = [
            "actor_0_wins",
            "actor_1_wins",
            "draws",
            "terminated",
            "truncated",
            "done",
        ]

        self.hit_ground_height = 0.3

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        material = materials.PhysicsMaterial(
            prim_path="/World/Physics_Materials/physics_material_0",
            restitution=0.8,
        )
        ball = objects.DynamicSphere(
            prim_path="/World/envs/env_0/ball",
            radius=self.ball_radius,
            mass=self.ball_mass,
            color=torch.tensor([1., .2, .2]),
            physics_material=material,
        )

        if self.use_local_usd:
            # use local usd resources
            usd_path = os.path.join(os.path.dirname(__file__), os.pardir, "assets", "default_environment.usd")
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                usd_path=usd_path
            )
        else:
            # use online usd resources
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )

        # placeholders
        drone_prims = self.drone.spawn(
            translations=[
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                [0.0, 0.0, 0.0],
            ]
        )

        material = UsdShade.Material(material.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("base_link/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

        return ["/World/defaultGroundPlane"]


    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]  # 23

        Opp_FirstPass_goto_observation_dim = drone_state_dim + 3
        Opp_FirstPass_observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 2
        Opp_FirstPass_hover_observation_dim = drone_state_dim + 3
        Opp_FirstPass_serve_observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 3
        Opp_FirstPass_serve_hover_observation_dim = drone_state_dim + 3
        Opp_FirstPass_receive_observation_dim = drone_state_dim + 3 + 3 + 3 + 2
        Opp_FirstPass_receive_hover_observation_dim = drone_state_dim + 3

        Opp_SecPass_goto_observation_dim = drone_state_dim + 3
        Opp_SecPass_observation_dim = drone_state_dim + 3 + 3 + 3 + 2
        Opp_SecPass_hover_observation_dim = drone_state_dim + 3
        
        Opp_Att_goto_observation_dim = drone_state_dim + 3
        Opp_Att_observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 2
        Opp_Att_hover_observation_dim = drone_state_dim + 3
        
        FirstPass_goto_observation_dim = drone_state_dim + 3
        FirstPass_observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 2
        FirstPass_hover_observation_dim = drone_state_dim + 3
        FirstPass_serve_observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 3
        FirstPass_serve_hover_observation_dim = drone_state_dim + 3
        FirstPass_receive_observation_dim = drone_state_dim + 3 + 3 + 3 + 2
        FirstPass_receive_hover_observation_dim = drone_state_dim + 3
        
        SecPass_goto_observation_dim = drone_state_dim + 3
        SecPass_observation_dim = drone_state_dim + 3 + 3 + 3 + 2
        SecPass_hover_observation_dim = drone_state_dim + 3
        
        Att_goto_observation_dim = drone_state_dim + 3
        Att_observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 2
        Att_hover_observation_dim = drone_state_dim + 3

        high_level_observation_dim = (3 + 4 + 3 + 3) * 6 + 3 + 3 + 12 + 2 + 2
        # 6 drones: pos, rot, vel, angular_vel
        # ball: pos, vel
        # already hit the ball: 12 dim
        #    our side: one-hot 2 dim * 3
        #    opponent side: one-hot 2 dim * 3
        # side: one-hot 2 dim
        # serve or rally: one-hot 2 dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "Opp_FirstPass_goto_observation": UnboundedContinuousTensorSpec((1, Opp_FirstPass_goto_observation_dim)),
                "Opp_FirstPass_observation": UnboundedContinuousTensorSpec((1, Opp_FirstPass_observation_dim)),
                "Opp_FirstPass_hover_observation": UnboundedContinuousTensorSpec((1, Opp_FirstPass_hover_observation_dim)),
                "Opp_FirstPass_serve_observation": UnboundedContinuousTensorSpec((1, Opp_FirstPass_serve_observation_dim)),
                "Opp_FirstPass_serve_hover_observation": UnboundedContinuousTensorSpec((1, Opp_FirstPass_serve_hover_observation_dim)),
                "Opp_FirstPass_receive_observation": UnboundedContinuousTensorSpec((1, Opp_FirstPass_receive_observation_dim)),
                "Opp_FirstPass_receive_hover_observation": UnboundedContinuousTensorSpec((1, Opp_FirstPass_receive_hover_observation_dim)),
                
                "Opp_SecPass_goto_observation": UnboundedContinuousTensorSpec((1, Opp_SecPass_goto_observation_dim)),
                "Opp_SecPass_observation": UnboundedContinuousTensorSpec((1, Opp_SecPass_observation_dim)),
                "Opp_SecPass_hover_observation": UnboundedContinuousTensorSpec((1, Opp_SecPass_hover_observation_dim)),
                
                "Opp_Att_goto_observation": UnboundedContinuousTensorSpec((1, Opp_Att_goto_observation_dim)),
                "Opp_Att_observation": UnboundedContinuousTensorSpec((1, Opp_Att_observation_dim)),
                "Opp_Att_hover_observation": UnboundedContinuousTensorSpec((1, Opp_Att_hover_observation_dim)),
                
                "FirstPass_goto_observation": UnboundedContinuousTensorSpec((1, FirstPass_goto_observation_dim)),
                "FirstPass_observation": UnboundedContinuousTensorSpec((1, FirstPass_observation_dim)),
                "FirstPass_hover_observation": UnboundedContinuousTensorSpec((1, FirstPass_hover_observation_dim)),
                "FirstPass_serve_observation": UnboundedContinuousTensorSpec((1, FirstPass_serve_observation_dim)),
                "FirstPass_serve_hover_observation": UnboundedContinuousTensorSpec((1, FirstPass_serve_hover_observation_dim)),
                "FirstPass_receive_observation": UnboundedContinuousTensorSpec((1, FirstPass_receive_observation_dim)),
                "FirstPass_receive_hover_observation": UnboundedContinuousTensorSpec((1, FirstPass_receive_hover_observation_dim)),
                
                "SecPass_goto_observation": UnboundedContinuousTensorSpec((1, SecPass_goto_observation_dim)),
                "SecPass_observation": UnboundedContinuousTensorSpec((1, SecPass_observation_dim)),
                "SecPass_hover_observation": UnboundedContinuousTensorSpec((1, SecPass_hover_observation_dim)),
                
                "Att_goto_observation": UnboundedContinuousTensorSpec((1, Att_goto_observation_dim)),
                "Att_observation": UnboundedContinuousTensorSpec((1, Att_observation_dim)),
                "Att_hover_observation": UnboundedContinuousTensorSpec((1, Att_hover_observation_dim)),

                "high_level_observation": UnboundedContinuousTensorSpec((2, high_level_observation_dim)),
            })
        }).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "Opp_FirstPass_goto_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Opp_FirstPass_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Opp_FirstPass_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Opp_FirstPass_serve_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Opp_FirstPass_serve_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Opp_FirstPass_receive_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Opp_FirstPass_receive_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),

                "Opp_SecPass_goto_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Opp_SecPass_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Opp_SecPass_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),

                "Opp_Att_goto_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Opp_Att_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Opp_Att_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),

                "FirstPass_goto_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "FirstPass_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "FirstPass_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "FirstPass_serve_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "FirstPass_serve_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "FirstPass_receive_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "FirstPass_receive_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),

                "SecPass_goto_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "SecPass_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "SecPass_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),

                "SecPass_new_action": torch.stack([self.drone.action_spec] * 1, dim=0),

                "Att_goto_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Att_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Att_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),

                # multi-head high_level_action
                # nvec: [6, 3, 4] -> [FirstPass has 6 skills, SecPass has 3 skills, Att has 4 skills]
                # shape: [2, 3] -> [2 teams, 3 drones: FirstPass, SecPass, Att]
                "high_level_old_action": MultiDiscreteTensorSpec(nvec=(6, 3, 4), shape=(2, 3), dtype=torch.long, device=self.device),
                "high_level_action": MultiDiscreteTensorSpec(nvec=(6, 4, 4), shape=(2, 3), dtype=torch.long, device=self.device)
            })
        }).expand(self.num_envs).to(self.device)

        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "Opp_FirstPass_reward": UnboundedContinuousTensorSpec((1, 1)),
                "Opp_SecPass_reward": UnboundedContinuousTensorSpec((1, 1)),
                "Opp_Att_reward": UnboundedContinuousTensorSpec((1, 1)),
                "FirstPass_reward": UnboundedContinuousTensorSpec((1, 1)),
                "SecPass_reward": UnboundedContinuousTensorSpec((1, 1)),
                "Att_reward": UnboundedContinuousTensorSpec((1, 1)),
                "high_level_reward": UnboundedContinuousTensorSpec((2, 1)),
            })
        }).expand(self.num_envs).to(self.device)

        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["Opp_FirstPass_goto"] = AgentSpec(
            "Opp_FirstPass_goto",
            1,
            observation_key=("agents", "Opp_FirstPass_goto_observation"),
            action_key=("agents", "Opp_FirstPass_goto_action"),
            reward_key=("agents", "Opp_FirstPass_reward"),
            state_key=("agents", "Opp_FirstPass_goto_state"),
        )
        self.agent_spec["Opp_FirstPass"] = AgentSpec(
            "Opp_FirstPass",
            1,
            observation_key=("agents", "Opp_FirstPass_observation"),
            action_key=("agents", "Opp_FirstPass_action"),
            reward_key=("agents", "Opp_FirstPass_reward"),
            state_key=("agents", "Opp_FirstPass_state"),
        )
        self.agent_spec["Opp_FirstPass_hover"] = AgentSpec(
            "Opp_FirstPass_hover",
            1,
            observation_key=("agents", "Opp_FirstPass_hover_observation"),
            action_key=("agents", "Opp_FirstPass_hover_action"),
            reward_key=("agents", "Opp_FirstPass_reward"),
            state_key=("agents", "Opp_FirstPass_hover_state"),
        )
        self.agent_spec["Opp_FirstPass_serve"] = AgentSpec(
            "Opp_FirstPass_serve",
            1,
            observation_key=("agents", "Opp_FirstPass_serve_observation"),
            action_key=("agents", "Opp_FirstPass_serve_action"),
            reward_key=("agents", "Opp_FirstPass_reward"),
            state_key=("agents", "Opp_FirstPass_serve_state"),
        )
        self.agent_spec["Opp_FirstPass_serve_hover"] = AgentSpec(
            "Opp_FirstPass_serve_hover",
            1,
            observation_key=("agents", "Opp_FirstPass_serve_hover_observation"),
            action_key=("agents", "Opp_FirstPass_serve_hover_action"),
            reward_key=("agents", "Opp_FirstPass_reward"),
            state_key=("agents", "Opp_FirstPass_serve_hover_state"),
        )
        self.agent_spec["Opp_FirstPass_receive"] = AgentSpec(
            "Opp_FirstPass_receive",
            1,
            observation_key=("agents", "Opp_FirstPass_receive_observation"),
            action_key=("agents", "Opp_FirstPass_receive_action"),
            reward_key=("agents", "Opp_FirstPass_reward"),
            state_key=("agents", "Opp_FirstPass_receive_state"),
        )
        self.agent_spec["Opp_FirstPass_receive_hover"] = AgentSpec(
            "Opp_FirstPass_receive_hover",
            1,
            observation_key=("agents", "Opp_FirstPass_receive_hover_observation"),
            action_key=("agents", "Opp_FirstPass_receive_hover_action"),
            reward_key=("agents", "Opp_FirstPass_reward"),
            state_key=("agents", "Opp_FirstPass_receive_hover_state"),
        )

        self.agent_spec["Opp_SecPass_goto"] = AgentSpec(
            "Opp_SecPass_goto",
            1,
            observation_key=("agents", "Opp_SecPass_goto_observation"),
            action_key=("agents", "Opp_SecPass_goto_action"),
            reward_key=("agents", "Opp_SecPass_reward"),
            state_key=("agents", "Opp_SecPass_goto_state"),
        )
        self.agent_spec["Opp_SecPass"] = AgentSpec(
            "Opp_SecPass",
            1,
            observation_key=("agents", "Opp_SecPass_observation"),
            action_key=("agents", "Opp_SecPass_action"),
            reward_key=("agents", "Opp_SecPass_reward"),
            state_key=("agents", "Opp_SecPass_state"),
        )
        self.agent_spec["Opp_SecPass_hover"] = AgentSpec(
            "Opp_SecPass_hover",
            1,
            observation_key=("agents", "Opp_SecPass_hover_observation"),
            action_key=("agents", "Opp_SecPass_hover_action"),
            reward_key=("agents", "Opp_SecPass_reward"),
            state_key=("agents", "Opp_SecPass_hover_state"),
        )

        self.agent_spec["Opp_Att_goto"] = AgentSpec(
            "Opp_Att_goto",
            1,
            observation_key=("agents", "Opp_Att_goto_observation"),
            action_key=("agents", "Opp_Att_goto_action"),
            reward_key=("agents", "Opp_Att_reward"),
            state_key=("agents", "Opp_Att_goto_state"),
        )
        self.agent_spec["Opp_Att"] = AgentSpec(
            "Opp_Att",
            1,
            observation_key=("agents", "Opp_Att_observation"),
            action_key=("agents", "Opp_Att_action"),
            reward_key=("agents", "Opp_Att_reward"),
            state_key=("agents", "Opp_Att_state"),
        )
        self.agent_spec["Opp_Att_hover"] = AgentSpec(
            "Opp_Att_hover",
            1,
            observation_key=("agents", "Opp_Att_hover_observation"),
            action_key=("agents", "Opp_Att_hover_action"),
            reward_key=("agents", "Opp_Att_reward"),
            state_key=("agents", "Opp_Att_hover_state"),
        )

        self.agent_spec["FirstPass_goto"] = AgentSpec(
            "FirstPass_goto",
            1,
            observation_key=("agents", "FirstPass_goto_observation"),
            action_key=("agents", "FirstPass_goto_action"),
            reward_key=("agents", "FirstPass_reward"),
            state_key=("agents", "FirstPass_goto_state"),
        )
        self.agent_spec["FirstPass"] = AgentSpec(
            "FirstPass",
            1,
            observation_key=("agents", "FirstPass_observation"),
            action_key=("agents", "FirstPass_action"),
            reward_key=("agents", "FirstPass_reward"),
            state_key=("agents", "FirstPass_state"),
        )
        self.agent_spec["FirstPass_serve"] = AgentSpec(
            "FirstPass_serve",
            1,
            observation_key=("agents", "FirstPass_serve_observation"),
            action_key=("agents", "FirstPass_serve_action"),
            reward_key=("agents", "FirstPass_reward"),
            state_key=("agents", "FirstPass_serve_state"),
        )
        self.agent_spec["FirstPass_serve_hover"] = AgentSpec(
            "FirstPass_serve_hover",
            1,
            observation_key=("agents", "FirstPass_serve_hover_observation"),
            action_key=("agents", "FirstPass_serve_hover_action"),
            reward_key=("agents", "FirstPass_reward"),
            state_key=("agents", "FirstPass_serve_hover_state"),
        )
        self.agent_spec["FirstPass_receive"] = AgentSpec(
            "FirstPass_receive",
            1,
            observation_key=("agents", "FirstPass_receive_observation"),
            action_key=("agents", "FirstPass_receive_action"),
            reward_key=("agents", "FirstPass_reward"),
            state_key=("agents", "FirstPass_receive_state"),
        )
        self.agent_spec["FirstPass_receive_hover"] = AgentSpec(
            "FirstPass_receive_hover",
            1,
            observation_key=("agents", "FirstPass_receive_hover_observation"),
            action_key=("agents", "FirstPass_receive_hover_action"),
            reward_key=("agents", "FirstPass_reward"),
            state_key=("agents", "FirstPass_receive_hover_state"),
        )
        self.agent_spec["FirstPass_hover"] = AgentSpec(
            "FirstPass_hover",
            1,
            observation_key=("agents", "FirstPass_hover_observation"),
            action_key=("agents", "FirstPass_hover_action"),
            reward_key=("agents", "FirstPass_reward"),
            state_key=("agents", "FirstPass_hover_state"),
        )

        self.agent_spec["SecPass_goto"] = AgentSpec(
            "SecPass_goto",
            1,
            observation_key=("agents", "SecPass_goto_observation"),
            action_key=("agents", "SecPass_goto_action"),
            reward_key=("agents", "SecPass_reward"),
            state_key=("agents", "SecPass_goto_state"),
        )
        self.agent_spec["SecPass"] = AgentSpec(
            "SecPass",
            1,
            observation_key=("agents", "SecPass_observation"),
            action_key=("agents", "SecPass_action"),
            reward_key=("agents", "SecPass_reward"),
            state_key=("agents", "SecPass_state"),
        )
        self.agent_spec["SecPass_hover"] = AgentSpec(
            "SecPass_hover",
            1,
            observation_key=("agents", "SecPass_hover_observation"),
            action_key=("agents", "SecPass_hover_action"),
            reward_key=("agents", "SecPass_reward"),
            state_key=("agents", "SecPass_hover_state"),
        )

        self.agent_spec["SecPass_new"] = AgentSpec(
            "SecPass_new",
            1,
            observation_key=("agents", "SecPass_observation"),
            action_key=("agents", "SecPass_new_action"),
            reward_key=("agents", "SecPass_reward"),
            state_key=("agents", "SecPass_state"),
        )

        self.agent_spec["Att_goto"] = AgentSpec(
            "Att_goto",
            1,
            observation_key=("agents", "Att_goto_observation"),
            action_key=("agents", "Att_goto_action"),
            reward_key=("agents", "Att_reward"),
            state_key=("agents", "Att_goto_state"),
        )
        self.agent_spec["Att"] = AgentSpec(
            "Att",
            1,
            observation_key=("agents", "Att_observation"),
            action_key=("agents", "Att_action"),
            reward_key=("agents", "Att_reward"),
            state_key=("agents", "Att_state"),
        )
        self.agent_spec["Att_hover"] = AgentSpec(
            "Att_hover",
            1,
            observation_key=("agents", "Att_hover_observation"),
            action_key=("agents", "Att_hover_action"),
            reward_key=("agents", "Att_reward"),
            state_key=("agents", "Att_hover_state"),
        )

        self.agent_spec["high_level"] = AgentSpec(
            "high_level",
            2,
            observation_key=("agents", "high_level_observation"),
            action_key=("agents", "high_level_action"),
            reward_key=("agents", "high_level_reward"),
            state_key=("agents", "high_level_state"),
        )

        self.agent_spec["high_level_old"] = AgentSpec(
            "high_level_old",
            2,
            observation_key=("agents", "high_level_observation"),
            action_key=("agents", "high_level_old_action"),
            reward_key=("agents", "high_level_reward"),
            state_key=("agents", "high_level_state"),
        )

        

        _stats_spec = CompositeSpec({
            # "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),

            "done": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
            "terminated": UnboundedContinuousTensorSpec(1),
           
            "num_Opp_FirstPass_hit": UnboundedContinuousTensorSpec(1),
            "num_Opp_SecPass_hit": UnboundedContinuousTensorSpec(1),
            "num_Opp_Att_hit": UnboundedContinuousTensorSpec(1),
            "num_FirstPass_hit": UnboundedContinuousTensorSpec(1),
            "num_SecPass_hit": UnboundedContinuousTensorSpec(1),
            "num_Att_hit": UnboundedContinuousTensorSpec(1),
            "num_hit": UnboundedContinuousTensorSpec(1),

            "Att_hit_left": UnboundedContinuousTensorSpec(1),
            "Att_hit_right": UnboundedContinuousTensorSpec(1),
            "FirstPass_goto_left": UnboundedContinuousTensorSpec(1),
            "FirstPass_goto_right": UnboundedContinuousTensorSpec(1),
            "Opp_Att_hit_left": UnboundedContinuousTensorSpec(1),
            "Opp_Att_hit_right": UnboundedContinuousTensorSpec(1),
            "Opp_FirstPass_goto_left": UnboundedContinuousTensorSpec(1),
            "Opp_FirstPass_goto_right": UnboundedContinuousTensorSpec(1),
            
            "Att_hit_empty_spot": UnboundedContinuousTensorSpec(1),
            "Att_hit_wrong_spot": UnboundedContinuousTensorSpec(1),
            "Opp_Att_hit_empty_spot": UnboundedContinuousTensorSpec(1),
            "Opp_Att_hit_wrong_spot": UnboundedContinuousTensorSpec(1),

            "done_drone_hit_the_ground": UnboundedContinuousTensorSpec(1),
            "done_drone_pass_net": UnboundedContinuousTensorSpec(1),
            "done_drone_wrong_hit_turn": UnboundedContinuousTensorSpec(1),
            "done_drong_wrong_hit_racket": UnboundedContinuousTensorSpec(1),
            "done_ball_hit_the_ground": UnboundedContinuousTensorSpec(1),
            "done_ball_hit_the_net": UnboundedContinuousTensorSpec(1),

            "actor_0_wins": UnboundedContinuousTensorSpec(1),
            "actor_1_wins": UnboundedContinuousTensorSpec(1),
            "draws": UnboundedContinuousTensorSpec(1),

            "actor_0_serve": UnboundedContinuousTensorSpec(1),
            "actor_0_serve_and_win": UnboundedContinuousTensorSpec(1),
            "actor_0_serve_and_lose": UnboundedContinuousTensorSpec(1),
            "actor_1_not_serve_and_win": UnboundedContinuousTensorSpec(1),
            "actor_1_not_serve_and_lose": UnboundedContinuousTensorSpec(1),
            
            "actor_1_serve": UnboundedContinuousTensorSpec(1),
            "actor_1_serve_and_win": UnboundedContinuousTensorSpec(1),
            "actor_1_serve_and_lose": UnboundedContinuousTensorSpec(1),
            "actor_0_not_serve_and_win": UnboundedContinuousTensorSpec(1),
            "actor_0_not_serve_and_lose": UnboundedContinuousTensorSpec(1),

            "FirstPass_high_level_0": UnboundedContinuousTensorSpec(1),
            "FirstPass_high_level_1": UnboundedContinuousTensorSpec(1),
            "FirstPass_high_level_2": UnboundedContinuousTensorSpec(1),
            "FirstPass_high_level_3": UnboundedContinuousTensorSpec(1),
            "FirstPass_high_level_4": UnboundedContinuousTensorSpec(1),
            "FirstPass_high_level_5": UnboundedContinuousTensorSpec(1),
            "SecPass_high_level_0": UnboundedContinuousTensorSpec(1),
            "SecPass_high_level_1": UnboundedContinuousTensorSpec(1),
            "SecPass_high_level_2": UnboundedContinuousTensorSpec(1),
            "Att_high_level_0": UnboundedContinuousTensorSpec(1),
            "Att_high_level_1": UnboundedContinuousTensorSpec(1),
            "Att_high_level_2": UnboundedContinuousTensorSpec(1),
            "Att_high_level_3": UnboundedContinuousTensorSpec(1),

            "Opp_FirstPass_high_level_0": UnboundedContinuousTensorSpec(1),
            "Opp_FirstPass_high_level_1": UnboundedContinuousTensorSpec(1),
            "Opp_FirstPass_high_level_2": UnboundedContinuousTensorSpec(1),
            "Opp_FirstPass_high_level_3": UnboundedContinuousTensorSpec(1),
            "Opp_FirstPass_high_level_4": UnboundedContinuousTensorSpec(1),
            "Opp_FirstPass_high_level_5": UnboundedContinuousTensorSpec(1),
            "Opp_SecPass_high_level_0": UnboundedContinuousTensorSpec(1),
            "Opp_SecPass_high_level_1": UnboundedContinuousTensorSpec(1),
            "Opp_SecPass_high_level_2": UnboundedContinuousTensorSpec(1),
            "Opp_Att_high_level_0": UnboundedContinuousTensorSpec(1),
            "Opp_Att_high_level_1": UnboundedContinuousTensorSpec(1),
            "Opp_Att_high_level_2": UnboundedContinuousTensorSpec(1),
            "Opp_Att_high_level_3": UnboundedContinuousTensorSpec(1),

            "SecPass_high_level_3": UnboundedContinuousTensorSpec(1)
        })

        self.stats_cfg: DictConfig = self.cfg.task.stats
        if self.stats_cfg.get("complete_reward_stats", False):
            _stats_spec["return_near_team"] = UnboundedContinuousTensorSpec(1)
            _stats_spec["reward_hit_near_team"] = UnboundedContinuousTensorSpec(1)
            _stats_spec["reward_FirstPass_hit_near_team"] = UnboundedContinuousTensorSpec(1)
            _stats_spec["penalty_too_close_near_team"] = UnboundedContinuousTensorSpec(1)
            _stats_spec["reward_win_near_team"] = UnboundedContinuousTensorSpec(1)

            _stats_spec["return_far_team"] = UnboundedContinuousTensorSpec(1)
            _stats_spec["reward_hit_far_team"] = UnboundedContinuousTensorSpec(1)
            _stats_spec["reward_FirstPass_hit_far_team"] = UnboundedContinuousTensorSpec(1)
            _stats_spec["penalty_too_close_far_team"] = UnboundedContinuousTensorSpec(1)
            _stats_spec["reward_win_far_team"] = UnboundedContinuousTensorSpec(1)
        
        stats_spec = _stats_spec.expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),

            "high_level_old_action": UnboundedContinuousTensorSpec((2, 3)),
            "high_level_action": UnboundedContinuousTensorSpec((2, 3)),

            "turn": UnboundedContinuousTensorSpec((1,), dtype=torch.long),
            "switch_turn": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            
            "Opp_FirstPass_hit": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "Opp_SecPass_hit": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "Opp_Att_hit": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "FirstPass_hit": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "SecPass_hit": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "Att_hit": DiscreteTensorSpec(2, (1,), dtype=torch.bool),

            "central_env_idx": UnboundedContinuousTensorSpec((1,), dtype=torch.long),
        }).expand(self.num_envs).to(self.device)
        
        self.observation_spec["info"] = info_spec
        
        self.info = info_spec.zero()
        self.info["central_env_idx"] = self.central_env_idx.long().item() * torch.ones((self.num_envs,), device=self.device, dtype=torch.long)
        
        self.info["high_level_old_action"] = - torch.ones((self.num_envs, 2, 3), device=self.device)
        self.info["high_level_action"] = - torch.ones((self.num_envs, 2, 3), device=self.device)

    
    def debug_draw_win(self, near_side_win, far_side_win, draw):
        env_id = self.central_env_idx
        ori = self.envs_positions[env_id].detach()
        
        point_win_near_side = torch.tensor([2., 0., 0.]).to(self.device) + ori
        point_win_far_side = torch.tensor([-2., 0., 0.]).to(self.device) + ori
        point_draw = torch.tensor([0., 0., 0.]).to(self.device) + ori

        colors = [(1, 1, 0, 1)]
        sizes = [20.]

        if near_side_win[env_id, 0].item() == 1:
            point = point_win_near_side
        elif far_side_win[env_id, 0].item() == 1:
            point = point_win_far_side
        elif draw[env_id, 0].item() == 1:
            point = point_draw
        point = [point.tolist()]

        self.draw.draw_points(point, colors, sizes)


    def _reset_idx(self, env_ids: torch.Tensor):
        self.is_rally[env_ids] = False
        self.serve_step[env_ids] = 0

        if self.random_turn:
            self.serve_turn[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device, dtype=torch.bool) # 1: far side team serve, 0: near side team serve
        else:
            self.serve_turn[env_ids] = torch.ones(len(env_ids), device=self.device, dtype=torch.bool) # always serve from far side
        
        self.ball_side[env_ids] = self.serve_turn[env_ids]
        self.last_hit_side[env_ids] = self.serve_turn[env_ids]
        
        self.Opp_SecPass_already_hit[env_ids] = (self.serve_turn[env_ids] == 1).unsqueeze(-1)
        self.Opp_Att_already_hit[env_ids] = (self.serve_turn[env_ids] == 1).unsqueeze(-1)
        
        self.SecPass_already_hit[env_ids] = (self.serve_turn[env_ids] == 0).unsqueeze(-1)
        self.Att_already_hit[env_ids] = (self.serve_turn[env_ids] == 0).unsqueeze(-1)

        self.drone._reset_idx(env_ids, self.training)

        Opp_FirstPass_pos = torch.where(
            self.serve_turn[env_ids].unsqueeze(-1).unsqueeze(-1), # (E', 1, 1)
            self.serve_pos_dist.sample(env_ids.shape).unsqueeze(1),
            self.receive_pos_dist.sample(env_ids.shape).unsqueeze(1),
        )
        Opp_FirstPass_rpy = self.Opp_FirstPass_rpy_dist.sample(env_ids.shape).unsqueeze(1)
        
        Opp_SecPass_pos = self.Opp_SecPass_pos_dist.sample(env_ids.shape).unsqueeze(1)
        Opp_SecPass_rpy = self.Opp_SecPass_rpy_dist.sample(env_ids.shape).unsqueeze(1)
        
        Opp_Att_pos = self.Opp_Att_pos_dist.sample(env_ids.shape).unsqueeze(1)
        Opp_Att_rpy = self.Opp_Att_rpy_dist.sample(env_ids.shape).unsqueeze(1)
        
        FirstPass_pos = torch.where(
            self.serve_turn[env_ids].unsqueeze(-1).unsqueeze(-1), # (E', 1, 1)
            self.receive_pos_dist.sample(env_ids.shape).unsqueeze(1),
            self.serve_pos_dist.sample(env_ids.shape).unsqueeze(1),
        )
        FirstPass_rpy = self.FirstPass_rpy_dist.sample(env_ids.shape).unsqueeze(1)
        
        SecPass_pos = self.SecPass_pos_dist.sample(env_ids.shape).unsqueeze(1)
        SecPass_rpy = self.SecPass_rpy_dist.sample(env_ids.shape).unsqueeze(1)
        
        Att_pos = self.Att_pos_dist.sample(env_ids.shape).unsqueeze(1)
        Att_rpy = self.Att_rpy_dist.sample(env_ids.shape).unsqueeze(1)

        # set to the far side of the court
        Opp_FirstPass_pos[..., 0:2] = - Opp_FirstPass_pos[..., 0:2]
        Opp_FirstPass_rpy[..., 2] += 3.1415926
        
        Opp_SecPass_pos[..., 0:2] = - Opp_SecPass_pos[..., 0:2]
        Opp_SecPass_rpy[..., 2] += 3.1415926
        
        Opp_Att_pos[..., 0:2] = - Opp_Att_pos[..., 0:2]
        Opp_Att_rpy[..., 2] += 3.1415926

        drone_pos = torch.cat([Opp_SecPass_pos, Opp_Att_pos, FirstPass_pos, SecPass_pos, Att_pos, Opp_FirstPass_pos], dim=1)
        drone_rpy = torch.cat([Opp_SecPass_rpy, Opp_Att_rpy, FirstPass_rpy, SecPass_rpy, Att_rpy, Opp_FirstPass_rpy], dim=1)
        drone_rot = euler_to_quaternion(drone_rpy)

        self.drone.set_world_poses(
            drone_pos +
            self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )

        Opp_FirstPass_lin_vel = self.Opp_FirstPass_lin_vel_dist.sample((*env_ids.shape, 1))
        Opp_FirstPass_ang_vel = self.Opp_FirstPass_ang_vel_dist.sample((*env_ids.shape, 1))
        Opp_FirstPass_vel = torch.cat((Opp_FirstPass_lin_vel, Opp_FirstPass_ang_vel), dim=-1)
        
        Opp_SecPass_lin_vel = self.Opp_SecPass_lin_vel_dist.sample((*env_ids.shape, 1))
        Opp_SecPass_ang_vel = self.Opp_SecPass_ang_vel_dist.sample((*env_ids.shape, 1))
        Opp_SecPass_vel = torch.cat((Opp_SecPass_lin_vel, Opp_SecPass_ang_vel), dim=-1)
        
        Opp_Att_lin_vel = self.Opp_Att_lin_vel_dist.sample((*env_ids.shape, 1))
        Opp_Att_ang_vel = self.Opp_Att_ang_vel_dist.sample((*env_ids.shape, 1))
        Opp_Att_vel = torch.cat((Opp_Att_lin_vel, Opp_Att_ang_vel), dim=-1)
        
        FirstPass_lin_vel = self.FirstPass_lin_vel_dist.sample((*env_ids.shape, 1))
        FirstPass_ang_vel = self.FirstPass_ang_vel_dist.sample((*env_ids.shape, 1))
        FirstPass_vel = torch.cat((FirstPass_lin_vel, FirstPass_ang_vel), dim=-1)
        
        SecPass_lin_vel = self.SecPass_lin_vel_dist.sample((*env_ids.shape, 1))
        SecPass_ang_vel = self.SecPass_ang_vel_dist.sample((*env_ids.shape, 1))
        SecPass_vel = torch.cat((SecPass_lin_vel, SecPass_ang_vel), dim=-1)
        
        Att_lin_vel = self.Att_lin_vel_dist.sample((*env_ids.shape, 1))
        Att_ang_vel = self.Att_ang_vel_dist.sample((*env_ids.shape, 1))
        Att_vel = torch.cat((Att_lin_vel, Att_ang_vel), dim=-1)

        drone_vel = torch.cat([Opp_SecPass_vel, Opp_Att_vel, FirstPass_vel, SecPass_vel, Att_vel, Opp_FirstPass_vel], dim=1)    
        self.drone.set_velocities(drone_vel, env_ids)

        ball_pos = self.ball_pos_dist.sample((*env_ids.shape, 1)) # (E', 1, 3)
        ball_rot = torch.tensor(
            [1., 0., 0., 0.], device=self.device).repeat(len(env_ids), 1)
        ball_lin_vel = self.ball_vel_dist.sample((*env_ids.shape, 1))

        # set to the far side of the court
        ball_pos[..., :2] = torch.where(
            self.serve_turn[env_ids].unsqueeze(-1).unsqueeze(-1), # (E', 1, 1)
            - ball_pos[..., :2],
            ball_pos[..., :2]
        )
        ball_lin_vel[..., :2] = torch.where(
            self.serve_turn[env_ids].unsqueeze(-1).unsqueeze(-1), # (E', 1, 1)
            - ball_lin_vel[..., :2],
            ball_lin_vel[..., :2]
        )
        
        ball_ang_vel = torch.zeros_like(ball_lin_vel)
        ball_vel = torch.cat((ball_lin_vel, ball_ang_vel), dim=-1)
        
        self.ball.set_world_poses(
            ball_pos +
            self.envs_positions[env_ids].unsqueeze(1), ball_rot, env_ids
        )
        self.ball.set_velocities(ball_vel, env_ids)
        self.ball.set_masses(torch.ones_like(env_ids,dtype=torch.float)*self.ball_mass, env_ids)
        self.ball_init_vel[env_ids] = ball_lin_vel
        self.ball_last_vel[env_ids] = ball_lin_vel

        self.serve_ball_anchor[env_ids, 0, 0] = torch.rand(len(env_ids), device=self.device) * 1 + 5  # x: [5, 6]
        self.serve_ball_anchor[env_ids, 0, 1] = torch.rand(len(env_ids), device=self.device) * 1 - 0.5  # y: [-0.5, 0.5]
        self.serve_ball_anchor[env_ids, 0, 2] = 0  # z: 1

        # some stats keys will not reset
        stats_list = []
        for i in self.not_reset_keys_in_stats:
            stats_list.append(self.stats[i][env_ids].clone())
        self.stats[env_ids] = 0.0
        for i, key in enumerate(self.not_reset_keys_in_stats):
            self.stats[key][env_ids] = stats_list[i]

        self.info["high_level_old_action"][env_ids] = -1
        self.info["high_level_action"][env_ids] = -1
        self.info["switch_turn"][env_ids] = True

        self.switch_turn[env_ids] = True
        self.ball_already_hit_the_ground[env_ids] = False
        self.ball_first_hit_the_ground[env_ids] = False
        self.racket_hit_ball[env_ids] = False

        self.Opp_FirstPass_already_hit[env_ids] = False
        self.Opp_FirstPass_last_hit_t[env_ids] = -10

        self.Opp_SecPass_already_hit[env_ids] = False
        self.Opp_SecPass_last_hit_t[env_ids] = -10

        self.Opp_Att_already_hit[env_ids] = False
        self.Opp_Att_last_hit_t[env_ids] = -10

        self.FirstPass_already_hit[env_ids] = False
        self.FirstPass_last_hit_t[env_ids] = -10

        self.SecPass_already_hit[env_ids] = False
        self.SecPass_last_hit_t[env_ids] = -10

        self.Att_already_hit[env_ids] = False
        self.Att_last_hit_t[env_ids] = -10

        self.racket_near_ball[env_ids] = False
        self.drone_near_ball[env_ids] = False

        self.ball_pass_net[env_ids] = False

        if (env_ids == self.central_env_idx).any() and self._should_render(0):
            self.ball_traj_vis.clear()
            self.draw.clear_lines()
            logging.info("=======================================")
            logging.info("Reset central environment.")
            if self.serve_turn[self.central_env_idx].item() == 0:
                logging.info("Near side serves.")
            else:
                logging.info("Far side serves.")

            point_list_1, point_list_2, colors, sizes = draw_court(
                self.W, self.L, self.H_NET, self.W_NET, n=2
            )
            point_list_1 = [
                _carb_float3_add(p, self.central_env_pos) for p in point_list_1
            ]
            point_list_2 = [
                _carb_float3_add(p, self.central_env_pos) for p in point_list_2
            ]
            self.draw.draw_lines(point_list_1, point_list_2, colors, sizes)

            # self.draw.clear_points()
            # self.debug_draw_turn()


    def _pre_sim_step(self, tensordict: TensorDictBase):
        # high-level action
        high_level_old_action = tensordict[("agents", "high_level_old_action")] # (E,2,3)
        high_level_action = tensordict[("agents", "high_level_action")] # (E,2,3)

        # # if Opp_Att attacks, only attacking left
        # high_level_action[:, 1, 2] = torch.where(
        #     high_level_action[:, 1, 2] == 2,
        #     1,
        #     high_level_action[:, 1, 2]
        # )

        # # if Opp_Att attacks, randomly choose left or right
        # high_level_action[:, 1, 2] = torch.where(
        #     ((high_level_action[:, 1, 2] == 1) | (high_level_action[:, 1, 2] == 2)),
        #     torch.randint_like(high_level_action[:, 1, 2], 1, 3),
        #     high_level_action[:, 1, 2]
        # )

        self.info["high_level_old_action"] = torch.where(
            self.info["switch_turn"].unsqueeze(-1).bool(), # (E,1,1)
            high_level_old_action,
            self.info["high_level_old_action"]
        ) # (E,2,3)

        self.info["high_level_action"] = torch.where(
            self.info["switch_turn"].unsqueeze(-1).bool(), # (E,1,1)
            high_level_action,
            self.info["high_level_action"]
        ) # (E,2,3)

        self.stats["FirstPass_high_level_0"] += self.info["switch_turn"] * (high_level_action[:, 0, 0] == 0).unsqueeze(-1) * self.is_rally.unsqueeze(-1)
        self.stats["FirstPass_high_level_1"] += self.info["switch_turn"] * (high_level_action[:, 0, 0] == 1).unsqueeze(-1) * self.is_rally.unsqueeze(-1)
        self.stats["FirstPass_high_level_2"] += self.info["switch_turn"] * (high_level_action[:, 0, 0] == 2).unsqueeze(-1) * self.is_rally.unsqueeze(-1)
        self.stats["FirstPass_high_level_3"] += self.info["switch_turn"] * (high_level_action[:, 0, 0] == 3).unsqueeze(-1) * self.is_rally.unsqueeze(-1)
        self.stats["FirstPass_high_level_4"] += self.info["switch_turn"] * (high_level_action[:, 0, 0] == 4).unsqueeze(-1) * self.is_rally.unsqueeze(-1)
        self.stats["FirstPass_high_level_5"] += self.info["switch_turn"] * (high_level_action[:, 0, 0] == 5).unsqueeze(-1) * self.is_rally.unsqueeze(-1)
        self.stats["SecPass_high_level_0"] += self.info["switch_turn"] * (high_level_action[:, 0, 1] == 0).unsqueeze(-1)
        self.stats["SecPass_high_level_1"] += self.info["switch_turn"] * (high_level_action[:, 0, 1] == 1).unsqueeze(-1)
        self.stats["SecPass_high_level_2"] += self.info["switch_turn"] * (high_level_action[:, 0, 1] == 2).unsqueeze(-1)
        self.stats["Att_high_level_0"] += self.info["switch_turn"] * (high_level_action[:, 0, 2] == 0).unsqueeze(-1)
        self.stats["Att_high_level_1"] += self.info["switch_turn"] * (high_level_action[:, 0, 2] == 1).unsqueeze(-1)
        self.stats["Att_high_level_2"] += self.info["switch_turn"] * (high_level_action[:, 0, 2] == 2).unsqueeze(-1)
        self.stats["Att_high_level_3"] += self.info["switch_turn"] * (high_level_action[:, 0, 2] == 3).unsqueeze(-1)

        self.stats["SecPass_high_level_3"] += self.info["switch_turn"] * (high_level_action[:, 0, 1] == 3).unsqueeze(-1)

        self.stats["Opp_FirstPass_high_level_0"] += self.info["switch_turn"] * (high_level_old_action[:, 1, 0] == 0).unsqueeze(-1) * self.is_rally.unsqueeze(-1)
        self.stats["Opp_FirstPass_high_level_1"] += self.info["switch_turn"] * (high_level_old_action[:, 1, 0] == 1).unsqueeze(-1) * self.is_rally.unsqueeze(-1)
        self.stats["Opp_FirstPass_high_level_2"] += self.info["switch_turn"] * (high_level_old_action[:, 1, 0] == 2).unsqueeze(-1) * self.is_rally.unsqueeze(-1)
        self.stats["Opp_FirstPass_high_level_3"] += self.info["switch_turn"] * (high_level_old_action[:, 1, 0] == 3).unsqueeze(-1) * self.is_rally.unsqueeze(-1)
        self.stats["Opp_FirstPass_high_level_4"] += self.info["switch_turn"] * (high_level_old_action[:, 1, 0] == 4).unsqueeze(-1) * self.is_rally.unsqueeze(-1)
        self.stats["Opp_FirstPass_high_level_5"] += self.info["switch_turn"] * (high_level_old_action[:, 1, 0] == 5).unsqueeze(-1) * self.is_rally.unsqueeze(-1)
        self.stats["Opp_SecPass_high_level_0"] += self.info["switch_turn"] * (high_level_old_action[:, 1, 1] == 0).unsqueeze(-1)
        self.stats["Opp_SecPass_high_level_1"] += self.info["switch_turn"] * (high_level_old_action[:, 1, 1] == 1).unsqueeze(-1)
        self.stats["Opp_SecPass_high_level_2"] += self.info["switch_turn"] * (high_level_old_action[:, 1, 1] == 2).unsqueeze(-1)
        self.stats["Opp_Att_high_level_0"] += self.info["switch_turn"] * (high_level_old_action[:, 1, 2] == 0).unsqueeze(-1)
        self.stats["Opp_Att_high_level_1"] += self.info["switch_turn"] * (high_level_old_action[:, 1, 2] == 1).unsqueeze(-1)
        self.stats["Opp_Att_high_level_2"] += self.info["switch_turn"] * (high_level_old_action[:, 1, 2] == 2).unsqueeze(-1)
        self.stats["Opp_Att_high_level_3"] += self.info["switch_turn"] * (high_level_old_action[:, 1, 2] == 3).unsqueeze(-1)

        # Opp_FirstPass
        Opp_FirstPass_goto_action = tensordict[("agents", "Opp_FirstPass_goto_action")]
        Opp_FirstPass_action = tensordict[("agents", "Opp_FirstPass_action")]
        Opp_FirstPass_hover_action = tensordict[("agents", "Opp_FirstPass_hover_action")]
        Opp_FirstPass_serve_action = tensordict[("agents", "Opp_FirstPass_serve_action")]
        Opp_FirstPass_serve_hover_action = tensordict[("agents", "Opp_FirstPass_serve_hover_action")]
        Opp_FirstPass_receive_action = tensordict[("agents", "Opp_FirstPass_receive_action")]
        Opp_FirstPass_receive_hover_action = tensordict[("agents", "Opp_FirstPass_receive_hover_action")]
        Opp_FirstPass_real_action = torch.where(
            (
                (self.info["high_level_old_action"][:, 1, 0] == 0) |
                (self.info["high_level_old_action"][:, 1, 0] == 1) |
                (self.info["high_level_old_action"][:, 1, 0] == 2)
            ).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
            Opp_FirstPass_goto_action,
            torch.where(
                (
                    (self.info["high_level_old_action"][:, 1, 0] == 3) |
                    (self.info["high_level_old_action"][:, 1, 0] == 4)
                ).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
                Opp_FirstPass_action,
                Opp_FirstPass_hover_action
            )
        )
        # serve
        Opp_FirstPass_real_action = torch.where(
            ((self.serve_turn == 0) & (self.serve_step == 0) & ~self.is_rally).unsqueeze(-1).unsqueeze(-1), # (E,1,1) 
            Opp_FirstPass_goto_action,
            Opp_FirstPass_real_action
        )
        Opp_FirstPass_real_action = torch.where(
            ((self.serve_turn == 0) & (self.serve_step == 1) & ~self.is_rally).unsqueeze(-1).unsqueeze(-1), # (E,1,1) 
            Opp_FirstPass_receive_action,
            Opp_FirstPass_real_action
        )
        Opp_FirstPass_receive_action = torch.where(
            ((self.serve_turn == 0) & (self.serve_step == 2) & ~self.is_rally).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
            Opp_FirstPass_receive_hover_action,
            Opp_FirstPass_real_action
        )
        Opp_FirstPass_real_action = torch.where(
            ((self.serve_turn == 1) & (self.serve_step == 0) & ~self.is_rally).unsqueeze(-1).unsqueeze(-1), # (E,1,1) 
            Opp_FirstPass_serve_action,
            Opp_FirstPass_real_action
        )
        Opp_FirstPass_real_action = torch.where(
            ((self.serve_turn == 1) & ((self.serve_step == 1) | (self.serve_step == 2)) & ~self.is_rally).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
            Opp_FirstPass_serve_hover_action,
            Opp_FirstPass_real_action
        )

        # Opp_SecPass
        Opp_SecPass_goto_action = tensordict[("agents", "Opp_SecPass_goto_action")]
        Opp_SecPass_action = tensordict[("agents", "Opp_SecPass_action")]
        Opp_SecPass_hover_action = tensordict[("agents", "Opp_SecPass_hover_action")]
        Opp_SecPass_real_action = torch.where(
            (self.info["high_level_old_action"][:, 1, 1] == 0).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
            Opp_SecPass_goto_action,
            torch.where(
                (self.info["high_level_old_action"][:, 1, 1] == 1).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
                Opp_SecPass_action,
                Opp_SecPass_hover_action
            )
        )
        
        # Opp_Att
        Opp_Att_goto_action = tensordict[("agents", "Opp_Att_goto_action")]
        Opp_Att_action = tensordict[("agents", "Opp_Att_action")]
        Opp_Att_hover_action = tensordict[("agents", "Opp_Att_hover_action")]
        Opp_Att_real_action = torch.where(
            (self.info["high_level_old_action"][:, 1, 2] == 0).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
            Opp_Att_goto_action,
            torch.where(
                (
                    (self.info["high_level_old_action"][:, 1, 2] == 1) |
                    (self.info["high_level_old_action"][:, 1, 2] == 2)
                ).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
                Opp_Att_action,
                Opp_Att_hover_action
            )
        )

        # FirstPass
        FirstPass_goto_action = tensordict[("agents", "FirstPass_goto_action")]
        FirstPass_action = tensordict[("agents", "FirstPass_action")]
        FirstPass_hover_action = tensordict[("agents", "FirstPass_hover_action")]
        FirstPass_serve_action = tensordict[("agents", "FirstPass_serve_action")]
        FirstPass_serve_hover_action = tensordict[("agents", "FirstPass_serve_hover_action")]
        FirstPass_receive_action = tensordict[("agents", "FirstPass_receive_action")]
        FirstPass_receive_hover_action = tensordict[("agents", "FirstPass_receive_hover_action")]
        FirstPass_real_action = torch.where(
            (
                (self.info["high_level_action"][:, 0, 0] == 0) |
                (self.info["high_level_action"][:, 0, 0] == 1) |
                (self.info["high_level_action"][:, 0, 0] == 2)
            ).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
            FirstPass_goto_action,
            torch.where(
                (
                    (self.info["high_level_action"][:, 0, 0] == 3) |
                    (self.info["high_level_action"][:, 0, 0] == 4)
                ).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
                FirstPass_action,
                FirstPass_hover_action
            )
        )
        FirstPass_real_action = torch.where(
            ((self.serve_turn == 1) & (self.serve_step == 0) & ~self.is_rally).unsqueeze(-1).unsqueeze(-1), # (E,1,1) 
            FirstPass_goto_action,
            FirstPass_real_action
        )
        FirstPass_real_action = torch.where(
            ((self.serve_turn == 1) & (self.serve_step == 1) & ~self.is_rally).unsqueeze(-1).unsqueeze(-1), # (E,1,1) 
            FirstPass_receive_action,
            FirstPass_real_action
        )
        FirstPass_receive_action = torch.where(
            ((self.serve_turn == 1) & (self.serve_step == 2) & ~self.is_rally).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
            FirstPass_receive_hover_action,
            FirstPass_real_action
        )
        FirstPass_real_action = torch.where(
            ((self.serve_turn == 0) & (self.serve_step == 0) & ~self.is_rally).unsqueeze(-1).unsqueeze(-1), # (E,1,1) 
            FirstPass_serve_action,
            FirstPass_real_action
        )
        FirstPass_real_action = torch.where(
            ((self.serve_turn == 0) & ((self.serve_step == 1) | (self.serve_step == 2)) & ~self.is_rally).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
            FirstPass_serve_hover_action,
            FirstPass_real_action
        )

        # SecPass
        SecPass_goto_action = tensordict[("agents", "SecPass_goto_action")]
        SecPass_action = tensordict[("agents", "SecPass_action")]
        SecPass_hover_action = tensordict[("agents", "SecPass_hover_action")]
        SecPass_new_action = tensordict[("agents", "SecPass_new_action")]
        SecPass_real_action = torch.where(
            (self.info["high_level_action"][:, 0, 1] == 0).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
            SecPass_goto_action,
            torch.where(
                (self.info["high_level_action"][:, 0, 1] == 1).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
                SecPass_action,
                torch.where(
                    (self.info["high_level_action"][:, 0, 1] == 3).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
                    SecPass_new_action,
                    SecPass_hover_action
                )
            )
        )

        # Att
        Att_goto_action = tensordict[("agents", "Att_goto_action")]
        Att_action = tensordict[("agents", "Att_action")]
        Att_hover_action = tensordict[("agents", "Att_hover_action")]
        Att_real_action = torch.where(
            (self.info["high_level_action"][:, 0, 2] == 0).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
            Att_goto_action,
            torch.where(
                (
                    (self.info["high_level_action"][:, 0, 2] == 1) |
                    (self.info["high_level_action"][:, 0, 2] == 2)
                ).unsqueeze(-1).unsqueeze(-1), # (E,1,1)
                Att_action,
                Att_hover_action
            )
        )

        # overall low-level actions
        actions = torch.cat(
            [
                Opp_SecPass_real_action,
                Opp_Att_real_action,
                FirstPass_real_action,
                SecPass_real_action,
                Att_real_action,
                Opp_FirstPass_real_action,
            ],
            dim=1
        )
        self.effort = self.drone.apply_action(actions)
    

    def check_ball_near_racket(self, racket_radius, cylinder_height_coeff):
        
        z_direction_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        z_direction_world = quat_rotate(self.drone_rot, z_direction_local) # (E,N,3)

        normal_vector_world = z_direction_world / torch.norm(z_direction_world, dim=-1).unsqueeze(-1)  # (E,N,3)

        cylinder_bottom_center = self.drone.pos  # (E,N,3) cylinder bottom center
        cylinder_axis = cylinder_height_coeff * self.ball_radius * normal_vector_world 

        ball_to_bottom = self.ball_pos - cylinder_bottom_center  # (E,N,3)
        projection_ratio = torch.sum(ball_to_bottom * cylinder_axis, dim=-1) / torch.sum(cylinder_axis * cylinder_axis, dim=-1)  # (E,N) projection of ball_to_bottom on cylinder_axis / cylinder_axis
        within_height = (projection_ratio >= 0) & (projection_ratio <= 1)  # (E,N)

        projection_point = cylinder_bottom_center + projection_ratio.unsqueeze(-1) * cylinder_axis  # (E,N,3)
        distance_to_axis = torch.norm(self.ball_pos - projection_point, dim=-1)  # (E,N)
        within_radius = distance_to_axis <= racket_radius  # (E,N)

        return (within_height & within_radius)  # (E,N)
    

    def check_hit(self, sim_dt, racket_radius=0.2, cylinder_height_coeff=2.0):
        
        racket_near_ball_last_step = self.racket_near_ball.clone()
        drone_near_ball_last_step = self.drone_near_ball.clone()

        self.racket_near_ball = self.check_ball_near_racket(racket_radius=racket_radius, cylinder_height_coeff=cylinder_height_coeff)  # (E,6)
        self.drone_near_ball = (torch.norm(self.rpos, dim=-1) < 0.4) # (E,6)
        
        ball_vel_z_change = ((self.ball_vel[..., 2] - self.ball_last_vel[..., 2]) > 9.8 * sim_dt) # (E,1)
        ball_vel_x_y_change = (self.ball_vel[..., :2] - self.ball_last_vel[..., :2]).norm(dim=-1) > 0.5 # (E,1)
        ball_vel_change = ball_vel_z_change | ball_vel_x_y_change # (E,1)

        self.drone_hit_ball = (drone_near_ball_last_step | self.drone_near_ball) & ball_vel_change # (E,6)
        self.racket_hit_ball = (racket_near_ball_last_step | self.racket_near_ball) & ball_vel_change # (E,6)

        self.racket_hit_ball[:, 0] = self.racket_hit_ball[:, 0] & (self.progress_buf - self.Opp_SecPass_last_hit_t > 3) # (E,)
        self.racket_hit_ball[:, 1] = self.racket_hit_ball[:, 1] & (self.progress_buf - self.Opp_Att_last_hit_t > 3) # (E,)
        self.racket_hit_ball[:, 2] = self.racket_hit_ball[:, 2] & (self.progress_buf - self.FirstPass_last_hit_t > 3) # (E,)
        self.racket_hit_ball[:, 3] = self.racket_hit_ball[:, 3] & (self.progress_buf - self.SecPass_last_hit_t > 3) # (E,)
        self.racket_hit_ball[:, 4] = self.racket_hit_ball[:, 4] & (self.progress_buf - self.Att_last_hit_t > 3) # (E,)
        self.racket_hit_ball[:, 5] = self.racket_hit_ball[:, 5] & (self.progress_buf - self.Opp_FirstPass_last_hit_t > 3) # (E,)
        
        self.drone_hit_ball[:, 0] = self.drone_hit_ball[:, 0] & (self.progress_buf - self.Opp_SecPass_last_hit_t > 3)
        self.drone_hit_ball[:, 1] = self.drone_hit_ball[:, 1] & (self.progress_buf - self.Opp_Att_last_hit_t > 3)
        self.drone_hit_ball[:, 2] = self.drone_hit_ball[:, 2] & (self.progress_buf - self.FirstPass_last_hit_t > 3)
        self.drone_hit_ball[:, 3] = self.drone_hit_ball[:, 3] & (self.progress_buf - self.SecPass_last_hit_t > 3)
        self.drone_hit_ball[:, 4] = self.drone_hit_ball[:, 4] & (self.progress_buf - self.Att_last_hit_t > 3)
        self.drone_hit_ball[:, 5] = self.drone_hit_ball[:, 5] & (self.progress_buf - self.Opp_FirstPass_last_hit_t > 3)


    def update_hit_info(self):
        """
        check if the ball is hit by the drone or the racket
        update the last hit time for each racket
        """
        
        self.check_hit(sim_dt=self.dt)
        self.wrong_hit_racket = self.drone_hit_ball & ~self.racket_hit_ball # (E,6)

        Opp_FirstPass_hit = self.racket_hit_ball[:, 5] # (E,)
        Opp_SecPass_hit = self.racket_hit_ball[:, 0] # (E,)
        Opp_Att_hit = self.racket_hit_ball[:, 1] # (E,)
        FirstPass_hit = self.racket_hit_ball[:, 2] # (E,)
        SecPass_hit = self.racket_hit_ball[:, 3] # (E,)
        Att_hit = self.racket_hit_ball[:, 4] # (E,)

        self.is_rally = torch.where(
            Opp_SecPass_hit | SecPass_hit,
            True,
            self.is_rally
        )
        self.serve_step += (1 - self.is_rally.long()) * (FirstPass_hit | SecPass_hit | Opp_FirstPass_hit | SecPass_hit)

        self.last_hit_side = torch.where(
            (Opp_FirstPass_hit | Opp_SecPass_hit | Opp_Att_hit),
            True,
            torch.where(
                (FirstPass_hit | SecPass_hit | Att_hit),
                False,
                self.last_hit_side
            )
        ) # (E,)

        self.Opp_FirstPass_last_hit_t = torch.where(Opp_FirstPass_hit, self.progress_buf, self.Opp_FirstPass_last_hit_t)
        self.Opp_SecPass_last_hit_t = torch.where(Opp_SecPass_hit, self.progress_buf, self.Opp_SecPass_last_hit_t)
        self.Opp_Att_last_hit_t = torch.where(Opp_Att_hit, self.progress_buf, self.Opp_Att_last_hit_t)
        self.FirstPass_last_hit_t = torch.where(FirstPass_hit, self.progress_buf, self.FirstPass_last_hit_t)
        self.SecPass_last_hit_t = torch.where(SecPass_hit, self.progress_buf, self.SecPass_last_hit_t)
        self.Att_last_hit_t = torch.where(Att_hit, self.progress_buf, self.Att_last_hit_t)
        
        self.ball_last_vel = self.ball_vel.clone()

        self.info["Opp_FirstPass_hit"] = Opp_FirstPass_hit.unsqueeze(-1)
        self.info["Opp_SecPass_hit"] = Opp_SecPass_hit.unsqueeze(-1)
        self.info["Opp_Att_hit"] = Opp_Att_hit.unsqueeze(-1)
        self.info["FirstPass_hit"] = FirstPass_hit.unsqueeze(-1)
        self.info["SecPass_hit"] = SecPass_hit.unsqueeze(-1)
        self.info["Att_hit"] = Att_hit.unsqueeze(-1)

        if self._should_render(0):
            if Opp_FirstPass_hit[self.central_env_idx].item() == 1:
                logging.info("Opponent_FirstPass hit the ball.")
            elif Opp_SecPass_hit[self.central_env_idx].item() == 1:
                logging.info("Opponent_SecPass hit the ball.")
            elif Opp_Att_hit[self.central_env_idx].item() == 1:
                logging.info("Opponent_Att hit the ball.")
            elif FirstPass_hit[self.central_env_idx].item() == 1:
                logging.info("FirstPass hit the ball.")
            elif SecPass_hit[self.central_env_idx].item() == 1:
                logging.info("SecPass hit the ball.")
            elif Att_hit[self.central_env_idx].item() == 1:
                logging.info("Att hit the ball.")

    
    def update_already_hit_info(self):
        Opp_FirstPass_hit = self.racket_hit_ball[:, 5] # (E,)
        Opp_SecPass_hit = self.racket_hit_ball[:, 0] # (E,)
        Opp_Att_hit = self.racket_hit_ball[:, 1] # (E,)
        FirstPass_hit = self.racket_hit_ball[:, 2] # (E,)
        SecPass_hit = self.racket_hit_ball[:, 3] # (E,)
        Att_hit = self.racket_hit_ball[:, 4] # (E,)

        self.Opp_FirstPass_already_hit = torch.where(Opp_FirstPass_hit.unsqueeze(-1) == 1, True, self.Opp_FirstPass_already_hit)
        self.Opp_SecPass_already_hit = torch.where(Opp_SecPass_hit.unsqueeze(-1) == 1, True, self.Opp_SecPass_already_hit)
        self.Opp_Att_already_hit = torch.where(Opp_Att_hit.unsqueeze(-1) == 1, True, self.Opp_Att_already_hit)
        self.FirstPass_already_hit = torch.where(FirstPass_hit.unsqueeze(-1) == 1, True, self.FirstPass_already_hit)
        self.SecPass_already_hit = torch.where(SecPass_hit.unsqueeze(-1) == 1, True, self.SecPass_already_hit)
        self.Att_already_hit = torch.where(Att_hit.unsqueeze(-1) == 1, True, self.Att_already_hit)

    
    def update_ball_pass_net(self):
        ball_at_far_side = (self.ball_pos[..., 0] < 0).squeeze(-1) # (E,)

        ball_from_near_side_to_far_side = (ball_at_far_side & ~self.ball_side).unsqueeze(-1) # (E,1)
        self.FirstPass_already_hit = torch.where(
            ball_from_near_side_to_far_side,
            False,
            self.FirstPass_already_hit
        ) # (E,1)
        self.SecPass_already_hit = torch.where(
            ball_from_near_side_to_far_side,
            False,
            self.SecPass_already_hit
        ) # (E,1)
        self.Att_already_hit = torch.where(
            ball_from_near_side_to_far_side,
            False,
            self.Att_already_hit
        ) # (E,1)

        ball_from_far_side_to_near_side = (~ball_at_far_side & self.ball_side).unsqueeze(-1) # (E,1)
        self.Opp_FirstPass_already_hit = torch.where(
            ball_from_far_side_to_near_side,
            False,
            self.Opp_FirstPass_already_hit
        ) # (E,1)
        self.Opp_SecPass_already_hit = torch.where(
            ball_from_far_side_to_near_side,
            False,
            self.Opp_SecPass_already_hit
        ) # (E,1)
        self.Opp_Att_already_hit = torch.where(
            ball_from_far_side_to_near_side,
            False,
            self.Opp_Att_already_hit
        ) # (E,1)
        
        self.ball_pass_net = (ball_from_near_side_to_far_side | ball_from_far_side_to_near_side).squeeze(-1) # (E,)
        
        self.ball_side = torch.where(
            self.ball_pass_net,
            ~self.ball_side,
            self.ball_side
        ) # (E,)

        if self._should_render(0):
            if ball_from_near_side_to_far_side[self.central_env_idx].item() == 1:
                logging.info("Ball passed the net from near side to far side.")
            elif ball_from_far_side_to_near_side[self.central_env_idx].item() == 1:
                logging.info("Ball passed the net from far side to near side.")


    def _check_available_action(self, action: torch.Tensor) -> bool:
        return ((action == 0) | (action == 1) | (action == 2) | (action == 3) | (action == 4) | (action == 5)).all().item()
    

    def compute_attacking_target_and_goto_pos(self):
        # last step hit info
        Opp_FirstPass_hit = self.racket_hit_ball[:, 5] 
        Opp_SecPass_hit = self.racket_hit_ball[:, 0]
        Opp_Att_hit = self.racket_hit_ball[:, 1]
        FirstPass_hit = self.racket_hit_ball[:, 2]
        SecPass_hit = self.racket_hit_ball[:, 3]
        Att_hit = self.racket_hit_ball[:, 4]

        # Opp_FirstPass_goto
        self.Opp_FirstPass_goto_pos_before_hit = torch.where(
            (self.info["high_level_old_action"][:, 1, 0] == 0).unsqueeze(-1).unsqueeze(-1),
            self.Opp_FirstPass_goto_pos_left_before_hit,
            torch.where(
                (self.info["high_level_old_action"][:, 1, 0] == 1).unsqueeze(-1).unsqueeze(-1),
                self.Opp_FirstPass_hover_pos_after_hit,
                torch.where(
                    (self.info["high_level_old_action"][:, 1, 0] == 2).unsqueeze(-1).unsqueeze(-1),
                    self.Opp_FirstPass_goto_pos_right_before_hit,
                    self.Opp_FirstPass_goto_pos_before_hit
                )
            )
        )
        self.Opp_FirstPass_goto_pos_before_hit = torch.where(
            ~self.is_rally.unsqueeze(-1).unsqueeze(-1),
            self.Opp_FirstPass_goto_receive_pos,
            self.Opp_FirstPass_goto_pos_before_hit
        )

        # Opp_FirstPass
        self.Opp_FirstPass_obs_attacking_target = torch.where(
            (self.info["high_level_old_action"][:, 1, 0] == 3).unsqueeze(-1).unsqueeze(-1),
            self.FirstPass_obs_left_attacking_target,
            torch.where(
                (self.info["high_level_old_action"][:, 1, 0] == 4).unsqueeze(-1).unsqueeze(-1),
                self.FirstPass_obs_right_attacking_target,
                self.Opp_FirstPass_obs_attacking_target
            )
        )

        # FirstPass_goto
        self.FirstPass_goto_pos_before_hit = torch.where(
            (self.info["high_level_action"][:, 0, 0] == 0).unsqueeze(-1).unsqueeze(-1),
            self.FirstPass_goto_pos_left_before_hit,
            torch.where(
                (self.info["high_level_action"][:, 0, 0] == 1).unsqueeze(-1).unsqueeze(-1),
                self.FirstPass_hover_pos_after_hit,
                torch.where(
                    (self.info["high_level_action"][:, 0, 0] == 2).unsqueeze(-1).unsqueeze(-1),
                    self.FirstPass_goto_pos_right_before_hit,
                    self.FirstPass_goto_pos_before_hit
                )
            )
        )
        self.FirstPass_goto_pos_before_hit = torch.where(
            ~self.is_rally.unsqueeze(-1).unsqueeze(-1),
            self.FirstPass_goto_receive_pos,
            self.FirstPass_goto_pos_before_hit
        )

        # FirstPass
        self.FirstPass_obs_attacking_target = torch.where(
            (self.info["high_level_action"][:, 0, 0] == 3).unsqueeze(-1).unsqueeze(-1),
            self.FirstPass_obs_left_attacking_target,
            torch.where(
                (self.info["high_level_action"][:, 0, 0] == 4).unsqueeze(-1).unsqueeze(-1),
                self.FirstPass_obs_right_attacking_target,
                self.FirstPass_obs_attacking_target
            )
        )

        # Opp_Att
        self.Opp_Att_attacking_target = torch.where(
            self.info["high_level_old_action"][:, 1, 2] == 1,
            0,
            torch.where(
                self.info["high_level_old_action"][:, 1, 2] == 2,
                1,
                self.Opp_Att_attacking_target
            )
        )
        
        # Att
        self.Att_attacking_target = torch.where(
            self.info["high_level_action"][:, 0, 2] == 1,
            0,
            torch.where(
                self.info["high_level_action"][:, 0, 2] == 2,
                1,
                self.Att_attacking_target
            )
        )


    def compute_turn(self):

        self.Opp_FirstPass_turn = torch.where(
            (
                (self.info["high_level_old_action"][:, 1, 0] == 3) |
                (self.info["high_level_old_action"][:, 1, 0] == 4)
            ), # (E,)
            1,
            0
        )

        self.Opp_SecPass_turn = torch.where(
            (self.info["high_level_old_action"][:, 1, 1] == 1), # (E,)
            1,
            0
        )

        self.Opp_Att_turn = torch.where(
            (
                (self.info["high_level_old_action"][:, 1, 2] == 1) |
                (self.info["high_level_old_action"][:, 1, 2] == 2)
            ), # (E,)
            1,
            0
        )

        self.FirstPass_turn = torch.where(
            (
                (self.info["high_level_action"][:, 0, 0] == 3) |
                (self.info["high_level_action"][:, 0, 0] == 4)
            ), # (E,)
            1,
            0
        )

        self.SecPass_turn = torch.where(
            (
                (self.info["high_level_action"][:, 0, 1] == 1) |
                (self.info["high_level_action"][:, 0, 1] == 3)   
            ), # (E,)
            1,
            0
        )

        self.Att_turn = torch.where(
            (
                (self.info["high_level_action"][:, 0, 2] == 1) |
                (self.info["high_level_action"][:, 0, 2] == 2)
            ), # (E,)
            1,
            0
        )


    def _compute_state_and_obs(self, is_step=False):
        self.root_state = self.drone.get_state() # (E, 6, 23)
        self.info["drone_state"][:] = self.root_state[..., :13]
        
        self.ball_pos, _ = self.get_env_poses(self.ball.get_world_poses()) # (E, 1, 3)
        self.ball_vel = self.ball.get_velocities()[..., :3] # (E, 1, 3)

        self.rpos = self.ball_pos - self.drone.pos # (E, 6, 3)

        pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )
        rot = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot, rot) # make sure w of (w,x,y,z) is positive
        self.root_state = torch.cat([pos, rot, vel, angular_vel, heading, up, throttle], dim=-1)
        
        rpy = quaternion_to_euler(rot)
        
        self.drone_rot = rot

        # save important information for reward computing

        # for the far side of the court
        ball_pos_transfer = self.ball_pos.clone()
        ball_pos_transfer[..., :2] = - ball_pos_transfer[..., :2]
        ball_vel_transfer = self.ball_vel.clone()
        ball_vel_transfer[..., :2] = - ball_vel_transfer[..., :2]

        serve_ball_anchor_transfer = self.serve_ball_anchor.clone()
        serve_ball_anchor_transfer[..., :2] = - serve_ball_anchor_transfer[..., :2]
        self.Opp_FirstPass_serve_ball_rpos = ball_pos_transfer - serve_ball_anchor_transfer

        self.FirstPass_serve_ball_rpos = self.ball_pos - serve_ball_anchor_transfer # (E, 1, 3)

        # root_state: (E, 1, 23)
        Opp_FirstPass_root_state = transfer_root_state_to_the_other_side(self.root_state[:, 5, :].unsqueeze(1))
        Opp_SecPass_root_state = transfer_root_state_to_the_other_side(self.root_state[:, 0, :].unsqueeze(1))
        Opp_Att_root_state = transfer_root_state_to_the_other_side(self.root_state[:, 1, :].unsqueeze(1))
        FirstPass_root_state = self.root_state[:, 2, :].unsqueeze(1)
        SecPass_root_state = self.root_state[:, 3, :].unsqueeze(1)
        Att_root_state = self.root_state[:, 4, :].unsqueeze(1)

        # relative pos to ball
        Opp_FirstPass_rpos = self.rpos[:, 5, :].clone().unsqueeze(1)
        Opp_FirstPass_rpos[..., :2] = - Opp_FirstPass_rpos[..., :2]
        Opp_SecPass_rpos = self.rpos[:, 0, :].clone().unsqueeze(1)
        Opp_SecPass_rpos[..., :2] = - Opp_SecPass_rpos[..., :2]
        Opp_Att_rpos = self.rpos[:, 1, :].clone().unsqueeze(1)
        Opp_Att_rpos[..., :2] = - Opp_Att_rpos[..., :2]
        FirstPass_rpos = self.rpos[:, 2, :].unsqueeze(1)
        SecPass_rpos = self.rpos[:, 3, :].unsqueeze(1)
        Att_rpos = self.rpos[:, 4, :].unsqueeze(1)

        # update attacking target and goto pos [high level action]
        if is_step:
            self.compute_attacking_target_and_goto_pos()
            self.compute_turn()

        # Opp_FirstPass_goto_obs
        Opp_FirstPass_goto_rpos = self.Opp_FirstPass_goto_pos_before_hit - Opp_FirstPass_root_state[..., :3] # (E, 1, 3)
        Opp_FirstPass_goto_rheading = self.hover_target_heading - Opp_FirstPass_root_state[..., 13:16] # (E, 1, 3)
        Opp_FirstPass_goto_obs = [
            Opp_FirstPass_goto_rpos, # (E, 1, 3)
            Opp_FirstPass_root_state[..., 3:], # (E, 1, root_state_dim-3)
            Opp_FirstPass_goto_rheading, # (E, 1, 3)
        ]
        Opp_FirstPass_goto_obs = torch.cat(Opp_FirstPass_goto_obs, dim=-1)

        # Opp_FirstPass_obs
        Opp_FirstPass_obs = [
            Opp_FirstPass_root_state, # (E, 1, 23)
            ball_pos_transfer, # (E, 1, 3)
            Opp_FirstPass_rpos, # (E, 1, 3)
            ball_vel_transfer, # (E, 1, 3)
            turn_to_obs(self.Opp_FirstPass_turn), # (E, 1, 2)
            self.Opp_FirstPass_obs_attacking_target # (E, 1, 2)
        ]
        Opp_FirstPass_obs = torch.cat(Opp_FirstPass_obs, dim=-1)

        # Opp_FirstPass_hover_obs
        Opp_FirstPass_hover_rpos = self.Opp_FirstPass_hover_pos_after_hit - Opp_FirstPass_root_state[..., :3] # (E, 1, 3)
        Opp_FirstPass_hover_rheading = self.hover_target_heading - Opp_FirstPass_root_state[..., 13:16] # (E, 1, 3)
        Opp_FirstPass_hover_obs = [
            Opp_FirstPass_hover_rpos, # (E, 1, 3)
            Opp_FirstPass_root_state[..., 3:], # (E, 1, root_state_dim-3)
            Opp_FirstPass_hover_rheading, # (E, 1, 3)
        ]
        Opp_FirstPass_hover_obs = torch.cat(Opp_FirstPass_hover_obs, dim=-1)

        # Opp_FirstPass_serve_obs
        Opp_FirstPass_serve_obs = [
            Opp_FirstPass_root_state, # (E, 1, 23)
            ball_pos_transfer, # (E, 1, 3)
            Opp_FirstPass_rpos, # (E, 1, 3)
            ball_vel_transfer, # (E, 1, 3)
            turn_to_obs(torch.ones(self.num_envs, device=self.device)), # (E, 1, 2)
            self.Opp_FirstPass_serve_ball_rpos,
        ]
        Opp_FirstPass_serve_obs = torch.cat(Opp_FirstPass_serve_obs, dim=-1)

        # Opp_FirstPass_serve_hover_obs
        Opp_FirstPass_serve_hover_rpos = self.Opp_FirstPass_hover_pos_after_hit -  Opp_FirstPass_root_state[..., :3] # (E, 1, 3)
        Opp_FirstPass_serve_hover_rheading = self.hover_target_heading - Opp_FirstPass_root_state[..., 13:16] # (E, 1, 3)
        Opp_FirstPass_serve_hover_obs = [
            Opp_FirstPass_serve_hover_rpos, # (E, 1, 3)
            Opp_FirstPass_root_state[..., 3:], # (E, 1, root_state_dim - 3)
            Opp_FirstPass_serve_hover_rheading, # (E, 1, 3)
        ]
        Opp_FirstPass_serve_hover_obs = torch.cat(Opp_FirstPass_serve_hover_obs, dim=-1)

        # Opp_FirstPass_receive_obs
        Opp_FirstPass_receive_obs = [
            Opp_FirstPass_root_state, # (E, 1, 23)
            ball_pos_transfer, # (E, 1, 3)
            Opp_FirstPass_rpos, # (E, 1, 3)
            ball_vel_transfer, # (E, 1, 3)
            turn_to_obs(torch.ones(self.num_envs, device=self.device)), # (E, 1, 2)
        ]
        Opp_FirstPass_receive_obs = torch.cat(Opp_FirstPass_receive_obs, dim=-1)

        # Opp_FirstPass_receive_hover_obs
        Opp_FirstPass_receive_hover_rpos = self.Opp_FirstPass_hover_pos_after_hit - Opp_FirstPass_root_state[..., :3] # (E, 1, 3)
        Opp_FirstPass_receive_hover_rheading = self.hover_target_heading - Opp_FirstPass_root_state[..., 13:16] # (E, 1, 3)
        Opp_FirstPass_receive_hover_obs = [
            Opp_FirstPass_receive_hover_rpos, # (E, 1, 3)
            Opp_FirstPass_root_state[..., 3:], # (E, 1, root_state_dim-3)
            Opp_FirstPass_receive_hover_rheading, # (E, 1, 3)
        ]
        Opp_FirstPass_receive_hover_obs = torch.cat(Opp_FirstPass_receive_hover_obs, dim=-1)

        # Opp_SecPass_goto_obs
        Opp_SecPass_goto_rpos = self.Opp_SecPass_goto_pos_before_hit - Opp_SecPass_root_state[..., :3] # (E, 1, 3)
        Opp_SecPass_goto_rheading = self.hover_target_heading - Opp_SecPass_root_state[..., 13:16] # (E, 1, 3)
        Opp_SecPass_goto_obs = [
            Opp_SecPass_goto_rpos, # (E, 1, 3)
            Opp_SecPass_root_state[..., 3:], # (E, 1, root_state_dim-3)
            Opp_SecPass_goto_rheading, # (E, 1, 3)
        ]
        Opp_SecPass_goto_obs = torch.cat(Opp_SecPass_goto_obs, dim=-1)

        # Opp_SecPass_obs
        Opp_SecPass_obs = [
            Opp_SecPass_root_state, # (E, 1, 23)
            ball_pos_transfer, # (E, 1, 3)
            Opp_SecPass_rpos, # (E, 1, 3)
            ball_vel_transfer, # (E, 1, 3)
            turn_to_obs(self.Opp_SecPass_turn), # (E, 1, 2)
        ]
        Opp_SecPass_obs = torch.cat(Opp_SecPass_obs, dim=-1)

        # Opp_SecPass_hover_obs
        Opp_SecPass_hover_rpos = self.Opp_SecPass_hover_pos_after_hit - Opp_SecPass_root_state[..., :3] # (E, 1, 3)
        Opp_SecPass_hover_rheading = self.hover_target_heading - Opp_SecPass_root_state[..., 13:16] # (E, 1, 3)
        Opp_SecPass_hover_obs = [
            Opp_SecPass_hover_rpos, # (E, 1, 3)
            Opp_SecPass_root_state[..., 3:], # (E, 1, root_state_dim - 3)
            Opp_SecPass_hover_rheading, # (E, 1, 3)
        ]
        Opp_SecPass_hover_obs = torch.cat(Opp_SecPass_hover_obs, dim=-1)

        # Opp_Att_goto_obs
        Opp_Att_goto_rpos = self.Opp_Att_goto_pos_before_hit - Opp_Att_root_state[..., :3] # (E, 1, 3)
        Opp_Att_goto_rheading = self.hover_target_heading - Opp_Att_root_state[..., 13:16] # (E, 1, 3)
        Opp_Att_goto_obs = [
            Opp_Att_goto_rpos, # (E, 1, 23)
            Opp_Att_root_state[..., 3:], # (E, 1, root_state_dim-3)
            Opp_Att_goto_rheading, # (E, 1, 3)
        ]
        Opp_Att_goto_obs = torch.cat(Opp_Att_goto_obs, dim=-1)

        # Opp_Att_obs
        Opp_Att_obs = [
            Opp_Att_root_state, # (E, 1, 23)
            ball_pos_transfer, # (E, 1, 3)
            Opp_Att_rpos, # (E, 1, 3)
            ball_vel_transfer, # (E, 1, 3)
            turn_to_obs(self.Opp_Att_turn), # (E, 1, 2)
            attacking_target_to_obs(self.Opp_Att_attacking_target) # (E, 1, 2)
        ]
        Opp_Att_obs = torch.cat(Opp_Att_obs, dim=-1)

        # Opp_Att_hover_obs
        Opp_Att_hover_rpos = self.Opp_Att_hover_pos_after_hit - Opp_Att_root_state[..., :3] # (E, 1, 3)
        Opp_Att_hover_rheading = self.hover_target_heading - Opp_Att_root_state[..., 13:16] # (E, 1, 3)
        Opp_Att_hover_obs = [
            Opp_Att_hover_rpos, # (E, 1, 3)
            Opp_Att_root_state[..., 3:], # (E, 1, root_state_dim-3)
            Opp_Att_hover_rheading, # (E, 1, 3)
        ]
        Opp_Att_hover_obs = torch.cat(Opp_Att_hover_obs, dim=-1)

        # FirstPass_goto_obs
        FirstPass_goto_rpos = self.FirstPass_goto_pos_before_hit - FirstPass_root_state[..., :3] # (E, 1, 3)
        FirstPass_goto_rheading = self.hover_target_heading - FirstPass_root_state[..., 13:16] # (E, 1, 3)
        FirstPass_goto_obs = [
            FirstPass_goto_rpos, # (E, 1, 3)
            FirstPass_root_state[..., 3:], # (E, 1, root_state_dim-3)
            FirstPass_goto_rheading, # (E, 1, 3)
        ]
        FirstPass_goto_obs = torch.cat(FirstPass_goto_obs, dim=-1)

        # FirstPass_obs
        FirstPass_obs = [
            FirstPass_root_state, # (E, 1, 23)
            self.ball_pos, # (E, 1, 3)
            FirstPass_rpos, # (E, 1, 3)
            self.ball_vel, # (E, 1, 3)
            turn_to_obs(self.FirstPass_turn), # (E, 1, 2)
            self.FirstPass_obs_attacking_target # (E, 1, 2)
        ]
        FirstPass_obs = torch.cat(FirstPass_obs, dim=-1)

        # FirstPass_hover_obs
        FirstPass_hover_rpos = self.FirstPass_hover_pos_after_hit - FirstPass_root_state[..., :3] # (E, 1, 3)
        FirstPass_hover_rheading = self.hover_target_heading - FirstPass_root_state[..., 13:16] # (E, 1, 3)
        FirstPass_hover_obs = [
            FirstPass_hover_rpos, # (E, 1, 3)
            FirstPass_root_state[..., 3:], # (E, 1, root_state_dim-3)
            FirstPass_hover_rheading, # (E, 1, 3)
        ]
        FirstPass_hover_obs = torch.cat(FirstPass_hover_obs, dim=-1)

        # FirstPass_serve_obs
        FirstPass_serve_obs = [
            FirstPass_root_state, # (E, 1, 23)
            self.ball_pos, # (E, 1, 3)
            FirstPass_rpos, # (E, 1, 3)
            self.ball_vel, # (E, 1, 3)
            turn_to_obs(torch.ones(self.num_envs, device=self.device)), # (E, 1, 2)
            self.FirstPass_serve_ball_rpos,
        ]
        FirstPass_serve_obs = torch.cat(FirstPass_serve_obs, dim=-1)

        # FirstPass_serve_hover_obs
        FirstPass_serve_hover_rpos = self.FirstPass_hover_pos_after_hit -  FirstPass_root_state[..., :3] # (E, 1, 3)
        FirstPass_serve_hover_rheading = self.hover_target_heading - FirstPass_root_state[..., 13:16] # (E, 1, 3)
        FirstPass_serve_hover_obs = [
            FirstPass_serve_hover_rpos, # (E, 1, 3)
            FirstPass_root_state[..., 3:], # (E, 1, root_state_dim - 3)
            FirstPass_serve_hover_rheading, # (E, 1, 3)
        ]
        FirstPass_serve_hover_obs = torch.cat(FirstPass_serve_hover_obs, dim=-1)

        # FirstPass_receive_obs
        FirstPass_receive_obs = [
            FirstPass_root_state, # (E, 1, 23)
            self.ball_pos, # (E, 1, 3)
            FirstPass_rpos, # (E, 1, 3)
            self.ball_vel, # (E, 1, 3)
            turn_to_obs(torch.ones(self.num_envs, device=self.device)), # (E, 1, 2)
        ]
        FirstPass_receive_obs = torch.cat(FirstPass_receive_obs, dim=-1)

        # FirstPass_receive_hover_obs
        FirstPass_receive_hover_rpos = self.FirstPass_hover_pos_after_hit - FirstPass_root_state[..., :3] # (E, 1, 3)
        FirstPass_receive_hover_rheading = self.hover_target_heading - FirstPass_root_state[..., 13:16] # (E, 1, 3)
        FirstPass_receive_hover_obs = [
            FirstPass_receive_hover_rpos, # (E, 1, 3)
            FirstPass_root_state[..., 3:], # (E, 1, root_state_dim-3)
            FirstPass_receive_hover_rheading, # (E, 1, 3)
        ]
        FirstPass_receive_hover_obs = torch.cat(FirstPass_receive_hover_obs, dim=-1)

        # SecPass_goto_obs
        SecPass_goto_rpos = self.SecPass_goto_pos_before_hit - SecPass_root_state[..., :3] # (E, 1, 3)
        SecPass_goto_rheading = self.hover_target_heading - SecPass_root_state[..., 13:16] # (E, 1, 3)
        SecPass_goto_obs = [
            SecPass_goto_rpos, # (E, 1, 3)
            SecPass_root_state[..., 3:], # (E, 1, root_state_dim-3)
            SecPass_goto_rheading, # (E, 1, 3)
        ]
        SecPass_goto_obs = torch.cat(SecPass_goto_obs, dim=-1)

        # SecPass_obs
        SecPass_obs = [
            SecPass_root_state, # (E, 1, 23)
            self.ball_pos, # (E, 1, 3)
            SecPass_rpos, # (E, 1, 3)
            self.ball_vel, # (E, 1, 3)
            turn_to_obs(self.SecPass_turn), # (E, 1, 2)
        ]
        SecPass_obs = torch.cat(SecPass_obs, dim=-1)

        # SecPass_hover_obs
        SecPass_hover_rpos = self.SecPass_hover_pos_after_hit - SecPass_root_state[..., :3] # (E, 1, 3)
        SecPass_hover_rheading = self.hover_target_heading - SecPass_root_state[..., 13:16] # (E, 1, 3)
        SecPass_hover_obs = [
            SecPass_hover_rpos, # (E, 1, 3)
            SecPass_root_state[..., 3:], # (E, 1, root_state_dim-3)
            SecPass_hover_rheading, # (E, 1, 3)
        ]
        SecPass_hover_obs = torch.cat(SecPass_hover_obs, dim=-1)

        # Att_goto_obs
        Att_goto_rpos = self.Att_goto_pos_before_hit - Att_root_state[..., :3] # (E, 1, 3)
        Att_goto_rheading = self.hover_target_heading - Att_root_state[..., 13:16] # (E, 1, 3)
        Att_goto_obs = [
            Att_goto_rpos, # (E, 1, 3)
            Att_root_state[..., 3:], # (E, 1, root_state_dim-3)
            Att_goto_rheading, # (E, 1, 3)
        ]
        Att_goto_obs = torch.cat(Att_goto_obs, dim=-1)

        # Att_obs
        Att_obs = [
            Att_root_state, # (E, 1, 23)
            self.ball_pos, # (E, 1, 3)
            Att_rpos, # (E, 1, 3)
            self.ball_vel, # (E, 1, 3)
            turn_to_obs(self.Att_turn), # (E, 1, 2)
            attacking_target_to_obs(self.Att_attacking_target)
        ]
        Att_obs = torch.cat(Att_obs, dim=-1)

        # Att_hover_obs
        Att_hover_rpos = self.Att_hover_pos_after_hit - Att_root_state[..., :3] # (E, 1, 3)
        Att_hover_rheading = self.hover_target_heading - Att_root_state[..., 13:16] # (E, 1, 3)
        Att_hover_obs = [
            Att_hover_rpos, # (E, 1, 3)
            Att_root_state[..., 3:], # (E, 1, root_state_dim-3)
            Att_hover_rheading, # (E, 1, 3)
        ]
        Att_hover_obs = torch.cat(Att_hover_obs, dim=-1)

        # high_level_obs
        far_side_Opp_FirstPass_root_state = self.root_state[:, 5, :].unsqueeze(1)
        far_side_Opp_SecPass_root_state = self.root_state[:, 0, :].unsqueeze(1)
        far_side_Opp_Att_root_state = self.root_state[:, 1, :].unsqueeze(1)
        far_side_FirstPass_root_state = transfer_root_state_to_the_other_side(self.root_state[:, 2, :].unsqueeze(1))
        far_side_SecPass_root_state = transfer_root_state_to_the_other_side(self.root_state[:, 3, :].unsqueeze(1))
        far_side_Att_root_state = transfer_root_state_to_the_other_side(self.root_state[:, 4, :].unsqueeze(1))

        near_side_already_hit = torch.cat([
            torch.nn.functional.one_hot(self.FirstPass_already_hit.long(), num_classes=2),
            torch.nn.functional.one_hot(self.SecPass_already_hit.long(), num_classes=2),
            torch.nn.functional.one_hot(self.Att_already_hit.long(), num_classes=2),
        ], dim=1).reshape(self.num_envs, 6) # (E, 6)

        far_side_already_hit = torch.cat([
            torch.nn.functional.one_hot(self.Opp_FirstPass_already_hit.long(), num_classes=2),
            torch.nn.functional.one_hot(self.Opp_SecPass_already_hit.long(), num_classes=2),
            torch.nn.functional.one_hot(self.Opp_Att_already_hit.long(), num_classes=2),
        ], dim=1).reshape(self.num_envs, 6) # (E, 6)

        already_hit_obs = torch.stack([
            torch.cat([near_side_already_hit, far_side_already_hit], dim=1), # (E, 12)
            torch.cat([far_side_already_hit, near_side_already_hit], dim=1), # (E, 12)
        ], dim=1) # (E, 2, 12)

        high_level_obs = [
            torch.cat([FirstPass_root_state[..., 0:13], Opp_FirstPass_root_state[..., 0:13]], dim=1), # (E, 2, 13)
            torch.cat([SecPass_root_state[..., 0:13], Opp_SecPass_root_state[..., 0:13]], dim=1), # (E, 2, 13)
            torch.cat([Att_root_state[..., 0:13], Opp_Att_root_state[..., 0:13]], dim=1), # (E, 2, 13)
            torch.cat([far_side_Opp_FirstPass_root_state[..., 0:13], far_side_FirstPass_root_state[..., 0:13]], dim=1), # (E, 2, 13)
            torch.cat([far_side_Opp_SecPass_root_state[..., 0:13], far_side_SecPass_root_state[..., 0:13]], dim=1), # (E, 2, 13)
            torch.cat([far_side_Opp_Att_root_state[..., 0:13], far_side_Att_root_state[..., 0:13]], dim=1), # (E, 2, 13)
            torch.cat([self.ball_pos, ball_pos_transfer], dim=1), # (E, 2, 3)
            torch.cat([self.ball_vel, ball_vel_transfer], dim=1), # (E, 2, 3)
            already_hit_obs, # (E, 2, 12)
            ball_side_to_obs(self.ball_pos), # (E, 2, 2)
            serve_or_rally_to_obs(self.is_rally) # (E, 2, 2)
        ]
        high_level_obs = torch.cat(high_level_obs, dim=-1)

        if self._should_render(0):
            central_env_pos = self.envs_positions[self.central_env_idx]
            ball_plot_pos = (
                self.ball_pos[self.central_env_idx] + central_env_pos).tolist()  # [2, 3]
            if len(self.ball_traj_vis) > 1:
                point_list_0 = self.ball_traj_vis[-1]
                point_list_1 = ball_plot_pos
                colors = [(.1, 1., .1, 1.)]
                sizes = [1.5]
                self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

            self.ball_traj_vis.append(ball_plot_pos)

        return TensorDict(
            {
                "agents": {
                    "Opp_FirstPass_goto_observation": Opp_FirstPass_goto_obs,
                    "Opp_FirstPass_observation": Opp_FirstPass_obs,
                    "Opp_FirstPass_hover_observation": Opp_FirstPass_hover_obs,
                    "Opp_FirstPass_serve_observation": Opp_FirstPass_serve_obs,
                    "Opp_FirstPass_serve_hover_observation": Opp_FirstPass_serve_hover_obs,
                    "Opp_FirstPass_receive_observation": Opp_FirstPass_receive_obs,
                    "Opp_FirstPass_receive_hover_observation": Opp_FirstPass_receive_hover_obs,

                    "Opp_SecPass_goto_observation": Opp_SecPass_goto_obs,
                    "Opp_SecPass_observation": Opp_SecPass_obs,
                    "Opp_SecPass_hover_observation": Opp_SecPass_hover_obs,

                    "Opp_Att_goto_observation": Opp_Att_goto_obs,
                    "Opp_Att_observation": Opp_Att_obs,
                    "Opp_Att_hover_observation": Opp_Att_hover_obs,

                    "FirstPass_goto_observation": FirstPass_goto_obs,
                    "FirstPass_observation": FirstPass_obs,
                    "FirstPass_hover_observation": FirstPass_hover_obs,
                    "FirstPass_serve_observation": FirstPass_serve_obs,
                    "FirstPass_serve_hover_observation": FirstPass_serve_hover_obs,
                    "FirstPass_receive_observation": FirstPass_receive_obs,
                    "FirstPass_receive_hover_observation": FirstPass_receive_hover_obs,

                    "SecPass_goto_observation": SecPass_goto_obs,
                    "SecPass_observation": SecPass_obs,
                    "SecPass_hover_observation": SecPass_hover_obs,

                    "Att_goto_observation": Att_goto_obs,
                    "Att_observation": Att_obs,
                    "Att_hover_observation": Att_hover_obs,

                    "high_level_observation": high_level_obs,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.num_envs,
        )


    def _compute_reward_and_done(self):
        Opp_FirstPass_reward = torch.zeros((self.num_envs, 1), device=self.device)
        Opp_SecPass_reward = torch.zeros((self.num_envs, 1), device=self.device)
        Opp_Att_reward = torch.zeros((self.num_envs, 1), device=self.device)
        FirstPass_reward = torch.zeros((self.num_envs, 1), device=self.device)
        SecPass_reward = torch.zeros((self.num_envs, 1), device=self.device)
        Att_reward = torch.zeros((self.num_envs, 1), device=self.device)

        Opp_FirstPass_hit = self.racket_hit_ball[:, 5] # (E,)
        Opp_SecPass_hit = self.racket_hit_ball[:, 0] # (E,)
        Opp_Att_hit = self.racket_hit_ball[:, 1] # (E,)
        FirstPass_hit = self.racket_hit_ball[:, 2] # (E,)
        SecPass_hit = self.racket_hit_ball[:, 3] # (E,)
        Att_hit = self.racket_hit_ball[:, 4] # (E,)

        # End checking

        ball_hit_the_ground = (self.ball_pos[..., 2] < self.hit_ground_height) # (E,1)
        self.ball_first_hit_the_ground = ball_hit_the_ground & ~self.ball_already_hit_the_ground.bool()
        self.ball_already_hit_the_ground = torch.where(ball_hit_the_ground, True, self.ball_already_hit_the_ground)
        
        ball_hit_the_net = calculate_ball_hit_the_net(self.ball_pos, self.ball_radius, self.W, self.H_NET) # (E,1)

        drone_hit_the_ground = (self.drone.pos[..., 2] < self.hit_ground_height).unsqueeze(-1).any(dim=1) # (E,1)

        near_side_drone_pass_net = calculate_drone_pass_the_net(self.drone.pos[:, self.near_side_drone_idx, :], near_side=True)
        far_side_drone_pass_net = calculate_drone_pass_the_net(self.drone.pos[:, self.far_side_drone_idx, :], near_side=False)
        drone_pass_net = near_side_drone_pass_net | far_side_drone_pass_net
        
        Opp_FirstPass_wrong_hit_turn = ((self.Opp_FirstPass_already_hit == True) & Opp_FirstPass_hit.unsqueeze(-1))
        Opp_SecPass_wrong_hit_turn = ((self.Opp_SecPass_already_hit == True) & Opp_SecPass_hit.unsqueeze(-1))
        Opp_Att_wrong_hit_turn = ((self.Opp_Att_already_hit == True) & Opp_Att_hit.unsqueeze(-1))
        FirstPass_wrong_hit_turn = ((self.FirstPass_already_hit == True) & FirstPass_hit.unsqueeze(-1))
        SecPass_wrong_hit_turn = ((self.SecPass_already_hit == True) & SecPass_hit.unsqueeze(-1))
        Att_wrong_hit_turn = ((self.Att_already_hit == True) & Att_hit.unsqueeze(-1))
        
        wrong_hit_turn = torch.cat([Opp_SecPass_wrong_hit_turn, Opp_Att_wrong_hit_turn, FirstPass_wrong_hit_turn, SecPass_wrong_hit_turn, Att_wrong_hit_turn, Opp_FirstPass_wrong_hit_turn], dim=1) # (E, 6)
        drone_wrong_hit_turn = wrong_hit_turn.any(dim=-1, keepdim=True) # (E,1)

        drone_wrong_hit_racket = self.wrong_hit_racket.any(dim=-1, keepdim=True) # (E,1)

        terminated = (
            drone_hit_the_ground |
            drone_pass_net |
            drone_wrong_hit_turn |
            drone_wrong_hit_racket |
            ball_hit_the_ground |
            ball_hit_the_net
        )
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # (E,1)
        done = truncated | terminated

        if terminated[self.central_env_idx].item() and self._should_render(0):
            if drone_hit_the_ground[self.central_env_idx].item():
                logging.info("Drone hit the ground")
            if drone_pass_net[self.central_env_idx].item():
                logging.info("Drone pass the net")
            if drone_wrong_hit_turn[self.central_env_idx].item():
                logging.info("Drone hit the ball in wrong turn")
            if drone_wrong_hit_racket[self.central_env_idx].item():
                logging.info("Drone hit the ball not by racket")
            if ball_hit_the_ground[self.central_env_idx].item():
                logging.info("Ball hit the ground")
            if ball_hit_the_net[self.central_env_idx].item():
                logging.info("Ball hit the net")

        # reward
        true_FirstPass_hit = FirstPass_hit & ((self.info["high_level_action"][:, 0, 0] == 3) | (self.info["high_level_action"][:, 0, 0] == 4))
        true_SecPass_hit = SecPass_hit & (self.info["high_level_action"][:, 0, 1] == 1)
        true_Att_hit = Att_hit & ((self.info["high_level_action"][:, 0, 2] == 1) | (self.info["high_level_action"][:, 0, 2] == 2))
        
        true_Opp_FirstPass_hit = Opp_FirstPass_hit & ((self.info["high_level_old_action"][:, 1, 0] == 3) | (self.info["high_level_old_action"][:, 1, 0] == 4))
        true_Opp_SecPass_hit = Opp_SecPass_hit & (self.info["high_level_old_action"][:, 1, 1] == 1)
        true_Opp_Att_hit = Opp_Att_hit & ((self.info["high_level_old_action"][:, 1, 2] == 1) | (self.info["high_level_old_action"][:, 1, 2] == 2))

        near_team_hit = (true_FirstPass_hit | true_SecPass_hit | true_Att_hit)
        far_team_hit = (true_Opp_FirstPass_hit | true_Opp_SecPass_hit | true_Opp_Att_hit)
        _reward_hit_coef = 2.0
        reward_hit = _reward_hit_coef * torch.stack([near_team_hit, far_team_hit], dim=1) # (E, 2)

        _reward_FP_hit_coef = 1.0
        reward_FP_hit = _reward_FP_hit_coef * torch.stack([true_FirstPass_hit, true_Opp_FirstPass_hit], dim=1) # (E, 2)

        _penalty_too_close_coef = 0.0
        too_close = self._compute_drone_too_close(threshold=1.0)
        penalty_too_close = _penalty_too_close_coef * too_close

        _reward_win_coef = 20.0
        near_side_win, far_side_win, both_win = determine_game_result_3v3(
            self.drone.pos, 
            self.wrong_hit_racket, 
            wrong_hit_turn, 
            self.ball_pos, 
            self.W, 
            self.L, 
            self.H_NET, 
            self.drone_idx_dict, 
            self.last_hit_side,
            self.hit_ground_height
        )
        game_draw = both_win | (done & ~(near_side_win | far_side_win))

        if self._should_render(0) and (near_side_win | far_side_win | game_draw)[self.central_env_idx].item():
            self.draw.clear_points()
            self.debug_draw_win(near_side_win, far_side_win, game_draw)
        result = torch.concat([near_side_win.float(), -near_side_win.float()], dim=1) + torch.concat([-far_side_win.float(), far_side_win.float()], dim=1)
        reward_win = _reward_win_coef * result # (E, 2)

        high_level_reward = reward_hit + reward_win + reward_FP_hit - penalty_too_close # (E, 2)

        if self._should_render(0):
            if near_side_win[self.central_env_idx].item():
                logging.info("Near side wins.")
            elif far_side_win[self.central_env_idx].item():
                logging.info("Far side wins.")
            elif game_draw[self.central_env_idx].item():
                logging.info("Draw.")

        self.switch_turn = (
            Opp_FirstPass_hit | Opp_SecPass_hit | Opp_Att_hit | FirstPass_hit | SecPass_hit | Att_hit |
            self.ball_pass_net
        ) # (E,)
        self.info["switch_turn"] = self.switch_turn.unsqueeze(-1) # (E,1)

        # log
        ep_len = self.progress_buf.unsqueeze(-1)

        self.stats["episode_len"][:] = ep_len

        self.stats["done"] = done.float()
        self.stats["truncated"] = truncated.float()
        self.stats["terminated"] = terminated.float()

        self.stats["num_Opp_FirstPass_hit"] += Opp_FirstPass_hit.unsqueeze(-1).float()
        self.stats["num_Opp_SecPass_hit"] += Opp_SecPass_hit.unsqueeze(-1).float()
        self.stats["num_Opp_Att_hit"] += Opp_Att_hit.unsqueeze(-1).float()
        self.stats["num_FirstPass_hit"] += FirstPass_hit.unsqueeze(-1).float()
        self.stats["num_SecPass_hit"] += SecPass_hit.unsqueeze(-1).float()
        self.stats["num_Att_hit"] += Att_hit.unsqueeze(-1).float()
        self.stats["num_hit"] += self.racket_hit_ball.sum(dim=-1, keepdim=True).float()

        self.stats["done_drone_hit_the_ground"] = drone_hit_the_ground.float()
        self.stats["done_drone_pass_net"] = drone_pass_net.float()
        self.stats["done_drone_wrong_hit_turn"] = drone_wrong_hit_turn.float()
        self.stats["done_drone_wrong_hit_racket"] = drone_wrong_hit_racket.float()
        self.stats["done_ball_hit_the_ground"] = ball_hit_the_ground.float()
        self.stats["done_ball_hit_the_net"] = ball_hit_the_net.float()

        self.stats["actor_0_wins"] = near_side_win.float()
        self.stats["actor_1_wins"] = far_side_win.float()
        self.stats["draws"] = game_draw.float()

        done_idx = done.nonzero(as_tuple=True)[0]

        self.stats["actor_0_serve"] = (~self.serve_turn).float().unsqueeze(-1)
        self.stats["actor_1_serve"] = self.serve_turn.float().unsqueeze(-1)

        self.stats["actor_0_serve_and_win"] = (~self.serve_turn.unsqueeze(-1) & near_side_win).float()
        self.stats["actor_0_serve_and_lose"] = (~self.serve_turn.unsqueeze(-1) & far_side_win).float()
        self.stats["actor_1_not_serve_and_win"] = (~self.serve_turn.unsqueeze(-1) & far_side_win).float()
        self.stats["actor_1_not_serve_and_lose"] = (~self.serve_turn.unsqueeze(-1) & near_side_win).float()

        self.stats["actor_1_serve_and_win"] = (self.serve_turn.unsqueeze(-1) & far_side_win).float()
        self.stats["actor_1_serve_and_lose"] = (self.serve_turn.unsqueeze(-1) & near_side_win).float()
        self.stats["actor_0_not_serve_and_win"] = (self.serve_turn.unsqueeze(-1) & near_side_win).float()
        self.stats["actor_0_not_serve_and_lose"] = (self.serve_turn.unsqueeze(-1) & far_side_win).float()
        
        if self.stats_cfg.get("complete_reward_stats", False):
            self.stats["return_near_team"].add_(high_level_reward[:, 0].unsqueeze(-1))
            self.stats["reward_hit_near_team"].add_(reward_hit[:, 0].unsqueeze(-1))
            self.stats["reward_FirstPass_hit_near_team"].add_(reward_FP_hit[:, 0].unsqueeze(-1))
            self.stats["penalty_too_close_near_team"].add_(penalty_too_close[:, 0].unsqueeze(-1))
            self.stats["reward_win_near_team"].add_(reward_win[:, 0].unsqueeze(-1))

            self.stats["return_far_team"].add_(high_level_reward[:, 1].unsqueeze(-1))
            self.stats["reward_hit_far_team"].add_(reward_hit[:, 1].unsqueeze(-1))
            self.stats["reward_FirstPass_hit_far_team"].add_(reward_FP_hit[:, 1].unsqueeze(-1))
            self.stats["penalty_too_close_far_team"].add_(penalty_too_close[:, 1].unsqueeze(-1))
            self.stats["reward_win_far_team"].add_(reward_win[:, 1].unsqueeze(-1))

        return TensorDict(
            {
                "agents": { 
                    "Opp_FirstPass_reward": Opp_FirstPass_reward,
                    "Opp_SecPass_reward": Opp_SecPass_reward,
                    "Opp_Att_reward": Opp_Att_reward,
                    "FirstPass_reward": FirstPass_reward,
                    "SecPass_reward": SecPass_reward,
                    "Att_reward": Att_reward,
                    "high_level_reward": high_level_reward
                },
                "done": done,
                "terminated": terminated,
                "truncated": truncated
            },
            self.num_envs,
        )

    
    def _compute_drone_too_close(self, threshold=0.8):
        drone_pos = self.drone.pos  # (E, 6, 3)

        # team 0
        drone_pos_FP = drone_pos[:, 2, :]  # (E, 3) 
        drone_pos_SP = drone_pos[:, 3, :]  # (E, 3)
        drone_pos_Att = drone_pos[:, 4, :]  # (E, 3)
        too_close_01 = torch.norm(drone_pos_FP - drone_pos_SP, dim=-1) < threshold  # (E,)
        too_close_02 = torch.norm(drone_pos_FP - drone_pos_Att, dim=-1) < threshold  # (E,)
        too_close_12 = torch.norm(drone_pos_SP - drone_pos_Att, dim=-1) < threshold  # (E,)
        team_0_too_close = too_close_01 | too_close_02 | too_close_12  # (E,)

        # team 1
        drone_pos_OppFP = drone_pos[:, 5, :]  # (E, 3)
        drone_pos_OppSP = drone_pos[:, 0, :]  # (E, 3)
        drone_pos_OppAtt = drone_pos[:, 1, :]  # (E, 3)
        too_close_34 = torch.norm(drone_pos_OppFP - drone_pos_OppSP, dim=-1) < threshold  # (E,)
        too_close_35 = torch.norm(drone_pos_OppFP - drone_pos_OppAtt, dim=-1) < threshold  # (E,)
        too_close_45 = torch.norm(drone_pos_OppSP - drone_pos_OppAtt, dim=-1) < threshold  # (E,)
        team_1_too_close = too_close_34 | too_close_35 | too_close_45  # (E,)

        too_close = torch.stack([team_0_too_close, team_1_too_close], dim=1)  # (E, 2)

        return too_close
    

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        for substep in range(self.substeps):
            self._pre_sim_step(tensordict) 
            self.sim.step(self._should_render(substep))
        self._post_sim_step(tensordict)
        self.progress_buf += 1
        tensordict_out = TensorDict({}, self.batch_size, device=self.device)
        tensordict_out.update(self._compute_state_and_obs(is_step=True)) 
        self.update_hit_info()
        self.update_ball_pass_net()
        tensordict_out.update(self._compute_reward_and_done())
        self.update_already_hit_info()

        return tensordict_out
    

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
        else:
            env_mask = torch.ones(self.num_envs, dtype=bool, device=self.device)
        env_ids = env_mask.cpu().nonzero().squeeze(-1).to(self.device)
        self._reset_idx(env_ids)
        self.sim._physics_sim_view.flush()
        self.progress_buf[env_ids] = 0.
        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(self._compute_state_and_obs())
        tensordict.set("truncated", (self.progress_buf > self.max_episode_length).unsqueeze(1))
        return tensordict

    