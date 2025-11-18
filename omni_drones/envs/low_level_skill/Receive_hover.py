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
    UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
)
from pxr import UsdShade, PhysxSchema
import logging
from carb import Float3
from omni.isaac.debug_draw import _debug_draw
from omni_drones.utils.volleyball.common import (
    rectangular_cuboid_edges,_carb_float3_add, 
    draw_court, 
    calculate_ball_hit_the_net, 
    calculate_drone_hit_the_net,
    calculate_ball_in_side,
    turn_to_obs, 
    target_to_obs, 
    attacking_target_to_obs, 
    quaternion_multiply, 
    transfer_root_state_to_the_other_side,
    quat_rotate,
)
from omni_drones.utils.torchrl.transforms import append_to_h5
from omni.isaac.orbit.sensors import ContactSensorCfg, ContactSensor
from omegaconf import DictConfig
from typing import Tuple, List
from abc import abstractmethod
import os
from torch.distributions import MultivariateNormal
import pandas as pd


class Receive_hover(IsaacEnv):

    def __init__(self, cfg, headless):

        self.L: float = cfg.task.court.L
        self.W: float = cfg.task.court.W
        self.H_NET: float = cfg.task.court.H_NET # height of the net
        self.W_NET: float = cfg.task.court.W_NET # not width of the net, but the width of the net's frame

        self.ball_mass: float = cfg.task.ball_mass
        self.ball_radius: float = cfg.task.ball_radius

        self.done_FirstPass_hit_the_ground = cfg.task.done_FirstPass_hit_the_ground
        
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

        # # contact sensor from Orbit
        # contact_sensor_cfg = ContactSensorCfg(prim_path="/World/envs/env_.*/ball")
        # self.contact_sensor: ContactSensor = contact_sensor_cfg.class_type(contact_sensor_cfg)
        # self.contact_sensor._initialize_impl()

        self.ball_pos_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.ball_pos_dist.low, device=self.device), 
            high=torch.tensor(cfg.task.initial.ball_pos_dist.high, device=self.device)
        )
        self.ball_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.ball_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.ball_vel_dist.high, device=self.device)
        )

        self.Opp_Server_pos_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_Server_pos_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_Server_pos_dist.high, device=self.device)
        )
        self.Opp_Server_rpy_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_Server_rpy_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_Server_rpy_dist.high, device=self.device)
        )
        self.Opp_Server_lin_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_Server_lin_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_Server_lin_vel_dist.high, device=self.device)
        )
        self.Opp_Server_ang_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.Opp_Server_ang_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.Opp_Server_ang_vel_dist.high, device=self.device)
        )
        
        self.FirstPass_pos_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.FirstPass_pos_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.FirstPass_pos_dist.high, device=self.device)
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
        
        self.ball_traj_vis = []
        self.draw = _debug_draw.acquire_debug_draw_interface()

        self.turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.switch_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.ball_init_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.ball_last_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.ball_already_hit_the_ground = torch.zeros((self.num_envs, 1), device=self.device)
        self.ball_first_hit_the_ground = torch.zeros((self.num_envs, 1), device=self.device)
        self.done_ball_hit_the_ground = cfg.task.done_ball_hit_the_ground

        hover_target_rpy = torch.zeros((self.num_envs, 1, 3), device=self.device) # (E, 1, 3)
        hover_target_rot = euler_to_quaternion(hover_target_rpy) # (E, 1, 4)
        self.hover_target_heading = quat_axis(hover_target_rot.squeeze(1), 0).unsqueeze(1) # (E, 1, 3)
        
        # q_m = torch.tensor([[0, 0, 0, 1]] * hover_target_rot.shape[0], device=hover_target_rot.device) # (E, 4)
        # hover_target_rot_transfer = quaternion_multiply(q_m, hover_target_rot.squeeze(1)) # (E, 4)
        # hover_target_rot_transfer = torch.where((hover_target_rot_transfer[..., 0] < 0).unsqueeze(-1), -hover_target_rot_transfer, hover_target_rot_transfer) # (E, 4)
        # self.hover_target_heading_transfer = quat_axis(hover_target_rot_transfer, 0).unsqueeze(1) # (E, 1, 3)
        
        self.Opp_Server_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: hover; 1: my turn
        self.Opp_Server_hit = torch.zeros((self.num_envs, 1), device=self.device)
        self.Opp_Server_last_hit_t = torch.zeros(self.num_envs, device=self.device)
        self.Opp_Server_hover_pos_after_hit = torch.tensor(cfg.task.get("Opp_Server_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)

        self.FirstPass_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.FirstPass_hit = torch.zeros((self.num_envs, 1), device=self.device)
        self.FirstPass_last_hit_t = torch.zeros(self.num_envs, device=self.device)
        # self.FirstPass_goto_pos_left_before_hit = torch.tensor(cfg.task.get("FirstPass_goto_pos_before_hit"), device=self.device).expand(self.num_envs, 1, 3)
        # self.FirstPass_goto_pos_right_before_hit = self.FirstPass_goto_pos_left_before_hit.clone()
        # self.FirstPass_goto_pos_right_before_hit[..., 1] = - self.FirstPass_goto_pos_right_before_hit[..., 1]
        # self.FirstPass_goto_pos_target_before_hit = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.FirstPass_goto_pos = torch.tensor(cfg.task.get("FirstPass_goto_pos"), device=self.device).expand(self.num_envs, 1, 3)
        self.FirstPass_hover_pos_after_hit = torch.tensor(cfg.task.get("FirstPass_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)
        
        self.FirstPass_ball_anchor = torch.tensor(cfg.task.FirstPass_ball_anchor, device=self.device).expand(self.num_envs, 1, 3)
        self.FirstPass_highest_ball_pos_after_my_turn = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.flag_reward_SecPass_hit_pos = torch.zeros((self.num_envs, 1), device=self.device)

        self.Server_ball_rpos = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.Server_ball_anchor = torch.zeros((self.num_envs, 1, 3), device=self.device)        

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
        # cr_api = PhysxSchema.PhysxContactReportAPI.Apply(ball.prim)
        # cr_api.CreateThresholdAttr().Set(0.)

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
                (1.0, -1.0, 1.0),
                (1.0, 1.0, 2.0),
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

        Opp_Server_observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 3
        Opp_Server_hover_observation_dim = drone_state_dim + 3
        FirstPass_goto_observation_dim = drone_state_dim + 3
        FirstPass_observation_dim = drone_state_dim + 3 + 3 + 3 + 2
        FirstPass_hover_observation_dim = drone_state_dim + 3

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "Opp_Server_observation": UnboundedContinuousTensorSpec((1, Opp_Server_observation_dim)),
                "Opp_Server_hover_observation": UnboundedContinuousTensorSpec((1, Opp_Server_hover_observation_dim)),
                "FirstPass_goto_observation": UnboundedContinuousTensorSpec((1, FirstPass_goto_observation_dim)),
                "FirstPass_observation": UnboundedContinuousTensorSpec((1, FirstPass_observation_dim)),
                "FirstPass_hover_observation": UnboundedContinuousTensorSpec((1, FirstPass_hover_observation_dim)),
            })
        }).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "Opp_Server_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Opp_Server_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "FirstPass_goto_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "FirstPass_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "FirstPass_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),
            })
        }).expand(self.num_envs).to(self.device)

        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "Opp_Server_reward": UnboundedContinuousTensorSpec((1, 1)),
                "FirstPass_reward": UnboundedContinuousTensorSpec((1, 1)),
            })
        }).expand(self.num_envs).to(self.device)

        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["Opp_Server"] = AgentSpec(
            "Opp_Server",
            1,
            observation_key=("agents", "Opp_Server_observation"),
            action_key=("agents", "Opp_Server_action"),
            reward_key=("agents", "Opp_Server_reward"),
            state_key=("agents", "Opp_Server_state"),
        )

        self.agent_spec["Opp_Server_hover"] = AgentSpec(
            "Opp_Server_hover",
            1,
            observation_key=("agents", "Opp_Server_hover_observation"),
            action_key=("agents", "Opp_Server_hover_action"),
            reward_key=("agents", "Opp_Server_reward"),
            state_key=("agents", "Opp_Server_hover_state"),
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

        self.agent_spec["FirstPass_hover"] = AgentSpec(
            "FirstPass_hover",
            1,
            observation_key=("agents", "FirstPass_hover_observation"),
            action_key=("agents", "FirstPass_hover_action"),
            reward_key=("agents", "FirstPass_reward"),
            state_key=("agents", "FirstPass_hover_state"),
        )

        _stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
            "terminated": UnboundedContinuousTensorSpec(1),
            "Opp_Server_hit": UnboundedContinuousTensorSpec(1),
            "FirstPass_hit": UnboundedContinuousTensorSpec(1),
            # "ball_hit_the_ground_vel_x": UnboundedContinuousTensorSpec(1),
            # "ball_hit_the_ground_vel_y": UnboundedContinuousTensorSpec(1),
            # "ball_hit_the_ground_vel_z": UnboundedContinuousTensorSpec(1),
            # "ball_hit_the_ground_vel": UnboundedContinuousTensorSpec(1),
            # "ball_hit_the_ground": UnboundedContinuousTensorSpec(1),
            "ball_hit_the_net": UnboundedContinuousTensorSpec(1),
            # "in_side": UnboundedContinuousTensorSpec(1),
        })
        self.stats_cfg: DictConfig = self.cfg.task.stats
        if self.stats_cfg.get("complete_reward_stats", False):
            _stats_spec.set("reward_hover", UnboundedContinuousTensorSpec(1))
            #_stats_spec.set("reward_up", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_pose", UnboundedContinuousTensorSpec(1))
            #_stats_spec.set("reward_spin", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_end", UnboundedContinuousTensorSpec(1))
            #_stats_spec.set("reward_pitch", UnboundedContinuousTensorSpec(1))
            #_stats_spec.set("reward_pos_z", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_drone_hit_the_net", UnboundedContinuousTensorSpec(1))
            #_stats_spec.set("penalty_drone_hit_the_ground", UnboundedContinuousTensorSpec(1))


        stats_spec = _stats_spec.expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["info"] = info_spec
        self.info = info_spec.zero()


    def debug_draw_turn(self):

        turn = self.turn[self.central_env_idx]
        FirstPass_turn = self.FirstPass_turn[self.central_env_idx]
        
        ori = self.envs_positions[self.central_env_idx].detach()
        
        points_turn = torch.tensor([2., 0., 0.]).to(self.device) + ori
        points_turn = [points_turn.tolist()]

        points_FirstPass_turn = torch.tensor([4., 0., 0.]).to(self.device) + ori
        points_FirstPass_turn = [points_FirstPass_turn.tolist()]

        if turn.item() == 0:
            colors_turn = [(1, 0, 0, 1)]
            logging.info("Central env turn: all hover")
        elif turn.item() == 1:
            colors_turn = [(0, 1, 0, 1)]
            logging.info("Central env turn: Opp_Server turn")
        elif turn.item() == 2:
            colors_turn = [(1, 1, 0, 1)]
            logging.info("Central env turn: FirstPass turn")
        
        if FirstPass_turn.item() == 1:
            colors_FirstPass_turn = [(0, 1, 0, 1)]
            # logging.info("Central env FirstPass turn: FirstPass turn")
        else:
            colors_FirstPass_turn = [(1, 0, 0, 1)]
            # logging.info("Central env FirstPass turn: FirstPass Hover turn")
        
        sizes = [15.]

        self.draw.draw_points(points_turn, colors_turn, sizes)
        self.draw.draw_points(points_FirstPass_turn, colors_FirstPass_turn, sizes)
    
    
    def debug_draw_target_point(self):
        ori = self.envs_positions[self.central_env_idx].detach()
        target_point = self.FirstPass_ball_anchor[0, 0, :].clone().detach() + ori
        target_point = [target_point.tolist()]
        siezes = [15.]
        colors = [(1, 0, 0, 1)]
        self.draw.draw_points(target_point, colors, siezes)


    def _reset_idx(self, env_ids: torch.Tensor):

        self.turn[env_ids] = torch.ones(len(env_ids), device=self.device).long() 

        self.Opp_Server_turn[env_ids] =  torch.ones(len(env_ids), device=self.device).long()
        self.FirstPass_turn[env_ids] = torch.zeros(len(env_ids), device=self.device).long()

        self.FirstPass_goto_pos_target_before_hit = self.FirstPass_goto_pos

        self.drone._reset_idx(env_ids, self.training)

        Opp_Server_pos = self.Opp_Server_pos_dist.sample(env_ids.shape).unsqueeze(1)
        Opp_Server_rpy = self.Opp_Server_rpy_dist.sample(env_ids.shape).unsqueeze(1)

        FirstPass_pos = self.FirstPass_pos_dist.sample(env_ids.shape).unsqueeze(1)
        FirstPass_rpy = self.FirstPass_rpy_dist.sample(env_ids.shape).unsqueeze(1)

        # set to the far side of the court
        Opp_Server_pos[..., 0:2] = - Opp_Server_pos[..., 0:2]
        Opp_Server_rpy[..., 2] += 3.1415926


        drone_pos = torch.cat([Opp_Server_pos, FirstPass_pos], dim=1)
        drone_rpy = torch.cat([Opp_Server_rpy, FirstPass_rpy], dim=1)
        drone_rot = euler_to_quaternion(drone_rpy)

        self.drone.set_world_poses(
            drone_pos +
            self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )

        Opp_Server_lin_vel = self.Opp_Server_lin_vel_dist.sample((*env_ids.shape, 1))
        Opp_Server_ang_vel = self.Opp_Server_ang_vel_dist.sample((*env_ids.shape, 1))
        Opp_Server_vel = torch.cat((Opp_Server_lin_vel, Opp_Server_ang_vel), dim=-1)

        FirstPass_lin_vel = self.FirstPass_lin_vel_dist.sample((*env_ids.shape, 1))
        FirstPass_ang_vel = self.FirstPass_ang_vel_dist.sample((*env_ids.shape, 1))
        FirstPass_vel = torch.cat((FirstPass_lin_vel, FirstPass_ang_vel), dim=-1)

        drone_vel = torch.cat([Opp_Server_vel, FirstPass_vel], dim=1)    
        self.drone.set_velocities(drone_vel, env_ids)

        ball_pos = self.ball_pos_dist.sample((*env_ids.shape, 1))
        ball_rot = torch.tensor([1., 0., 0., 0.], device=self.device).repeat(len(env_ids), 1)
        ball_lin_vel = self.ball_vel_dist.sample((*env_ids.shape, 1))
        ball_lin_vel[..., 1] -= (ball_pos[..., 1]) * 0.5
        ball_ang_vel = torch.zeros_like(ball_lin_vel)

        # set to the far side of the court
        ball_pos[..., 0:2] = - ball_pos[..., 0:2]
        ball_lin_vel[..., :2] = - ball_lin_vel[..., :2]
            
        ball_vel = torch.cat((ball_lin_vel, ball_ang_vel), dim=-1)
        
        self.ball.set_world_poses(
            ball_pos +
            self.envs_positions[env_ids].unsqueeze(1), ball_rot, env_ids
        )
        self.ball.set_velocities(ball_vel, env_ids)
        
        self.ball.set_masses(torch.ones_like(env_ids,dtype=torch.float)*self.ball_mass, env_ids)
        self.ball_init_vel[env_ids] = ball_lin_vel
        self.ball_last_vel[env_ids] = ball_lin_vel

        self.Server_ball_anchor[env_ids, 0, 0] = torch.rand(len(env_ids), device=self.device) * 1.5 + 4.5  # x: [4, 6]
        self.Server_ball_anchor[env_ids, 0, 1] = torch.rand(len(env_ids), device=self.device) * 3.8 - 1.9  # y: [-2, 2]
        # self.Server_ball_anchor[env_ids, 0, 0] = 4.5  # x: [4, 6]
        # self.Server_ball_anchor[env_ids, 0, 1] = 1.9  # y: [-2, 2]
        self.Server_ball_anchor[env_ids, 0, 2] = 0  # z: 1

        self.stats[env_ids] = 0.

        self.switch_turn[env_ids] = 0
        self.ball_already_hit_the_ground[env_ids] = 0
        self.ball_first_hit_the_ground[env_ids] = 0

        self.Opp_Server_hit[env_ids] = 0
        self.Opp_Server_last_hit_t[env_ids] = -10

        self.FirstPass_hit[env_ids] = 0
        self.FirstPass_last_hit_t[env_ids] = -10
        self.FirstPass_highest_ball_pos_after_my_turn[env_ids] = 0
        self.flag_reward_SecPass_hit_pos[env_ids] = 0

        if (env_ids == self.central_env_idx).any() and self._should_render(0):
            self.ball_traj_vis.clear()
            self.draw.clear_lines()

            logging.info("Reset central environment")

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

            self.draw.clear_points()
            self.debug_draw_turn()
            self.debug_draw_target_point()


    def _pre_sim_step(self, tensordict: TensorDictBase):
        Opp_Server_action = tensordict[("agents", "Opp_Server_action")]
        Opp_Server_hover_action = tensordict[("agents", "Opp_Server_hover_action")]
        Opp_Server_real_action = torch.where(
            tensordict["stats"]["Opp_Server_hit"].unsqueeze(-1).bool(), 
            Opp_Server_hover_action, 
            Opp_Server_action
        )

        FirstPass_goto_action = tensordict[("agents", "FirstPass_goto_action")]
        FirstPass_action = tensordict[("agents", "FirstPass_action")]
        FirstPass_hover_action = tensordict[("agents", "FirstPass_hover_action")]
        FirstPass_real_action = torch.where(
            tensordict["stats"]["Opp_Server_hit"].unsqueeze(-1).bool(), 
            FirstPass_action, 
            FirstPass_goto_action
        )
        FirstPass_real_action = torch.where(
                tensordict["stats"]["FirstPass_hit"].unsqueeze(-1).bool(), 
                FirstPass_hover_action, 
                FirstPass_real_action
            )        

        actions = torch.cat([Opp_Server_real_action, FirstPass_real_action], dim=1)
        self.effort = self.drone.apply_action(actions)


    # def _post_sim_step(self, tensordict: TensorDictBase):
    #     self.contact_sensor.update(self.dt)


    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state() # (E, 3, 23)
        self.info["drone_state"][:] = self.root_state[..., :13]
        
        self.ball_pos, _ = self.get_env_poses(self.ball.get_world_poses()) # (E, 1, 3)
        self.ball_vel = self.ball.get_velocities()[..., :3] # (E, 1, 3)

        self.rpos = self.ball_pos - self.drone.pos # (E, 3, 3)

        pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )
        rot = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot, rot) # make sure w of (w,x,y,z) is positive
        self.root_state = torch.cat([pos, rot, vel, angular_vel, heading, up, throttle], dim=-1)
        
        rpy = quaternion_to_euler(rot)
        
        self.drone_rot = rot

        # save important information for reward computing
        self.FirstPass_rpos = self.rpos[:, 1, :].unsqueeze(1)
        self.FirstPass_pos = pos[:, 1, :].unsqueeze(1)
        self.FirstPass_yaw = rpy[:, 1, 2].unsqueeze(-1)
        self.FirstPass_roll = rpy[:, 1, 0].unsqueeze(-1)
        self.FirstPass_vel = vel[:, 1, :].unsqueeze(1)

        # for the far side of the court
        ball_pos_transfer = self.ball_pos.clone()
        ball_pos_transfer[..., :2] = - ball_pos_transfer[..., :2]
        ball_vel_transfer = self.ball_vel.clone()
        ball_vel_transfer[..., :2] = - ball_vel_transfer[..., :2]

        ball_anchor_transfer = self.Server_ball_anchor.clone()
        ball_anchor_transfer[..., :2] = - ball_anchor_transfer[..., :2]        
        self.Server_ball_rpos = ball_pos_transfer - ball_anchor_transfer

        # root_state: (E, 1, 23)
        Opp_Server_root_state = transfer_root_state_to_the_other_side(self.root_state[:, 0, :].unsqueeze(1))
        FirstPass_root_state = self.root_state[:, 1, :].unsqueeze(1)

        # relative pos to ball
        Opp_Server_rpos = self.rpos[:, 0, :].clone().unsqueeze(1)
        Opp_Server_rpos[..., :2] = - Opp_Server_rpos[..., :2]
        FirstPass_rpos = self.rpos[:, 1, :].unsqueeze(1)

        # Opp_Server_obs
        Opp_Server_obs = [
            Opp_Server_root_state, # (E, 1, 23)
            ball_pos_transfer, # (E, 1, 3)
            Opp_Server_rpos, # (E, 1, 3)
            ball_vel_transfer, # (E, 1, 3)
            turn_to_obs(self.Opp_Server_turn), # (E, 1, 2)
            self.Server_ball_rpos,
        ]
        Opp_Server_obs = torch.cat(Opp_Server_obs, dim=-1)

        # Opp_Server_hover_obs
        Opp_Server_hover_rpos = self.Opp_Server_hover_pos_after_hit - Opp_Server_root_state[..., :3] # (E, 1, 3)
        Opp_Server_hover_rheading = self.hover_target_heading - Opp_Server_root_state[..., 13:16] # (E, 1, 3)
        Opp_Server_hover_obs = [
            Opp_Server_hover_rpos, # (E, 1, 3)
            Opp_Server_root_state[..., 3:], # (E, 1, root_state_dim - 3)
            Opp_Server_hover_rheading, # (E, 1, 3)
        ]
        Opp_Server_hover_obs = torch.cat(Opp_Server_hover_obs, dim=-1)

        # FirstPass_goto_obs
        FirstPass_goto_rpos = self.FirstPass_goto_pos_target_before_hit - FirstPass_root_state[..., :3] # (E, 1, 3)
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

            if self.switch_turn[self.central_env_idx]:
                self.debug_draw_turn()
                self.debug_draw_target_point()

        return TensorDict(
            {
                "agents": {
                    "Opp_Server_observation": Opp_Server_obs,
                    "Opp_Server_hover_observation": Opp_Server_hover_obs,

                    "FirstPass_goto_observation": FirstPass_goto_obs,
                    "FirstPass_observation": FirstPass_obs,
                    "FirstPass_hover_observation": FirstPass_hover_obs,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.num_envs,
        )
    

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

        racket_near_ball = self.check_ball_near_racket(racket_radius=racket_radius, cylinder_height_coeff=cylinder_height_coeff)  # (E,3)
        drone_near_ball = (torch.norm(self.rpos, dim=-1) < 0.4) # (E,3)
        
        ball_vel_z_change = ((self.ball_vel[..., 2] - self.ball_last_vel[..., 2]) > 9.8 * sim_dt) # (E,1)
        ball_vel_x_y_change = (self.ball_vel[..., :2] - self.ball_last_vel[..., :2]).norm(dim=-1) > 0.5 # (E,1)
        ball_vel_change = ball_vel_z_change | ball_vel_x_y_change # (E,1)

        drone_hit_ball = drone_near_ball & ball_vel_change # (E,1)
        racket_hit_ball = racket_near_ball & ball_vel_change # (E,1)

        racket_hit_ball[:, 0] = racket_hit_ball[:, 0] & (self.progress_buf - self.Opp_Server_last_hit_t > 3) # (E,)
        racket_hit_ball[:, 1] = racket_hit_ball[:, 1] & (self.progress_buf - self.FirstPass_last_hit_t > 3) # (E,)
        
        drone_hit_ball[:, 0] = drone_hit_ball[:, 0] & (self.progress_buf - self.Opp_Server_last_hit_t > 3)
        drone_hit_ball[:, 1] = drone_hit_ball[:, 1] & (self.progress_buf - self.FirstPass_last_hit_t > 3)

        return racket_hit_ball, drone_hit_ball
    

    def _compute_reward_and_done(self):
        Opp_Server_reward = torch.zeros((self.num_envs, 1), device=self.device)
        
        racket_hit_ball, drone_hit_ball = self.check_hit(sim_dt=self.dt)
        wrong_hit_racket = drone_hit_ball & ~racket_hit_ball # (E,1)
        
        self.Opp_Server_last_hit_t = torch.where(racket_hit_ball[:, 0], self.progress_buf, self.Opp_Server_last_hit_t)
        self.FirstPass_last_hit_t = torch.where(racket_hit_ball[:, 1], self.progress_buf, self.FirstPass_last_hit_t)
        
        self.Opp_Server_hit = torch.where(racket_hit_ball[:, 0].unsqueeze(-1) == 1, 1, self.Opp_Server_hit)
        self.FirstPass_hit = torch.where(racket_hit_ball[:, 1].unsqueeze(-1) == 1, 1, self.FirstPass_hit)
        
        self.ball_last_vel = self.ball_vel

        # reward 1: hover
        rpos_hover = self.drone.pos[:, 1, :].unsqueeze(1) - self.FirstPass_hover_pos_after_hit
        rheading =  (self.drone.heading[:, 1, :].unsqueeze(1) - torch.tensor([1., 0., 0.], device=self.device).unsqueeze(0).unsqueeze(0))
        distance = torch.norm(torch.cat([rpos_hover, rheading], dim=-1), dim=-1)
        reward_pose = 1 / (1.0 + torch.square(1.2 * distance))
        reward_up = torch.square((self.drone.up[:, 1, 2] + 1) / 2).unsqueeze(-1)
        spinnage = torch.square(self.drone.vel[:, 1, -1]).unsqueeze(-1)
        reward_spin = 1 / (1.0 + torch.square(spinnage))

        reward_hover = 3 * (
            reward_pose 
            + reward_pose * (reward_up + reward_spin) 
        ) # (E,1)

        # End checking

        ball_hit_the_ground = (self.ball_pos[..., 2] < 0.2) # (E,1)
        self.ball_first_hit_the_ground = ball_hit_the_ground & ~self.ball_already_hit_the_ground.bool()
        self.ball_already_hit_the_ground = torch.where(ball_hit_the_ground, 1, self.ball_already_hit_the_ground)
        ball_hit_the_net = calculate_ball_hit_the_net(self.ball_pos, self.ball_radius, self.W, self.H_NET) # (E,1)
        FirstPass_hit_the_ground = (self.FirstPass_pos[..., 2] < 0.35) # (E,1)

        terminated = (
            (self.drone.pos[..., 2] > 4.0).any(dim=1).unsqueeze(-1) # z direction
            |(self.drone.pos[..., 2] < 0.30).any(dim=1).unsqueeze(-1) # z direction
            | (self.ball_pos[..., 2] > 5.) # z direction
            | ((self.Opp_Server_turn == 0) & racket_hit_ball[:, 0]).unsqueeze(-1) # hover and hit
            | ((self.FirstPass_turn == 0) & racket_hit_ball[:, 1]).unsqueeze(-1) # hover and hit
            | wrong_hit_racket.any(dim=1).unsqueeze(-1) # not racket hit the ball
        )
        if self.done_ball_hit_the_ground:
            terminated = terminated | ball_hit_the_ground
        if self.done_FirstPass_hit_the_ground:
            terminated = terminated | FirstPass_hit_the_ground

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # (E,1)
        done = truncated | terminated

        FirstPass_not_hit_and_ball_in_side = self.Opp_Server_hit * (1 - self.FirstPass_hit) * ball_hit_the_ground * calculate_ball_in_side(self.ball_pos, self.W, self.L, near_or_far="near").float() # (E,1)
        FirstPass_not_hit_and_ball_out_side = self.Opp_Server_hit * (1 - self.FirstPass_hit) * ball_hit_the_ground * (1 - calculate_ball_in_side(self.ball_pos, self.W, self.L, near_or_far="near").float()) # (E,1)

        # reward 3: End reward

        _penalty_drone_hit_the_net_coeff = 5.0
        drone_hit_the_net = calculate_drone_hit_the_net(self.drone.pos, self.W, self.H_NET)[:, 1].unsqueeze(-1) # (E,1)
        penalty_drone_hit_the_net = self.FirstPass_hit * _penalty_drone_hit_the_net_coeff * drone_hit_the_net.float() # (E,1)

        _penalty_drone_hit_the_ground_coeff = 10.0 
        drone_hit_the_ground = (self.drone.pos[:, 1, 2].unsqueeze(-1) < 0.30)
        penalty_drone_hit_the_ground = self.FirstPass_hit * _penalty_drone_hit_the_ground_coeff * drone_hit_the_ground.float() # (E,1)

        _penalty_drone_wrong_side_coeff = 5.0 #
        drone_wrong_side = (self.drone.pos[:, 1, 0].unsqueeze(-1) < 0)
        penalty_drone_wrong_side =  _penalty_drone_wrong_side_coeff * drone_wrong_side.float()

        _penalty_end_drone_pos_z_coeff = 10
        penalty_end_drone_pos_z = _penalty_end_drone_pos_z_coeff * (self.drone.pos[:, 1, 2].unsqueeze(-1) < 0.35).float()        
        
        reward_end = (- penalty_drone_hit_the_net - penalty_drone_hit_the_ground - penalty_drone_wrong_side - penalty_end_drone_pos_z)# (E,1)

        reward_end = torch.where(
            done, 
            reward_end, 
            torch.zeros_like(done, device=self.device)
        ) # (E,1)

        # Overall reward
        FirstPass_reward_1 = torch.zeros((self.num_envs, 1), device=self.device)
        FirstPass_reward_2 = reward_hover + reward_end # (E,1)

        FirstPass_reward = torch.where(self.FirstPass_hit.bool(), FirstPass_reward_2, FirstPass_reward_1)

        # change turn
        Opp_Server_hit = racket_hit_ball[:, 0] # (E,)
        FirstPass_hit = racket_hit_ball[:, 1] # (E,)
        last_turn = self.turn.clone() # (E,)

        self.turn = torch.where((self.turn == 1) & Opp_Server_hit, 2, self.turn) # (E,)
        self.turn = torch.where((self.turn == 2) & FirstPass_hit, 0, self.turn) # (E,)
        self.Opp_Server_turn = torch.where((self.turn == 1), 1, 0) # (E,)
        self.FirstPass_turn = torch.where((self.turn == 2), 1, 0)
        self.switch_turn = torch.where((self.turn ^ last_turn).bool(), 1, 0) # (E,)

        # log
        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["return"].add_(FirstPass_reward)
        self.stats["episode_len"][:] = ep_len

        self.stats["truncated"].add_(truncated.float())
        self.stats["terminated"].add_(terminated.float())

        self.stats["Opp_Server_hit"] = self.Opp_Server_hit
        self.stats["FirstPass_hit"] = self.FirstPass_hit

        # self.stats["ball_hit_the_ground"].add_(self.ball_first_hit_the_ground)
        self.stats["ball_hit_the_net"].add_(ball_hit_the_net)
        # self.stats["in_side"].add_(in_side)

        # self.stats["ball_hit_the_ground_vel_x"].add_(in_side * self.ball_vel[..., 0])
        # self.stats["ball_hit_the_ground_vel_y"].add_(in_side * self.ball_vel[..., 1])
        # self.stats["ball_hit_the_ground_vel_z"].add_(in_side * self.ball_vel[..., 2])
        # self.stats["ball_hit_the_ground_vel"].add_(in_side * self.ball_vel.norm(dim=-1))
        
        if self.stats_cfg.get("complete_reward_stats", False):
            self.stats["reward_hover"].add_(reward_hover)
            #self.stats["reward_up"].add_(reward_up)
            self.stats["reward_pose"].add_(reward_pose)
            #self.stats["reward_spin"].add_(reward_spin)
            
            self.stats["reward_end"].add_(reward_end)

        return TensorDict(
            {
                "agents": {
                    "Opp_Server_reward": Opp_Server_reward,
                    "FirstPass_reward": FirstPass_reward,
                },
                "done": done,
                "terminated": terminated,
                "truncated": truncated
            },
            self.num_envs,
        )