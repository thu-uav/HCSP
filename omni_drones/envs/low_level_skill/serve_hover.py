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
    calculate_ball_in_side,
    calculate_drone_hit_the_net,
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


class Serve_hover(IsaacEnv):

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
        self.Server_pos_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.drone_pos_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.drone_pos_dist.high, device=self.device)
        )
        self.Server_rpy_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.drone_rpy_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.drone_rpy_dist.high, device=self.device)
        )
        self.Server_lin_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.drone_lin_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.drone_lin_vel_dist.high, device=self.device)
        )
        self.Server_ang_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.drone_ang_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.drone_ang_vel_dist.high, device=self.device)
        )

        self.ball_traj_vis = []
        self.draw = _debug_draw.acquire_debug_draw_interface()

        self.turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: hover; 1: my turn
        self.switch_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.attacking_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: left; 1: mid; 2:right

        self.Server_hit = torch.zeros((self.num_envs, 1), device=self.device)
        self.Server_hit_t = torch.zeros((self.num_envs, 1), device=self.device)
        self.Server_last_hit_t = torch.zeros((self.num_envs, 1), device=self.device)

        self.ball_hit_ground = torch.zeros((self.num_envs, 1), device=self.device)
        self.ball_already_hit_the_ground = torch.zeros((self.num_envs, 1), device=self.device)
        self.in_side = torch.zeros((self.num_envs, 1), device=self.device)
        self.ball_anchor = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.ball_rpos = torch.zeros((self.num_envs, 1, 3), device=self.device)

        self.highest_ball_pos_after_my_turn = torch.zeros((self.num_envs, 1, 3), device=self.device)

        self.last_drone_positions = torch.zeros(size=(self.num_envs, 1, 3), device=self.device)

        self.ball_last_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.ball_init_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)

        self.done_ball_hit_ground = cfg.task.get("done_ball_hit_ground", False)

        hover_target_rpy = torch.zeros((self.num_envs, 1, 3), device=self.device) # (E, 1, 3)
        hover_target_rot = euler_to_quaternion(hover_target_rpy) # (E, 1, 4)
        self.hover_target_heading = quat_axis(hover_target_rot.squeeze(1), 0).unsqueeze(1) # (E, 1, 3)

        self.Server_hover_pos_after_hit = torch.tensor(cfg.task.get("Server_hover_pos_after_hit"), device=self.device)
        
        
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
            ]
        )

        material = UsdShade.Material(material.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("base_link/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

        return ["/World/defaultGroundPlane"]


    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]

        Server_observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 3
        Server_hover_observation_dim = drone_state_dim + 3 

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "Server_observation": UnboundedContinuousTensorSpec((1, Server_observation_dim)),
                "Server_hover_observation": UnboundedContinuousTensorSpec((1, Server_hover_observation_dim)),
            })
        }).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "Server_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Server_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),
            })
        }).expand(self.num_envs).to(self.device)
        
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "Server_reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)
        
        self.agent_spec["Server"] = AgentSpec(
            "Server",
            1,
            observation_key=("agents", "Server_observation"),
            action_key=("agents", "Server_action"),
            reward_key=("agents", "Server_reward"),
            state_key=("agents", "Server_state"),
        )

        self.agent_spec["Server_hover"] = AgentSpec(
            "Server_hover",
            1,
            observation_key=("agents", "Server_hover_observation"),
            action_key=("agents", "Server_hover_action"),
            reward_key=("agents", "Server_reward"),
            state_key=("agents", "Server_state"),
        )
        
        _stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
            "terminated": UnboundedContinuousTensorSpec(1),
            "done": UnboundedContinuousTensorSpec(1),
            "Server_hit": UnboundedContinuousTensorSpec(1),
            "num_hits": UnboundedContinuousTensorSpec(1),
        })
        self.stats_cfg: DictConfig = self.cfg.task.stats
        if self.stats_cfg.get("complete_reward_stats", False):
            _stats_spec.set("reward_hover", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_up", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_pose", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_spin", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_end", UnboundedContinuousTensorSpec(1))
            # _stats_spec.set("reward_pitch", UnboundedContinuousTensorSpec(1))
            # _stats_spec.set("reward_pos_z", UnboundedContinuousTensorSpec(1))
            # _stats_spec.set("penalty_drone_hit_the_net", UnboundedContinuousTensorSpec(1))
            # _stats_spec.set("penalty_drone_hit_the_ground", UnboundedContinuousTensorSpec(1))

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
        ori = self.envs_positions[self.central_env_idx].detach()
        
        points = torch.tensor([4., 0., 0.]).to(self.device) + ori
        points = [points.tolist()]

        if turn.item() == 0:
            colors = [(1, 0, 0, 1)]
        else:
            colors = [(0, 1, 0, 1)]

        sizes = [15.]
        self.draw.draw_points(points, colors, sizes)
        
        if turn == 0:
            logging.info("Central env turn: Hover")
        else:
            logging.info("Central env turn: Server turn")


    def debug_draw_attacking_target(self):

        ori = self.envs_positions[self.central_env_idx].detach()
        
        # Compute the target points
        target_point = self.ball_anchor[self.central_env_idx].clone().detach() + ori
        target_point = target_point.squeeze(0).tolist()

        target_hover_point = torch.tensor([4.5, 0.0, 2.0]).to(self.device) + ori
        target_hover_point = [target_hover_point.tolist()]
        colors = [(1, 0, 0, 1)]

        sizes = [15.]

        self.draw.draw_points([target_point], colors, sizes)
        self.draw.draw_points(target_hover_point, [(1, 0, 0, 1)], sizes)
        
        logging.info(f"Central env attacking target: {target_point}")



    def _reset_idx(self, env_ids: torch.Tensor):
        
        self.drone._reset_idx(env_ids, self.training)
        Server_pos = self.Server_pos_dist.sample(env_ids.shape).unsqueeze(1)
        Server_rpy = self.Server_rpy_dist.sample(env_ids.shape).unsqueeze(1)

        drone_pos = Server_pos
        drone_rpy = Server_rpy
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos +
            self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )

        Server_lin_vel = self.Server_lin_vel_dist.sample((*env_ids.shape, 1))
        Server_ang_vel = self.Server_ang_vel_dist.sample((*env_ids.shape, 1))
        Server_vel = torch.cat((Server_lin_vel, Server_ang_vel), dim=-1)
        self.drone.set_velocities(Server_vel, env_ids)
        
        ball_pos = self.ball_pos_dist.sample((*env_ids.shape, 1))
        ball_rot = torch.tensor(
            [1., 0., 0., 0.], device=self.device).repeat(len(env_ids), 1)
        ball_lin_vel = self.ball_vel_dist.sample((*env_ids.shape, 1))

        ball_lin_vel[..., 1] -= (ball_pos[..., 1] ) * 0.5

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
        
        self.turn[env_ids] = torch.ones(len(env_ids), device=self.device, dtype=torch.long) # always Server turn
        #self.attacking_target[env_ids] = torch.randint(0, 3, (len(env_ids),), device=self.device, dtype=torch.long)
        self.ball_anchor[env_ids, 0, 0] = torch.rand(len(env_ids), device=self.device) * 1 - 5.5  # x: [-6, -3]
        self.ball_anchor[env_ids, 0, 1] = torch.rand(len(env_ids), device=self.device) * 1 - 0.5  # y: [-3, 3]
        self.ball_anchor[env_ids, 0, 2] = 0  # z: 1


        self.stats[env_ids] = 0.
        self.Server_hit[env_ids] = 0
        self.Server_hit_t[env_ids] = -10
        self.Server_last_hit_t[env_ids] = -10
        self.ball_hit_ground[env_ids] = 0
        self.ball_already_hit_the_ground[env_ids] = 0
        self.switch_turn[env_ids] = 0
        self.in_side[env_ids] = 0
        self.highest_ball_pos_after_my_turn[env_ids] = 0
        
        self.last_drone_positions[env_ids] = drone_pos

        # Draw
        if (env_ids == self.central_env_idx).any() and self._should_render(0):
            self.ball_traj_vis.clear()
            self.draw.clear_lines()
            # self.debug_draw_region()

            logging.info("Reset central environment")

            point_list_1, point_list_2, colors, sizes = draw_court(
                self.W, self.L, self.H_NET, self.W_NET
            )
            point_list_1 = [
                _carb_float3_add(p, self.central_env_pos) for p in point_list_1
            ]
            point_list_2 = [
                _carb_float3_add(p, self.central_env_pos) for p in point_list_2
            ]
            self.draw.draw_lines(point_list_1, point_list_2, colors, sizes)

            self.draw.clear_points()
            # self.debug_draw_turn()
            # self.debug_draw_attacking_target()


    def _pre_sim_step(self, tensordict: TensorDictBase):
        
        Server_action = tensordict[("agents", "Server_action")]
        Server_hover_action = tensordict[("agents", "Server_hover_action")]
        Server_action = torch.where(
            tensordict["stats"]["Server_hit"].unsqueeze(-1).bool(),
            Server_hover_action,
            Server_action
        )
        self.effort = self.drone.apply_action(Server_action)



    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]
        
        self.ball_pos, ball_rot = self.get_env_poses(self.ball.get_world_poses())
        self.ball_vel = self.ball.get_velocities()

        # relative position and heading
        self.rpos = self.ball_pos - self.drone.pos # (E,1,3)
        self.ball_rpos = self.ball_pos - self.ball_anchor
        
        pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )
        rot = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot, rot) # make sure w of (w,x,y,z) is positive
        rpy = quaternion_to_euler(rot)
        
        self.drone_rot = rot.squeeze(1)
        self.drone_pos = pos.squeeze(1)
        self.drone_rpy = rpy.squeeze(1)
        self.drone_vel = vel.squeeze(1)
        self.drone_angular_vel = angular_vel.squeeze(1)
        self.roll = rpy[..., 0]
        self.pitch = rpy[..., 1]
        self.yaw = rpy[..., 2]
        
        self.info["roll"] = self.roll
        self.info["pitch"] = self.pitch
        self.info["yaw"] = self.yaw

        Server_obs = [
            pos,
            rot, # w of (w,x,y,z) is positive
            vel,
            angular_vel,
            heading,
            up,
            throttle,
            self.ball_pos, # [E, 1, 3]
            self.rpos, # [E, 1, 3]
            self.ball_vel[..., :3], # [E, 1, 3]
            turn_to_obs(self.turn), # [E, 1, 2]
            self.ball_rpos
        ] # obs_dim: root_state + rpos(3) + ball_pos(3) + ball_vel(3) + turn(2) + start_point(2)
        Server_obs = torch.cat(Server_obs, dim=-1)


        Server_hover_rpos = self.Server_hover_pos_after_hit - self.root_state[..., :3] # (E, 1, 3)
        Server_hover_rheading = self.hover_target_heading - self.root_state[..., 13:16] # (E, 1, 3)
        Server_hover_obs = [
            Server_hover_rpos, # (E, 1, 3)
            self.root_state[..., 3:], # (E, 1, root_state_dim-3)
            Server_hover_rheading, # (E, 1, 3)
        ]
        Server_hover_obs = torch.cat(Server_hover_obs, dim=-1)
        
        if self._should_render(0):
            central_env_pos = self.envs_positions[self.central_env_idx]
            ball_plot_pos = (
                self.ball_pos[self.central_env_idx]+central_env_pos).tolist()  # [2, 3]
            if len(self.ball_traj_vis) > 1:
                point_list_0 = self.ball_traj_vis[-1]
                point_list_1 = ball_plot_pos
                colors = [(.1, 1., .1, 1.)]
                sizes = [3.0]
                self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

            self.ball_traj_vis.append(ball_plot_pos)

            # if self.switch_turn[self.central_env_idx]:
            #     self.debug_draw_turn()

        return TensorDict(
            {
                "agents": {
                    "Server_observation": Server_obs,
                    "Server_hover_observation": Server_hover_obs,
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

        racket_hit_ball = racket_hit_ball & (self.progress_buf - self.Server_last_hit_t > 3) # (E,)

        drone_hit_ball = drone_hit_ball & (self.progress_buf - self.Server_last_hit_t > 3)

        return racket_hit_ball, drone_hit_ball
    

    def _compute_reward_and_done(self):

        # drone near ball
        drone_near_ball = (torch.norm(self.rpos, dim=-1) < 0.4) # (E,1)
        
        # racket near ball
        self.racket_r = 0.2
        ball_pos = self.ball_pos.squeeze(1) # (E,3)
        
        z_direction_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.z_direction_world = quat_rotate(self.drone_rot, z_direction_local) # (E,3)

        normal_vector_world = self.z_direction_world / torch.norm(self.z_direction_world, dim=-1).unsqueeze(-1)  # (E,3)

        self.drone_racket_center = self.drone_pos # (E,3) cylinder low center

        self.drone_cylinder_top_center = self.drone_racket_center + 2.0 * self.ball_radius * normal_vector_world # (E,3) cylinder top center
        cylinder_axis = self.drone_cylinder_top_center - self.drone_racket_center  # (E, 3)

        ball_to_bottom = ball_pos - self.drone_racket_center  # (E, 3)
        t = torch.sum(ball_to_bottom * cylinder_axis, dim=-1) / torch.sum(cylinder_axis * cylinder_axis, dim=-1)  # (E, ) projection of ball_to_bottom on cylinder_axis   
        within_height = (t >= 0) & (t <= 1)  # (E,)

        projection_point = self.drone_racket_center + t.unsqueeze(-1) * cylinder_axis  # (E, 3)
        distance_to_axis = torch.norm(ball_pos - projection_point, dim=-1)  # (E,)
        within_radius = distance_to_axis <= self.racket_r  # (E,)

        racket_near_ball = (within_height & within_radius).unsqueeze(-1)  # (E,)

        # racket hit ball
        ball_near_ground = (self.ball_pos[..., 2] < 0.2) # (E,1)
        
        ball_vel_z_change = ((self.ball_vel[..., 2] - self.ball_last_vel[..., 2]) > -0.196)
        hit_1 = racket_near_ball & ball_vel_z_change # (E,1) # 
        maybe_wrong_hit_1 = drone_near_ball & ball_vel_z_change
        hit_ground_1 = ball_near_ground & ball_vel_z_change # (E,1) # 

        ball_vel_x_y_change = (self.ball_vel[..., :2] - self.ball_init_vel[..., :2]).norm(dim=-1) > 0.5 # (E,1)
        hit_2 = racket_near_ball & ball_vel_x_y_change & ~self.Server_hit.bool() # (E,1) # 
        maybe_wrong_hit_2 = drone_near_ball & ball_vel_x_y_change & ~self.Server_hit.bool()
        hit_ground_2 = ball_near_ground & ball_vel_x_y_change

        hit = (hit_1 | hit_2) & ~hit_ground_1 & ~hit_ground_2 & ~self.ball_already_hit_the_ground.bool() # racket hit ball
        hit = hit & (self.progress_buf.unsqueeze(-1) - self.Server_last_hit_t > 3) # (E,1)
        maybe_wrong_hit = (maybe_wrong_hit_1 | maybe_wrong_hit_2) & ~hit_ground_1 & ~hit_ground_2 & ~self.ball_already_hit_the_ground.bool()
        maybe_wrong_hit = maybe_wrong_hit & (self.progress_buf.unsqueeze(-1) - self.Server_last_hit_t > 3) # (E,1)
        wrong_hit_1 = (hit == False) & (maybe_wrong_hit == True) # not racket hit ball but drone hit ball

        self.Server_last_hit_t = torch.where(hit, self.progress_buf.unsqueeze(-1), self.Server_last_hit_t)
        
        self.Server_hit = torch.where(hit, 1, self.Server_hit)
        
        self.ball_last_vel = self.ball_vel[..., :3].clone()

        # reward 1: hover
        rpos_hover = self.drone.pos - self.Server_hover_pos_after_hit
        rheading = self.drone.heading[:, 0, :].unsqueeze(1) - torch.tensor([1., 0., 0.], device=self.device).unsqueeze(0).unsqueeze(0)
        distance = torch.norm(torch.cat([rpos_hover, rheading], dim=-1), dim=-1)
        reward_pose = 1.0 / (1.0 + torch.square(1.2 * distance))
        reward_up = torch.square((self.drone.up[:, 0, 2] + 1) / 2).unsqueeze(-1)
        spinnage = torch.square(self.drone.vel[:, 0, -1]).unsqueeze(-1)
        reward_spin = 1.0 / (1.0 + torch.square(spinnage))
        reward_effort = 0.1 * torch.exp(-self.effort)
        reward_hover = 3 * (
            reward_pose 
            + reward_pose * (reward_up + reward_spin) 
        ) # (E,1)

        # End checking

        ball_hit_the_ground = (self.ball_pos[..., 2] < 0.2) # (E,1)
        self.ball_first_hit_the_ground = ball_hit_the_ground & ~self.ball_already_hit_the_ground.bool()
        self.ball_already_hit_the_ground = torch.where(ball_hit_the_ground, 1, self.ball_already_hit_the_ground)
        ball_hit_the_net = calculate_ball_hit_the_net(self.ball_pos, self.ball_radius, self.W, self.H_NET) # (E,1)

        wrong_hit_2 = (self.turn == 0).unsqueeze(-1) & hit

        terminated = (
            (self.drone.pos[..., 2] < 0.1).any(dim=1).unsqueeze(-1) 
            | (self.drone.pos[..., 2] > 4.0).any(dim=1).unsqueeze(-1) # z direction
            | (self.ball_pos[..., 2] > 5.) # z direction
            | wrong_hit_1 # not racket hit the ball
            | wrong_hit_2 # hover and hit
        )
        if self.done_ball_hit_ground:
            terminated = terminated | ball_hit_the_ground

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # (E,1)
        done = truncated | terminated

        # reward 3: End reward

        _penalty_end_drone_pos_z_coeff = 10
        penalty_end_drone_pos_z = _penalty_end_drone_pos_z_coeff * (self.drone.pos[..., 2] < 0.3)
        
        reward_end = - penalty_end_drone_pos_z# (E,1)

        reward_end = torch.where(
            done, 
            reward_end, 
            torch.zeros_like(done, device=self.device)
        ) # (E,1)

        # Overall reward
        Server_reward_1 = torch.zeros((self.num_envs, 1), device=self.device)
        Server_reward_2 =  reward_hover + reward_end # (E,1)

        Server_reward = torch.where(self.Server_hit.bool(), Server_reward_2, Server_reward_1)

        # change turn
        hit_index = hit.squeeze(-1)

        last_turn = self.turn.clone()
        self.turn[hit_index] = 0
        self.switch_turn = torch.where((self.turn ^ last_turn).bool(), 1, 0)

        self.Server_hit_t[hit_index] = self.progress_buf[hit_index].unsqueeze(-1)
        self.last_drone_positions[hit_index] = self.drone_pos[hit_index].unsqueeze(1)             

        ep_len = self.progress_buf.unsqueeze(-1)

        # log
        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["return"].add_(Server_reward)
        self.stats["episode_len"][:] = ep_len

        self.stats["truncated"].add_(truncated.float())
        self.stats["done"].add_(done.float())
        self.stats["terminated"].add_(terminated.float())
        self.stats["Server_hit"] = self.Server_hit
        self.stats["num_hits"].add_(hit.float())
        
        if self.stats_cfg.get("complete_reward_stats", False):
            self.stats["reward_hover"].add_(reward_hover)
            self.stats["reward_up"].add_(reward_up)
            self.stats["reward_pose"].add_(reward_pose)
            self.stats["reward_spin"].add_(reward_spin)
            
            self.stats["reward_end"].add_(reward_end)


        return TensorDict(
            {
                "agents": {
                    "Server_reward": Server_reward,
                },
                "done": done,
                "terminated": terminated,
                "truncated": truncated
            },
            self.num_envs,
        )