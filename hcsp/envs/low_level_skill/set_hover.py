from typing import Optional
import functorch

import omni.isaac.core.utils.torch as torch_utils
import hcsp.utils.kit as kit_utils
from hcsp.utils.torch import euler_to_quaternion, quat_axis, quaternion_to_euler
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.objects as objects
import omni.isaac.core.materials as materials
import torch
import torch.distributions as D

from hcsp.envs.isaac_env import AgentSpec, IsaacEnv
from hcsp.robots.drone import MultirotorBase
from hcsp.views import RigidPrimView
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
)
from pxr import UsdShade, PhysxSchema
import logging
from carb import Float3
from omni.isaac.debug_draw import _debug_draw
from hcsp.utils.volleyball.common import rectangular_cuboid_edges,_carb_float3_add, draw_court, calculate_ball_hit_the_net, turn_to_obs, target_to_obs, attacking_target_to_obs
from hcsp.utils.torchrl.transforms import append_to_h5
from omni.isaac.orbit.sensors import ContactSensorCfg, ContactSensor
from omegaconf import DictConfig
from typing import Tuple, List
from abc import abstractmethod
import os


class Set_hover(IsaacEnv):
    """
    Set drone: 
        first load a trained Set policy checkpoint
        then train a hover policy
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

        # contact sensor from Orbit
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
        
        self.ball_traj_vis = []
        self.draw = _debug_draw.acquire_debug_draw_interface()

        self.switch_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.ball_init_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.ball_last_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)
        
        self.SecPass_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: hover; 1: my turn
        self.SecPass_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: left; 1: right
        self.SecPass_hit = torch.zeros((self.num_envs, 1), device=self.device)
        self.SecPass_last_hit_t = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        hover_target_rpy = torch.zeros((self.num_envs, 1, 3), device=self.device)
        hover_target_rot = euler_to_quaternion(hover_target_rpy)
        self.hover_target_heading = quat_axis(hover_target_rot.squeeze(1), 0).unsqueeze(1)
        self.SecPass_hover_pos_after_hit = torch.tensor(cfg.task.get("SecPass_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)


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
        drone_state_dim = self.drone.state_spec.shape[-1]  # 23

        SecPass_observation_dim = drone_state_dim + 3 + 3 + 3 + 2
        SecPass_hover_observation_dim = drone_state_dim + 3

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "SecPass_observation": UnboundedContinuousTensorSpec((1, SecPass_observation_dim)),
                "SecPass_hover_observation": UnboundedContinuousTensorSpec((1, SecPass_hover_observation_dim)),
            })
        }).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "SecPass_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "SecPass_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),
            })
        }).expand(self.num_envs).to(self.device)

        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "SecPass_reward": UnboundedContinuousTensorSpec((1, 1)),
            })
        }).expand(self.num_envs).to(self.device)

        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)

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
            state_key=("agents", "SecPass_state"),
        )

        _stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
            "terminated": UnboundedContinuousTensorSpec(1),
            "SecPass_hit": UnboundedContinuousTensorSpec(1),
            "ball_hit_the_ground_vel_x": UnboundedContinuousTensorSpec(1),
            "ball_hit_the_ground_vel_y": UnboundedContinuousTensorSpec(1),
            "ball_hit_the_ground_vel_z": UnboundedContinuousTensorSpec(1),
            "ball_hit_the_ground_vel": UnboundedContinuousTensorSpec(1),
            "ball_hit_the_ground": UnboundedContinuousTensorSpec(1),
            "ball_hit_the_net": UnboundedContinuousTensorSpec(1),
            "in_side": UnboundedContinuousTensorSpec(1),
        })

        stats_spec = _stats_spec.expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["info"] = info_spec
        self.info = info_spec.zero()


    def debug_draw_turn(self):

        SecPass_turn = self.SecPass_turn[self.central_env_idx]
        
        ori = self.envs_positions[self.central_env_idx].detach()
        
        points = torch.tensor([4., 0., 0.]).to(self.device) + ori
        points = [points.tolist()]

        if SecPass_turn.item() == 1:
            colors = [(0, 1, 0, 1)]
            logging.info("Central env turn: SecPass turn")
        elif SecPass_turn.item() == 0:
            colors = [(0, 0, 1, 1)]
            logging.info("Central env turn: SecPass Hover turn")
        sizes = [15.]
        self.draw.draw_points(points, colors, sizes)
    

    def _reset_idx(self, env_ids: torch.Tensor):

        self.SecPass_turn[env_ids] = torch.ones(len(env_ids), device=self.device).long()

        self.drone._reset_idx(env_ids, self.training)

        SecPass_pos = self.SecPass_pos_dist.sample(env_ids.shape).unsqueeze(1)
        SecPass_rpy = self.SecPass_rpy_dist.sample(env_ids.shape).unsqueeze(1)

        drone_pos = SecPass_pos
        drone_rpy = SecPass_rpy
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos +
            self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )

        SecPass_lin_vel = self.SecPass_lin_vel_dist.sample((*env_ids.shape, 1))
        SecPass_ang_vel = self.SecPass_ang_vel_dist.sample((*env_ids.shape, 1))
        SecPass_vel = torch.cat((SecPass_lin_vel, SecPass_ang_vel), dim=-1)
        self.drone.set_velocities(SecPass_vel, env_ids)

        ball_pos = self.ball_pos_dist.sample((*env_ids.shape, 1))
        ball_rot = torch.tensor(
            [1., 0., 0., 0.], device=self.device).repeat(len(env_ids), 1)
        ball_lin_vel = self.ball_vel_dist.sample((*env_ids.shape, 1))

        ball_lin_vel[..., 1] -= (ball_pos[..., 1] - (-1.5)) * 0.5

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

        self.stats[env_ids] = 0.
        self.SecPass_hit[env_ids] = 0
        self.switch_turn[env_ids] = 0
        self.SecPass_last_hit_t[env_ids] = -10

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


    def _pre_sim_step(self, tensordict: TensorDictBase):
        SecPass_action = tensordict[("agents", "SecPass_action")]
        SecPass_hover_action = tensordict[("agents", "SecPass_hover_action")]
        SecPass_real_action = torch.where(tensordict["stats"]["SecPass_hit"].unsqueeze(-1).bool(), SecPass_hover_action, SecPass_action)

        self.effort = self.drone.apply_action(SecPass_real_action)


    def _post_sim_step(self, tensordict: TensorDictBase):
        # self.contact_sensor.update(self.dt)
        pass


    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state() # (E, 1, 23)
        self.info["drone_state"][:] = self.root_state[..., :13]
        self.ball_pos, _ = self.get_env_poses(self.ball.get_world_poses()) # (E, 1, 3)
        self.ball_vel = self.ball.get_velocities()[..., :3] # (E, 1, 3)

        self.rpos = self.ball_pos - self.drone.pos # (E, 1, 3)

        pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )
        rot = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot, rot) # make sure w of (w,x,y,z) is positive
        rpy = quaternion_to_euler(rot)

        SecPass_obs = [
            pos,
            rot, # w of (w,x,y,z) is positive
            vel,
            angular_vel,
            heading,
            up,
            throttle,
            self.ball_pos,
            self.rpos,
            self.ball_vel,
            turn_to_obs(self.SecPass_turn), # (E, 1, 2)
        ]
        SecPass_obs = torch.cat(SecPass_obs, dim=-1)

        SecPass_hover_rpos = self.SecPass_hover_pos_after_hit - self.root_state[:, 0, :3].unsqueeze(1) # (E, 1, 3)
        SecPass_hover_rheading = self.hover_target_heading - self.root_state[:, 0, 13:16].unsqueeze(1) # (E, 1, 3)
        SecPass_hover_obs = [
            SecPass_hover_rpos, # (E, 1, 23)
            self.root_state[:, 0, 3:].unsqueeze(1), # (E, 1, root_state_dim-3)
            SecPass_hover_rheading, # (E, 1, 3)
        ]
        SecPass_hover_obs = torch.cat(SecPass_hover_obs, dim=-1)

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
        
        return TensorDict(
            {
                "agents": {
                    "SecPass_observation": SecPass_obs,
                    "SecPass_hover_observation": SecPass_hover_obs,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.num_envs,
        )
            

    def _compute_reward_and_done(self):

        rpos_hover = self.drone.pos - self.SecPass_hover_pos_after_hit
        rheading = self.drone.heading[:, 0, :].unsqueeze(1) - torch.tensor([1., 0., 0.], device=self.device).unsqueeze(0).unsqueeze(0)
        distance = torch.norm(torch.cat([rpos_hover, rheading], dim=-1), dim=-1)
        reward_pose = 1.0 / (1.0 + torch.square(1.2 * distance))
        reward_up = torch.square((self.drone.up[:, 0, 2] + 1) / 2).unsqueeze(-1)
        spinnage = torch.square(self.drone.vel[:, 0, -1]).unsqueeze(-1)
        reward_spin = 1.0 / (1.0 + torch.square(spinnage))
        # reward_effort = 0.1 * torch.exp(-self.effort)
        reward_hover = 3 * (
            reward_pose 
            + reward_pose * (reward_up + reward_spin) 
            # + reward_effort 
        ) # (E,1)
        SecPass_reward_1 = torch.zeros((self.num_envs, 1), device=self.device) # already trained SecPass policy
        SecPass_reward_2 = reward_hover # hover policy to be trained
        SecPass_reward = torch.where(self.SecPass_hit.bool(), SecPass_reward_2, SecPass_reward_1)

        drone_near_ball = ((self.drone.pos - self.ball_pos).norm(dim=-1) < 0.5) # (E, 1)
        ball_vel_z_change = ((self.ball_vel[..., 2] - self.ball_last_vel[..., 2]) > -0.196) # (E, 1)
        hit_1 = drone_near_ball & ball_vel_z_change
        ball_vel_x_y_change = ((self.ball_vel[..., :2] - self.ball_init_vel[..., :2]).norm(dim=-1) > 0.5) # (E, 1)
        hit_2 = drone_near_ball & ball_vel_x_y_change & ~self.SecPass_hit.bool()
        hit = hit_1 | hit_2 # (E, 1)

        hit[:, 0] = hit[:, 0] & (self.progress_buf - self.SecPass_last_hit_t > 3) # (E,)
        self.SecPass_last_hit_t = torch.where(hit[:, 0], self.progress_buf, self.SecPass_last_hit_t)

        self.SecPass_hit = torch.where(hit[:, 0].unsqueeze(-1) == 1, 1, self.SecPass_hit)
        self.ball_last_vel = self.ball_vel

        ball_hit_the_ground = (self.ball_pos[..., 2] < 0.2) # (E,1)
        ball_hit_the_net = calculate_ball_hit_the_net(self.ball_pos, self.ball_radius, self.W, self.H_NET) # (E,1)

        terminated = (
            (self.drone.pos[..., 2] < 0.3).any(dim=1).unsqueeze(-1) | (self.drone.pos[..., 2] > 3.7).any(dim=1).unsqueeze(-1) # z direction
            | (self.ball_pos[..., 2] > 5.) # z direction
            | ((self.SecPass_turn == 0) & hit[:, 0]).unsqueeze(-1) # hover and hit
            | ball_hit_the_net
            # | ball_hit_the_ground            
        )

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # (E,1)
        done = truncated | terminated

        # change turn
        SecPass_hit = hit[:, 0] # (E,)
        last_turn = self.SecPass_turn.clone() # (E,)
        self.SecPass_turn = torch.where((self.SecPass_turn == 1) & SecPass_hit, 0, self.SecPass_turn) # (E,)
        self.switch_turn = torch.where((self.SecPass_turn ^ last_turn).bool(), 1, 0) # (E,)

        # log
        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["return"].add_(SecPass_reward)
        self.stats["episode_len"][:] = ep_len
        self.stats["truncated"].add_(truncated.float())
        self.stats["terminated"].add_(terminated.float())
        self.stats["SecPass_hit"] = self.SecPass_hit

        return TensorDict(
            {
                "agents": {
                    "SecPass_reward": SecPass_reward,
                },
                "done": done,
                "terminated": terminated,
                "truncated": truncated
            },
            self.num_envs,
        )