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
import os
import csv

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
from .common import rectangular_cuboid_edges,_carb_float3_add
from omni_drones.utils.torchrl.transforms import append_to_h5

from omni.isaac.orbit.sensors import ContactSensorCfg, ContactSensor

from omegaconf import DictConfig

from typing import Tuple, List

from .volley_env import VolleyEnv, draw_court

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

_COLOR_T = Tuple[float, float, float, float]

class Serve(VolleyEnv):
    def __init__(self, cfg, headless):

        self.add_reward_hover = cfg.task.get("add_reward_hover", False)
        
        super().__init__(cfg, headless)

        self.turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: hover; 1: my turn
        self.switch_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.attacking_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: left; 1: mid; 2:right

        self.Server_hit = torch.zeros((self.num_envs, 1), device=self.device)
        self.Server_hit_t = torch.zeros((self.num_envs, 1), device=self.device)
        self.Server_last_hit_t = torch.zeros((self.num_envs, 1), device=self.device)

        self.ball_hit_ground = torch.zeros((self.num_envs, 1), device=self.device)
        self.ball_already_hit_the_ground = torch.zeros((self.num_envs, 1), device=self.device)
        self.ball_anchor = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.ball_rpos = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.in_side = torch.zeros((self.num_envs, 1), device=self.device)
        # self.ball_anchor_target_0 = torch.tensor(cfg.task.ball_anchor_target_0, device=self.device)
        # self.ball_anchor_target_1 = torch.tensor(cfg.task.ball_anchor_target_1, device=self.device)
        # self.ball_anchor_target_2 = torch.tensor(cfg.task.ball_anchor_target_2, device=self.device)
        self.highest_ball_pos_after_my_turn = torch.zeros((self.num_envs, 1, 3), device=self.device)

        self.last_drone_positions = torch.zeros(size=(self.num_envs, 1, 3), device=self.device)

        self.hover_pos_after_hit = torch.tensor(cfg.task.hover_pos_after_hit, device=self.device)

        self.ball_last_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.ball_init_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)

        self.done_ball_hit_ground = cfg.task.get("done_ball_hit_ground", False)


    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 3

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim))
            })
        }).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": torch.stack([self.drone.action_spec] * 1, dim=0),
            })
        }).expand(self.num_envs).to(self.device)
        
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)
        
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )
        
        _stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
            "done": UnboundedContinuousTensorSpec(1),
            "hit": UnboundedContinuousTensorSpec(1),
            "num_hits": UnboundedContinuousTensorSpec(1),
        })
        self.stats_cfg: DictConfig = self.cfg.task.stats
        if self.stats_cfg.get("complete_reward_stats", False):
            if self.add_reward_hover:
                _stats_spec.set("reward_hover", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_my_turn", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_after_my_turn", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_end", UnboundedContinuousTensorSpec(1))

            _stats_spec.set("reward_rpos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_hit", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_pos_z", UnboundedContinuousTensorSpec(1))
            #_stats_spec.set("penalty_yaw", UnboundedContinuousTensorSpec(1))
            #_stats_spec.set("penalty_drone_x", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_roll", UnboundedContinuousTensorSpec(1))

            _stats_spec.set("reward_rpos_anchor", UnboundedContinuousTensorSpec(1))
            
            _stats_spec.set("reward_highest_rpos_anchor", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("in_side", UnboundedContinuousTensorSpec(1))
            #_stats_spec.set("penalty_end_drone_pos_z", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_end_wrong_hit", UnboundedContinuousTensorSpec(1))

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
        drone_pos = self.drone_pos_dist.sample((*env_ids.shape, 1))
        drone_rpy = self.drone_rpy_dist.sample((*env_ids.shape, 1))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos +
            self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(torch.zeros(
            len(env_ids), 6, device=self.device), env_ids)
        
        ball_pos = self.ball_pos_dist.sample((*env_ids.shape, 1))
        ball_rot = torch.tensor(
            [1., 0., 0., 0.], device=self.device).repeat(len(env_ids), 1)
        ball_lin_vel = self.ball_vel_dist.sample((*env_ids.shape, 1))

        ball_lin_vel[..., 1] -= ball_pos[..., 1] * 0.5
        
        ball_ang_vel = torch.zeros_like(ball_lin_vel)
        ball_vel = torch.cat((ball_lin_vel, ball_ang_vel), dim=-1)
        self.ball.set_world_poses(
            ball_pos +
            self.envs_positions[env_ids].unsqueeze(1), ball_rot, env_ids
        )
        self.ball.set_velocities(ball_vel, env_ids)
        self.ball.set_masses(torch.ones_like(env_ids,dtype=torch.float)*self.ball_mass, env_ids)
        self.ball_last_vel[env_ids] = ball_lin_vel
        self.ball_init_vel[env_ids] = ball_lin_vel
        
        self.turn[env_ids] = torch.ones(len(env_ids), device=self.device, dtype=torch.long) # always Server turn
        #self.attacking_target[env_ids] = torch.randint(0, 3, (len(env_ids),), device=self.device, dtype=torch.long)
        self.ball_anchor[env_ids, 0, 0] = torch.rand(len(env_ids), device=self.device) * 1 - 5.5  
        self.ball_anchor[env_ids, 0, 1] = torch.rand(len(env_ids), device=self.device) * 1 - 0.5  
        self.ball_anchor[env_ids, 0, 2] = 0  # z: 1

        self.stats[env_ids] = 0.
        self.Server_hit[env_ids] = 0
        self.Server_hit_t[env_ids] = -10
        self.Server_last_hit_t[env_ids] = -10
        self.ball_hit_ground[env_ids] = 0
        self.ball_already_hit_the_ground[env_ids] = 0
        self.in_side[env_ids] = 0
        self.switch_turn[env_ids] = 0
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

        # debug
        #print("Ball Anchor (self.ball_anchor) shape:", self.ball_anchor.shape)

        obs = [
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
            #attacking_target_to_obs_3(self.attacking_target) # [E, 1, 3]
        ] # obs_dim: root_state + rpos(3) + ball_pos(3) + ball_vel(3) + turn(2) + start_point(2)
        

        obs = torch.cat(obs, dim=-1)
        
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
                    "observation": obs,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.num_envs,
        )


    def _compute_reward_and_done(self):

        # reward 1: Turn=0(after hit): hover
        rpos_last = self.drone_pos.unsqueeze(1) - self.hover_pos_after_hit
        rheading = self.drone.heading - torch.tensor([1., 0., 0.], device=self.device).unsqueeze(0)
        distance = torch.norm(torch.cat([rpos_last, rheading], dim=-1), dim=-1)
        reward_pose = 1.0 / (1.0 + torch.square(1.2 * distance))
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)
        spinnage = torch.square(self.drone.vel[..., -1])
        reward_spin = 1.0 / (1.0 + torch.square(spinnage))
        reward_effort = 0.1 * torch.exp(-self.effort)

        if self.add_reward_hover:
            reward_hover = 3 * (
                reward_pose 
                + reward_pose * (reward_up + reward_spin) 
            ) # (E,1)
            reward_hover = self.Server_hit * (self.turn == 0).unsqueeze(-1) * reward_hover # (E,1)


        # reward 2: Turn=1: my turn

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
        
        _reward_hit_coeff = 10
        reward_hit = (self.turn == 1).unsqueeze(-1) * _reward_hit_coeff * torch.where(hit, 1.0, 0.0)

        _reward_rpos_coeff = 0.16
        reward_rpos = (self.turn == 1).unsqueeze(-1) * _reward_rpos_coeff / (1.0 + torch.square(0.5 * torch.norm(self.rpos, dim=-1)))
        
        delta_z = (self.drone.pos[..., 2] - 1.9).abs() # (E,1)
        _penalty_pos_z_coeff = 0.02
        penalty_pos_z = (self.turn == 1).unsqueeze(-1) * _penalty_pos_z_coeff * delta_z

        _penalty_yaw_coeff = 0.03
        penalty_yaw = (self.turn == 1).unsqueeze(-1) * _penalty_yaw_coeff * self.yaw.abs()

        roll_angle = self.roll  # (E, 1)
        _penalty_roll_coeff = 0.5
        penalty_roll = (self.turn == 1).unsqueeze(-1) * _penalty_roll_coeff * torch.where(roll_angle.abs() > 1.57, roll_angle.abs() - 1.57, torch.tensor(0.0))

        drone_x = self.drone.pos[..., 0]
        _penalty_drone_x_coeff = 150
        penalty_drone_x = (self.turn == 1).unsqueeze(-1) * _penalty_drone_x_coeff * torch.where((drone_x < 3), 1.0, 0.0)
        
        reward_my_turn = (1 - self.ball_already_hit_the_ground) * (reward_rpos + reward_hit - penalty_pos_z - penalty_yaw - penalty_drone_x - penalty_roll) # (E,1)
        
 
        # reward 3: Reward after my turn
        self.highest_ball_pos_after_my_turn = torch.where(
            (self.Server_hit.bool() & (self.ball_pos[..., 2] > self.highest_ball_pos_after_my_turn[..., 2])).unsqueeze(-1).expand(-1, -1, 3),
            self.ball_pos,
            self.highest_ball_pos_after_my_turn
        )
        #reward_rpos: hit the ball to the target point
        rpos_anchor = self.ball_pos - self.ball_anchor
        _reward_rpos_anchor_coeff = 16.0
        reward_rpos_anchor = self.Server_hit * _reward_rpos_anchor_coeff / (1.0 + torch.square(1.2 * torch.norm(rpos_anchor, dim=-1)))
        
        reward_after_my_turn = (1 - self.ball_already_hit_the_ground) * reward_rpos_anchor # (E,1)


        # reward 4: End reward

        ball_hit_the_ground = (self.ball_pos[..., 2] < 0.2) # (E,1)
        self.ball_already_hit_the_ground = torch.where(
            ball_hit_the_ground, 
            torch.ones_like(ball_hit_the_ground, device=self.device), 
            self.ball_already_hit_the_ground
        )

        temp_in_side = (
            (self.ball_pos[..., 0] > -self.L/2) & (self.ball_pos[..., 0] < 0) &
            (self.ball_pos[..., 1] > -self.W/2) & (self.ball_pos[..., 1] < self.W/2) 
        )

        self.in_side = torch.where(
            self.ball_already_hit_the_ground.bool(),
            self.in_side,
            temp_in_side
        )

        wrong_hit_2 = (self.turn == 0).unsqueeze(-1) & hit

        terminated = (
            (self.drone.pos[..., 2] < 0.3) | (self.drone.pos[..., 2] > 3.7) # z direction
            | (self.ball_pos[..., 2] > 10.) # z direction
            | wrong_hit_1 # not racket hit the ball
            | wrong_hit_2 # hover and hit
        )
        if self.done_ball_hit_ground:
            terminated = terminated | ball_hit_the_ground
        
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # (E,1)
        done = truncated | terminated # (E,1)

        reward_inside = 10 * self.in_side

        _reward_highest_rpos_anchor_coeff = 1.5
        reward_highest_rpos_anchor = done * self.Server_hit * _reward_highest_rpos_anchor_coeff * torch.where((self.highest_ball_pos_after_my_turn[..., 2] > 3), 1.0, 0.0)

        _penalty_end_drone_pos_z_coeff = 0.1
        penalty_end_drone_pos_z = _penalty_end_drone_pos_z_coeff * (self.drone.pos[..., 2] < 0.3)

        _penalty_end_wrong_hit_coeff = 10.0
        penalty_end_wrong_hit = _penalty_end_wrong_hit_coeff * (wrong_hit_1 | wrong_hit_2)
        
        reward_end = reward_inside + reward_highest_rpos_anchor - penalty_end_drone_pos_z - penalty_end_wrong_hit # (E,1)
        reward_end = torch.where(
            done, 
            reward_end, 
            torch.zeros_like(done, device=self.device)
        ) # (E,1)

        # Overall reward
        reward = reward_my_turn + reward_after_my_turn + reward_end # (E,1)
        if self.add_reward_hover:
            reward += reward_hover

        # change turn
        hit_index = hit.squeeze(-1)

        last_turn = self.turn.clone()
        self.turn[hit_index] = 0
        self.switch_turn = torch.where((self.turn ^ last_turn).bool(), 1, 0)

        self.Server_hit_t[hit_index] = self.progress_buf[hit_index].unsqueeze(-1)
        self.last_drone_positions[hit_index] = self.drone_pos[hit_index].unsqueeze(1)             

        ep_len = self.progress_buf.unsqueeze(-1)

        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = ep_len
        self.stats["truncated"].add_(truncated.float())
        self.stats["done"].add_(done.float())
        self.stats["hit"] = self.Server_hit
        self.stats["num_hits"].add_(hit.float())

        if self.stats_cfg.get("complete_reward_stats", False):
            if self.add_reward_hover:
                self.stats["reward_hover"].add_(reward_hover)
            self.stats["reward_my_turn"].add_(reward_my_turn)
            self.stats["reward_after_my_turn"].add_(reward_after_my_turn)
            self.stats["reward_end"].add_(reward_end)

            self.stats["reward_rpos"].add_(reward_rpos)
            self.stats["reward_hit"].add_(reward_hit)
            self.stats["penalty_pos_z"].add_(penalty_pos_z)
            #self.stats["penalty_yaw"].add_(penalty_yaw)
            #self.stats["penalty_drone_x"].add_(penalty_drone_x)
            self.stats["penalty_roll"].add_(penalty_roll)

            self.stats["reward_rpos_anchor"].add_(reward_rpos_anchor)
            
            self.stats["reward_highest_rpos_anchor"].add_(reward_highest_rpos_anchor)
            #self.stats["penalty_end_drone_pos_z"].add_(penalty_end_drone_pos_z)
            self.stats["penalty_end_wrong_hit"].add_(penalty_end_wrong_hit)
            self.stats["in_side"] = self.in_side

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": done,
                "terminated": terminated,
                "truncated": truncated
            },
            self.num_envs,
        )
