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
from omni_drones.utils.volleyball.common import rectangular_cuboid_edges,_carb_float3_add
from omni_drones.utils.torchrl.transforms import append_to_h5

from omni.isaac.orbit.sensors import ContactSensorCfg, ContactSensor

from omegaconf import DictConfig

from typing import Tuple, List

from omni_drones.envs.low_level_skill.volley_env import VolleyEnv, draw_court

_COLOR_T = Tuple[float, float, float, float]

def turn_to_obs(t: torch.Tensor) -> torch.Tensor:
    """convert representation of drone turn to one-hot vector

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env, 1, 2)
    """
    table = torch.tensor(
        [
            [
                [0.0, 1.0], # hover
            ],
            [
                [1.0, 0.0], # my turn
            ]
        ],
        device=t.device,
    )
    if t.dtype != torch.long:
        t = t.long()

    return table[t]


def target_to_obs(t: torch.Tensor) -> torch.Tensor:
    """convert representation of ball target to one-hot vector

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env, 1, 2)
    """
    table = torch.tensor(
        [
            [
                [0.0, 1.0], # left
            ],
            [
                [1.0, 0.0], # right
            ]
        ],
        device=t.device,
    )
    if t.dtype != torch.long:
        t = t.long()

    return table[t]


class Set(VolleyEnv):
    def __init__(self, cfg, headless):

        self.add_reward_hover = cfg.task.get("add_reward_hover", False)
        
        super().__init__(cfg, headless)

        self.turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: hover; 1: my turn
        self.init_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: hover; 1: my turn
        self.target = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: left; 1: right

        self.ball_anchor_0 = torch.tensor(cfg.task.ball_anchor_0, device=self.device)
        self.ball_anchor_1 = torch.tensor(cfg.task.ball_anchor_1, device=self.device)
        self.ball_anchor = torch.zeros((self.num_envs, 1, 3), device=self.device)

        self.second_pass_hit = torch.zeros((self.num_envs, 1), device=self.device)
        self.second_pass_hit_t = torch.zeros((self.num_envs, 1), device=self.device)
        self.highest_ball_pos_after_my_turn = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.switch_turn = torch.zeros((self.num_envs, 1), device=self.device)
        self.last_drone_positions = torch.zeros(size=(self.num_envs, 1, 3), device=self.device)
        self.ball_already_hit_the_ground = torch.zeros((self.num_envs, 1), device=self.device)
        self.flag_reward_attacker_hit_pos = torch.zeros((self.num_envs, 1), device=self.device)

        self.ball_last_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.ball_init_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)

        self.drone_model_name = cfg.task.drone_model
        assert self.drone_model_name in ["Iris"]

        self.done_ball_hit_ground = cfg.task.get("done_ball_hit_ground", False)
        self.alpha_coeffient = cfg.task.get("alpha_coeffient", False)
        self.drone_hover_pos_after_hit = torch.tensor(cfg.task.get("drone_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)

        self.save_ball_state = cfg.task.get("save_ball_state", False)
        if self.save_ball_state:
            self.csv_path = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, "scripts", "shell", "ball_state", "secpass_ball_state.csv")
            with open(self.csv_path, 'w', newline='') as f:
                pass
            self.ball_state_idx = 0


    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3 + 3 + 3 + 2

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
            "hit_pos_x": UnboundedContinuousTensorSpec(1),
            "hit_pos_y": UnboundedContinuousTensorSpec(1),
            "hit_pos_z": UnboundedContinuousTensorSpec(1),
            "hit_ball_vel_x": UnboundedContinuousTensorSpec(1),
            "hit_ball_vel_y": UnboundedContinuousTensorSpec(1),
            "hit_ball_vel_z": UnboundedContinuousTensorSpec(1),
            "highest_ball_pos_x": UnboundedContinuousTensorSpec(1),
            "highest_ball_pos_y": UnboundedContinuousTensorSpec(1),
            "highest_ball_pos_z": UnboundedContinuousTensorSpec(1),
        })
        self.stats_cfg: DictConfig = self.cfg.task.stats
        if self.stats_cfg.get("complete_reward_stats", False):
            if self.add_reward_hover:
                _stats_spec.set("reward_hover_1", UnboundedContinuousTensorSpec(1))
                _stats_spec.set("reward_hover_2", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_my_turn", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_after_my_turn", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_end", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_hit", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_hit_pos_z", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_hit_pos_xy", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_rpos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_pos_z", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_yaw", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_attacker_hit_pos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_highest_pos_z", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_end_drone_pos_z", UnboundedContinuousTensorSpec(1))
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
        
        points = torch.tensor([1.5, -1.5, 0.]).to(self.device) + ori
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
            logging.info("Central env turn: Second_pass turn")


    def debug_draw_target(self):

        target = self.target[self.central_env_idx]
        ori = self.envs_positions[self.central_env_idx].detach()
        
        if target.item() == 0:
            points = self.ball_anchor_0.clone()
            points[..., 2] = 0.
            points += ori
        else:
            points = self.ball_anchor_1.clone()
            points[..., 2] = 0.
            points += ori
        points = [points.tolist()]

        colors = [(1, 1, 0, 1)]

        sizes = [15.]
        self.draw.draw_points(points, colors, sizes)
        
        logging.info(f"Central env target: {'Left' if target.item() == 0 else 'Right'}")


    def _reset_idx(self, env_ids: torch.Tensor):
        
        self.drone._reset_idx(env_ids, self.training)
        drone_pos = self.drone_pos_dist.sample((*env_ids.shape, 1))
        drone_rpy = self.drone_rpy_dist.sample((*env_ids.shape, 1))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos +
            self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        drone_lin_vel = self.drone_lin_vel_dist.sample((*env_ids.shape, 1))
        drone_ang_vel = self.drone_ang_vel_dist.sample((*env_ids.shape, 1))
        drone_vel = torch.cat((drone_lin_vel, drone_ang_vel), dim=-1)
        self.drone.set_velocities(drone_vel, env_ids)
        
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
        self.ball_last_vel[env_ids] = ball_lin_vel
        self.ball_init_vel[env_ids] = ball_lin_vel

        self.last_drone_positions[env_ids] = drone_pos

        # self.turn[env_ids] = torch.bernoulli(0.8 * torch.ones(len(env_ids), device=self.device)).long() # 0: hover; 1: my turn; 80% probability to be my turn
        # self.turn[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device)
        # self.turn[env_ids] = torch.zeros(len(env_ids), device=self.device) # always hover
        self.turn[env_ids] = torch.ones(len(env_ids), device=self.device).long() # always my turn
        self.init_turn[env_ids] = self.turn[env_ids]
        self.target[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device)
        
        self.stats[env_ids] = 0.
        self.second_pass_hit[env_ids] = 0
        self.second_pass_hit_t[env_ids] = 0
        self.ball_already_hit_the_ground[env_ids] = 0
        self.highest_ball_pos_after_my_turn[env_ids] = torch.zeros((len(env_ids), 1, 3), device=self.device)
        self.switch_turn[env_ids] = 0
        self.flag_reward_attacker_hit_pos[env_ids] = 0

        self.ball_anchor[env_ids] = torch.where(self.target[env_ids].unsqueeze(1).unsqueeze(2).expand(-1, 1, 3).bool(), self.ball_anchor_1, self.ball_anchor_0)

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
            self.debug_draw_turn()
            self.debug_draw_target()


    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]
        
        self.ball_pos, ball_rot = self.get_env_poses(self.ball.get_world_poses())
        self.ball_vel = self.ball.get_velocities()

        # relative position and heading
        self.rpos = self.ball_pos - self.drone.pos # (E,1,3)
        
        pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )
        rot = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot, rot) # make sure w of (w,x,y,z) is positive
        rpy = quaternion_to_euler(rot)
        
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
            # target_to_obs(self.target) # [E, 1, 2]
        ] # obs_dim: root_state + rpos(3) + ball_pos(3) + ball_vel(3) + turn(2) + target(2)
        
        obs = torch.cat(obs, dim=-1)
        
        if self._should_render(0):
            central_env_pos = self.envs_positions[self.central_env_idx]
            ball_plot_pos = (
                self.ball_pos[self.central_env_idx]+central_env_pos).tolist()  # [2, 3]
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
                    "observation": obs,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.num_envs,
        )


    def _compute_reward_and_done(self):

        # reward 1: Turn=0: hover
        
        rpos_hover = self.drone_pos.unsqueeze(1) - self.drone_hover_pos_after_hit
        rheading = self.drone.heading - torch.tensor([1., 0., 0.], device=self.device).unsqueeze(0)
        distance = torch.norm(torch.cat([rpos_hover, rheading], dim=-1), dim=-1)
        reward_pose = 2.0 / (1.0 + torch.square(1.2 * distance))
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)
        spinnage = torch.square(self.drone.vel[..., -1])
        reward_spin = 1.0 / (1.0 + torch.square(spinnage))
        reward_effort = 0.02 * torch.exp(-self.effort)

        delta_t = self.progress_buf.unsqueeze(-1) - self.second_pass_hit_t
        alpha = self.alpha_coeffient * delta_t / (self.max_episode_length - self.second_pass_hit_t) 
        
        if self.add_reward_hover:
            reward_hover = alpha * (self.turn == 0).unsqueeze(-1) * (
                reward_pose 
                + reward_pose * (reward_up + reward_spin) 
                + reward_effort 
            ) # (E,1)

            # hover_1: after hit the ball before the ball hit the ground
            reward_hover_1 = self.second_pass_hit * (1 - self.ball_already_hit_the_ground) * reward_hover # (E,1)

            # hover_2: after the ball hit the ground and hit in_side
            reward_hover_2 = self.second_pass_hit * self.ball_already_hit_the_ground * reward_hover # (E,1)
        

        # reward 2: Turn=1: my turn
        ball_near_ground = (self.ball_pos[..., 2] < 0.2) # (E,1)
        drone_near_ball = ((self.drone_pos.unsqueeze(1) - self.ball_pos).norm(dim=-1) < 0.5) # (E,1)
        ball_vel_z_change = ((self.ball_vel[..., 2] - self.ball_last_vel[..., 2]) > -0.196)
        # ball_vel_z_change = (self.ball_vel[..., 2] > 0) & (self.ball_last_vel[..., 2] <= 0) # (E,1)
        hit_1 = drone_near_ball & ball_vel_z_change # (E,1) 
        hit_ground_1 = ball_near_ground & ball_vel_z_change # (E,1) 
        ball_vel_x_y_change = (self.ball_vel[..., :2] - self.ball_init_vel[..., :2]).norm(dim=-1) > 0.5 # (E,1)
        hit_2 = drone_near_ball & ball_vel_x_y_change & ~self.second_pass_hit.bool() # (E,1) 
        hit_ground_2 = ball_near_ground & ball_vel_x_y_change
        hit = (hit_1 | hit_2) & ~hit_ground_1 & ~hit_ground_2 & ~self.ball_already_hit_the_ground.bool() # drone hit ball
        
        if hit.any().item() & self.save_ball_state:
            hit_index = hit.squeeze(-1)
            state_ball_pos = self.ball_pos[hit_index, 0, :]
            state_ball_lin_vel = self.ball_vel[hit_index, 0, :3]
            state_ball = torch.cat((state_ball_pos, state_ball_lin_vel), dim=-1)
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(state_ball.cpu().numpy())

        self.stats["hit_pos_x"].add_(hit * (1 - self.second_pass_hit) * self.ball_pos[..., 0])
        self.stats["hit_pos_y"].add_(hit * (1 - self.second_pass_hit) * self.ball_pos[..., 1])
        self.stats["hit_pos_z"].add_(hit * (1 - self.second_pass_hit) * self.ball_pos[..., 2])

        self.second_pass_hit = torch.where(hit, 1, self.second_pass_hit)
        self.ball_last_vel = self.ball_vel[..., :3].clone()

        reward_hit = (1 - self.ball_already_hit_the_ground) * (self.turn == 1).unsqueeze(-1) * 40 * hit.float() # (E,1) 

        penalty_hit_pos_z = (1 - self.ball_already_hit_the_ground) * (self.turn == 1).unsqueeze(-1) * 40 * (self.drone.pos[..., 2] - 2.2).abs() * hit.float() # (E,1)
        penalty_hit_pos_xy = (1 - self.ball_already_hit_the_ground) * (self.turn == 1).unsqueeze(-1) * 20 * (self.drone.pos[..., :2] - torch.tensor([1.5, -1.5], device=self.device)).norm(dim=-1) * hit.float() # (E,1)

        reward_rpos = (1 - self.ball_already_hit_the_ground) * (self.turn == 1).unsqueeze(-1) * 2.0 / (1.0 + torch.square(1.2 * torch.norm(self.rpos, dim=-1)))
        
        delta_z = (1 - self.ball_already_hit_the_ground) * (self.drone.pos[..., 2] - 1.8).abs() # (E,1)
        penalty_pos_z = (1 - self.ball_already_hit_the_ground) * (self.turn == 1).unsqueeze(-1) * 0.02 * delta_z
        
        penalty_yaw = (1 - self.ball_already_hit_the_ground) * (self.turn == 1).unsqueeze(-1) * 2 * self.yaw.abs()
        
        reward_my_turn = reward_rpos + reward_hit - penalty_hit_pos_z - penalty_hit_pos_xy - penalty_pos_z - penalty_yaw # (E,1)
        

        # reward 3: Reward after my turn
        self.highest_ball_pos_after_my_turn = torch.where(
            (self.second_pass_hit.bool() & (self.rpos[..., 2] > self.highest_ball_pos_after_my_turn[..., 2])).unsqueeze(-1).expand(-1, -1, 3),
            self.ball_pos,
            self.highest_ball_pos_after_my_turn
        )

        rpos_anchor = self.ball_pos - self.ball_anchor
        reward_rpos_anchor = (1 - self.ball_already_hit_the_ground) * self.second_pass_hit * (
            10.0 / (1.0 + torch.square(1.2 * torch.norm(rpos_anchor, dim=-1)))
        )

        reward_attacker_hit_pos = (self.flag_reward_attacker_hit_pos == 0) * (1 - self.ball_already_hit_the_ground) * self.second_pass_hit * (self.highest_ball_pos_after_my_turn[... ,2] > 3.8) * ((self.ball_pos[..., 2] - 3.6).abs() < 0.1) * (
            200.0 / (1.0 + 2.0 * torch.norm(self.ball_pos[..., 0:2] - torch.tensor([1.5, 1.5], device=self.device), dim=-1))
        ) # only considered once at most in one episode
        self.flag_reward_attacker_hit_pos = torch.where(
            reward_attacker_hit_pos > 0, 
            torch.ones_like(reward_attacker_hit_pos, device=self.device), 
            self.flag_reward_attacker_hit_pos
        )
        
        reward_after_my_turn = reward_rpos_anchor + reward_attacker_hit_pos # (E,1)


        # reward 4: End reward
        highest_pos_z = self.highest_ball_pos_after_my_turn[..., 2]
        
        ball_hit_the_ground = (self.ball_pos[..., 2] < 0.2) # (E,1)
        self.ball_already_hit_the_ground = torch.where(
            ball_hit_the_ground, 
            torch.ones_like(ball_hit_the_ground, device=self.device), 
            self.ball_already_hit_the_ground
        )

        terminated = (
            (self.drone.pos[..., 2] < 0.3) | 
            (self.drone.pos[..., 2] > 3.7) # z direction
            | (self.ball_pos[..., 2] > 5.) # z direction
            | ((self.turn == 0).unsqueeze(-1) & hit) # hover and hit
        )
        if self.done_ball_hit_ground:
            terminated = terminated | ball_hit_the_ground

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # (E,1)
        done = truncated | terminated # (E,1)

        reward_highest_pos_z = done * self.second_pass_hit * 400.0 / (1.0 + 2.0 * torch.abs(highest_pos_z - 4.5))
        penalty_end_drone_pos_z = done * 10 * (self.drone.pos[..., 2] < 0.3)
        penalty_end_wrong_hit = done * 10 * ((self.turn == 0).unsqueeze(-1) & hit)
        
        reward_end = reward_highest_pos_z - penalty_end_drone_pos_z - penalty_end_wrong_hit # (E,1)

        # Overall reward
        if self.add_reward_hover:
            reward = reward_hover_1 + reward_hover_2 + reward_my_turn + reward_after_my_turn + reward_end # (E,1)
        else:
            reward = reward_my_turn + reward_after_my_turn + reward_end # (E,1)

        # change turn
        hit_index = hit.squeeze(-1)
        last_turn = self.turn.clone()
        self.turn[hit_index.squeeze(-1)] = 0
        self.switch_turn = torch.where((self.turn ^ last_turn).bool(), 1, 0)
        self.second_pass_hit_t[hit_index] = self.progress_buf[hit_index].unsqueeze(-1)
        self.last_drone_positions[hit_index] = self.drone_pos[hit_index].unsqueeze(1)

        ep_len = self.progress_buf.unsqueeze(-1)

        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = ep_len

        self.stats["done"].add_(done.float())
        self.stats["truncated"].add_(truncated.float())

        self.stats["hit"] = self.second_pass_hit
        self.stats["num_hits"].add_(hit.float())

        self.stats["hit_ball_vel_x"].add_((delta_t == 3).float() * self.second_pass_hit * self.ball_vel[..., 0])
        self.stats["hit_ball_vel_y"].add_((delta_t == 3).float() * self.second_pass_hit * self.ball_vel[..., 1])
        self.stats["hit_ball_vel_z"].add_((delta_t == 3).float() * self.second_pass_hit * self.ball_vel[..., 2])

        self.stats["highest_ball_pos_x"] = self.highest_ball_pos_after_my_turn[..., 0]
        self.stats["highest_ball_pos_y"] = self.highest_ball_pos_after_my_turn[..., 1]
        self.stats["highest_ball_pos_z"] = self.highest_ball_pos_after_my_turn[..., 2]

        if self.stats_cfg.get("complete_reward_stats", False):
            if self.add_reward_hover:
                self.stats["reward_hover_1"].add_(reward_hover_1)
                self.stats["reward_hover_2"].add_(reward_hover_2)
            self.stats["reward_my_turn"].add_(reward_my_turn)
            self.stats["reward_after_my_turn"].add_(reward_after_my_turn)
            self.stats["reward_end"].add_(reward_end)
            self.stats["reward_hit"].add_(reward_hit)
            self.stats["penalty_hit_pos_z"].add_(penalty_hit_pos_z)
            self.stats["penalty_hit_pos_xy"].add_(penalty_hit_pos_xy)
            self.stats["reward_rpos"].add_(reward_rpos)
            self.stats["penalty_pos_z"].add_(penalty_pos_z)
            self.stats["penalty_yaw"].add_(penalty_yaw)
            self.stats["reward_attacker_hit_pos"].add_(reward_attacker_hit_pos)
            self.stats["reward_highest_pos_z"].add_(reward_highest_pos_z)
            self.stats["penalty_end_drone_pos_z"].add_(penalty_end_drone_pos_z)
            self.stats["penalty_end_wrong_hit"].add_(penalty_end_wrong_hit)

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
