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
    rectangular_cuboid_edges,
    _carb_float3_add,
    draw_court, 
    calculate_ball_hit_the_net,
    calculate_drone_pass_the_net,
    turn_to_obs, 
    target_to_obs, 
    attacking_target_to_obs,
    quat_rotate,
)
from omni_drones.utils.torchrl.transforms import append_to_h5
from omni.isaac.orbit.sensors import ContactSensorCfg, ContactSensor
from omegaconf import DictConfig
from typing import Tuple, List
from abc import abstractmethod
import os


class Attack_hover(IsaacEnv):

    def __init__(self, cfg, headless):

        self.L: float = cfg.task.court.L
        self.W: float = cfg.task.court.W
        self.H_NET: float = cfg.task.court.H_NET # height of the net
        self.W_NET: float = cfg.task.court.W_NET # not width of the net, but the width of the net's frame

        self.ball_mass: float = cfg.task.ball_mass
        self.ball_radius: float = cfg.task.ball_radius

        self.done_Att_pass_the_net = cfg.task.get("done_Att_pass_the_net", False)
        
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

        self.turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) 
        # 0: before SecPass_hit_the_ball; 1: after SecPass_hit_the_ball before Att_hit_the_ball; 2: after Att_hit_the_ball
        self.switch_turn = torch.zeros(self.num_envs, device=self.device)
        self.ball_init_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.ball_last_vel = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.ball_already_hit_the_ground = torch.zeros((self.num_envs, 1), device=self.device)
        self.ball_first_hit_the_ground = torch.zeros((self.num_envs, 1), device=self.device)
        self.ball_already_hit_the_ground_in_side = torch.zeros((self.num_envs, 1), device=self.device)
        self.done_ball_hit_ground = cfg.task.get("done_ball_hit_ground", True)
        self.ball_anchor_0 = torch.tensor(cfg.task.ball_anchor_0, device=self.device)
        self.ball_anchor_1 = torch.tensor(cfg.task.ball_anchor_1, device=self.device)
        self.ball_anchor = torch.zeros((self.num_envs, 1, 3), device=self.device)
        
        self.SecPass_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: hover; 1: my turn
        self.SecPass_hit = torch.zeros((self.num_envs, 1), device=self.device)
        self.SecPass_last_hit_t = torch.zeros(self.num_envs, device=self.device)
        self.SecPass_hover_pos_after_hit = torch.tensor(cfg.task.get("SecPass_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)
        
        self.Att_turn = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: hover; 1: my turn
        self.Att_attacking_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 0: left; 1: right
        self.Att_hit = torch.zeros((self.num_envs, 1), device=self.device)
        self.Att_last_hit_t = torch.zeros(self.num_envs, device=self.device)
        self.Att_goto_pos_before_hit = torch.tensor(cfg.task.get("Att_goto_pos_before_hit"), device=self.device).expand(self.num_envs, 1, 3)
        self.Att_hover_pos_after_hit = torch.tensor(cfg.task.get("Att_hover_pos_after_hit"), device=self.device).expand(self.num_envs, 1, 3)

        hover_target_rpy = torch.zeros((self.num_envs, 1, 3), device=self.device)
        hover_target_rot = euler_to_quaternion(hover_target_rpy)
        self.hover_target_heading = quat_axis(hover_target_rot.squeeze(1), 0).unsqueeze(1)

        
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
        cr_api = PhysxSchema.PhysxContactReportAPI.Apply(ball.prim)
        cr_api.CreateThresholdAttr().Set(0.)

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

        SecPass_observation_dim = drone_state_dim + 3 + 3 + 3 + 2
        SecPass_hover_observation_dim = drone_state_dim + 3
        Att_goto_observation_dim = drone_state_dim + 3
        Att_observation_dim = drone_state_dim + 3 + 3 + 3 + 2 + 2
        Att_hover_observation_dim = drone_state_dim + 3

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "SecPass_observation": UnboundedContinuousTensorSpec((1, SecPass_observation_dim)),
                "SecPass_hover_observation": UnboundedContinuousTensorSpec((1, SecPass_hover_observation_dim)),
                "Att_goto_observation": UnboundedContinuousTensorSpec((1, Att_goto_observation_dim)),
                "Att_observation": UnboundedContinuousTensorSpec((1, Att_observation_dim)),
                "Att_hover_observation": UnboundedContinuousTensorSpec((1, Att_hover_observation_dim)),
            })
        }).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "SecPass_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "SecPass_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Att_goto_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Att_action": torch.stack([self.drone.action_spec] * 1, dim=0),
                "Att_hover_action": torch.stack([self.drone.action_spec] * 1, dim=0),
            })
        }).expand(self.num_envs).to(self.device)

        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "SecPass_reward": UnboundedContinuousTensorSpec((1, 1)),
                "Att_reward": UnboundedContinuousTensorSpec((1, 1)),
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
            state_key=("agents", "SecPass_hover_state"),
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

        _stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
            "terminated": UnboundedContinuousTensorSpec(1),
            "Att_hit": UnboundedContinuousTensorSpec(1),
            "SecPass_hit": UnboundedContinuousTensorSpec(1),
            "ball_hit_the_ground_vel_x": UnboundedContinuousTensorSpec(1),
            "ball_hit_the_ground_vel_y": UnboundedContinuousTensorSpec(1),
            "ball_hit_the_ground_vel_z": UnboundedContinuousTensorSpec(1),
            "ball_hit_the_ground_vel": UnboundedContinuousTensorSpec(1),
            "ball_hit_the_ground": UnboundedContinuousTensorSpec(1),
            "ball_hit_the_net": UnboundedContinuousTensorSpec(1),
            "in_side": UnboundedContinuousTensorSpec(1),
            "hit_pos_x": UnboundedContinuousTensorSpec(1),
            "hit_pos_y": UnboundedContinuousTensorSpec(1),
            "hit_pos_z": UnboundedContinuousTensorSpec(1),
            "Att_hit_the_net": UnboundedContinuousTensorSpec(1),
            "Att_pass_the_net": UnboundedContinuousTensorSpec(1),
        })
        self.stats_cfg: DictConfig = self.cfg.task.stats
        if self.stats_cfg.get("complete_reward_stats", False):
            _stats_spec.set("reward_hover", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_Att_hit_the_net", UnboundedContinuousTensorSpec(1))
            if self.done_Att_pass_the_net:
                _stats_spec.set("penalty_Att_pass_the_net", UnboundedContinuousTensorSpec(1))

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
            logging.info("Central env turn: Before SecPass hit the ball")
        elif turn.item() == 1:
            colors = [(0, 1, 0, 1)]
            logging.info("Central env turn: After SecPass hit the ball before Att hit the ball")
        elif turn.item() == 2:
            colors = [(0, 0, 1, 1)]
            logging.info("Central env turn: After Att hit the ball")
        sizes = [15.]
        self.draw.draw_points(points, colors, sizes)
    

    def debug_draw_Att_hitting_point(self):

        ori = self.envs_positions[self.central_env_idx].detach()
        
        points = torch.tensor([1.5, 1.5, 0.]).to(self.device) + ori
       
        points = [points.tolist()]

        colors = [(1, 1, 0, 1)]

        sizes = [15.]
        self.draw.draw_points(points, colors, sizes)
        

    def debug_draw_Att_attacking_target(self):

        attacking_target = self.Att_attacking_target[self.central_env_idx]
        ori = self.envs_positions[self.central_env_idx].detach()
        
        if attacking_target.item() == 0:
            points = torch.tensor([0., -1.5, 0.]).to(self.device) + ori
        else:
            points = torch.tensor([0., 1.5, 0.]).to(self.device) + ori
        points = [points.tolist()]

        colors = [(1, 1, 0, 1)]

        sizes = [15.]
        self.draw.draw_points(points, colors, sizes)
        
        logging.info(f"Central env Attacker attacking target: {'Left' if attacking_target.item() == 0 else 'Right'}")
    

    def calculate_drone_hit_the_net(self, W: float, H_NET: float, r: float=0.25) -> torch.Tensor:
        """The function is kinematically incorrect and only applicable to non-boundary cases.
        But it's efficient and very easy to implement

        Args:
            r (float): radius of the ball
            W (float): width of the imaginary net
            H_NET (float): height of the imaginary net

        Returns:
            torch.Tensor: (E,1)
        """
        tmp = (
            (self.drone.pos[..., 0].abs() < 3 * r) # * 3 is to avoid the case where the ball hits the net without being reported due to simulation steps
            & (self.drone.pos[..., 1].abs() < W / 2)
            & (self.drone.pos[..., 2] < H_NET)
        )  # (E,1)
        return tmp
    

    def _reset_idx(self, env_ids: torch.Tensor):

        self.turn[env_ids] = torch.zeros(len(env_ids), device=self.device).long()
        self.SecPass_turn[env_ids] = torch.ones(len(env_ids), device=self.device).long()
        self.Att_turn[env_ids] = torch.zeros(len(env_ids), device=self.device).long()

        self.Att_attacking_target[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device)

        self.drone._reset_idx(env_ids, self.training)

        SecPass_pos = self.SecPass_pos_dist.sample(env_ids.shape).unsqueeze(1)
        SecPass_rpy = self.SecPass_rpy_dist.sample(env_ids.shape).unsqueeze(1)
        Att_pos = self.Att_pos_dist.sample(env_ids.shape).unsqueeze(1)
        Att_rpy = self.Att_rpy_dist.sample(env_ids.shape).unsqueeze(1)

        drone_pos = torch.cat([SecPass_pos, Att_pos], dim=1)
        drone_rpy = torch.cat([SecPass_rpy, Att_rpy], dim=1)
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos +
            self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )

        SecPass_lin_vel = self.SecPass_lin_vel_dist.sample((*env_ids.shape, 1))
        SecPass_ang_vel = self.SecPass_ang_vel_dist.sample((*env_ids.shape, 1))
        SecPass_vel = torch.cat((SecPass_lin_vel, SecPass_ang_vel), dim=-1)
        Att_lin_vel = self.Att_lin_vel_dist.sample((*env_ids.shape, 1))
        Att_ang_vel = self.Att_ang_vel_dist.sample((*env_ids.shape, 1))
        Att_vel = torch.cat((Att_lin_vel, Att_ang_vel), dim=-1)
        
        drone_vel = torch.cat([SecPass_vel, Att_vel], dim=1)
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
        self.ball_init_vel[env_ids] = ball_lin_vel
        self.ball_last_vel[env_ids] = ball_lin_vel

        self.ball_anchor[env_ids] = torch.where(self.Att_attacking_target[env_ids].unsqueeze(1).unsqueeze(2).expand(-1, 1, 3).bool(), self.ball_anchor_1, self.ball_anchor_0)

        self.stats[env_ids] = 0.

        self.switch_turn[env_ids] = 0
        self.ball_already_hit_the_ground[env_ids] = 0
        self.ball_first_hit_the_ground[env_ids] = 0
        self.ball_already_hit_the_ground_in_side[env_ids] = 0

        self.SecPass_hit[env_ids] = 0
        self.SecPass_last_hit_t[env_ids] = -10
        
        self.Att_hit[env_ids] = 0
        self.Att_last_hit_t[env_ids] = -10

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
            self.debug_draw_Att_hitting_point()
            self.debug_draw_Att_attacking_target()
            self.debug_draw_turn()


    def _pre_sim_step(self, tensordict: TensorDictBase):
        SecPass_action = tensordict[("agents", "SecPass_action")]
        SecPass_hover_action = tensordict[("agents", "SecPass_hover_action")]
        SecPass_real_action = torch.where(tensordict["stats"]["SecPass_hit"].unsqueeze(-1).bool(), SecPass_hover_action, SecPass_action)
        
        Att_goto_action = tensordict[("agents", "Att_goto_action")]
        Att_action = tensordict[("agents", "Att_action")]
        Att_hover_action = tensordict[("agents", "Att_hover_action")]
        Att_real_action = torch.where(tensordict["stats"]["SecPass_hit"].unsqueeze(-1).bool(), Att_action, Att_goto_action)
        Att_real_action = torch.where(tensordict["stats"]["Att_hit"].unsqueeze(-1).bool(), Att_hover_action, Att_real_action)

        actions = torch.cat([SecPass_real_action, Att_real_action], dim=1)
        self.effort = self.drone.apply_action(actions)


    def _post_sim_step(self, tensordict: TensorDictBase):
        # self.contact_sensor.update(self.dt)
        pass


    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state() # (E, 2, 23)
        self.info["drone_state"][:] = self.root_state[..., :13]
        self.ball_pos, _ = self.get_env_poses(self.ball.get_world_poses()) # (E, 1, 3)
        self.ball_vel = self.ball.get_velocities()[..., :3] # (E, 1, 3)

        self.rpos = self.ball_pos - self.drone.pos # (E, 2, 3)

        pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )
        rpy = quaternion_to_euler(rot)
        rot = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot, rot)
        self.root_state = torch.cat([pos, rot, vel, angular_vel, heading, up, throttle], dim=-1)
        self.drone_rot = rot

        self.SecPass_pos = pos[:, 0, :].unsqueeze(1) # (E, 1, 3)
        self.Att_pos = pos[:, 1, :].unsqueeze(1) # (E, 1, 3)

        self.SecPass_heading = heading[:, 0, :].unsqueeze(1) # (E, 1, 3)
        self.Att_heading = heading[:, 1, :].unsqueeze(1) # (E, 1, 3)

        self.SecPass_up = up[:, 0, :].unsqueeze(1) # (E, 1, 3)
        self.Att_up = up[:, 1, :].unsqueeze(1) # (E, 1, 3)

        self.SecPass_vel = vel[:, 0, :].unsqueeze(1) # (E, 1, 3)
        self.Att_vel = vel[:, 1, :].unsqueeze(1) # (E, 1, 3)

        self.SecPass_rpos = self.rpos[:, 0, :].unsqueeze(1)
        self.Att_rpos = self.rpos[:, 1, :].unsqueeze(1)

        self.SecPass_rpy = rpy[:, 0, :].unsqueeze(1)
        self.Att_rpy = rpy[:, 1, :].unsqueeze(1)
        self.Att_roll = self.Att_rpy[..., 0]
        self.Att_pitch = self.Att_rpy[..., 1]
        self.Att_yaw = self.Att_rpy[..., 2]

        SecPass_obs = [
            self.root_state[:, 0, :].unsqueeze(1), # (E, 1, 23)
            self.ball_pos, # (E, 1, 3)
            self.SecPass_rpos, # (E, 1, 3)
            self.ball_vel, # (E, 1, 3)
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

        Att_goto_rpos = self.Att_goto_pos_before_hit - self.Att_pos # (E, 1, 3)
        Att_goto_rheading = self.hover_target_heading - self.Att_heading # (E, 1, 3)
        Att_goto_obs = [
            Att_goto_rpos, # (E, 1, 23)
            self.root_state[:, 1, 3:].unsqueeze(1), # (E, 1, root_state_dim-3)
            Att_goto_rheading, # (E, 1, 3)
        ]
        Att_goto_obs = torch.cat(Att_goto_obs, dim=-1)

        Att_obs = [
            self.root_state[:, 1, :].unsqueeze(1), # (E, 1, 23)
            self.ball_pos, # (E, 1, 3)
            self.Att_rpos, # (E, 1, 3)
            self.ball_vel, # (E, 1, 3)
            turn_to_obs(self.Att_turn), # (E, 1, 2)
            attacking_target_to_obs(self.Att_attacking_target) # (E, 1, 2)
        ]
        Att_obs = torch.cat(Att_obs, dim=-1)

        Att_hover_rpos = self.Att_hover_pos_after_hit - self.root_state[:, 1, :3].unsqueeze(1) # (E, 1, 3)
        Att_hover_rheading = self.hover_target_heading - self.root_state[:, 1, 13:16].unsqueeze(1) # (E, 1, 3)
        Att_hover_obs = [
            Att_hover_rpos, # (E, 1, 23)
            self.root_state[:, 1, 3:].unsqueeze(1), # (E, 1, root_state_dim-3)
            Att_hover_rheading, # (E, 1, 3)
        ]
        Att_hover_obs = torch.cat(Att_hover_obs, dim=-1)

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
                    "SecPass_observation": SecPass_obs,
                    "SecPass_hover_observation": SecPass_hover_obs,
                    "Att_goto_observation": Att_goto_obs,
                    "Att_observation": Att_obs,
                    "Att_hover_observation": Att_hover_obs,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.num_envs,
        )
            

    def _compute_reward_and_done(self):

        # reward 1: hover
        SecPass_reward = torch.zeros((self.num_envs, 1), device=self.device)
        
        rpos_hover = self.drone.pos[:, 1, :].unsqueeze(1) - self.Att_hover_pos_after_hit
        rheading = self.drone.heading[:, 1, :].unsqueeze(1) - torch.tensor([1., 0., 0.], device=self.device).unsqueeze(0).unsqueeze(0)
        distance = torch.norm(torch.cat([rpos_hover, rheading], dim=-1), dim=-1)
        reward_pose = 1.0 / (1.0 + torch.square(1.2 * distance))
        reward_up = torch.square((self.drone.up[:, 1, 2] + 1) / 2).unsqueeze(-1)
        spinnage = torch.square(self.drone.vel[:, 1, -1]).unsqueeze(-1)
        reward_spin = 1.0 / (1.0 + torch.square(spinnage))
        reward_hover = 3 * (
            reward_pose 
            + reward_pose * (reward_up + reward_spin)  
        ) # (E,1)

        # reward 2: Att_turn=1: my turn
        self.racket_r = 0.2

        z_direction_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.z_direction_world = quat_rotate(self.drone_rot, z_direction_local) # (E,2,3)

        normal_vector_world = self.z_direction_world / torch.norm(self.z_direction_world, dim=-1).unsqueeze(-1)  # (E,2,3)

        self.drone_racket_center = self.drone.pos # (E,2,3) cylinder low center
        self.drone_cylinder_top_center = self.drone_racket_center + 2.0 * self.ball_radius * normal_vector_world # (E,2,3) cylinder top center
        cylinder_axis = self.drone_cylinder_top_center - self.drone_racket_center  # (E,2,3)

        ball_to_bottom = self.ball_pos - self.drone_racket_center  # (E,2,3)
        t = torch.sum(ball_to_bottom * cylinder_axis, dim=-1) / torch.sum(cylinder_axis * cylinder_axis, dim=-1)  # (E,2) projection of ball_to_bottom on cylinder_axis   
        within_height = (t >= 0) & (t <= 1)  # (E,2)

        projection_point = self.drone_racket_center + t.unsqueeze(-1) * cylinder_axis  # (E,2,3)
        distance_to_axis = torch.norm(self.ball_pos - projection_point, dim=-1)  # (E,2)
        within_radius = distance_to_axis <= self.racket_r  # (E,2)

        racket_near_ball = (within_height & within_radius)  # (E,2)

        # racket hit ball
        ball_near_ground = (self.ball_pos[..., 2] < 0.2) # (E,1)
        ball_vel_z_change = ((self.ball_vel[..., 2] - self.ball_last_vel[..., 2]) > -0.196) # (E,1)
        hit_1 = racket_near_ball & ball_vel_z_change # (E,2) # 
        hit_ground_1 = ball_near_ground & ball_vel_z_change # (E,1) # 
        ball_vel_x_y_change = (self.ball_vel[..., :2] - self.ball_init_vel[..., :2]).norm(dim=-1) > 0.5 # (E,1)
        hit_2 = racket_near_ball & ball_vel_x_y_change & ~torch.cat((self.SecPass_hit, self.Att_hit), dim=-1).bool() # (E,2) 
        hit_ground_2 = ball_near_ground & ball_vel_x_y_change # (E,1)
        hit = (hit_1 | hit_2) & ~hit_ground_1 & ~hit_ground_2 & ~self.ball_already_hit_the_ground.bool() # (E,2) racket hit ball

        self.stats["hit_pos_x"].add_(hit[:, 1].unsqueeze(-1) * (1 - self.Att_hit) * self.ball_pos[..., 0])
        self.stats["hit_pos_y"].add_(hit[:, 1].unsqueeze(-1) * (1 - self.Att_hit) * self.ball_pos[..., 1])
        self.stats["hit_pos_z"].add_(hit[:, 1].unsqueeze(-1) * (1 - self.Att_hit) * self.ball_pos[..., 2])

        hit[:, 0] = hit[:, 0] & (self.progress_buf - self.SecPass_last_hit_t > 3) # (E,)
        hit[:, 1] = hit[:, 1] & (self.progress_buf - self.Att_last_hit_t > 3) # (E,)
        self.SecPass_last_hit_t = torch.where(hit[:, 0], self.progress_buf, self.SecPass_last_hit_t)
        self.Att_last_hit_t = torch.where(hit[:, 1], self.progress_buf, self.Att_last_hit_t)

        self.SecPass_hit = torch.where(hit[:, 0].unsqueeze(-1) == 1, 1, self.SecPass_hit)
        self.Att_hit = torch.where(hit[:, 1].unsqueeze(-1) == 1, 1, self.Att_hit)
        self.ball_last_vel = self.ball_vel
        

        # reward 4: End reward
        ball_hit_the_ground = (self.ball_pos[..., 2] < 0.2) # (E,1)
        self.ball_first_hit_the_ground = ball_hit_the_ground & ~self.ball_already_hit_the_ground.bool()
        self.ball_already_hit_the_ground = torch.where(ball_hit_the_ground, 1, self.ball_already_hit_the_ground)
        ball_hit_the_net = calculate_ball_hit_the_net(self.ball_pos, self.ball_radius, self.W, self.H_NET) # (E,1)

        Att_pass_the_net = calculate_drone_pass_the_net(self.drone.pos[:, 1, :].unsqueeze(1)) # (E,1)
        
        
        Att_hit_the_net = self.calculate_drone_hit_the_net(self.W, self.H_NET)[:, 1].unsqueeze(-1) # (E,1)
        _penalty_Att_hit_the_net_coef = 50.0
        penalty_Att_hit_the_net = _penalty_Att_hit_the_net_coef * Att_hit_the_net

        terminated = (
            (self.drone.pos[..., 2] < 0.3).any(dim=1).unsqueeze(-1) | 
            (self.drone.pos[..., 2] > 4.0).any(dim=1).unsqueeze(-1) # z direction
            | (self.ball_pos[..., 2] > 5.) # z direction
            | ((self.SecPass_turn == 0) & hit[:, 0]).unsqueeze(-1) # hover and hit
            | ((self.Att_turn == 0) & hit[:, 1]).unsqueeze(-1) # hover and hit
            | ball_hit_the_net
            | (hit[:, 0] & hit[:, 1]).unsqueeze(-1) # both hit
            | Att_hit_the_net         
        )
        if self.done_Att_pass_the_net:
            terminated = terminated | Att_pass_the_net
            penalty_Att_pass_the_net = 10 * Att_pass_the_net
            if self.stats_cfg.get("complete_reward_stats", False):
                self.stats["penalty_Att_pass_the_net"].add_(penalty_Att_pass_the_net)
        if self.done_ball_hit_ground:
            terminated = terminated | ball_hit_the_ground

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # (E,1)
        done = truncated | terminated

        in_side = self.ball_first_hit_the_ground * (
            (self.ball_pos[..., 0] > -self.L/2) & (self.ball_pos[..., 0] < 0) &
            (self.ball_pos[..., 1] > -self.W/2) & (self.ball_pos[..., 1] < self.W/2) 
        )
        self.ball_already_hit_the_ground_in_side = torch.where(self.ball_first_hit_the_ground, in_side, self.ball_already_hit_the_ground_in_side)
    

        # final reward
        reward = reward_hover - penalty_Att_hit_the_net
        if self.done_Att_pass_the_net:
            reward -= penalty_Att_pass_the_net

        Att_reward_1 = torch.zeros((self.num_envs, 1), device=self.device) # already trained GoTo and Attack policy
        Att_reward_2 = reward # attack_hover policy to be trained
        Att_reward = torch.where(self.SecPass_hit.bool(), Att_reward_2, Att_reward_1)

        # change turn
        SecPass_hit_index = hit[:, 0] # (E,)
        Att_hit_index = hit[:, 1] # (E,)
        last_turn = self.turn.clone() # (E,)
        self.turn = torch.where((self.turn == 0) & SecPass_hit_index, 1, self.turn) # (E,)
        self.turn = torch.where((self.turn == 1) & Att_hit_index, 2, self.turn) # (E,)
        self.SecPass_turn = torch.where((self.turn == 0), 1, 0) # (E,)
        self.Att_turn = torch.where((self.turn == 1), 1, 0) # (E,)
        self.switch_turn = torch.where((self.turn ^ last_turn).bool(), 1, 0) # (E,)

        # log
        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["return"].add_(Att_reward)
        self.stats["episode_len"][:] = ep_len

        self.stats["truncated"].add_(truncated.float())
        self.stats["terminated"].add_(terminated.float())

        self.stats["Att_hit"] = self.Att_hit
        self.stats["SecPass_hit"] = self.SecPass_hit

        self.stats["Att_pass_the_net"] = torch.where(Att_pass_the_net, torch.ones_like(Att_pass_the_net), self.stats["Att_pass_the_net"])

        self.stats["ball_hit_the_ground"].add_(self.ball_first_hit_the_ground)
        self.stats["ball_hit_the_net"].add_(ball_hit_the_net)
        self.stats["in_side"].add_(in_side)

        self.stats["ball_hit_the_ground_vel_x"].add_(in_side * self.ball_vel[..., 0])
        self.stats["ball_hit_the_ground_vel_y"].add_(in_side * self.ball_vel[..., 1])
        self.stats["ball_hit_the_ground_vel_z"].add_(in_side * self.ball_vel[..., 2])
        self.stats["ball_hit_the_ground_vel"].add_(in_side * self.ball_vel.norm(dim=-1))

        self.stats["Att_hit_the_net"].add_(Att_hit_the_net)

        if self.stats_cfg.get("complete_reward_stats", False):
            self.stats["reward_hover"].add_(reward_hover)
            self.stats["penalty_Att_hit_the_net"].add_(penalty_Att_hit_the_net)

        return TensorDict(
            {
                "agents": {
                    "SecPass_reward": SecPass_reward,
                    "Att_reward": Att_reward,
                },
                "done": done,
                "terminated": terminated,
                "truncated": truncated
            },
            self.num_envs,
        )