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
from .common import rectangular_cuboid_edges,_carb_float3_add
from omni_drones.utils.torchrl.transforms import append_to_h5
from omni.isaac.orbit.sensors import ContactSensorCfg, ContactSensor
from omegaconf import DictConfig
from typing import Tuple, List
from abc import abstractmethod
import os

_COLOR_T = Tuple[float, float, float, float]


def _draw_net(
    W: float,
    H_NET: float,
    W_NET: float,
    color_mesh: _COLOR_T = (1.0, 1.0, 1.0, 1.0),
    color_post: _COLOR_T = (1.0, 0.729, 0, 1.0),
    size_mesh_line: float = 3.0,
    size_post: float = 10.0,
    n: int = 30,
):
    if n < 2:
        raise ValueError("n should be greater than 1")
    point_list_1 = [Float3(0, -W / 2, i * W_NET / (n - 1) + H_NET - W_NET)
                    for i in range(n)]
    point_list_2 = [Float3(0, W / 2, i * W_NET / (n - 1) + H_NET - W_NET)
                    for i in range(n)]

    point_list_1.append(Float3(0, W / 2, 0))
    point_list_1.append(Float3(0, -W / 2, 0))

    point_list_2.append(Float3(0, W / 2, H_NET))
    point_list_2.append(Float3(0, -W / 2, H_NET))

    colors = [color_mesh for _ in range(n)]
    sizes = [size_mesh_line for _ in range(n)]
    colors.append(color_post)
    colors.append(color_post)
    sizes.append(size_post)
    sizes.append(size_post)

    return point_list_1, point_list_2, colors, sizes


def _draw_board(
    W: float, L: float, color: _COLOR_T = (1.0, 1.0, 1.0, 1.0), line_size: float = 10.0
):
    point_list_1 = [
        Float3(-L / 2, -W / 2, 0),
        Float3(-L / 2, W / 2, 0),
        Float3(-L / 2, -W / 2, 0),
        Float3(L / 2, -W / 2, 0),
    ]
    point_list_2 = [
        Float3(L / 2, -W / 2, 0),
        Float3(L / 2, W / 2, 0),
        Float3(-L / 2, W / 2, 0),
        Float3(L / 2, W / 2, 0),
    ]

    colors = [color for _ in range(4)]
    sizes = [line_size for _ in range(4)]

    return point_list_1, point_list_2, colors, sizes


def _draw_lines_args_merger(*args):
    buf = [[] for _ in range(4)]
    for arg in args:
        buf[0].extend(arg[0])
        buf[1].extend(arg[1])
        buf[2].extend(arg[2])
        buf[3].extend(arg[3])

    return (
        buf[0],
        buf[1],
        buf[2],
        buf[3],
    )


def draw_court(W: float, L: float, H_NET: float, W_NET: float, n: int = 30):
    return _draw_lines_args_merger(_draw_net(W, H_NET, W_NET, n=n), _draw_board(W, L))


class VolleyEnv(IsaacEnv):
    @abstractmethod
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
        contact_sensor_cfg = ContactSensorCfg(prim_path="/World/envs/env_.*/ball")
        self.contact_sensor: ContactSensor = contact_sensor_cfg.class_type(contact_sensor_cfg)
        self.contact_sensor._initialize_impl()

        self.ball_pos_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.ball_pos_dist.low, device=self.device), 
            high=torch.tensor(cfg.task.initial.ball_pos_dist.high, device=self.device)
        )
        self.ball_vel_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.ball_vel_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.ball_vel_dist.high, device=self.device)
        )
        self.drone_pos_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.drone_pos_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.drone_pos_dist.high, device=self.device)
        )
        self.drone_rpy_dist = D.Uniform(
            low=torch.tensor(cfg.task.initial.drone_rpy_dist.low, device=self.device),
            high=torch.tensor(cfg.task.initial.drone_rpy_dist.high, device=self.device)
        )
        if cfg.task.initial.get("drone_lin_vel_dist"):
            self.drone_lin_vel_dist = D.Uniform(
                low=torch.tensor(cfg.task.initial.drone_lin_vel_dist.low, device=self.device),
                high=torch.tensor(cfg.task.initial.drone_lin_vel_dist.high, device=self.device)
            )
        if cfg.task.initial.get("drone_ang_vel_dist"):
            self.drone_ang_vel_dist = D.Uniform(
                low=torch.tensor(cfg.task.initial.drone_ang_vel_dist.low, device=self.device),
                high=torch.tensor(cfg.task.initial.drone_ang_vel_dist.high, device=self.device)
            )
        
        self.ball_traj_vis = []
        self.draw = _debug_draw.acquire_debug_draw_interface()


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
        drone_prims = self.drone.spawn(translations=[(0.0, 0.0, 2.)])

        material = UsdShade.Material(material.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("base_link/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

        return ["/World/defaultGroundPlane"]

    @abstractmethod
    def _set_specs(self):
        pass
    

    @abstractmethod
    def _reset_idx(self, env_ids: torch.Tensor):
        pass


    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)


    def _post_sim_step(self, tensordict: TensorDictBase):
        self.contact_sensor.update(self.dt)


    @abstractmethod
    def _compute_state_and_obs(self):
        pass


    @abstractmethod
    def _compute_reward_and_done(self):
        pass