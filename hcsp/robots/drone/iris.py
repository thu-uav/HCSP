import torch

from omni.isaac.core.prims import RigidPrimView
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from hcsp.robots.drone.multirotor import MultirotorBase
from hcsp.robots.robot import ASSET_PATH


class Iris(MultirotorBase):

    usd_path: str = ASSET_PATH + "/usd/iris_batVisualOnly.usd"
    param_path: str = ASSET_PATH + "/usd/iris.yaml"
