import torch
from carb import Float3
from typing import List,Tuple
from omni_drones.utils.torch import quat_axis


def _carb_float3_add(a: Float3, b: Float3) -> Float3:
    return Float3(a.x + b.x, a.y + b.y, a.z + b.z)


def rectangular_cuboid_edges(length:float,width:float,z_low:float,height:float)->Tuple[List[Float3],List[Float3]]:
    """the rectangular cuboid is 
    """
    z=Float3(0,0,height)
    vertices=[
        Float3(-length/2,width/2,z_low),
        Float3(length/2,width/2,z_low),
        Float3(length/2,-width/2,z_low),
        Float3(-length/2,-width/2,z_low),
    ]
    points_start=[
        vertices[0],
            vertices[1],
            vertices[2],
            vertices[3],
            vertices[0],
            vertices[1],
            vertices[2],
            vertices[3],
            _carb_float3_add(vertices[0], z),
            _carb_float3_add(vertices[1] , z),
            _carb_float3_add(vertices[2] , z),
            _carb_float3_add(vertices[3] , z),
    ]
    
    points_end=[
        vertices[1],
            vertices[2],
            vertices[3],
            vertices[0],
            _carb_float3_add(vertices[0] , z),
            _carb_float3_add(vertices[1] , z),
            _carb_float3_add(vertices[2] , z),
            _carb_float3_add(vertices[3] , z),
            _carb_float3_add(vertices[1] , z),
            _carb_float3_add(vertices[2] , z),
            _carb_float3_add(vertices[3] , z),
            _carb_float3_add(vertices[0] , z),
    ]
    
    return points_start,points_end


_COLOR_T = Tuple[float, float, float, float]


def draw_net(
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


def draw_board(
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


def draw_lines_args_merger(*args):
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
    return draw_lines_args_merger(draw_net(W, H_NET, W_NET, n=n), draw_board(W, L))


def calculate_ball_hit_the_net(
    ball_pos: torch.Tensor, r: float, W: float, H_NET: float
) -> torch.Tensor:
    """The function is kinematically incorrect and only applicable to non-boundary cases.
    But it's efficient and very easy to implement

    Args:
        ball_pos (torch.Tensor): (E,1,3)
        r (float): radius of the ball
        W (float): width of the imaginary net
        H_NET (float): height of the imaginary net

    Returns:
        torch.Tensor: (E,1)
    """
    tmp = (
        (ball_pos[:, :, 0].abs() < 3 * r) # * 3 is to avoid the case where the ball hits the net without being reported due to simulation steps
        & (ball_pos[:, :, 1].abs() < W / 2)
        & (ball_pos[:, :, 2] < H_NET)
    )  # (E,1)
    return tmp


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


def attacking_target_to_obs(t: torch.Tensor) -> torch.Tensor:
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


def quaternion_multiply(q1, q2):
    assert q1.shape == q2.shape and q1.shape[-1] == 4
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def transfer_root_state_to_the_other_side(root_state):
    """Transfer the root state to the other side of the court

    Args:
        root_state: [E, 1, 23]

    Returns:
        root_state: [E, 1, 23]
    """
    
    assert len(root_state.shape) == 3
    assert root_state.shape[1] == 1
    assert root_state.shape[2] == 23
    
    pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
        root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
    )
    
    pos[..., :2] = - pos[..., :2]

    q_m = torch.tensor([[0, 0, 0, 1]] * rot.shape[0], device=rot.device)
    rot[:, 0, :] = quaternion_multiply(q_m, rot[:, 0, :])
    rot = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot, rot)

    vel[..., :2] = - vel[..., :2]

    angular_vel[..., :2] = - angular_vel[..., :2]

    heading = quat_axis(rot, axis=0)

    up = quat_axis(rot, axis=2)

    return torch.cat([pos, rot, vel, angular_vel, heading, up, throttle], dim=-1)


def quat_rotate(q, v):
    """
    Rotate vector v by quaternion q
    q: quaternion [w, x, y, z]
    v: vector [x, y, z]
    Returns the rotated vector
    """

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Quaternion rotation formula
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    
    # Construct the rotation matrix from the quaternion
    rot_matrix = torch.stack([
        ww + xx - yy - zz,
        2 * (xy - wz),
        2 * (xz + wy),
        
        2 * (xy + wz),
        ww - xx + yy - zz,
        2 * (yz - wx),
        
        2 * (xz - wy),
        2 * (yz + wx),
        ww - xx - yy + zz
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)

    v_expanded = v.expand(*q.shape[:-1], 3)
    
    # Rotate the vector using the rotation matrix
    return torch.matmul(rot_matrix, v_expanded.unsqueeze(-1)).squeeze(-1)