import torch
from carb import Float3
from typing import List,Tuple
from omni_drones.utils.torch import quat_axis
import torch.nn.functional as F


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


def calculate_ball_pass_the_net(
    ball_pos: torch.Tensor, r: float
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
    )  # (E,1)
    return tmp


def calculate_drone_pass_the_net(
    drone_pos: torch.Tensor, near_side: bool = True
) -> torch.Tensor:
    """The function is kinematically incorrect and only applicable to non-boundary cases.
    But it's efficient and very easy to implement

    Args:
        drone_pos (torch.Tensor): (E,1,3)
        near_side (bool): whether the drone is on the near side

    Returns:
        torch.Tensor: (E,1)
    """
    if near_side:
        tmp = (drone_pos[:, :, 0] < 0.2).any(dim=1).unsqueeze(-1)  # (E,1)
    else:
        tmp = (drone_pos[:, :, 0] > -0.2).any(dim=1).unsqueeze(-1) # (E,1)
    return tmp


def calculate_ball_in_side(
    ball_pos: torch.Tensor, W: float, L: float, near_or_far: str = "far"
) -> torch.Tensor:
    """The function is kinematically incorrect and only applicable to non-boundary cases.
    But it's efficient and very easy to implement

    Args:
        ball_pos (torch.Tensor): (E,1,3)
        W (float): width of the pitch
        L (float): length of the pitch
        near_or_far: ball hit to the near side or far side

    Returns:
        torch.Tensor: (E,1)
    """
    if near_or_far == "far":
        tmp = (
            (ball_pos[..., 0] > - L / 2) 
            & (ball_pos[..., 0] < 0) 
            & (ball_pos[..., 1] > - W / 2) 
            & (ball_pos[..., 1] < W / 2) 
        )  # (E,1)
    elif near_or_far == "near":
        tmp = (
            (ball_pos[..., 0] > 0) 
            & (ball_pos[..., 0] < L / 2) 
            & (ball_pos[..., 1] > - W / 2) 
            & (ball_pos[..., 1] < W / 2) 
        )
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


def attacking_target_to_obs(t: torch.Tensor, Att_or_FirstPass=True) -> torch.Tensor:
    """convert representation of ball target to one-hot vector

    Args:
        t (torch.Tensor): (n_env,)
        Att_or_FirstPass (bool):
            True: Att
            False: FirstPass

    Returns:
        torch.Tensor: (n_env, 1, 2)
    """
    if Att_or_FirstPass:
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
    else:
        table = torch.tensor(
            [
                [
                    [1.0, 0.0], # left
                ],
                [
                    [0.0, 1.0], # right
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
        root_state.clone(), split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
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


def calculate_drone_hit_the_net(drone_pos: torch.tensor, W: float, H_NET: float, r: float=0.1) -> torch.Tensor:
        """The function is kinematically incorrect and only applicable to non-boundary cases.
        But it's efficient and very easy to implement

        Args:
            drone_pos (torch.Tensor): (E,N,3)
            W (float): width of the imaginary net
            r (float): radius of the ball
            H_NET (float): height of the imaginary net

        Returns:
            torch.Tensor: (E,N)
        """
        tmp = (
            (drone_pos[..., 0].abs() < 2 * r) # * 2 is to avoid the case where the ball hits the net without being reported due to simulation steps
            & (drone_pos[..., 1].abs() < W / 2)
            & (drone_pos[..., 2] < H_NET)
        )  # (E,N)
        return tmp


def ball_side_to_obs(ball_pos: torch.Tensor) -> torch.Tensor:
    """convert representation of ball side to one-hot vector

    Args:
        ball_pos (torch.Tensor): (n_env, 1, 3)

    Returns:
        torch.Tensor: (n_env, 2, 2)
    """

    table = torch.tensor(
            [
                [
                    [0.0, 1.0], # lose the ball control
                ],
                [
                    [1.0, 0.0], # control the ball
                ]
            ],
            device=ball_pos.device,
        )

    near_side_control_ball = (ball_pos[..., 0] > 0).squeeze(1).long()  # (n_env,)
    far_side_control_ball = (ball_pos[..., 0] <= 0).squeeze(1).long()  # (n_env,)

    obs_near = table[near_side_control_ball]
    obs_far = table[far_side_control_ball]

    obs = torch.cat([obs_near, obs_far], dim=1)

    return obs

def serve_or_rally_to_obs(serve_or_rally: torch.Tensor) -> torch.Tensor:
    """
        convert representation of serve or rally to one-hot vector

    Args:
        serve_or_rally (torch.Tensor): (n_env,)
    """
    obs = F.one_hot(serve_or_rally.long(), num_classes=2).float() # (n_env, 2)
    obs = torch.stack([obs, obs], dim=1) # (n_env, 2, 2)
    return obs # (n_env, 2, 2)
    

def minor_turn_to_obs(FirstPass_turn, SecPass_turn, Att_turn, Opp_FirstPass_turn, Opp_SecPass_turn, Opp_Att_turn):
    """convert representation of minor turn to one-hot vector

    Args:
        FirstPass_turn (torch.Tensor): (n_env,)
        SecPass_turn (torch.Tensor): (n_env,)
        Att_turn (torch.Tensor): (n_env,)
        Opp_FirstPass_turn (torch.Tensor): (n_env,)
        Opp_SecPass_turn (torch.Tensor): (n_env,)
        Opp_Att_turn (torch.Tensor): (n_env,)
    Returns:
        torch.Tensor: (n_env, 2, 6)
    """
    assert (FirstPass_turn + SecPass_turn + Att_turn + Opp_FirstPass_turn + Opp_SecPass_turn + Opp_Att_turn == 1).all()

    obs_near = F.one_hot(1 * SecPass_turn + 2 * Att_turn + 3 * Opp_FirstPass_turn + 4 * Opp_SecPass_turn + 5 * Opp_Att_turn, num_classes=6)
    obs_far = F.one_hot(1 * Opp_SecPass_turn + 2 * Opp_Att_turn + 3 * FirstPass_turn + 4 * SecPass_turn + 5 * Att_turn, num_classes=6)

    obs = torch.stack((obs_near, obs_far), dim=1).float()

    return obs


def determine_game_result_3v3(drone_pos, wrong_hit_racket, wrong_hit_turn, ball_pos, W, L, H_NET, drone_idx_dict, last_hit_side, hit_ground_height):
    """Determine the game result of the 3v3 volleyball game

    Args:
        drone_pos (torch.Tensor): (E, N, 3)
        wrong_hit_racket (torch.Tensor): (E, N)
        wrong_hit_turn (torch.Tensor): (E, N)
        ball_pos (torch.Tensor): (E, 1, 3)
        W (float): width of the pitch
        L (float): length of the pitch
        H_NET (float): height of the imaginary net
        near_side_drone_idx (List[int]): index of the near side drones
        far_side_drone_idx (List[int]): index of the far side drones
        last_hit_side (torch.Tensor): (E,)
        hit_ground_height (float): threshold of the height of hitting the ground

    Returns:
        torch.Tensor: (E, 1)
    """
    assert len(drone_pos.shape) == 3
    assert len(wrong_hit_racket.shape) == 2
    assert len(wrong_hit_turn.shape) == 2
    assert len(ball_pos.shape) == 3

    assert drone_pos.shape[0] == wrong_hit_racket.shape[0] == wrong_hit_turn.shape[0] == ball_pos.shape[0] == last_hit_side.shape[0]
    assert drone_pos.shape[1] == wrong_hit_racket.shape[1] == wrong_hit_turn.shape[1]
    assert ball_pos.shape[1] == 1

    near_side_drone_idx = [
        drone_idx_dict["FirstPass"],
        drone_idx_dict["SecPass"],
        drone_idx_dict["Att"],
    ]

    far_side_drone_idx = [
        drone_idx_dict["Opp_FirstPass"],
        drone_idx_dict["Opp_SecPass"],
        drone_idx_dict["Opp_Att"],
    ]

    near_side_drone_hit_the_ground = (drone_pos[:, near_side_drone_idx, 2] < hit_ground_height).any(dim=1).unsqueeze(-1)
    far_side_drone_hit_the_ground = (drone_pos[:, far_side_drone_idx, 2] < hit_ground_height).any(dim=1).unsqueeze(-1)

    near_side_drone_pass_the_net = calculate_drone_pass_the_net(drone_pos[:, near_side_drone_idx], near_side=True).any(dim=1).unsqueeze(-1)
    far_side_drone_pass_the_net = calculate_drone_pass_the_net(drone_pos[:, far_side_drone_idx], near_side=False).any(dim=1).unsqueeze(-1)

    near_side_drone_wrong_hit_racket = wrong_hit_racket[:, near_side_drone_idx].any(dim=1).unsqueeze(-1)
    far_side_drone_wrong_hit_racket = wrong_hit_racket[:, far_side_drone_idx].any(dim=1).unsqueeze(-1)

    near_side_drone_wrong_hit_turn = wrong_hit_turn[:, near_side_drone_idx].any(dim=1).unsqueeze(-1)
    far_side_drone_wrong_hit_turn = wrong_hit_turn[:, far_side_drone_idx].any(dim=1).unsqueeze(-1)

    ball_hit_the_ground = (ball_pos[:, :, 2] < hit_ground_height) # (E, 1)
    # ball hit the ground in side
    near_side_ball_hit_the_ground = ball_hit_the_ground & calculate_ball_in_side(ball_pos, W, L, near_or_far="near")
    far_side_ball_hit_the_ground = ball_hit_the_ground & calculate_ball_in_side(ball_pos, W, L, near_or_far="far")

    # ball hit the ground out side
    ball_out_side = ball_hit_the_ground & ~calculate_ball_in_side(ball_pos, W, L, near_or_far="near") & ~calculate_ball_in_side(ball_pos, W, L, near_or_far="far")
    far_side_hit = [
        drone_idx_dict["FirstPass"], # Opp_Att has hit
        drone_idx_dict["Opp_SecPass"], # Opp_FirstPass has hit
        drone_idx_dict["Opp_Att"], # Opp_SecPass has hit
    ]
    near_side_hit = [
        drone_idx_dict["Opp_FirstPass"], # Att has hit
        drone_idx_dict["SecPass"], # FirstPass has hit
        drone_idx_dict["Att"], # SecPass has hit
    ]
    near_side_hit_the_ball_out_side = ball_out_side & ~last_hit_side.unsqueeze(-1)
    far_side_hit_the_ball_out_side = ball_out_side & last_hit_side.unsqueeze(-1)

    # ball hit the net
    ball_hit_the_net = calculate_ball_hit_the_net(ball_pos, r=0.1, W=W, H_NET=H_NET) # (E, 1)
    near_side_hit_the_ball_on_net = ball_hit_the_net & ~last_hit_side.unsqueeze(-1)
    far_side_hit_the_ball_on_net = ball_hit_the_net & last_hit_side.unsqueeze(-1)

    # result
    near_side_win = (
        far_side_drone_hit_the_ground | 
        far_side_drone_pass_the_net | 
        far_side_drone_wrong_hit_racket | 
        far_side_drone_wrong_hit_turn | 
        far_side_ball_hit_the_ground |
        far_side_hit_the_ball_out_side |
        far_side_hit_the_ball_on_net
    ) # (E, 1)

    far_side_win = (
        near_side_drone_hit_the_ground | 
        near_side_drone_pass_the_net | 
        near_side_drone_wrong_hit_racket | 
        near_side_drone_wrong_hit_turn | 
        near_side_ball_hit_the_ground |
        near_side_hit_the_ball_out_side |
        near_side_hit_the_ball_on_net
    ) # (E, 1)

    draw = (near_side_win & far_side_win)
    near_side_win = near_side_win & ~draw
    far_side_win = far_side_win & ~draw

    return near_side_win, far_side_win, draw


def determine_game_result_3v3_strict(drone_pos, wrong_hit_racket, wrong_hit_turn, ball_pos, W, L, H_NET, drone_idx_dict, last_hit_side, hit_ground_height, strict_side: str="near"):
    """Determine the game result of the 3v3 volleyball game

    Args:
        drone_pos (torch.Tensor): (E, N, 3)
        wrong_hit_racket (torch.Tensor): (E, N)
        wrong_hit_turn (torch.Tensor): (E, N)
        ball_pos (torch.Tensor): (E, 1, 3)
        W (float): width of the pitch
        L (float): length of the pitch
        H_NET (float): height of the imaginary net
        near_side_drone_idx (List[int]): index of the near side drones
        far_side_drone_idx (List[int]): index of the far side drones
        last_hit_side (torch.Tensor): (E,)
        hit_ground_height (float): threshold of the height of hitting the ground

    Returns:
        torch.Tensor: (E, 1)
    """
    assert len(drone_pos.shape) == 3
    assert len(wrong_hit_racket.shape) == 2
    assert len(wrong_hit_turn.shape) == 2
    assert len(ball_pos.shape) == 3

    assert drone_pos.shape[0] == wrong_hit_racket.shape[0] == wrong_hit_turn.shape[0] == ball_pos.shape[0] == last_hit_side.shape[0]
    assert drone_pos.shape[1] == wrong_hit_racket.shape[1] == wrong_hit_turn.shape[1]
    assert ball_pos.shape[1] == 1

    near_side_drone_idx = [
        drone_idx_dict["FirstPass"],
        drone_idx_dict["SecPass"],
        drone_idx_dict["Att"],
    ]

    far_side_drone_idx = [
        drone_idx_dict["Opp_FirstPass"],
        drone_idx_dict["Opp_SecPass"],
        drone_idx_dict["Opp_Att"],
    ]

    near_side_drone_hit_the_ground = (drone_pos[:, near_side_drone_idx, 2] < hit_ground_height).any(dim=1).unsqueeze(-1)
    far_side_drone_hit_the_ground = (drone_pos[:, far_side_drone_idx, 2] < hit_ground_height).any(dim=1).unsqueeze(-1)

    near_side_drone_pass_the_net = calculate_drone_pass_the_net(drone_pos[:, near_side_drone_idx], near_side=True).any(dim=1).unsqueeze(-1)
    far_side_drone_pass_the_net = calculate_drone_pass_the_net(drone_pos[:, far_side_drone_idx], near_side=False).any(dim=1).unsqueeze(-1)

    near_side_drone_wrong_hit_racket = wrong_hit_racket[:, near_side_drone_idx].any(dim=1).unsqueeze(-1)
    far_side_drone_wrong_hit_racket = wrong_hit_racket[:, far_side_drone_idx].any(dim=1).unsqueeze(-1)

    near_side_drone_wrong_hit_turn = wrong_hit_turn[:, near_side_drone_idx].any(dim=1).unsqueeze(-1)
    far_side_drone_wrong_hit_turn = wrong_hit_turn[:, far_side_drone_idx].any(dim=1).unsqueeze(-1)

    ball_hit_the_ground = (ball_pos[:, :, 2] < hit_ground_height) # (E, 1)
    # ball hit the ground in side
    near_side_ball_hit_the_ground = ball_hit_the_ground & calculate_ball_in_side(ball_pos, W, L, near_or_far="near")
    far_side_ball_hit_the_ground = ball_hit_the_ground & calculate_ball_in_side(ball_pos, W, L, near_or_far="far")

    # ball hit the ground out side
    ball_out_side = ball_hit_the_ground & ~calculate_ball_in_side(ball_pos, W, L, near_or_far="near") & ~calculate_ball_in_side(ball_pos, W, L, near_or_far="far")
    far_side_hit = [
        drone_idx_dict["FirstPass"], # Opp_Att has hit
        drone_idx_dict["Opp_SecPass"], # Opp_FirstPass has hit
        drone_idx_dict["Opp_Att"], # Opp_SecPass has hit
    ]
    near_side_hit = [
        drone_idx_dict["Opp_FirstPass"], # Att has hit
        drone_idx_dict["SecPass"], # FirstPass has hit
        drone_idx_dict["Att"], # SecPass has hit
    ]
    near_side_hit_the_ball_out_side = ball_out_side & ~last_hit_side.unsqueeze(-1)
    far_side_hit_the_ball_out_side = ball_out_side & last_hit_side.unsqueeze(-1)

    # ball hit the net
    ball_hit_the_net = calculate_ball_hit_the_net(ball_pos, r=0.1, W=W, H_NET=H_NET) # (E, 1)
    near_side_hit_the_ball_on_net = ball_hit_the_net & ~last_hit_side.unsqueeze(-1)
    far_side_hit_the_ball_on_net = ball_hit_the_net & last_hit_side.unsqueeze(-1)

    # result
    assert strict_side in ["near", "far"], "strict_side should be 'near' or 'far'"
    if strict_side == "near":
        near_side_win = (
            # far_side_drone_hit_the_ground | 
            # far_side_drone_pass_the_net | 
            # far_side_drone_wrong_hit_racket | 
            # far_side_drone_wrong_hit_turn | 
            far_side_ball_hit_the_ground |
            far_side_hit_the_ball_out_side |
            far_side_hit_the_ball_on_net
        ) # (E, 1)
    else:
        near_side_win = (
            far_side_drone_hit_the_ground | 
            far_side_drone_pass_the_net | 
            far_side_drone_wrong_hit_racket | 
            far_side_drone_wrong_hit_turn | 
            far_side_ball_hit_the_ground |
            far_side_hit_the_ball_out_side |
            far_side_hit_the_ball_on_net
        ) # (E, 1)

    if strict_side == "far":
        far_side_win = (
            # near_side_drone_hit_the_ground | 
            # near_side_drone_pass_the_net | 
            # near_side_drone_wrong_hit_racket | 
            # near_side_drone_wrong_hit_turn | 
            near_side_ball_hit_the_ground |
            near_side_hit_the_ball_out_side |
            near_side_hit_the_ball_on_net
        )
    else:
        far_side_win = (
            near_side_drone_hit_the_ground | 
            near_side_drone_pass_the_net | 
            near_side_drone_wrong_hit_racket | 
            near_side_drone_wrong_hit_turn | 
            near_side_ball_hit_the_ground |
            near_side_hit_the_ball_out_side |
            near_side_hit_the_ball_on_net
        ) # (E, 1)

    draw = (near_side_win & far_side_win)
    near_side_win = near_side_win & ~draw
    far_side_win = far_side_win & ~draw

    return near_side_win, far_side_win, draw