CUDA_VISIBLE_DEVICES=0 python ../train.py headless=true \
    total_frames=1000000000 \
    task=Serve \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    task.ball_mass=0.005 \
    task.ball_radius=0.1 \
    eval_interval=100 \
    save_interval=100 \
    only_eval=false \
    task.add_reward_hover=false \
    task.done_ball_hit_ground=true \
    # wandb.mode=disabled # debug