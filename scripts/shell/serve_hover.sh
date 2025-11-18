CUDA_VISIBLE_DEVICES=0 python ../train_serve_hover.py headless=true \
    total_frames=500000000 \
    task=Serve_hover \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    task.ball_mass=0.005 \
    task.ball_radius=0.1 \
    eval_interval=100 \
    save_interval=100 \
    only_eval=true \
    Server_policy_checkpoint_path="checkpoint/checkpoint_serve.pt" \
    task.done_ball_hit_ground=false \
    # wandb.mode=disabled \