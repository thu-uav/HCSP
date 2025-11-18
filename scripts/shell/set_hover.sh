CUDA_VISIBLE_DEVICES=0 python ../train_set_hover.py headless=true \
    total_frames=100000000 \
    task=Set_hover \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    task.ball_mass=0.005 \
    task.ball_radius=0.1 \
    eval_interval=100 \
    save_interval=100 \
    only_eval=false \
    SecPass_policy_checkpoint_path="checkpoint/checkpoint_secpass.pt" \
    # wandb.mode=disabled \