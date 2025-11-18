CUDA_VISIBLE_DEVICES=1 python ../train.py headless=true \
    total_frames=20000000 \
    task=Hover \
    task.drone_model=Iris \
    task.env.num_envs=1024 \
    eval_interval=30 \
    save_interval=30 \
    task.action_transform=PIDrate \
    task.time_encoding=true \
    task.throttles_in_obs=false \
    # wandb.mode=disabled # debug