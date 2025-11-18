CUDA_VISIBLE_DEVICES=0 python ../train.py headless=true \
    total_frames=50000000 \
    task=Goto \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    eval_interval=30 \
    save_interval=30 \
    only_eval=false \
    only_eval_one_traj=false \
    task.save_drone_state=true \
    # wandb.mode=disabled