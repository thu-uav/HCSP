CUDA_VISIBLE_DEVICES=0 python ../train_attack.py headless=true \
    total_frames=1000000000 \
    task=Attack \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    task.ball_mass=0.005 \
    task.ball_radius=0.1 \
    eval_interval=300 \
    save_interval=300 \
    only_eval=false \
    SecPass_policy_checkpoint_path="checkpoint/checkpoint_secpass_2.pt" \
    SecPass_hover_policy_checkpoint_path="checkpoint/checkpoint_secpass_hover_2.pt" \
    Att_goto_policy_checkpoint_path="checkpoint/checkpoint_goto.pt" \
    task.done_ball_hit_ground=true \
    task.done_Att_pass_the_net=true \
    # wandb.mode=disabled \
    