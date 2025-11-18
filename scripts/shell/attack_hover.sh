CUDA_VISIBLE_DEVICES=0 python ../train_attack_hover.py headless=true \
    total_frames=500000000 \
    task=Attack_hover \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    task.ball_mass=0.005 \
    task.ball_radius=0.1 \
    eval_interval=100 \
    save_interval=100 \
    only_eval=false \
    SecPass_policy_checkpoint_path="checkpoint/checkpoint_secpass.pt" \
    SecPass_hover_policy_checkpoint_path="checkpoint/checkpoint_secpass_hover.pt" \
    Att_goto_policy_checkpoint_path="checkpoint/checkpoint_goto.pt" \
    Att_policy_checkpoint_path="checkpoint/checkpoint_att.pt" \
    task.done_ball_hit_ground=false \
    task.done_Att_pass_the_net=true \
    # wandb.mode=disabled \