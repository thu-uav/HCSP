CUDA_VISIBLE_DEVICES=0 python ../train_pass_hover.py headless=true \
    total_frames=500000000 \
    task=Pass_hover \
    task.drone_model=Iris \
    task.env.num_envs=4096\
    task.ball_mass=0.005 \
    task.ball_radius=0.1 \
    eval_interval=100 \
    save_interval=100 \
    only_eval=false \
    Opp_SecPass_policy_checkpoint_path="checkpoint/checkpoint_secpass_2.pt" \
    Opp_SecPass_hover_policy_checkpoint_path="checkpoint/checkpoint_secpass_hover_2.pt" \
    Opp_Att_goto_policy_checkpoint_path="checkpoint/checkpoint_goto.pt" \
    Opp_Att_policy_checkpoint_path="checkpoint/checkpoint_att_2.pt" \
    Opp_Att_hover_policy_checkpoint_path="checkpoint/checkpoint_att_hover_2.pt" \
    FirstPass_goto_policy_checkpoint_path="checkpoint/checkpoint_goto.pt" \
    FirstPass_policy_checkpoint_path="checkpoint/checkpoint_firstpass_3.pt" \
    task.done_ball_hit_ground=false \
    algo.actor.lr=0.000001 \
    algo.critic.lr=0.000001 \
    task.sim.dt=0.02 \
    task.sim.substeps=1 \
    task.use_trained_state=true \
    # wandb.mode=disabled \
    
    
