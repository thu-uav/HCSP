CUDA_VISIBLE_DEVICES=0 python ../train_receive_hover.py headless=true \
    total_frames=500000000 \
    task=Receive_hover \
    task.drone_model=Iris \
    task.env.num_envs=4096 \
    task.ball_mass=0.005 \
    task.ball_radius=0.1 \
    eval_interval=100 \
    save_interval=100 \
    only_eval=false \
    Opp_Server_policy_checkpoint_path="checkpoint/checkpoint_serve.pt" \
    Opp_Server_hover_policy_checkpoint_path="checkpoint/checkpoint_serve_hover.pt" \
    FirstPass_goto_policy_checkpoint_path="checkpoint/checkpoint_goto.pt" \
    FirstPass_policy_checkpoint_path="checkpoint/checkpoint_firstpass_receive.pt" \
    task.done_ball_hit_the_ground=false \
    task.done_FirstPass_hit_the_ground=true \
    algo.actor.lr=0.0001 \
    algo.critic.lr=0.0001 \
    task.sim.dt=0.02 \
    # wandb.mode=disabled \
    
    
    
    