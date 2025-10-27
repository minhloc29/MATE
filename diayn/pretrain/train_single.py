import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import numpy as np
import random
from collections import deque
import mate
from mate.agents import GreedyTargetAgent, GreedyCameraAgent
import matplotlib.pyplot as plt
from config import *
from model_single import DIAYNAgent

def main():
    skill_reward_history = [[] for _ in range(NUM_SKILLS)]

    print(f"--- Stage 1: Pre-training {NUM_SKILLS} Skills for Single Camera using MultiCamera env ---")
    print(f"--- Using Device: {DEVICE} ---")
    
    base_env = gym.make('MultiAgentTracking-v0', config=ENV_CONFIG)
    env = mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())

    # Set seeds for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Environment Loaded: Training camera 0, ObsDim={OBS_DIM}, ActDim={ACTION_DIM}")
    agent = DIAYNAgent(OBS_DIM, ACTION_DIM, NUM_SKILLS)

    obs, _ = env.reset(seed=seed)
    print(f"Local Observation:\n{obs}Observation Shape:\n{obs.shape}")
    current_skill = np.random.randint(NUM_SKILLS)
    episode_steps, total_steps, update_steps = 0, 0, 0

    # Take only first camera's observation
    obs = obs[0]  

    disc_loss_history = []
    critic_loss_history = [[] for _ in range(NUM_SKILLS)]
    actor_loss_history = [[] for _ in range(NUM_SKILLS)]

    print(f"Starting pre-training for {MAX_TIMESTEPS} total steps...")

    while total_steps < MAX_TIMESTEPS:
        if total_steps % 500 == 0:
            current_skill = np.random.randint(NUM_SKILLS)

        # Get action for single camera
        action_norm = agent.select_action(obs, current_skill)
        action = np.clip(action_norm * ACTION_SCALE.cpu().numpy(),
                        [-ROTATION_MAX, -ZOOM_MAX], [ROTATION_MAX, ZOOM_MAX])
        
        # Create action array for all cameras but only control first one
        actions = np.zeros((NUM_AGENTS, ACTION_DIM))
        actions[0] = action

        if total_steps % 2000 == 0:
            print(f"Step {total_steps:,}: sample action {action} (skill={current_skill})")

        next_obs, reward, done, trunc, info = env.step(actions)
        
        # Take only first camera's next observation
        next_obs_local = next_obs[0]
        
        # Calculate pseudo-reward for single camera
        skill_tensor = torch.LongTensor([current_skill]).to(DEVICE)
        pseudo_reward = agent.calculate_pseudo_reward(torch.FloatTensor(next_obs_local).to(DEVICE), skill_tensor)
        pseudo_reward_np = pseudo_reward.cpu().numpy().item()

        skill_reward_history[current_skill].append(pseudo_reward_np)

        if total_steps % 5000 == 0:
            print(f"[Skill {current_skill}] pseudo-reward: {pseudo_reward_np:.3f}")

        # Store single transition for first camera
        done_flag = done or trunc
        agent.replay_buffer.add(obs, action, pseudo_reward_np, next_obs_local, done_flag, current_skill)

        obs = next_obs_local
        episode_steps += 1
        total_steps += 1

        if done_flag or episode_steps >= MAX_EPISODE_STEPS:
            next_obs, _ = env.reset(seed=None)
            obs = next_obs[0]  # Take first camera's observation
            episode_steps = 0

        # ...existing training code...

    print("🎯 Pre-training complete.")
    agent.save_models("diayn_single_camera_skills_v1.pth")

    # === PLOTTING ===
    plt.figure(figsize=(8, 5))
    for z in range(NUM_SKILLS):
        plt.plot(skill_reward_history[z], label=f"Skill {z}")
    plt.xlabel("Training updates (~5000 steps per point)")
    plt.ylabel("Pseudo-Reward")
    plt.title("DIAYN Single Camera Skill Pseudo-Reward Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("diayn_single_camera_reward_curve.png")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(disc_loss_history)
    plt.xlabel("Training updates")
    plt.ylabel("Discriminator Loss")
    plt.title("DIAYN Single Camera Discriminator Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("diayn_single_camera_disc_loss_curve.png")
    plt.show()

    env.close()

if __name__ == "__main__":
    main()