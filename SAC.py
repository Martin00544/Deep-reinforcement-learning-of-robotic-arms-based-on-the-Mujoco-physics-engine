# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 20:58:45 2026

@author: Martin
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import mujoco
import mujoco.viewer
import time
import random
from collections import deque
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PiperUpwardEnv:
    def __init__(self, model_path="piper_scene.xml", render=False):

        with open(model_path, 'r') as f:
            xml_string = f.read()
        
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        
        self.render = render
        self.viewer = None
        
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                         for name in self.joint_names]
        
        self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link6')
        
        self.upward_config = np.array([0.0, 1.27, -2.7, 0.0, 0.0, 0.0])#[0.0, 1.57, -1.57, 0.0, 0.0, 0.0]
        
        mujoco.mj_resetData(self.model, self.data)
        
        init_state = self.get_state()
        self.state_dim = len(init_state)
        
        self.action_dim = len(self.joint_ids) - 1 
        
        self.action_high = np.array([2.0, 2.0, 2.0, 1.5, 1.0, 2.0])
        self.action_low = -self.action_high
        
        self.reset()
    
    def get_gripper_pos(self):
        return self.data.xpos[self.gripper_body_id].copy()
    
    def get_gripper_orientation(self):
        return self.data.xquat[self.gripper_body_id].copy()
    
    def get_state(self):
        qpos = self.data.qpos[self.joint_ids]
        qvel = self.data.qvel[self.joint_ids]
        gripper_pos = self.get_gripper_pos()
        gripper_quat = self.get_gripper_orientation()
        
        qw, qx, qy, qz = gripper_quat

        z_vec = np.array([
            2*(qx*qz - qw*qy),
            2*(qy*qz + qw*qx),
            1 - 2*(qx*qx + qy*qy)
        ])
        

        gripper_height = gripper_pos[2]
        

        config_distance = np.sum(np.abs(qpos[:6] - self.upward_config))
        

        state = np.concatenate([
            qpos,
            qvel,
            z_vec,  
            [gripper_height], 
            [config_distance] 
        ])
        
        return state
    
    def calculate_upward_reward(self, qpos, gripper_pos, gripper_quat, action):
       
        height_reward = gripper_pos[2] * 2.0
        
       
        qw, qx, qy, qz = gripper_quat
        z_vec = np.array([
            2*(qx*qz - qw*qy),
            2*(qy*qz + qw*qx),
            1 - 2*(qx*qx + qy*qy)
        ])
        upward_reward = z_vec[2] * 5.0  
             
        config_distance = np.sum(np.abs(qpos[:6] - self.upward_config))
        config_reward = -config_distance * 0.5
        
        action_penalty = np.sum(np.square(action)) * 0.01
        
        velocity_penalty = np.sum(np.square(self.data.qvel[self.joint_ids])) * 0.001
        
        stability_reward = -np.sum(np.abs(self.data.qvel[self.joint_ids])) * 0.01
        
        total_reward = (
            height_reward + 
            upward_reward + 
            config_reward - 
            action_penalty - 
            velocity_penalty + 
            stability_reward
        )
        
        return total_reward, {
            'height': gripper_pos[2],
            'upward': z_vec[2],
            'config_distance': config_distance,
            'height_reward': height_reward,
            'upward_reward': upward_reward
        }
    
    def step(self, action):

        scaled_action = np.clip(action, -1, 1)
        scaled_action = self.action_low + (scaled_action + 1) * (self.action_high - self.action_low) / 2
        
        for i in range(self.action_dim):
            self.data.ctrl[i] = scaled_action[i]
        
        self.data.ctrl[self.action_dim] = 0.5 
        
        mujoco.mj_step(self.model, self.data)
        
        state = self.get_state()
        
        qpos = self.data.qpos[self.joint_ids]
        gripper_pos = self.get_gripper_pos()
        gripper_quat = self.get_gripper_orientation()
        
        reward, info = self.calculate_upward_reward(qpos, gripper_pos, gripper_quat, action)
        
        done = False
        success = False
        
        if (gripper_pos[2] > 0.3 and  
            info['upward'] > 0.95 and 
            info['config_distance'] < 0.5):  
            reward += 100.0 
            success = True
        
        if self.data.time > 15.0:  
            done = True
        
        if success:
            done = True
        
        if self.render and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        if self.render and self.viewer is not None:
            self.viewer.sync()
            time.sleep(0.01)
        
        return state, reward, done, info
    
    def reset(self):

        mujoco.mj_resetData(self.model, self.data)
        
        joint_ranges = {
            'joint1': (-2.618, 2.618),
            'joint2': (0, 3.14),
            'joint3': (-2.697, 0),
            'joint4': (-1.832, 1.832),
            'joint5': (-1.22, 1.22),
            'joint6': (-3.14, 3.14),
            'gripper': (-0.035, 0.035)
        }
        
        if np.random.random() < 0.3:
            for i, joint_name in enumerate(self.joint_names[:6]): 
                target = self.upward_config[i]
                noise = np.random.uniform(-0.5, 0.5)
                self.data.qpos[self.joint_ids[i]] = target + noise
        else:
            for i, joint_name in enumerate(self.joint_names):
                joint_range = joint_ranges[joint_name]
                self.data.qpos[self.joint_ids[i]] = np.random.uniform(
                    joint_range[0] + 0.1, 
                    joint_range[1] - 0.1
                )
        
        self.data.qpos[self.joint_ids[6]] = 0.0  
        
        mujoco.mj_step(self.model, self.data)
        
        return self.get_state()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()

class ImprovedSAC:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, hidden_dim=256).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim=256).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim=256).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        self.exploration_noise = 0.1
        self.noise_decay = 0.995
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
                return action.cpu().numpy()[0]
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(state)
                
                if self.exploration_noise > 0:
                    noise = torch.randn_like(action) * self.exploration_noise
                    action = torch.clamp(action + noise, -1, 1)
                
                self.exploration_noise *= self.noise_decay
                self.exploration_noise = max(self.exploration_noise, 0.01)
                
                return action.cpu().numpy()[0]
    
    def update(self, buffer, batch_size=256):
        if len(buffer) < batch_size:
            return 0, 0, 0
        
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        current_q1, current_q2 = self.critic(states, actions)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  
        self.critic_optimizer.step()
        
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0) 
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item(), alpha_loss.item()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)
    
    def forward(self, state):
        x = self.net(state)
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)  
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        
        x_t = normal.rsample()
        
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1_net(x), self.q2_net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(next_states),
            np.array(dones).reshape(-1, 1)
        )
    
    def __len__(self):
        return len(self.buffer)


def train_simple_sac(env, max_episodes=500, max_steps=200, batch_size=256, buffer_capacity=100000):
    set_seed(42)
    
    print(f"状态维度: {env.state_dim}, 动作维度: {env.action_dim}")
    
    agent = ImprovedSAC(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2
    )
    
    buffer = ReplayBuffer(buffer_capacity)
    
    episode_rewards = []
    episode_heights = []
    episode_upwards = []
    success_rates = []
    
    print("开始训练SAC算法...")
    print("目标：机械臂以最少动作向上伸直")
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_height = 0
        episode_upward = 0
        steps = 0
        success = False
        
        for step in range(max_steps):
            action = agent.select_action(state)
            
            next_state, reward, done, info = env.step(action)
            
            buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_height = max(episode_height, info['height'])
            episode_upward = max(episode_upward, info['upward'])
            steps += 1
            
            if len(buffer) > batch_size:
                critic_loss, actor_loss, alpha_loss = agent.update(buffer, batch_size)
            
            if done:
                success = (info['height'] > 0.3 and info['upward'] > 0.95)
                break
        
        episode_rewards.append(episode_reward)
        episode_heights.append(episode_height)
        episode_upwards.append(episode_upward)
        success_rates.append(1 if success else 0)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_height = np.mean(episode_heights[-10:])
            avg_upward = np.mean(episode_upwards[-10:])
            success_rate = np.mean(success_rates[-10:]) * 100
            
            print(f"Episode {episode + 1}/{max_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Height: {avg_height:.3f}, "
                  f"Avg Upward: {avg_upward:.3f}, "
                  f"Success Rate: {success_rate:.1f}%")
            
            if success_rate > 50 or avg_height > 0.25:
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'episode': episode,
                    'reward': avg_reward,
                    'height': avg_height,
                    'upward': avg_upward
                }, f'simple_sac_model_ep{episode+1}_.pth')
    
    print("训练完成!")
    return agent, episode_rewards, episode_heights, episode_upwards, success_rates


def test_upward_agent(env, agent, num_episodes=5):
    print("\n测试向上伸直智能体...")
    
    total_rewards = []
    final_heights = []
    final_upwards = []
    success_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        print(f"\n测试回合 {episode + 1}:")
        
        while not done and step_count < 200:
            action = agent.select_action(state, evaluate=True)
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if step_count % 50 == 0:
                print(f"  步骤 {step_count}: 高度={info['height']:.3f}, "
                      f"向上程度={info['upward']:.3f}, "
                      f"配置距离={info['config_distance']:.3f}")
            
            if env.render:
                time.sleep(0.01)
        
        success = (info['height'] > 0.3 and info['upward'] > 0.95)
        if success:
            success_count += 1
            print(f"  ✓ 成功！最终高度={info['height']:.3f}, 向上程度={info['upward']:.3f}")
        else:
            print(f"  ✗ 未成功。最终高度={info['height']:.3f}, 向上程度={info['upward']:.3f}")
        
        total_rewards.append(episode_reward)
        final_heights.append(info['height'])
        final_upwards.append(info['upward'])
    
    print(f"\n测试结果:")
    print(f"  成功次数: {success_count}/{num_episodes}")
    print(f"  平均奖励: {np.mean(total_rewards):.2f}")
    print(f"  平均最终高度: {np.mean(final_heights):.3f}")
    print(f"  平均最终向上程度: {np.mean(final_upwards):.3f}")


def main():
    print("Piper机器人向上伸直控制")
    print("=" * 50)
    print("目标：训练机械臂以最少动作向上伸直")
    print("=" * 50)
    
    print("\n请选择模式:")
    print("1: 训练模式 (训练向上伸直控制)")
    print("2: 测试模式 (加载预训练模型)")
    print("3: 演示模式 (随机动作观察)")
    
    choice = input("请输入选择 (1/2/3): ")
    
    if choice == "1":
        print("\n开始训练向上伸直控制...")
        env = PiperUpwardEnv(model_path="scene.xml", render=False)
        agent, rewards, heights, upwards, successes = train_simple_sac(
            env, 
            max_episodes=2000,  
            max_steps=100,
            batch_size=128
        )
        env.close()
        
        torch.save({
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'rewards': rewards,
            'heights': heights,
            'upwards': upwards,
            'successes': successes
        }, 'upward_sac_model_final_.pth')
        
        print("\n训练完成！最终模型已保存为 'upward_sac_model_final_.pth'")
        
        print("\n测试训练好的向上伸直智能体...")
        env = PiperUpwardEnv(model_path="scene.xml", render=True)
        test_upward_agent(env, agent, num_episodes=3)
        env.close()
    
    elif choice == "2":
        env = PiperUpwardEnv(model_path="scene.xml", render=True)

        agent = ImprovedSAC(
            state_dim=env.state_dim,
            action_dim=env.action_dim
        )

        try:
            checkpoint = torch.load("upward_sac_model_final_.pth", map_location=agent.device)
            agent.actor.load_state_dict(checkpoint['actor'])
            agent.critic.load_state_dict(checkpoint['critic'])
            print(f"加载向上伸直模型成功")
            if 'heights' in checkpoint:
                print(f"训练统计: 平均高度={np.mean(checkpoint['heights'][-50:]):.3f}, "
                      f"平均向上程度={np.mean(checkpoint['upwards'][-50:]):.3f}")
        except FileNotFoundError:
            print("未找到预训练模型，使用随机初始化的智能体")
        
        test_upward_agent(env, agent, num_episodes=10)
        env.close()
    
    else:
        env = PiperUpwardEnv(model_path="scene.xml", render=True)
        
        print("\n演示模式 - 观察机械臂状态")
        print("按Ctrl+C退出演示")
        
        try:
            for episode in range(2):
                state = env.reset()
                print(f"\n演示回合 {episode + 1}:")
                print(f"初始状态维度: {len(state)}")
                
                for step in range(50):
                    action = np.random.uniform(-0.5, 0.5, size=env.action_dim)
                    state, reward, done, info = env.step(action)
                    
                    if step % 10 == 0:
                        print(f"  步骤 {step}: 高度={info['height']:.3f}, "
                              f"向上程度={info['upward']:.3f}")
                    
                    if done:
                        break
                
                print(f"回合结束: 最终高度={info['height']:.3f}, 最终向上程度={info['upward']:.3f}")
                
        except KeyboardInterrupt:
            print("\n演示结束")
        finally:
            env.close()

if __name__ == "__main__":
    main()