import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class MAPPOAgent:
    def __init__(self, agent_id, state_dim, action_dim, hidden_dim=128, learning_rate=1e-4):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=learning_rate)
        
        self.experience_buffer = {
            'states': deque(maxlen=1000),
            'actions': deque(maxlen=1000),
            'rewards': deque(maxlen=1000),
            'next_states': deque(maxlen=1000),
            'dones': deque(maxlen=1000),
            'log_probs': deque(maxlen=1000),
            'values': deque(maxlen=1000)
        }

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_logits = self.actor(state_tensor)
            value = self.critic(state_tensor)
        
        action_probs = torch.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()

    def store_experience(self, state, action, reward, next_state, done, log_prob, value):
        self.experience_buffer['states'].append(state)
        self.experience_buffer['actions'].append(action)
        self.experience_buffer['rewards'].append(reward)
        self.experience_buffer['next_states'].append(next_state)
        self.experience_buffer['dones'].append(done)
        self.experience_buffer['log_probs'].append(log_prob)
        self.experience_buffer['values'].append(value)

    def compute_gae_advantages(self, gamma=0.98, lambda_gae=0.95):
        states = list(self.experience_buffer['states'])
        rewards = list(self.experience_buffer['rewards'])
        values = list(self.experience_buffer['values'])
        dones = list(self.experience_buffer['dones'])
        
        advantages = []
        gae_advantage = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            td_error = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae_advantage = td_error + gamma * lambda_gae * (1 - dones[t]) * gae_advantage
            
            advantages.insert(0, gae_advantage)
        
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages

    def update_networks(self, gamma=0.98, lambda_gae=0.95, epsilon_clip=0.2, num_epochs=3):
        if len(self.experience_buffer['states']) == 0:
            return 0.0, 0.0
        
        states = torch.FloatTensor(np.array(list(self.experience_buffer['states'])))
        actions = torch.LongTensor(list(self.experience_buffer['actions']))
        old_log_probs = torch.FloatTensor(list(self.experience_buffer['log_probs']))
        rewards = torch.FloatTensor(list(self.experience_buffer['rewards']))
        
        advantages = self.compute_gae_advantages(gamma, lambda_gae)
        returns = advantages + torch.FloatTensor(list(self.experience_buffer['values']))
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        for epoch in range(num_epochs):
            action_logits = self.actor(states)
            action_probs = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            values = self.critic(states).squeeze()
            critic_loss = nn.MSELoss()(values, returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        self.experience_buffer['states'].clear()
        self.experience_buffer['actions'].clear()
        self.experience_buffer['rewards'].clear()
        self.experience_buffer['next_states'].clear()
        self.experience_buffer['dones'].clear()
        self.experience_buffer['log_probs'].clear()
        self.experience_buffer['values'].clear()
        
        return total_actor_loss / num_epochs, total_critic_loss / num_epochs

    def get_state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])

