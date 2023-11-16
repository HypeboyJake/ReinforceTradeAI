import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class A2C(nn.Module):
    def __init__(self, input_shape, output_dim, shared_network=None):
        super().__init__()
        self.shared_network = shared_network
        self.flatten = nn.Flatten()
        self.input_dim = self._get_flat_input_dim(input_shape)
        
        self.policy_head = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=-1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  
        )

    def _get_flat_input_dim(self, input_shape):
        sample_input = torch.rand(1, *input_shape)
        sample_flattened = self.flatten(sample_input)
        return sample_flattened.shape[1]

    def forward(self, x):
        x = x.unsqueeze(0) if x.dim() == 1 else x
        if self.shared_network:
            x = self.shared_network(x)
        else:
            x = self.flatten(x) if x.dim() > 1 else x
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class PPO(nn.Module):
    def __init__(self, input_dim, output_dim, shared_network=None):
        super().__init__()
        self.shared_network = shared_network
        
        if isinstance(input_dim, tuple):
            input_dim = np.prod(input_dim)
        
        self.policy_head = nn.Sequential(
            nn.Linear(128 if shared_network else input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=-1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128 if shared_network else input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )


    def forward(self, x):
        if self.shared_network:
            x = self.shared_network(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def evaluate_actions(self, states, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long)

        policy, values = self(states)

        dist = torch.distributions.Categorical(policy)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values


class DDQN(nn.Module):
    def __init__(self, input_dim, output_dim, shared_network=None):
        super().__init__()
        self.shared_network = shared_network
        
        if isinstance(input_dim, tuple):
            input_dim = np.prod(input_dim)
        
        self.network_head = nn.Sequential(
            nn.Linear(128 if shared_network else input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        if self.shared_network:
            x = self.shared_network(x)
        x = self.network_head(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, shared_network=None):
        super().__init__()
        self.shared_network = shared_network
        if isinstance(input_dim, tuple):
            input_dim = np.prod(input_dim)
        self.lstm = nn.LSTM(input_size=128 if self.shared_network else input_dim,
                            hidden_size=256,
                            batch_first=True)
        self.actor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x, hidden_state=None):
        if self.shared_network:
            x = self.shared_network(x)
        x = x.unsqueeze(0) if x.dim() == 2 else x
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        lstm_out = lstm_out[:, -1, :]
        log_probs = self.actor_head(lstm_out)
        values = self.critic_head(lstm_out)
        return log_probs, values, hidden_state
    

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, shared_network=None):
        super().__init__()
        self.shared_network = shared_network
        if isinstance(input_dim, tuple):
            input_dim = np.prod(input_dim)
        self.network_head = nn.Sequential(
            nn.Linear(128 if shared_network else input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.Dropout(p=0.1),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        if self.shared_network:
            x = self.shared_network(x)
        x = self.network_head(x)
        return x

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, shared_network=None):
        super().__init__()
        self.shared_network = shared_network

        if isinstance(input_dim, tuple):
            input_dim = np.prod(input_dim)
        
        self.network_head = nn.Sequential(
            nn.Linear(128 if shared_network else input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.Dropout(p=0.1),
            nn.Linear(32, output_dim),
        )
    def forward(self, x):
        if self.shared_network:
            x = self.shared_network(x)
        x = self.network_head(x)
        return x

def calculate_dqn_loss(batch, model, target_model, gamma):
    states, actions, rewards, next_states, dones = batch
    current_q_values = model(states).gather(1, actions).squeeze(-1)
    next_q_values = target_model(next_states).max(1)[0].detach()
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))
    target_q_values = target_q_values.squeeze(-1) 
    loss = F.mse_loss(current_q_values, target_q_values)
    return loss


def calculate_a2c_loss(state, action, reward, next_state, done, log_probs, values, model, gamma):
    _, next_values = model(next_state)
    next_values = next_values.squeeze() 
    returns = reward + gamma * next_values * (1 - done)
    advantages = returns - values.squeeze()
    policy_loss = -(log_probs * advantages.detach()).mean()
    values = values.squeeze()
    returns = returns.detach().squeeze()
    if not values.size():
        values = values.unsqueeze(0)
    if not returns.size():
        returns = returns.unsqueeze(0)
    value_loss = F.mse_loss(values, returns.detach())
    loss = policy_loss + value_loss
    return loss


def calculate_log_probs_and_values(model, states, actions):
    if not isinstance(actions, torch.Tensor):
        actions = torch.tensor([actions], dtype=torch.int64)
    policy, values = model(states)
    dist = torch.distributions.Categorical(policy)
    log_probs = dist.log_prob(actions)
    return log_probs, values


def calculate_ppo_loss(states, actions, rewards, next_states, dones, old_log_probs, values, model, clip_ratio):
    new_log_probs, entropy, new_values = model.evaluate_actions(states, actions)
    ratios = torch.exp(new_log_probs - old_log_probs.detach())
    advantages = rewards + new_values * (1 - dones) - values.detach()
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(new_values.squeeze(), values.squeeze())
    loss = policy_loss + value_loss - entropy.mean() * 0.01  
    return loss


def calculate_lstm_loss(batch, model, value_loss_coef, entropy_coef):
    states, actions, rewards, values, log_probs, dones, hidden_states = batch
    rewards = rewards.expand_as(values)
    dones_tensor = dones.to(dtype=torch.float32).view(1, -1)
    while dones_tensor.size(1) < values.size(1):
        dones_tensor = torch.cat((dones_tensor, dones_tensor), dim=1)
    dones_tensor = dones_tensor[:, :values.size(1)]
    advantages = rewards + (1 - dones_tensor) * values - values.detach()
    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = F.mse_loss(rewards + (1 - dones_tensor) * values, values) * value_loss_coef
    entropy = -(torch.exp(log_probs) * log_probs).mean() * entropy_coef
    loss = policy_loss + value_loss - entropy
    return loss




def calculate_dnn_loss(outputs, targets):
    loss = F.mse_loss(outputs, targets)
    return loss


def calculate_ddqn_loss(batch, model, target_model, gamma):
    states, actions, rewards, next_states, dones = batch
    if actions.dim() == 1:
        actions = actions.unsqueeze(-1)
    elif actions.dim() > 2:
        actions = actions.squeeze()

    current_q_values = model(states).gather(1, actions).squeeze(-1)
    next_actions = model(next_states).max(1)[1].unsqueeze(1)
    next_q_values = target_model(next_states).gather(1, next_actions).squeeze(-1)
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))
    target_q_values = target_q_values.squeeze(-1)
    loss = F.mse_loss(current_q_values, target_q_values)
    return loss



def optimize_dqn(model, target_model, optimizer, batch, gamma):
    loss = calculate_dqn_loss(batch, model, target_model, gamma)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def optimize_a2c(model, optimizer, batch, gamma):
    state, action, reward, next_state, done, log_probs, values = batch
    loss = calculate_a2c_loss(state, action, reward, next_state, done, log_probs, values, model, gamma)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def optimize_ppo(model, optimizer, batch, clip_ratio):
    states, actions, rewards, next_states, dones, old_log_probs, ppo_values = batch
    loss = calculate_ppo_loss(states, actions, rewards, next_states, dones, old_log_probs, ppo_values, model, clip_ratio)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def optimize_ddqn(model, target_model, optimizer, batch, gamma):
    states, actions, rewards, next_states, dones = batch
    assert states.dim() == 2 and next_states.dim() == 2, "States and next_states must have a batch dimension"
    
    loss = calculate_ddqn_loss(batch, model, target_model, gamma)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def optimize_dnn(model, optimizer, batch):
    predicted_q_values, target_values = batch
    loss = calculate_dnn_loss(predicted_q_values, target_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def optimize_lstm(model, optimizer, batch, value_loss_coef, entropy_coef):
    loss = calculate_lstm_loss(batch, model, value_loss_coef, entropy_coef)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()