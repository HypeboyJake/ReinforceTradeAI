from sharing.models import DQN, A2C, PPO, LSTM, DNN, DDQN
from sharing.models import ( 
    optimize_dqn,
    optimize_ddqn,
    optimize_a2c,
    optimize_ppo,
    optimize_lstm,
    optimize_dnn,
    calculate_log_probs_and_values,
)
import torch
import torch.optim as optim
import numpy as np

# 모델과 옵티마이저 매핑
MODEL_MAPPING = {'A2C': A2C, 'PPO': PPO, 'DDQN': DDQN, 'LSTM': LSTM, 'DNN': DNN, 'DQN': DQN}
OPTIMIZER_MAPPING = {'Adam': optim.Adam, 'SGD': optim.SGD}

class Agent:
    def __init__(self, input_dim, output_dim, epsilon, gamma,
                # "사용할 모델을 선택하세요"
                # "Please select the model to use."
                 agent_type='PPO', 
                # "사용할 옵티마이저를 선택하세요"
                # "Please select the optimizer to use."
                 optimizer_type='Adam',
                 shared_network=False,
                 clip_ratio=0.2,
                 lr=0.0001, 
                 weight_decay=0, 
                 **kwargs):
        self.epsilon = epsilon
        self.gamma = gamma
        self.shared_network = shared_network
        self.clip_ratio = clip_ratio

        self.old_log_probs = []
        self.ppo_values = []

        self.value_loss_coef = 0.5  
        self.entropy_coef = 0.01  

        model_class = MODEL_MAPPING.get(agent_type)
        if not model_class:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        self.model = model_class(input_dim, output_dim, shared_network)

        optimizer_class = OPTIMIZER_MAPPING.get(optimizer_type)
        if not optimizer_class:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        self.optimizer = optimizer_class(self.model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)

        if agent_type in ['DQN', 'DDQN']:
            self.target_model = model_class(input_dim, output_dim, shared_network)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
        
    def select_action(self, state):
        # 0 : Buy, 1 : Sell, 3 : Hold
        action_space = [0, 1, 2] 
        state = state.float()

        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(action_space)
        else:
            if isinstance(self.model, DQN):
                with torch.no_grad():
                    model_output = self.model(state)
                    max_value, max_index = model_output.max(0)  
                    return max_index.item()
            elif isinstance(self.model, DDQN):
                with torch.no_grad():
                    state = state.unsqueeze(0).float()  
                    q_values = self.model(state) 
                    return q_values.max(1)[1].item()
            elif isinstance(self.model, A2C):
                with torch.no_grad():
                    state = state.unsqueeze(0).float() 
                    policy, value = self.model(state)
                    dist = torch.distributions.Categorical(policy)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    self.last_log_prob = log_prob  
                    self.last_value = value  
                    return action.item()
            elif isinstance(self.model, PPO):
                with torch.no_grad():
                    policy, value = self.model(state)
                    dist = torch.distributions.Categorical(policy)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    self.old_log_probs.append(log_prob)
                    self.ppo_values.append(value)
                    return action.item()
            elif isinstance(self.model, LSTM):
                with torch.no_grad():
                    state = state.unsqueeze(0).unsqueeze(1)
                    log_probs, _, _ = self.model(state)
                    probs = torch.exp(log_probs)
                    action = torch.multinomial(probs, 1).item()
                    return action
            elif isinstance(self.model, DNN):
                with torch.no_grad():
                    model_output = self.model(state)
                    max_value, max_index = model_output.max(0) 
                    return max_index.item()
            else:
                raise ValueError(f"Unsupported model type: {type(self.model)}")
            
    def learn(self, state, action, reward, next_state, done):
        if isinstance(self.model, DQN):
            state = state.unsqueeze(0) if state.dim() == 1 else state
            next_state = next_state.unsqueeze(0) if next_state.dim() == 1 else next_state
            action = torch.tensor([action], dtype=torch.int64).unsqueeze(0)
            reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
            done = torch.tensor([done], dtype=torch.float32).unsqueeze(0)
            dqn_batch = (state, action, reward, next_state, done)
            optimize_dqn(self.model, self.target_model, self.optimizer, dqn_batch, self.gamma)
        elif isinstance(self.model, DDQN):
            state = state.unsqueeze(0) if state.dim() == 1 else state
            next_state = next_state.unsqueeze(0) if next_state.dim() == 1 else next_state
            action = torch.tensor([action], dtype=torch.int64).unsqueeze(0)
            reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
            done = torch.tensor([done], dtype=torch.float32).unsqueeze(0)
            ddqn_batch = (state, action, reward, next_state, done)
            optimize_ddqn(self.model, self.target_model, self.optimizer, ddqn_batch, self.gamma)
        elif isinstance(self.model, A2C):
            reward_tensor = torch.tensor(reward).float()
            done_tensor = torch.tensor(done, dtype=torch.float32)
            state = state.float()
            action_tensor = torch.tensor(action).long()
            next_state = next_state.float()
            done_tensor = done_tensor.unsqueeze(0) if done_tensor.dim() == 0 else done_tensor
            log_probs, values = calculate_log_probs_and_values(self.model, state, action_tensor)
            a2c_batch = (state, action_tensor, reward_tensor, next_state, done_tensor, log_probs, values)
            optimize_a2c(self.model, self.optimizer, a2c_batch, self.gamma)
        elif isinstance(self.model, PPO):
            if not self.old_log_probs or not self.ppo_values:
                return
            old_log_probs = torch.stack(self.old_log_probs)
            ppo_values = torch.stack(self.ppo_values)
            ppo_batch = (state, action, reward, next_state, done, old_log_probs, ppo_values)
            optimize_ppo(self.model, self.optimizer, ppo_batch, self.clip_ratio)
            self.old_log_probs = []
            self.ppo_values = []
        elif isinstance(self.model, LSTM):
            state = state.clone().detach().unsqueeze(0).float() if not state.requires_grad else state.unsqueeze(0).float()
            next_state = next_state.clone().detach().unsqueeze(0).float() if not next_state.requires_grad else next_state.unsqueeze(0).float()
            action = torch.tensor([action], dtype=torch.int64).unsqueeze(0)
            reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
            done = torch.tensor([done], dtype=torch.float32).unsqueeze(0)
            log_probs, values, hidden_state = self.model(state)
            lstm_batch = (state, action, reward, next_state, done, log_probs, values)
            optimize_lstm(self.model, self.optimizer, lstm_batch, self.value_loss_coef, self.entropy_coef)
        elif isinstance(self.model, DNN):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) if not isinstance(state, torch.Tensor) else state
            next_state = next_state.unsqueeze(0) if next_state.dim() == 1 else next_state
            reward = torch.tensor([reward], dtype=torch.float32)
            done = torch.tensor([done], dtype=torch.float32)
            outputs = self.model(state)
            next_state_values = self.model(next_state).max(1)[0].detach()
            target_value = reward + (1 - done) * self.gamma * next_state_values
            target_value = target_value.unsqueeze(-1)  
            action = torch.tensor(action, dtype=torch.int64).unsqueeze(-1)
            predicted_q_value = outputs[action].unsqueeze(-1)
            optimize_dnn(self.model, self.optimizer, (predicted_q_value, target_value))
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")