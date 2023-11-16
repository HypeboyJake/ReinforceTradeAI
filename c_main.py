import torch
import pandas as pd
from tqdm import tqdm
from CoinTradingEnv import CoinTradingEnv 
from sharing.agent import Agent 
from sharing.visualization import create_folder, plot_all_charts

'''
파일의 형식은 0번째 날짜, 3번째 종가를 무조건 지켜주셔야합니다.
The file format must strictly follow: 
date as the 0th element, and closing price as the 3rd element.
'''

"실제 파일경로를 입력해주세요"
"Please enter the actual file path."
real_data = pd.read_csv("file_path")


#"data는 시각화에만 사용되고 학습할 때 사용하지 않습니다"
#"The data is used only for visualization and not for training purposes"
"write the correct feature name"
data = real_data.drop(['Date'], axis=1)
#data = real_data.drop(['Time'], axis=1)

"파일에 결측치가 있다면 각주를 풀어주세요.(평균으로 대체)"
"If there are missing values in the file, please annotate them. (Replace with average)"
# data.fillna(data.mean(), inplace=True) # NAN값 평균으로 대체
data = data.values

env = CoinTradingEnv(data)  

"파일의 피쳐수를 넣어주세요 (피쳐수, )형식을 유지해주세요."
"Please enter the number of features in the file, and maintain the format (number of features, )."
"ex ) state_dim = (7,)"
state_dim = 

action_dim = env.action_space.n
agent = Agent(input_dim=state_dim, output_dim=action_dim, epsilon=0.3 , gamma= 0.99) 

"학습을 몇번 반복 하시겠습니까?"
"How many times would you like to repeat the training?"
"ex ) num_episodes = 10000 "
num_episodes = 10000

# 학습 루프
for episode in tqdm(range(num_episodes+1)):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    total_reward = 0
    holdings = [] 
    actions = []  
    rewards = []  

    done = False
    while not done:
        action = agent.select_action(state) 
        next_state, reward, done, info = env.step(action)  
        next_state = torch.tensor(next_state, dtype=torch.float32) 
        final_action = info.get('final_action')  
        agent.learn(state, action, reward, next_state, done) 

        holdings.append(env.current_coin)  
        actions.append(final_action)  
        rewards.append(reward)  
        total_reward += reward

    # "몇번의 학습마다 시각화파일을 저장할까요?"
    # "After how many training iterations should a visualization file be saved?"
    if episode % 1000 == 0:
        folder_name = create_folder(episode)

        roe_values = env.roe_per_step  
        plot_all_charts(real_data, holdings, env.data, actions, roe_values, folder_name, episode)

    print(f"Episode {episode}: Total Reward: {total_reward}%")

# save model
torch.save(agent.model.state_dict(), 'jrl.pth')