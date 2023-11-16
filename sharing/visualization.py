import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os


def create_folder(epoch):
    folder_name = f'epoch_{epoch}'
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def plot_all_charts(old_data, holdings, data, actions, roe_values, folder_name, epoch):
    ohlcv_df = pd.DataFrame(old_data)
    ohlcv_df.iloc[:, 0] = pd.to_datetime(ohlcv_df.iloc[:, 0])
    ohlcv_df.set_index(ohlcv_df.iloc[:, 0], inplace=True)

    plt.figure(figsize=(15, 12))

    plt.subplot(4, 1, 1)
    plt.plot(ohlcv_df.iloc[:, 4], label='Close Price')  
    plt.title('Close Price Line Chart')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    
    plt.subplot(4, 1, 2)
    plt.plot(data[:-1, 3], label='Close Price', linewidth=1)
    for i, action in enumerate(actions):
        color = 'blue' if action == 0 else 'red' if action == 1 else 'yellow'
        marker = '^' if action == 0 else 'v' if action == 1 else 'o'
        plt.scatter(i, data[i, 3], color=color, marker=marker, alpha=0.7)
    plt.title('Agent Actions with Close Price')
    plt.xlabel('Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)


    plt.subplot(4, 1, 3)
    plt.plot(holdings, label='Holdings')
    plt.title('Holdings per Step')
    plt.xlabel('Step')
    plt.ylabel('Amount of Held Coins')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(roe_values, label='ROE')
    plt.title('Return on Equity over Steps')
    plt.xlabel('Step')
    plt.ylabel('ROE (%)')
    plt.legend()

    plt.tight_layout() 
    plt.savefig(f'{folder_name}/combined_chart_{epoch}.png')
    plt.close()
