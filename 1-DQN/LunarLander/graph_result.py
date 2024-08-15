import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np


def plot_rewards(filename):
    """Plot the rewards obtained during each epoch of training

    Args:
        filename (_type_): file containing the saved rewards
    """
    episodes, rewards = [], []

    with open(filename, mode='r') as file:
        for line in file:
            # Utiliser une expression régulière pour extraire les valeurs
            match = re.match(r"Epoch (\d+) : Reward : ([\-\+]?\d+\.\d+) / Loss : ([\-\+]?\d+\.\d+) ;", line.strip())
            if match:
                i = int(match.group(1))
                n = float(match.group(2))

                if n == -1:
                    n = 0
            
            episodes.append(i)
            rewards.append(n)

    short = 20
    medium = int(5 * short)
    large = int(2 * medium)

    # Calcul de la moyenne mobile avec Pandas
    rewards_series = pd.Series(rewards)
    rolling_mean = rewards_series.rolling(window=short).mean()
    large_rolling_mean = rewards_series.rolling(window=medium).mean()
    vlarge_rolling_mean = rewards_series.rolling(window=large).mean()
    #exp_rolling_mean = rewards_series.rolling(window=500).ewm(com=0.5).mean()

    # Plot des récompenses brutes
    plt.style.use('dark_background')
    plt.plot(episodes, rewards, label='Raw Rewards', color='red', alpha=0.3)
    
    # Plot de la moyenne mobile
    plt.plot(episodes, rolling_mean, label=f'Moving Average ({short})', color='yellow', alpha=0.5)
    plt.plot(episodes, large_rolling_mean, label=f'Moving Average ({medium})', color='yellow')
    # Plot de la moyenne mobile

    plt.plot(episodes, vlarge_rolling_mean, label=f'Moving Average ({large})', color='white')



    plt.grid(visible=None, which='major', axis='y', linestyle='--')

    plt.fill_between(episodes, large_rolling_mean, rolling_mean, where=(large_rolling_mean<rolling_mean), color="yellow", alpha=0.4, interpolate=True)
    plt.fill_between(episodes, large_rolling_mean, rolling_mean, where=(large_rolling_mean>rolling_mean), color="red", alpha=0.4, interpolate=True)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards during training')
    plt.legend()
    plt.show()

plot_rewards("Training_Data_1/rewards.txt")

#plot_rewards("Training_Data_2/rewards.txt")