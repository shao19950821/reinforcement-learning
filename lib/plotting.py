import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


def plot_blackjack_policy(P, title="Policy"):
    """
    Plots the policy .
    Args:
        P: the policy to display, map from state to action value
    """
    x_range = np.arange(1, 10) # dealer showing card
    y_range = np.arange(11, 21) # planer sum


    def plot_policy(D, title):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        y_0 = [player_sum for _, (player_sum , _, action) in enumerate(D) if action == 0]
        x_0 = [dealer_showing for _, (_, dealer_showing, action) in enumerate(D) if action == 0]
        y_1 = [player_sum for  _, (player_sum, _, action) in enumerate(D) if action == 1]
        x_1 = [dealer_showing for _, (_, dealer_showing, action) in enumerate(D) if action == 1]
        ax.scatter(x_0, y_0, c='r', marker='$H$', alpha=0.5)
        ax.scatter(x_1, y_1, c='b', marker='$S$', alpha=0.5)
        ax.axis([0,11,11,22])
        ax.set_ylabel('Player Sum')
        ax.set_xlabel('Dealer Showing')
        ax.set_title(title)
        ax.grid(True)
        fig.tight_layout()
        plt.show()

    # sorted by dealer_showing
    P_0 = sorted([(player_sum, dealer_showing, action) for (player_sum, dealer_showing, useable_ace), action in P.items() if useable_ace == True], key = lambda x:(x[1],x[2],x[0]))
    P_1 = sorted([(player_sum, dealer_showing, action) for (player_sum, dealer_showing, useable_ace), action in P.items() if useable_ace == False], key = lambda x:(x[1],x[2],x[0]))
    #print(f"P_0={P_0}")
    #print(f"P_1={P_1}")
    plot_policy(P_0, "{} (Usable Ace)".format(title))
    plot_policy(P_1, "{} (No Usable Ace)".format(title))
    #plot_policy(P, "{} (Usable Ace)".format(title))


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3
