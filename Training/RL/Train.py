import os
import time

from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor

from Training.RL.Simulation import Simulator
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


if __name__ == '__main__':
    env = Simulator('family', verbose=True)
    env = Monitor(env)

    models_dir = f"models/{int(time.time())}/"
    logdir = f"logs/{int(time.time())}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    model.learn(total_timesteps=10, progress_bar=True)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=2)
    print(f"Mean reward: {mean_reward}")
