import os
import random
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from rl.carla_env import CarlaEnv
from rl.ppo import PPO, ReplayBuffer
from rl.rl_utils import Normalization


if __name__ == '__main__':

    config = OmegaConf.load("utils/config_train.yaml")
    rl_cfg = config.rl

    log_dir = "runs"
    writer = SummaryWriter(log_dir)

    seed = 0
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = CarlaEnv(config)

    replay_buffer = ReplayBuffer(rl_cfg)
    rl = PPO(rl_cfg)

    if rl_cfg.use_state_norm:
        state_norm = Normalization(shape=rl_cfg.state_dim)

    step = 0

    for episode in range(0, rl_cfg.max_episode):
        ep_reward  = 0
        ep_len = 0
        done = False

        state = env.reset()

        if rl_cfg.use_state_norm:
            state = state_norm(state)

        while not done:
            env.render()
            action, action_logprob = rl.choose_action(state, True)
            next_state, reward, done, t, info = env.step(action[0])

            if rl_cfg.use_state_norm:
                next_state = state_norm(next_state)

            reach_capacity = replay_buffer.store((state, action, action_logprob, next_state, reward, done))

            if reach_capacity:
                print('update')
                rl.update(replay_buffer)

            state = next_state

            ep_reward += reward
            ep_len += 1
            step += 1

            writer.add_scalar("step/reward",reward,step)

            if done or t:
                break
        ep_avg_reward = ep_reward / ep_len
        writer.add_scalar("episode/reward", ep_reward, episode)
        writer.add_scalar("episode/length", ep_len, episode)
        writer.add_scalar("episode/average_reward", ep_avg_reward, episode)
        print(f"Episode {episode} | Average Reward: {ep_avg_reward:.2f}| Reward: {ep_reward:.2f} | Len: {ep_len}")

        if episode % rl_cfg.interval == 0:
            rl.save_model(os.path.join(log_dir, "model"),episode)

    writer.close()


