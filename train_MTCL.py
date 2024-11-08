import argparse
import os
import shutil
from random import random, randint

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris


class Config:
    def __init__(self):
        """Initialize with default configuration settings."""
        self.width = 10
        self.height = 20
        self.block_size = 30
        self.batch_size = 512
        self.lr = 1e-3
        self.gamma = 0.99
        self.initial_epsilon = 1
        self.final_epsilon = 1e-3
        self.num_decay_epochs = 2000
        self.num_epochs = 3000
        self.save_interval = 1000
        self.replay_memory_size = 30000
        self.log_path = "./tensorboard"
        self.saved_path = "trained_models"

    def __str__(self):
        """Return a string representation of the configuration."""
        return (f"Config(width={self.width}, height={self.height}, block_size={self.block_size}, "
                f"batch_size={self.batch_size}, lr={self.lr}, gamma={self.gamma}, "
                f"initial_epsilon={self.initial_epsilon}, final_epsilon={self.final_epsilon}, "
                f"num_decay_epochs={self.num_decay_epochs}, num_epochs={self.num_epochs}, "
                f"save_interval={self.save_interval}, replay_memory_size={self.replay_memory_size}, "
                f"log_path={self.log_path}, saved_path={self.saved_path})")

    def get_config_as_tuple(self):
        """Return the configuration values as a tuple."""
        return (self.width, self.height, self.block_size, self.batch_size, self.lr, self.gamma,
                self.initial_epsilon, self.final_epsilon, self.num_decay_epochs, self.num_epochs,
                self.save_interval, self.replay_memory_size, self.log_path, self.saved_path)


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        model.cuda()

    epoch = 0
    while epoch < opt.num_epochs:
        state = env.reset()
        episode = []
        done = False
        if torch.cuda.is_available():
            state = state.cuda()

        while not done:
            epsilon = max(opt.final_epsilon, opt.initial_epsilon *
                          (opt.num_decay_epochs - epoch) / opt.num_decay_epochs)
            u = random()
            random_action = u <= epsilon

            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            if torch.cuda.is_available():
                next_states = next_states.cuda()

            model.eval()
            with torch.no_grad():
                predictions = model(next_states)[:, 0]

            model.train()
            if random_action:
                index = randint(0, len(next_steps) - 1)
            else:
                index = torch.argmax(predictions).item()

            action = next_actions[index]
            reward, done = env.step(action, render=True)

            if torch.cuda.is_available():
                state = state.cuda()

            episode.append((state, action, reward))

            state = next_states[index, :]

        returns = []
        G = 0
        for _, _, reward in reversed(episode):
            G = reward + opt.gamma * G
            returns.insert(0, G)

        states, actions, rewards = zip(*episode)
        states = torch.stack(states)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)

        if torch.cuda.is_available():
            states = states.cuda()
            returns = returns.cuda()

        predictions = model(states)
        loss = criterion(predictions, returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log results
        final_score = env.score
        final_tetrominoes = env.tetrominoes
        final_cleared_lines = env.cleared_lines
        print(f"Epoch: {epoch + 1}/{opt.num_epochs}, Score: {final_score}, "
              f"Tetrominoes: {final_tetrominoes}, Cleared Lines: {final_cleared_lines}")

        writer.add_scalar('Train/Score', final_score, epoch)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, f"{opt.saved_path}/tetris_{epoch}")

        epoch += 1

    torch.save(model, f"{opt.saved_path}/tetris")


if __name__ == "__main__":
    opt = Config()
    train(opt)
