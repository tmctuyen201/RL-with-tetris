import argparse
import torch
import cv2
from src.tetris import Tetris


class Config:
    def __init__(self):
        # Define the configuration parameters with default values
        self.width = 10  # The common width for all images
        self.height = 20  # The common height for all images
        self.block_size = 30  # Size of a block
        self.fps = 300  # Frames per second
        self.saved_path = "trained_models"  # Path to save trained models
        self.output = "output.mp4"  # Output file for the game


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model = torch.load("{}/tetris".format(opt.saved_path))
    else:
        model = torch.load("{}/tetris".format(opt.saved_path),
                           map_location=lambda storage, loc: storage)
    model.eval()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()
    if torch.cuda.is_available():
        model.cuda()
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
                          (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True, video=out)

        if done:
            out.release()
            break


if __name__ == "__main__":
    opt = Config()
    test(opt)
