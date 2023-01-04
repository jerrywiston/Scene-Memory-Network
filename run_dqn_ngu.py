import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gym
import argparse
import time
import os
import json
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import cv2

from maze3d import maze
import ngu.maze_intrinsic_env
from rl import dqn, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(state, img_size, th=0.4):
    state = np.array(Image.fromarray(state).resize(img_size,Image.BILINEAR))
    state = state.astype(np.float) / 255.
    state = state.transpose(2,0,1)
    return state

def epsilon_compute(frame_id, epsilon_max=1, epsilon_min=0.2, epsilon_decay=100000):
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-frame_id / epsilon_decay)

def save_video(img_buffer, fname, video_path="video"):
    size = (img_buffer[0].shape[1], img_buffer[0].shape[0])
    out = cv2.VideoWriter(os.path.join(video_path, fname), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
    for i in range(len(img_buffer)):
        out.write(cv2.cvtColor(img_buffer[i], cv2.COLOR_BGR2RGB))
    out.release()

def train(env, agent, stack_frames, img_size, exp_path="experiments_rl", eps_steps=1000, max_steps=1000000):
    total_step = 0
    episode = 0

    save_path = os.path.join(exp_path, "save")
    video_path = os.path.join(exp_path, "video")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    train_record = []
    eval_record = []
    while True:
        # Reset environment.
        state, info = env.reset()
        state = preprocess(state, img_size=img_size)
        state = state.repeat(stack_frames, axis=0)
        state_dict = {"obs":state}

        # Initialize information.
        step = 0
        total_reward = 0
        loss = 0.

        # One episode.
        while True:
            # Select action.
            epsilon = epsilon_compute(total_step)
            action = agent.choose_action(state_dict, epsilon)

            # Get next stacked state.
            state_next, reward, done, info = env.step(action)
            state_next = preprocess(state_next, img_size=img_size)
            state_next = np.concatenate([state_next, state[1*3:]], 0)
            state_next_dict = {"obs":state_next}

            # Store transition and learn.
            agent.store_transition(state_dict, action, reward, state_next_dict, done)
            #env.render(show=True)
            #cv2.waitKey(10)

            if total_step > 4*agent.batch_size:
                loss = agent.learn()

            state_dict = state_next_dict
            step += 1
            total_step += 1
            total_reward += reward

            if total_step % 100 == 0 or step>eps_steps or done:
                print('\rEpisode: {:3d} | Step: {:3d} / {:3d} | Reward: {:.3f} / {:.3f} | Loss: {:.3f} | Epsilon: {:.3f}'\
                    .format(episode, step, total_step, reward, total_reward, loss, epsilon), end="")
            
            if total_step % 10000 == 0:
                print("\nSave Model ...")
                agent.save_model(path=save_path)
                print("Generate GIF ...")
                img_buffer, eval_total_reward, score = play(env, agent, stack_frames, img_size)
                eval_record.append({"eps":episode, "step": total_step, "score":eval_total_reward})
                with open(model_path+'eval_record.json', 'w') as file:
                    json.dump(eval_record, file)
                save_video(img_buffer, "train_" + str(total_step).zfill(6) + ".avi", video_path)
                print("Score:", score)
                print("Done !!")

            if done or step>=eps_steps:
                train_record.append({"eps":episode, "step": total_step, "score":total_reward})
                with open(model_path+'train_record.json', 'w') as file:
                    json.dump(train_record, file)
                episode += 1
                print()
                break
        
        if total_step > max_steps:
            break

def play(env, agent, stack_frames, img_size, eps_steps=1000, render=False, level_path=None):
    img_buffer = []

    # Reset environment.
    if level_path is None:
        state, info = env.reset()
    else:
        state, info = env.reset_from_file(level_path)
        
    state = preprocess(state, img_size=img_size)
    state = state.repeat(stack_frames, axis=0)
    state_dict = {"obs":state}
    img_buffer.append(env.render()[:,:,::-1])

    # Initialize information.
    step = 0
    frame = 0
    total_reward = 0
    loss = 0.

    # One episode.
    while True:
        # Select action.
        action = agent.choose_action(state_dict, 0.2)

        # Get next stacked state.
        state_next, reward, done, info = env.step(action)
        state_next = preprocess(state_next, img_size=img_size)
        state_next = np.concatenate([state_next,state[1*3:]], 0)
        state_next_dict = {"obs":state_next}

        img_buffer.append(env.render()[:,:,::-1])
        if render:
            env.render() # Can't use in colab.

        # Store transition and learn.
        total_reward += reward
        print('\rStep: {:3d} | Reward: {:.3f} / {:.3f}'\
            .format(step, reward, total_reward), end="")
            
        state_dict = state_next_dict
        step += 1
        if done or step>eps_steps:
            score = info["score"]
            print()
            break

    return img_buffer, total_reward, score

if __name__ == "__main__":
    ############ Parser ############
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    #parser.add_argument('--type', '-t', nargs='?', type=str, default="train", help='train / test')
    parser.add_argument('--exp_name', nargs='?', type=str, default="rltest" ,help='Experiment name.')
    parser.add_argument('--n_items', '-n', nargs='?', type=int, default=15 ,help='Number of items.')
    test = parser.parse_args().test
    exp_name = parser.parse_args().exp_name
    n_items = parser.parse_args().n_items

    ############ Create Folder ############
    if not test:
        now = datetime.datetime.now()
        tinfo = "%d-%d-%d"%(now.year, now.month, now.day)
        exp_path = "experiments_rl/"
        model_path = os.path.join(exp_path, tinfo + "_" + exp_name + "/")

        video_path = os.path.join(model_path, "video/")
        save_path = os.path.join(model_path, "save/")
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    ############ Create Env ############
    maze_obj = maze.MazeGridRandom2(size=(11,11), room_total=5)
    env = ngu.maze_intrinsic_env.MazeEpisodicIcmEnv(maze_obj, render_res=(64,64), n_items=n_items)
    stack_frames = 5
    img_size = (64,64)

    ############ Create Agent ############
    agent = dqn.DQNAgent(
        n_actions = 5,
        input_shape = [(stack_frames)*3, *img_size],
        qnet = models.QNetTSM,
        device = device,
        learning_rate = 2e-4, 
        reward_decay = 0.95,
        replace_target_iter = 1000, 
        memory_size = 5000,
        batch_size = 32,)

    if not test:
        train(env, agent, stack_frames, img_size, model_path, eps_steps=1000, max_steps=400000)
    else:
        agent.load_model(os.path.join(exp_name, "save"))
        eval_path = os.path.join(exp_name, "eval/")
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)
        score_rec = []
        for i in range(20):
            path = "level_save/" + str(i).zfill(4) + ".npz"
            img_buffer, eval_total_reward, score = play(env, agent, stack_frames, img_size, level_path=path)
            save_video(img_buffer, "test_" + str(i).zfill(4) + ".avi", eval_path)
            score_rec.append(score)
    
        print(score_rec)
        print(np.array(score_rec).mean(), "+/-", np.array(score_rec).std())
