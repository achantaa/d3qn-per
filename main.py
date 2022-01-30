import gym
import ale_py
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from models import D3QNAgent


MAX_FRAMES = 10000000  # Total number of frames to train on
EVAL_FREQUENCY = 200000  # Frequency of running the eval loop (in frames)
EVAL_FRAMES = 10000  # Number of frames to run eval loop for
MAX_EPISODE_LENGTH = 18000  # Maximum length of episodes (in frames)
REPLAY_MEM_START_SIZE = 50000  # Number of experiences in the buffer before starting training
REPLAY_MEM_MAX_SIZE = 1000000  # Size of the replay buffer
TARGET_UPDATE_FREQUENCY = 10000  # Number of actions after which target network has to be updated
MODEL_SAVE_FREQUENCY = 20000  # Frequency at which model should be saved


def train():
    frame_number = 0

    # For Tensorboard
    log_dir = "logs/"
    file_writer = tf.summary.create_file_writer(log_dir)
    file_writer.set_as_default()

    # Use v0 instead of v5 for frameskip compatibility with AtariPreprocessing Wrapper
    env = gym.make('MsPacmanNoFrameskip-v0')
    env = gym.wrappers.AtariPreprocessing(env, grayscale_newaxis=True, scale_obs=True)  # adds last dim, scales to 0..1
    input_shape = (env.screen_size, env.screen_size, 1)

    # Create Agent
    agent = D3QNAgent(0.0005,
                      0.99,
                      input_shape=input_shape,
                      action_space=env.action_space,
                      max_replay_buffer_size=REPLAY_MEM_MAX_SIZE,
                      max_frames=MAX_FRAMES)
    agent.compile_models()

    # Run till a total of MAX_FRAMES frames are seen
    while frame_number < MAX_FRAMES:
        epoch_frame_number = 0
        rewards = []
        loss_list = []

        # Train till EVAL_FREQUENCY frames, and then evaluate performance
        while epoch_frame_number < EVAL_FREQUENCY:
            episode_reward_sum = 0
            state = env.reset()

            # Each training episode lasts for MAX_EPISODE_LENGTH frames
            for _ in tqdm(range(MAX_EPISODE_LENGTH)):
                action = agent.choose_action(state=state, frame_number=frame_number)
                new_state, reward, done, info = env.step(action)
                frame_number += 1
                epoch_frame_number += 1
                episode_reward_sum += reward
                agent.remember(state, action, reward, new_state, done)

                # start learning after replay buffer is filled
                if frame_number > REPLAY_MEM_START_SIZE:
                    loss, td_errors = agent.train()
                    loss_list.append(loss)
                    tf.summary.scalar('loss', loss, step=frame_number)
                    if frame_number % MODEL_SAVE_FREQUENCY == 0:
                        model_filepath = r"<filepath-here>"
                        agent.main_QNet.save(model_filepath)
                        print(np.mean(loss_list[-100:]))
                    # update target net only after set number of actions are taken
                    if frame_number % TARGET_UPDATE_FREQUENCY == 0:
                        agent.update_target_net()

                # Break the loop when the game is over
                if done:
                    done = False
                    break
                state = new_state

            rewards.append(episode_reward_sum)
            tf.summary.scalar('reward', episode_reward_sum, step=len(rewards))

        tf.summary.scalar('avg_100_loss', np.mean(loss_list[-100:]), step=len(loss_list))
        tf.summary.scalar('avg_100_reward', np.mean(rewards[-100:]), step=len(rewards))

        # evaluate
        eval_frames = 0
        eval_rewards = []
        while eval_frames < EVAL_FRAMES:
            episode_eval_reward_sum = 0
            state = env.reset()
            for _ in tqdm(range(MAX_EPISODE_LENGTH)):
                action = agent.choose_action(state=state, frame_number=frame_number, evaluation=True)
                new_state, reward, done, info = env.step(action)
                eval_frames += 1
                episode_eval_reward_sum += reward
                state = new_state
                if done:
                    break
            eval_rewards.append(episode_eval_reward_sum)
        tf.summary.scalar('eval_rewards', np.mean(eval_rewards), step=frame_number)


if __name__ == '__main__':
    train()
