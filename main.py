import gym
import numpy as np
from collections import namedtuple

from config import EPISODES, BATCH_SIZE, INIT_REPLAY_MEMORY_SIZE
from agent import DQNAgent
from plot import plot_learning_history

if __name__ == '__main__':
    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    env = gym.make('CartPole-v1')
    agent = DQNAgent(env)
    state = env.reset()[0]
    state = np.reshape(state, [1, agent.state_size])

    # Filling up the replay memory
    for i in range(INIT_REPLAY_MEMORY_SIZE):
        action = agent.choose_action(state)
        next_state, reward, done, *_, = env.step(action)
        next_state = np.reshape(next_state, [1, agent.state_size])
        agent.remember(Transition(state, action, reward, next_state, done))

        if done:
            state = env.reset()[0]
            state = np.reshape(state, [1, agent.state_size])
        else:
            state = next_state

    total_rewards, losses = [], []
    for e in range(EPISODES):
        state = env.reset()[0]
        if e % 10 == 0:
            env.render()
        state = np.reshape(state, [1, agent.state_size])
        for i in range(500):
            action = agent.choose_action(state)
            next_state, reward, done, *_ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(Transition(state, action, reward, next_state, done))

            state = next_state
            if e % 10 == 0:
                env.render()
            if done:
                total_rewards.append(i)
                print(f'Episode: {e}/{EPISODES}, Total reward: {i}')
                break
            loss = agent.replay(BATCH_SIZE)
            losses.append(loss)

    plot_learning_history(total_rewards)
