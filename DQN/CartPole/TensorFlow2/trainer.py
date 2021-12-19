import gym

from agent import DQNAgent
from utils import reward_plotter

def main():
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env)
    history = agent.train(n_episodes=500)
    agent.save()
    reward_plotter(history,10)

if __name__ == "__main__":
    main()
