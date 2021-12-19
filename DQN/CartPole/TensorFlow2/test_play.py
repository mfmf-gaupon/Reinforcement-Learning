import gym

from agent import DQNAgent

def main():
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env)
    agent.load()
    agent.test_play(n=3)

if __name__ == "__main__":
    main()
