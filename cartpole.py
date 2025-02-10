import gym 
import time
env = gym.make('CartPole-v1')
env.reset()
_, _, done, _= env.step(1)

while True:
    print(time.perf_counter())
    env.render()
    print(env.state)