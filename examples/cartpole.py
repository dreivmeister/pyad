import math
import matplotlib.pyplot as plt
import random
import gymnasium as gym
from pyad import optim
from pyad.new_core import Tensor, smooth_l1_loss
from pyad.nn import Module, LinearLayer
import numpy as np

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1")


# Get the state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Define the policy network
class Policy(Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = LinearLayer(state_size, 32)
        self.fc2 = LinearLayer(32, action_size)

    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x.softmax(axis=-1)
    
    def parameters(self):
        return [*self.fc1.parameters(), *self.fc2.parameters()]

policy = Policy(state_size, action_size)
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# live-plot setup (non-blocking)
plt.ion()
scores = []
fig, ax = plt.subplots(figsize=(8,4))
line, = ax.plot([], [], '-o', markersize=3)
ax.set_xlabel("Episode")
ax.set_ylabel("Score")
ax.set_title("Training scores")
ax.grid(True)
plt.show(block=False)

# Define the update policy function
def update_policy(rewards, log_probs, optimizer):
    log_probs = Tensor.stack(log_probs)
    loss = -Tensor.mean(log_probs) * (sum(rewards) - 15)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
gamma = 0.99


# Training loop
for episode in range(5000):
    state, _ = env.reset()
    done = False
    score = 0
    log_probs = []
    rewards = []
    
    while not done:
        # Select action
        state = Tensor(state, dtype=np.float32).reshape((1, -1))
        probs = policy(state)
        action = np.random.multinomial(1, probs.data.flatten()).argmax() # Brrrr        
        log_prob = probs[0, action].log()

        # Take step
        next_state, reward, done, _, _ = env.step(action)
        score += reward
        
        rewards.append(reward)
        log_probs.append(log_prob)
        state = next_state

    # Update policy
    #print(f"Episode {episode}: {score}")
    update_policy(rewards, log_probs, optimizer)
    
    # update live plot
    scores.append(score)
    line.set_xdata(range(len(scores)))
    line.set_ydata(scores)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)
    