import torch
import torch.nn as nn
import torch.optim as optim

def cartpole_dynamics(state, force, params):
    """
    Compute the next state of the cart-pole system.
    """
    g = 9.81  # gravity
    m_cart, m_pole, l_pole = params
    x, x_dot, theta, theta_dot = state
    
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    total_mass = m_cart + m_pole
    pole_mass_length = m_pole * l_pole
    
    temp = (force + pole_mass_length * theta_dot**2 * sin_theta) / total_mass
    theta_acc = (g * sin_theta - cos_theta * temp) / (l_pole * (4/3 - m_pole * cos_theta**2 / total_mass))
    x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass
    
    return torch.tensor([x_dot, x_acc, theta_dot, theta_acc])

class CartPoleEnv:
    def __init__(self, dt=0.02, params=(1.0, 0.3, 0.5)):
        self.dt = dt
        self.params = torch.tensor(params, dtype=torch.float32)
        self.state = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    
    def step(self, action):
        next_state = self.state + self.dt * cartpole_dynamics(self.state, action, self.params)
        self.state = next_state
        return next_state

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
    
    def forward(self, state):
        return torch.tanh(self.fc(state)) * 20.0  # Force range [-20, 20]

def train():
    env = CartPoleEnv()
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    
    for episode in range(10000):
        env.state = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        total_loss = 0
        
        for t in range(100):
            state = env.state
            action = policy(state)
            next_state = env.step(action)
            
            loss = (next_state[0] - 1.0)**2 + (next_state[2] - torch.pi)**2 + action**2
            total_loss += loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {total_loss.item()}")

if __name__ == "__main__":
    train()
