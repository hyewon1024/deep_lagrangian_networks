import numpy as np
import gym
from IPython import display as ipythondisplay
from PIL import Image

def calculate_acceleration_cartpole(states, actions, env):
    x = states[:, 0]
    x_dot = states[:, 1]
    theta = states[:, 2]
    theta_dot = states[:, 3]
    torque = actions.squeeze()

    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (torque[:, 0] + env.polemass_length * theta_dot**2 * sintheta) / env.total_mass
    thetaacc = (env.gravity * sintheta - costheta * temp) / (
        env.length * (4.0 / 3.0 - env.masspole * costheta**2 / env.total_mass)
    )
    xacc = temp - env.polemass_length * thetaacc * costheta / env.total_mass

    return np.stack([xacc, thetaacc], axis=1)

def generate_rand_data_with_pd_control(env, ntrajs, traj_len, dt, kp=50, kr=10):
    assert traj_len > 1, "Trajectory length must be greater than 1"
    assert ntrajs >= 1, "Number of trajectories must be at least 1"

    # Trajectory data placeholders
    xs = np.zeros((ntrajs, traj_len, env.observation_space.shape[0]))
    uss = np.zeros((ntrajs, traj_len - 1,  2))

    for i in range(ntrajs):
        # Reset environment and initialize
        state = env.reset() 
        target = np.zeros_like(state)  # Target state: [0, 0, 0, 0] (modify if needed)

        xs[i, 0, :] = state  # Store initial state
        prev_error = target - state  # Initial error

        for t in range(traj_len - 1):
            # PD control calculation
            error = target - state
            error_derivative = (error - prev_error) / dt
            pd_control = kp * error + kr * error_derivative

            action = 0  # Action remains constant as per the description
            next_state, _, done, _ = env.step(action)

            xs[i, t + 1, :] = next_state
            uss[i, t, :] = np.array([action, 0]) #fx, Ftheta

            state = next_state
            prev_error = error

            if state[0] < -2.4 or state[0] > 2.4:
                break

    # Prepare data for output
    states = xs[:, :-1, :].reshape(-1, xs.shape[-1])
    torqueset = uss.reshape(-1, uss.shape[-1])

    # Extract q, qdot
    q = states[:, [0, 2]]  # Position variables
    qdot = states[:, [1, 3]]  # Velocity variables

    # Calculate qddot
    qddot = calculate_acceleration_cartpole(states, torqueset, env)

    return q, qdot, qddot, torqueset
# Example usage
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    ntrajs = 12
    traj_len = 3
    dt = 0.05
    q, qdot, qddot, torque = generate_rand_data_with_pd_control(env, ntrajs, traj_len, dt, kp=50, kr=10)

    print("q shape:", q.shape)
    print("qdot shape:", qdot.shape)
    print("qddot shape:", qddot.shape)
    print("torque shape:", torque.shape)