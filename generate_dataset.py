import numpy as np
import gym
from IPython import display as ipythondisplay
from PIL import Image
from data.continuous_cartpole import ContinuousCartPoleEnv
import random

def real_lagrangian_matrix(states, torqueset, env):
    #Calculate lagrangian metrix (real)
    x = states[:, 0]
    x_dot = states[:, 1]
    theta = states[:, 2]
    theta_dot = states[:, 3]
    N=x.shape[0]

    m11= np.full((N,), env.total_mass)
    m12= env.length * env.masspole * np.cos(theta)
    m21= m12
    m22=np.full((N,), env.masspole * (env.length ** 2) + env.polemass_length)

    mass_matrix = np.stack([m11, m12, m21, m22], axis=-1).reshape(N, 2, 2)
    
    c12= -env.length * env.masspole * theta_dot * np.sin(theta)
    zero=np.zeros((N,))
    corrioli_matrix= np.stack([zero, c12, zero, zero], axis=-1).reshape(N, 2, 2)

    gravitational_term = np.stack([zero, -env.masspole * env.gravity * env.length * np.sin(theta)], axis=-1).reshape(N, 2, 1)

    generalized_force = torqueset.reshape(N, 2, 1)
    return  mass_matrix, corrioli_matrix, gravitational_term, generalized_force

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

def generate_rand_data_with_pd_control(env, ntrajs, traj_len, dt, kp=50, kr=100):
    assert traj_len > 0, "Trajectory length must be greater than 1"
    assert ntrajs >= 1, "Number of trajectories must be at least 1"

    # Trajectory data placeholders
    xs = np.zeros((ntrajs, traj_len, env.observation_space.shape[0]))
    uss = np.zeros((ntrajs, traj_len - 1,  2))
    traj_state= np.zeros((0, 4))
    traj_uss= np.zeros((0, 2))
    
    for i in range(ntrajs):
        # Reset environment and initialize
        state = env.reset() 
        target = np.copy(state)  # Target state: [0, 0, 0, 0] (modify if needed)
        target[2]=-3.14
        xs[i, 0, :] = state  # Store initial state
        t_length = traj_len-1
        for t in range(traj_len - 1):
            #env.render()
            x_position = env.state[0]
            
            if t % 2 ==0:
                if x_position < -2.4 + 0.5:
                    action = 0.5* env.action_space.sample()+0.5  #-1.0~0 
                elif x_position > 2.4 - 0.5:
                    action= 0.5* env.action_space.sample()-0.5  #-1.0~0
                else:
                    action = env.action_space.sample()
                next_state, _, done, _ = env.step(action)
            else:
                action = 0*env.action_space.sample()
                next_state, _, done, _ = env.step(action)
            # PD control calculation
 
            # prev_error = target - state  # Initial error
            # error = (prev_error[0], prev_error[2])
            # error_derivative = (prev_error[1], prev_error[3]) 
            # pd_control = kp * error + kr * error_derivative
            # action = (pd_control - np.mean(pd_control))/ np.std(pd_control) 
            # #      # Action remains constant as per the description
            # next_state, _, done, _ = env.step(action)
            # state= next_state
            
            xs[i, t + 1, :] = env.state #next_state
            uss[i, t, :] = np.array([env.force_mag * float(action), 0]) #np.array([0, 0]) #fx, Ftheta

            state = next_state #next_state
            if state[0] < -2.4 or state[0] > 2.4: #done state일 때 
                t_length= t
                print(f'traj length is {t_length}')
                break

        xs_trim= xs[i, :t_length, :] #마지막 요소 제외 
        us_trim =uss[i, :t_length, :]

        x= np.vstack((traj_state, xs_trim))
        u= np.vstack((traj_uss, us_trim))
        traj_state=x
        traj_uss = u

    # Prepare data for output
    states= traj_state
    torqueset= traj_uss
    #states = xs[:, :-1, :].reshape(-1, xs.shape[-1])
    #torqueset = uss.reshape(-1, uss.shape[-1])

    # Extract q, qdot
    q = states[:, [0, 2]]  # Position variables
    qdot = states[:, [1, 3]]  # Velocity variables

    # Calculate qddot
    qddot = calculate_acceleration_cartpole(states, torqueset, env)

    # Extra calculation 
    mass_matrix, corrioli_matrix, gravitational_term, generalized_force= real_lagrangian_matrix(states, torqueset, env)
    mass_matrix_qddot= mass_matrix @ np.reshape(qddot, (qddot.shape+ (1,)))
    mass_matrix_qddot= mass_matrix_qddot.squeeze(axis=2)

    corrioli_force = corrioli_matrix @ np.reshape(qdot, (qdot.shape+ (1,)))
    corrioli_force = corrioli_force.squeeze(axis=2)

    gravitational_term = gravitational_term.squeeze(axis=2) #3D->2D
    generalized_force =  generalized_force.squeeze(axis=2)
    
    gravitational_force=gravitational_term*q
    real_torque = mass_matrix_qddot+corrioli_force+gravitational_force

    return q, qdot, qddot, torqueset, mass_matrix_qddot, corrioli_force, gravitational_force, generalized_force, real_torque

# Example usage
if __name__ == "__main__":
    #env = gym.make("CartPole-v1")
    env = ContinuousCartPoleEnv()
    ntrajs = 1
    traj_len = 100
    dt = 0.05
    
    q, qdot, qddot, torque, m, c, g, f, control_input = generate_rand_data_with_pd_control(env, ntrajs, traj_len, dt, kp=50, kr=10)
    _= -1
    print(f'{q[_]}, {qddot[_]} {torque[_]}')

    print("q shape:", q.shape)
    print("qdot shape:", qdot.shape)
    print("qddot shape:", qddot.shape)
    print("torque shape:", torque.shape)
    print("mass_matrix shape", m.shape)