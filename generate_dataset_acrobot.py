import numpy as np
import gym
from IPython import display as ipythondisplay
from PIL import Image
from data.continuous_acrobot import ContinuousAcrobotEnv
import random

def real_lagrangian_matrix(states, torqueset, env):
    #Calculate lagrangian metrix (real)
    theta1, theta2, theta1_dot, theta2_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3] 
    costheta1= np.cos(theta1)
    costheta2= np.cos(theta2)
    sintheta1= np.sin(theta1)
    sintheta2= np.sin(theta2)
    N=costheta1.shape[0]
 
    m1, m2 = env.LINK_MASS_1, env.LINK_MASS_2
    l1, l2 = env.LINK_LENGTH_1, env.LINK_LENGTH_2
    lc1, lc2 = env.LINK_COM_POS_1, env.LINK_COM_POS_2
    I1 = env.LINK_MOI 
    I2 = env.LINK_MOI
    g = 9.8

    # Mass matrix
    d11 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + l1 * lc2 * costheta2) + I1 + I2
    d12 = m2 * (lc1**2 + l1 * lc2 * costheta2) + I2
    d22 = m2 * lc2**2 + I2

    m11= np.full((N,), d11)
    m12= d12
    m21= m12
    m22=np.full((N,), d22)
    mass_matrix = np.stack([m11, m12, m21, m22], axis=-1).reshape(N, 2, 2)

    h = -m2 * l1 * lc2 * sintheta2
    c11= np.full((N,), h * theta2_dot)
    c12= c11
    c21= np.full((N,), -(0.5)*h * theta1_dot)
    zero=np.zeros((N,))
    corrioli_matrix= np.stack([c11, c12, c21, zero], axis=-1).reshape(N, 2, 2)

    g1 = np.full((N,), (m1 * lc1 + m2 * l1) * g * costheta1 + m2 * lc2 * g * (costheta1*costheta2-sintheta1*sintheta2))
    g2 = np.full((N,), m2 * lc2 * g * (costheta1*costheta2-sintheta1*sintheta2))
    gravitational_term = np.stack([g1, g2], axis=-1).reshape(N, 2, 1)

    generalized_force = torqueset.reshape(N, 2, 1)
    return  mass_matrix, corrioli_matrix, gravitational_term, generalized_force

def calculate_acceleration_cartpole(states, actions, env):
    theta1, theta2, theta1_dot, theta2_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3] 
    costheta1= np.cos(theta1)
    costheta2= np.cos(theta2)
    sintheta1= np.sin(theta1)
    sintheta2= np.sin(theta2)
    torque = actions.squeeze()
    N=costheta1.shape[0]

    m1, m2 = env.LINK_MASS_1, env.LINK_MASS_2
    l1, l2 = env.LINK_LENGTH_1, env.LINK_LENGTH_2
    lc1, lc2 = env.LINK_COM_POS_1, env.LINK_COM_POS_2
    I1 = env.LINK_MOI 
    I2 = env.LINK_MOI
    g = 9.8

    d1= np.full((N,), m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * costheta2) + I1 + I2)
    d2 = np.full((N,), m2 * (lc2**2 + l1 * lc2 * costheta2) + I2)
    phi2 = np.full((N,), m2 * lc2 * g * (sintheta1*costheta2+costheta1*sintheta2))
    phi1 = np.full((N,), (
        -m2 * l1 * lc2 * theta2_dot**2 * sintheta2
        - 2 * m2 * l1 * lc2 * theta2_dot * theta1_dot * sintheta2
        + (m1 * lc1 + m2 * l1) * g * sintheta1
        + phi2))
    ddtheta2 = np.full((N,),(
            torque[:, 1] + d2 / d1 * phi1 - m2 * l1 * lc2 * theta1_dot**2 * sintheta2 - phi2
        ) / (m2 * lc2**2 + I2 - d2**2 / d1))
    ddtheta1 =np.full((N,), -(d2 * ddtheta2 + phi1) / d1)
    

    return np.stack([ddtheta1, ddtheta2], axis=1)

def generate_rand_data_with_pd_control(env, ntrajs, traj_len, dt, kp=50, kr=100):
    assert traj_len > 0, "Trajectory length must be greater than 1"
    assert ntrajs >= 1, "Number of trajectories must be at least 1"

    # Trajectory data placeholders
    xs = np.zeros((ntrajs, traj_len, 4)) #6 states
    uss = np.zeros((ntrajs, traj_len - 1,  2))
    traj_state= np.zeros((0, 4))
    traj_uss= np.zeros((0, 2))
    
    for i in range(ntrajs):
        # Reset environment and initialize
        state = env.reset()
        xs[i, 0, :] = state  # Store initial state
        t_length = traj_len-1
        last_uss= 0
        for t in range(traj_len - 1):
            env.render()
            action= env.action_space.sample()
            next_state, _, done, _, _ = env.step(action)
            
            xs[i, t + 1, :] = env.state #next_state
            uss[i, t, :] = np.array([0, last_uss + float(action)]) #np.array([0, 0]) #fx, Ftheta
            last_uss=  uss[i, t, :][0]
            state = next_state #next_state
            #done state 상관 없이 진행 
        xs_trim= xs[i, :t_length, :] #마지막 요소 제외 
        us_trim =uss[i, :t_length, :]

        x= np.vstack((traj_state, xs_trim))
        u= np.vstack((traj_uss, us_trim))
        traj_state=x
        traj_uss = u

    # Prepare data for output
    states= traj_state
    torqueset= traj_uss  #force 

    # Extract q, qdot
    q = states[:, 0:2]  # Position variables
    qdot = states[:, 2:4]  # Velocity variables


    # Calculate qddot
    qddot = calculate_acceleration_cartpole(states, torqueset, env)

    # Extra calculation 
    mass_matrix, corrioli_matrix, gravitational_term, generalized_force= real_lagrangian_matrix(states, torqueset, env)

    corrioli_force = corrioli_matrix @ np.reshape(qdot, (qdot.shape+ (1,)))
    corrioli_force = corrioli_force.squeeze(axis=2)

    gravitational_force = gravitational_term.squeeze(axis=2) #3D->2D
    generalized_force =  generalized_force.squeeze(axis=2)

    new_qddot =np.zeros((0, 2))
    for _ in range(mass_matrix.shape[0]):
        ode_r = np.linalg.inv(mass_matrix[_]).dot(-corrioli_force[_] - gravitational_force[_]) + torqueset[_]
        new_qddot= np.vstack([new_qddot, ode_r])

    mass_matrix_qddot= mass_matrix @ np.reshape(new_qddot, (new_qddot.shape+ (1,)))
    mass_matrix_qddot= mass_matrix_qddot.squeeze(axis=2)

    return q, qdot, new_qddot, torqueset, mass_matrix_qddot, corrioli_force, gravitational_force, generalized_force, torqueset

# Example usage
if __name__ == "__main__":
    #env = gym.make("CartPole-v1")
    env = ContinuousAcrobotEnv(render_mode='human')
    ntrajs = 1
    traj_len = 2459
    dt = 0.05
    
    q, qdot, qddot, torque, m, c, g, f, control_input = generate_rand_data_with_pd_control(env, ntrajs, traj_len, dt, kp=50, kr=10)
    _= -1
    print(f'{q[_]}, {qddot[_]} {torque[_]}. {control_input[_]}')

    print("q shape:", q.shape)
    print("qdot shape:", qdot.shape)
    print("qddot shape:", qddot.shape)
    print("torque shape:", torque.shape)
