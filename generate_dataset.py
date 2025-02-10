import numpy as np
import gym
from IPython import display as ipythondisplay
from PIL import Image
from data.continuous_cartpole import ContinuousCartPoleEnv
import random
import os 
import time
import scipy.io as sio


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

    gravitational_term = np.stack([zero, env.masspole * env.gravity * env.length * np.sin(theta)], axis=-1).reshape(N, 2, 1) #원래 부호 -인데 수정함 

    generalized_force = torqueset.reshape(N, 2, 1)

    return  mass_matrix, corrioli_matrix, gravitational_term, generalized_force

def calculate_acceleration_cartpole(states, actions, env):
    g= 9.8
    mc= 1.0
    mp= 0.1
    x = states[:, 0]
    x_dot = states[:, 1]
    theta = states[:, 2]
    theta_dot = states[:, 3]
    torque = actions.squeeze()

    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp= (torque[:, 0] + mp* env.polemass_length * theta_dot**2 * sintheta) / env.total_mass

    denom = (mc + mp * (1 - costheta**2)) * env.length
    new_thetaacc = (mp * env.length * sintheta * costheta * theta_dot**2 - (mc + mp) * g * sintheta + costheta * torque[:, 0]) / denom
    new_xacc = (mp * env.length * theta_dot**2 * sintheta - mp * g * sintheta * costheta + torque[:, 0]) / (-mp * costheta**2 + mc + mp)
    
    thetaacc = (g * sintheta - costheta * temp) / \
        (env.length * (4.0/3.0 - mp * costheta * costheta / env.total_mass))  
    xacc = temp - env.length * thetaacc * costheta / env.total_mass

#(mp*(costheta**2)*0.5-(1.1)*0.5) 

    return np.stack([xacc, thetaacc], axis=1)

def generate_rand_data_with_pd_control(env, ntrajs, traj_len, dt, kp=5, kr=10):
    assert traj_len > 0, "Trajectory length must be greater than 1"
    assert ntrajs >= 1, "Number of trajectories must be at least 1"

    # Trajectory data placeholders
    xs = np.zeros((ntrajs, traj_len, env.observation_space.shape[0]))
    uss = np.zeros((ntrajs, traj_len,  2)) #traf_len-1
    acc= np.zeros((ntrajs, traj_len,  2))
    traj_state= np.zeros((0, 4))
    traj_uss= np.zeros((0, 2))
    traj_acc= np.zeros((0, 2))

    for i in range(ntrajs):
        # Reset environment and initialize

        state = env.reset() 
        last_state=state
        xs[i, 0, :] = state  # Store initial state
        uss[i, 0, :]= np.array([0, 0])
    
        acc[i, 0, :]= np.array([0, 0])
        t_length = traj_len-1
        save_action = env.action_space.sample()
        last_uss= float(save_action)*env.force_mag
        env.step(save_action)
        for t in range(1, traj_len - 1):
            xs[i, t, :] = env.state #next_state
            uss[i, t, :] = np.array([last_uss + float(save_action), 0])
            acc[i, t, :]= np.array([xs[i, t, :][1]-last_state[1], xs[i, t, :][3]-last_state[3]])/ 0.02 #dt 
            
            #env.render()
            x_position = env.state[0]
            
            #next_state, _, done, _ = env.step(0*env.action_space.sample())

            if t % 3 ==0:
                if x_position < -2.4 + 0.5:
                    action = 0.5* env.action_space.sample()+0.5  #-1.0~0 
                elif x_position > 2.4 - 0.5:
                    action= 0.5* env.action_space.sample()-0.5  #-1.0~0
                else:
                    action = env.action_space.sample()
                next_state, _, done, _ = env.step(action)
            else:
                action = 0* env.action_space.sample()
                next_state, _, done, _ = env.step(action)
                
            last_state=state
            state= next_state 
            save_action= action 
            
            # xs[i, t + 1, :] = env.state #next_state
            # uss[i, t, :] = np.array([last_uss + 0 * float(save_action), 0]) #np.array([0, 0]) #fx, Ftheta
            # print(f"last tau: {last_uss}")
            # print(f"new tau: {env.force_mag * float(action)}")
            last_uss= uss[i, t, :][0] #force 누적 
            if state[0] < -2.4 or state[0] > 2.4: #done state일 때 
                t_length= t
                print(f'traj length is {t_length}')
                break

        xs_trim= xs[i,  1:t_length, :] 
        us_trim =uss[i, 1:t_length, :]
        acc_trim =acc[i, 1:t_length, :]

        x= np.vstack((traj_state, xs_trim))
        u= np.vstack((traj_uss, us_trim))
        acc_= np.vstack((traj_acc, acc_trim))
        traj_state=x
        traj_uss = u
        traj_acc= acc_

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

    corrioli_force = corrioli_matrix @ np.reshape(qdot, (qdot.shape+ (1,)))
    corrioli_force = corrioli_force.squeeze(axis=2)

    gravitational_force = gravitational_term.squeeze(axis=2) #3D->2D
    generalized_force =  generalized_force.squeeze(axis=2)
    
    new_qddot =np.zeros((0, 2))
    for _ in range(mass_matrix.shape[0]):
        ode_r = np.linalg.inv(mass_matrix[_]).dot(-corrioli_force[_] - gravitational_force[_] + torqueset[_])
        new_qddot= np.vstack([new_qddot, ode_r])
    mass_matrix_qddot= mass_matrix @ np.reshape(new_qddot, (new_qddot.shape+ (1,)))
    mass_matrix_qddot= mass_matrix_qddot.squeeze(axis=2)
    return q, qdot, traj_acc, torqueset, mass_matrix_qddot, corrioli_force, gravitational_force, generalized_force, torqueset

# Example usage
if __name__ == "__main__":
    #env = gym.make("CartPole-v1")
    env = ContinuousCartPoleEnv()
    ntrajs = 1
    traj_len = 1500
    dt = 0.05
    
    q, qdot, qddot, torque, m, c, g, f, control_input = generate_rand_data_with_pd_control(env, ntrajs, traj_len, dt, kp=50, kr=10)
    _= -1
    for _ in range(q.shape[0]):
        print((q[_], qdot[_], qddot[_], torque[_]))


    metric_folder = "cartpole_metrics_traj"
    os.makedirs(metric_folder, exist_ok=True)

    all_data_traj = {key: [] for key in [
        'q','qd', 'qdd', 'tau']}

    for key, value in zip([
        'q','qd', 'qdd', 'tau'], [q, qdot, qddot, torque]):
        all_data_traj[key].append(value)

    output_path = os.path.join(metric_folder, 'metrics_delan_traj.mat')
    sio.savemat(output_path, all_data_traj)
