import argparse
import torch
import numpy as np
import gym
import os 
import scipy.io as sio
import time 
import random
import math

from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork
from deep_lagrangian_networks.replay_memory import PyTorchReplayMemory
from deep_lagrangian_networks.utils import load_dataset, init_env
from generate_dataset import generate_rand_data_with_pd_control
from data.continuous_cartpole import ContinuousCartPoleEnv
from data.continuous_acrobot import ContinuousAcrobotEnv


def replay_buffer(data, batch_size):
    # 데이터 변환: NumPy 배열 → PyTorch Tensor
    data_torch = [torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x for x in data]
    train_qp, train_qv, train_qa, train_torque = data_torch[:4]  # 데이터 분할

    total_samples = train_qp.shape[0]
    mem = [
        (train_qp[i:i + batch_size], 
         train_qv[i:i + batch_size], 
         train_qa[i:i + batch_size], 
         train_torque[i:i + batch_size])
        for i in range(0, total_samples, batch_size)
    ]
    
    return mem       


def compute_energy(q, qd):
    x, theta, x_dot, theta_dot= q[:,0], q[:,1], qd[:,0], qd[:,1]

    mc=1.0
    mp=0.1
    l=0.5
    g=9.8

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)   
    T= (0.5*mc + 0.5*mp) *(x_dot)**2 + x_dot* l*mp*theta_dot*cos_theta+0.5*mp*(l)**2+ (theta_dot)**2
    V= mp*g*l*cos_theta
    energy=T+V
    return energy

def compute_tau(q, qd, qdd):
    g = 9.8
    m1 = 1.1  # 질량 계수
    m2 = 0.1  # 추가 질량 계수
    l = 0.5   # 길이 계수
    x, theta, x_dot, theta_dot, x_ddot, theta_ddot = q[:, 0], q[:, 1], qd[:, 0], qd[:, 1], qdd[:,0], qdd[:, 1]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    tau1 = m1 * x_ddot +  m2 * l * theta_ddot * cos_theta -  m2 * l * (theta_dot**2) * sin_theta
    tau2 = m2 * l * x_ddot * cos_theta + m2 * (l**2) *  theta_ddot - m2 * g * l *  sin_theta
    
    tau = torch.stack([tau1, tau2], dim=1)  # (batch_size, 2) 형태로 반환
    return tau

if __name__ == "__main__":

    # Read Command Line Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[True, ], help="Training using CUDA.")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0, ], help="Set the CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[42, ], help="Set the random seed")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[1, ], help="Render the figure")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[0, ], help="Load the DeLaN model")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[0, ], help="Save the DeLaN model")
    seed, cuda, render, load_model, save_model = init_env(parser.parse_args())
    n_dof=2
    #Set hyper Params 
    hyper = {'n_width': 128,
             'n_depth': 2,
             'diagonal_epsilon': 0.01,
             'activation': 'SoftPlus',
             'b_init': 1.e-4,
             'b_diag_init': 0.001,
             'w_init': 'xavier_normal',
             'gain_hidden': np.sqrt(2.),
             'gain_output': 0.1,
             'ntrajs': 1,
             'traj_len': 1500,
             'dt' : 0.05,
             'n_minibatch': 512,
             'learning_rate': 1.e-04,
             'weight_decay': 1.e-5,
             'max_epoch': 10000} #10000
    
    #Generate train/test dataset 
    #env = gym.make("CartPole-v1")
    #env = ContinuousAcrobotEnv("human")
    env= ContinuousCartPoleEnv()
    train_data= generate_rand_data_with_pd_control(env, ntrajs=hyper["ntrajs"], traj_len=hyper["traj_len"], dt=hyper["dt"], kp=50, kr=10)
    test_data= generate_rand_data_with_pd_control(env, ntrajs=2, traj_len=70, dt=0.05, kp=50, kr=10)
    train_qp, train_qv,  train_qa, train_tau, train_m, train_c, train_g, train_f, train_control_input= train_data #prediction model과의 torque, dE/dt 차이를 계산
    test_qp, test_qv, test_qa, test_tau, test_m, test_c, test_g, test_f, test_control_input = test_data
    ##
    metric_folder = "cartpole_metrics_traj"
    os.makedirs(metric_folder, exist_ok=True)

    all_data_traj = {key: [] for key in [
        'q','qd', 'qdd', 'tau']}

    for key, value in zip([
        'q','qd', 'qdd', 'tau'], [test_qp, test_qv, test_qa, test_tau]):
        all_data_traj[key].append(value)

    output_path = os.path.join(metric_folder, 'metrics_delan_traj.mat')
    sio.savemat(output_path, all_data_traj)
##
    if load_model:
        load_file = "data/delan_model.torch"
        state = torch.load(load_file)

        delan_model = DeepLagrangianNetwork(n_dof, **state['hyper'])
        delan_model.load_state_dict(state['state_dict'])
        delan_model = delan_model.cuda() if cuda else delan_model.cpu()

    else:
        # Construct DeLaN:
        delan_model = DeepLagrangianNetwork(n_dof, **hyper)
        delan_model = delan_model.cuda() if cuda else delan_model.cpu()

    # Generate & Initialize the Optimizer:
    optimizer = torch.optim.Adam(delan_model.parameters(),
                                 lr=hyper["learning_rate"],
                                 weight_decay=hyper["weight_decay"],
                                 amsgrad=True)
    # Generate Replay Memory:
    mem= replay_buffer(train_data, batch_size=hyper['n_minibatch'])

    # Start Training Loop:
    t0_start = time.perf_counter()
    Kp = 10
    Kd = 1

    epoch_i = 0

    while epoch_i < hyper['max_epoch'] and not load_model:
        l_mem_mean_inv_dyn, l_mem_var_inv_dyn = 0.0, 0.0
        l_mem_mean_dEdt, l_mem_var_dEdt = 0.0, 0.0
        l_mem, n_batches = 0.0, 0.0

        # with torch.no_grad():
        #     sample_num = hyper['n_minibatch']
        #     q_0= np.random.uniform(-4.8, 4.8, size=(sample_num, 1))
        #     q_1= np.random.uniform(-3.14, 3.14, size=(sample_num, 1))  # Mean 0, Std 1 
        #     q_target= torch.from_numpy(np.hstack([q_0, q_1])).float()

        #     qd_0=  np.random.normal(0, 1, size=(sample_num, 1))
        #     qd_1= np.random.normal(0, 1, size=(sample_num, 1))  # Mean 0, Std 1
        #     qd_target= torch.from_numpy(np.hstack([qd_0, qd_1])).float()
        #     qdd_target=torch.from_numpy(np.random.normal(0, 1, size= (sample_num, 2))).float()

        #     tau_ff = delan_model(q_target, qd_target, qdd_target)[0]
        #     last_error=np.zeros_like(q_target)

        for i, (q, qd, qdd, tau) in enumerate(mem):
            t0_batch = time.perf_counter()

            optimizer.zero_grad()
            # error= q- q_target
            # d_error = error - last_error
            # #x, theta, x_dot, theta_dot, x_ddot, theta_ddot = q[:, 0], q[:, 1], qd[:, 0], qd[:, 1], qdd[:,0], qdd[:, 1]
            # #tau =compute_tau(x, theta, x_dot, theta_dot, x_ddot, theta_ddot) #Kp * (q_target - q) + Kd *(qd_target-qd)+tau_save[-1]
            # last_error=error 

            # Reset gradients:

            #energy controller regulates the systems energy 
            #For the training dataset, we use the energy controller to swing-up the pendulum, stabilize the pendulum at the top and let the pendulum fall down after 2s. Once the pendulum settles the process repeated until about 200s of data is collected.
            # energy_ = compute_energy(q, qd)
            # energy_target = compute_energy(q_target, qd_target)
            # e_loss= (energy_-energy_target).unsqueeze(1)
            # ke= 10
            # kp =20
            # tau = ke * e_loss* torch.sign(qd * torch.cos(q)) - kp*q 
            
            # qdd= delan_model.for_dyn(q, qd, tau)
            tau_hat, dEdt_hat = delan_model(q, qd, qdd)


            # Compute the loss of the Euler-Lagrange Differential Equation:
            err_inv = torch.sum((tau_hat - tau) ** 2, dim=1)
            l_mean_inv_dyn = torch.mean(err_inv)
            l_var_inv_dyn = torch.var(err_inv)

            # Compute the loss of the Power Conservation:
            dEdt = torch.matmul(qd.view(-1, 2, 1).transpose(dim0=1, dim1=2), tau.view(-1, 2, 1)).view(-1)
            err_dEdt = (dEdt_hat - dEdt) ** 2
            l_mean_dEdt = torch.mean(err_dEdt)
            l_var_dEdt = torch.var(err_dEdt)

            # Compute gradients & update the weights:
            loss = l_mean_inv_dyn + l_mean_dEdt #수정 원래 l_mem_mean_dEdt
            loss.backward()
            optimizer.step()
            
            # Update internal data:
            n_batches += 1
            l_mem += loss.item()
            l_mem_mean_inv_dyn += l_mean_inv_dyn.item()
            l_mem_var_inv_dyn += l_var_inv_dyn.item()
            l_mem_mean_dEdt += l_mean_dEdt.item()
            l_mem_var_dEdt += l_var_dEdt.item()

            t_batch = time.perf_counter() - t0_batch

        # Update Epoch Loss & Computation Time:
        l_mem_mean_inv_dyn /= float(n_batches)
        l_mem_var_inv_dyn /= float(n_batches)
        l_mem_mean_dEdt /= float(n_batches)
        l_mem_var_dEdt /= float(n_batches)
        l_mem /= float(n_batches)
        epoch_i += 1
        if epoch_i == 1 or np.mod(epoch_i, 10) == 0:
            print("Epoch {0:05d}: ".format(epoch_i), end=" ")
            print("Time = {0:05.1f}s".format(time.perf_counter() - t0_start), end=", ")
            print("Loss = {0:.3e}".format(l_mem), end=", ")
            print("Inv Dyn = {0:.3e} \u00B1 {1:.3e}".format(l_mem_mean_inv_dyn, 1.96 * np.sqrt(l_mem_var_inv_dyn)), end=", ")
            print("Power Con = {0:.3e} \u00B1 {1:.3e}".format(l_mem_mean_dEdt, 1.96 * np.sqrt(l_mem_var_dEdt)))
    
    #####   Model Test 과정     #####
    
    # Compute the inertial, centrifugal & gravitational torque using batched samples
    t0_batch = time.perf_counter()

    # Convert NumPy samples to torch:
    q = torch.from_numpy(test_qp).float().to(delan_model.device)
    qd = torch.from_numpy(test_qv).float().to(delan_model.device)
    qdd = torch.from_numpy(test_qa).float().to(delan_model.device)

    zeros = torch.zeros_like(q).float().to(delan_model.device)


    # Compute the torque decomposition:
    with torch.no_grad():
        delan_g = delan_model.inv_dyn(q, zeros, zeros).cpu().numpy().squeeze()
        delan_c = delan_model.inv_dyn(q, qd, zeros).cpu().numpy().squeeze() - delan_g
        delan_m = delan_model.inv_dyn(q, zeros, qdd).cpu().numpy().squeeze() - delan_g

    t_batch = (time.perf_counter() - t0_batch) / (3. * float(test_qp.shape[0]))
    
        # Move model to the CPU:
    delan_model.cpu()

    # Compute the joint torque using single samples on the CPU. The results is done using only single samples to
    # imitate the online control-loop. These online computation are performed on the CPU as this is faster for single
    # samples.

    delan_tau, delan_dEdt = np.zeros(test_qp.shape), np.zeros((test_qp.shape[0], 1))
    t0_evaluation = time.perf_counter()
    test_save_tau=[]
    for i in range(test_qp.shape[0]):

        with torch.no_grad():

            # Convert NumPy samples to torch:
            q = torch.from_numpy(test_qp[i]).float().view(1, -1)
            qd = torch.from_numpy(test_qv[i]).float().view(1, -1)
            qdd = torch.from_numpy(test_qa[i]).float().view(1, -1)

            # Compute predicted torque:
            out = delan_model(q, qd, qdd)
            delan_tau[i] = out[0].cpu().numpy().squeeze()
            delan_dEdt[i] = out[1].cpu().numpy()

            q_target =q.clone() # x 값은 유지, 각도는 0 
            q_target[:, 1] = 0  
            qd_target = torch.zeros_like(qd)
            test_save_tau.append(compute_tau(q, qd, qdd))

    #tau 저장하기 
    metric_folder = "cartpole_metrics_DeLan"
    os.makedirs(metric_folder, exist_ok=True)

    all_data = {key: [] for key in [
        'q','qd', 'qdd', 'delan_tau', 'delan_dEdt', 'delan_m', 'delan_c', 'delan_g', 'test_m', 'test_c', 'test_g', 'test_tau', 'test_torque'
    ]}

    for key, value in zip([
        'q','qd', 'qdd', 'delan_tau', 'delan_dEdt', 'delan_m', 'delan_c', 'delan_g', 'test_m', 'test_c', 'test_g', 'test_tau', 'test_torque'
    ], [
    test_qp, test_qv,  test_qa,  delan_tau, delan_dEdt, delan_m, delan_c, delan_g, test_m, test_c, test_g, test_tau, test_control_input
    ]):
        all_data[key].append(value)

    output_path = os.path.join(metric_folder, 'metrics_delan.mat')
    sio.savemat(output_path, all_data)

    #####
    t_eval = (time.perf_counter() - t0_evaluation) / float(test_qp.shape[0])
    # Compute Errors:
    test_dEdt = np.sum(test_tau * test_qv, axis=1).reshape((-1, 1))
    err_g = 1. / float(test_qp.shape[0]) * np.sum((delan_g - test_g) ** 2)
    err_m = 1. / float(test_qp.shape[0]) * np.sum((delan_m - test_m) ** 2)
    err_c = 1. / float(test_qp.shape[0]) * np.sum((delan_c - test_c) ** 2)
    err_tau = 1. / float(test_qp.shape[0]) * np.sum((delan_tau - test_tau) ** 2)
    err_dEdt = 1. / float(test_qp.shape[0]) * np.sum((delan_dEdt - test_dEdt) ** 2)

    #mass matrix, corroli matrix 비교

    print("\nPerformance:")
    print("                Torque MSE = {0:.3e}".format(err_tau))
    print("              Inertial MSE = {0:.3e}".format(err_m))
    print("Coriolis & Centrifugal MSE = {0:.3e}".format(err_c))
    print("         Gravitational MSE = {0:.3e}".format(err_g))
    print("    Power Conservation MSE = {0:.3e}".format(err_dEdt))
    print("      Comp Time per Sample = {0:.3e}s / {1:.1f}Hz".format(t_eval, 1./t_eval))
