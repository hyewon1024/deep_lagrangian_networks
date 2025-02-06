import argparse
import torch
import numpy as np
import gym
import os 
import scipy.io as sio
import time 

from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork
from deep_lagrangian_networks.replay_memory import PyTorchReplayMemory
from deep_lagrangian_networks.utils import load_dataset, init_env
from generate_dataset import generate_rand_data_with_pd_control

def replay_buffer(data, batch_size):
   
    data_torch = [
        torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x
        for x in data
    ]
    
    train_qp, train_qv, train_qa, torque = data_torch[:4]
    total_samples= train_qp.shape[0]

    # 배치 리스트 생성
    mem = [
        (train_qp[i:i + batch_size],
         train_qv[i:i + batch_size],
         train_qa[i:i + batch_size],
         torque[i:i + batch_size])
        for i in range(0, total_samples, batch_size)
    ]
    
    return mem        

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
    hyper = {'n_width': 64,
             'n_depth': 2,
             'diagonal_epsilon': 0.01,
             'activation': 'SoftPlus',
             'b_init': 1.e-4,
             'b_diag_init': 0.001,
             'w_init': 'xavier_normal',
             'gain_hidden': np.sqrt(2.),
             'gain_output': 0.1,
             'ntrajs': 100,
             'traj_len': 50,
             'dt' : 0.05,
             'n_minibatch': 50,
             'learning_rate': 5.e-04,
             'weight_decay': 1.e-5,
             'max_epoch': 100} #10000
    
    #Generate train/test dataset 
    env = gym.make("Acrobot-v1")
    train_data= generate_rand_data_with_pd_control(env, ntrajs=hyper["ntrajs"], traj_len=hyper["traj_len"], dt=hyper["dt"], kp=50, kr=10)
    test_data= generate_rand_data_with_pd_control(env, ntrajs=5, traj_len=40, dt=0.05, kp=50, kr=10)
    train_qp, train_qv,  train_qa, train_tau, train_m, train_c, train_g, train_f, train_control_input= train_data #prediction model과의 torque, dE/dt 차이를 계산
    test_qp, test_qv,  test_qa, test_tau, test_m, test_c, test_g, test_f, test_control_input = test_data

     # Load existing model parameters:
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
    mem_dim = ((n_dof, ), (n_dof, ), (n_dof, ), (n_dof, ))
    mem= replay_buffer(train_data, batch_size=hyper['n_minibatch'])

    # Start Training Loop:
    t0_start = time.perf_counter()

    epoch_i = 0
    while epoch_i < hyper['max_epoch'] and not load_model:
        l_mem_mean_inv_dyn, l_mem_var_inv_dyn = 0.0, 0.0
        l_mem_mean_dEdt, l_mem_var_dEdt = 0.0, 0.0
        l_mem, n_batches = 0.0, 0.0

        for q, qd, qdd, tau in mem:
            t0_batch = time.perf_counter()

            # Reset gradients:
            optimizer.zero_grad()

            # Compute the Rigid Body Dynamics Model:
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
            loss = l_mean_inv_dyn + l_mem_mean_dEdt
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

    print("\nPerformance:")
    print("                Torque MSE = {0:.3e}".format(err_tau))
    print("              Inertial MSE = {0:.3e}".format(err_m))
    print("Coriolis & Centrifugal MSE = {0:.3e}".format(err_c))
    print("         Gravitational MSE = {0:.3e}".format(err_g))
    print("    Power Conservation MSE = {0:.3e}".format(err_dEdt))
    print("      Comp Time per Sample = {0:.3e}s / {1:.1f}Hz".format(t_eval, 1./t_eval))
