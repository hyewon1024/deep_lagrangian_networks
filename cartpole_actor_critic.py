import sys
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 원본과 동일한 환경을 사용 (ContinuousCartPoleEnv)
from data.continuous_cartpole import ContinuousCartPoleEnv

EPISODES = 10000

# ------------------------------
# Actor 네트워크: 상태를 입력받아 행동 확률 분포 출력
# ------------------------------
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # softmax를 통해 확률 분포 생성 (행동 선택)
        return F.softmax(x, dim=-1)

# ------------------------------
# Critic 네트워크: 상태를 입력받아 해당 상태의 가치를 출력
# ------------------------------
class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ------------------------------
# A2C 에이전트 (PyTorch 버전)
# ------------------------------
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        
        self.state_size = state_size
        self.action_size = action_size  # Keras 코드에서는 env.action_space.shape[0] 사용
        
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        
        # Actor와 Critic 네트워크 생성
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        
        # 옵티마이저 생성 (각 네트워크별)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # 저장된 모델이 있다면 불러오기
        if self.load_model:
            self.actor.load_state_dict(torch.load("./save_model/cartpole_actor_trained.pth"))
            self.critic.load_state_dict(torch.load("./save_model/cartpole_critic_trained.pth"))
    
    # 현재 상태에서 Actor의 확률 분포에 따라 행동 선택
    def get_action(self, state):
        # state: numpy array, shape = (1, state_size)
        state_tensor = torch.FloatTensor(state)
        probs = self.actor(state_tensor)  # shape: [1, action_size]
        # 확률 분포로부터 샘플링 (배치 사이즈 1)
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        # 원래 코드는 ' + 0*env.action_space.sample()' 를 붙여줬으므로 그대로 둠
        return action  + 0 * env.action_space.sample()
    
    # 한 타임스텝마다 Actor와 Critic 네트워크 업데이트
    def train_model(self, state, action, reward, next_state, done):
        # numpy array를 torch tensor로 변환 (배치 크기 1)
        state_tensor = torch.FloatTensor(state)          # shape: [1, state_size]
        next_state_tensor = torch.FloatTensor(next_state)  # shape: [1, state_size]
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
        
        # Critic: 현재 상태와 다음 상태의 가치 추정
        value = self.critic(state_tensor)         # shape: [1, 1]
        next_value = self.critic(next_state_tensor) # shape: [1, 1]
        
        # 타깃 값 계산 (종료 상태인 경우 reward만, 그렇지 않으면 Bellman target)
        if done:
            target = reward_tensor
        else:
            target = reward_tensor + self.discount_factor * next_value.detach()
        
        advantage = target - value  # advantage 계산
        
        # Actor 업데이트
        probs = self.actor(state_tensor)  # 행동 확률 분포, shape: [1, action_size]
        m = torch.distributions.Categorical(probs)
        # 선택된 행동에 대한 로그 확률 계산
        log_prob = m.log_prob(torch.tensor([action]))
        # Actor 손실: -log(prob) * advantage (advantage는 Critic의 오차로부터 산출)
        actor_loss = -log_prob * advantage.detach().squeeze()
        
        # Critic 업데이트: MSE 손실
        critic_loss = F.mse_loss(value, target)
        
        # Actor 네트워크 업데이트
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Critic 네트워크 업데이트
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

# ------------------------------
# 메인 루프
# ------------------------------
if __name__ == "__main__":
    env = ContinuousCartPoleEnv()
    state_size = env.observation_space.shape[0]
    # 하게 action_size는 env.action_space.shape[0]으로 사용
    action_size = env.action_space.shape[0]
    
    agent = A2CAgent(state_size, action_size)
    
    scores, episodes = [], []

    for e in range(EPISODES):
        state = env.reset() #pole을 떨어뜨린 채로 시작 
        state = np.reshape(state, [1, state_size])
        done = False
        score = 0
        
        while not done:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # # 에피소드 중간 종료시 -100 보상 ()
            # eward = reward if not done or score == 499 else -100

            
            agent.train_model(state, action, reward, next_state, done)
            
            score += reward
            state = next_state
            balancing = (state[:, 2] > -0.05 and state[:, 2] < 0.05) 
            balancing = bool(balancing)
            if done:
                # 에피소드 종료 후 점수 처리 ()
                score_print = score if score == 500.0 else score + 100
                scores.append(score_print)
                episodes.append(e)
                print("episode:", e, "  score:", score_print)
                
                # 최근 10 에피소드 평균 점수가 490 이상이면 모델 저장 후 종료
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    print("max_score")
                    torch.save(agent.actor.state_dict(), "./save_model/cartpole_actor.pth")
                    torch.save(agent.critic.state_dict(), "./save_model/cartpole_critic.pth")
                    sys.exit()
                break
