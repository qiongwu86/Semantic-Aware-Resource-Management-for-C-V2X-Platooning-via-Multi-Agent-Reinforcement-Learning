import os
import numpy as np
import torch as T
import torch.nn.functional as F
from networks import CriticNetwork
from buffer import ReplayBuffer
import copy


class Global_Critic():
    def __init__(self, beta, input_dims, tau, n_actions, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                 batch_size, n_agents, update_actor_interval, noise):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.beta = beta
        self.number_agents = n_agents
        self.number_actions = n_actions
        self.number_states = input_dims
        self.update_actor_iter = update_actor_interval
        self.learn_step_counter = 0
        self.noise = noise
        self.Global_Loss = []

        self.global_critic1 = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                            n_actions=n_actions, name='global_critic1', agent_label='global_critic1')

        self.global_critic2 = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                            n_actions=n_actions, name='global_critic2', agent_label='global_critic2')

        self.global_target_critic1 = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                                   n_actions=n_actions, name='global_target_critic1',
                                                   agent_label='global_target_critic1')

        self.global_target_critic2 = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                                   n_actions=n_actions, name='global_target_critic2',
                                                   agent_label='global_target_critic2')

        self.update_global_network_parameters(tau=1)

    def save_models(self):
        self.global_critic1.save_checkpoint()
        self.global_critic2.save_checkpoint()
        self.global_target_critic1.save_checkpoint()
        self.global_target_critic2.save_checkpoint()

    def load_models(self):
        self.global_critic1.load_checkpoint()
        self.global_critic2.load_checkpoint()
        self.global_target_critic1.load_checkpoint()
        self.global_target_critic2.load_checkpoint()

    def global_learn(self, agents_nets, state, action, reward_g, reward_l, state_, terminal):

        self.agents_networks = agents_nets

        states = T.tensor(state, dtype=T.float).to(self.global_critic1.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.global_critic1.device)
        actions = T.tensor(action, dtype=T.float).to(self.global_critic1.device)
        rewards_g = T.tensor(reward_g, dtype=T.float).to(self.global_critic1.device)
        rewards_l = T.tensor(reward_l, dtype=T.float).to(self.global_critic1.device)
        done = T.tensor(terminal).to(self.global_critic1.device)

        for i in range(self.number_agents):
            self.agents_networks[i].target_actor.eval()
            self.agents_networks[i].target_critic.eval()
        self.global_target_critic1.eval()
        self.global_target_critic2.eval()
        self.global_critic1.eval()
        self.global_critic2.eval()

        target_actions = T.zeros([self.batch_size, self.number_actions * self.number_agents])
        for i in range(self.number_agents):
            target_actions[:, i * self.number_actions:(i + 1) * self.number_actions] = \
                self.agents_networks[i].target_actor.forward(
                    states_[:, i * self.number_states:(i + 1) * self.number_states])

        target_actions = target_actions + T.clamp((T.randn_like(target_actions) * self.noise), -0.5, 0.5)
        target_actions = T.clamp(target_actions, -0.999, 0.999)

        q1_ = self.global_target_critic1.forward(states_, target_actions.to(self.global_critic1.device))
        q2_ = self.global_target_critic2.forward(states_, target_actions.to(self.global_critic1.device))

        q1 = self.global_critic1.forward(states, actions)
        q2 = self.global_critic2.forward(states, actions)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = rewards_g + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        self.global_critic1.train()
        self.global_critic2.train()
        self.global_critic1.optimizer.zero_grad()
        self.global_critic2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.global_critic1.optimizer.step()
        self.global_critic2.optimizer.step()
        self.global_critic1.eval()
        self.global_critic2.eval()
        self.Global_Loss.append(critic_loss.detach().cpu().numpy())

        self.update_global_network_parameters()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_actor_iter != 0:
            return

        for i in range(self.number_agents):
            state_i = states[:, i * self.number_states:(i + 1) * self.number_states]
            action_i = actions[:, i * self.number_actions:(i + 1) * self.number_actions]
            reward_l_i = rewards_l[:, i]
            state_next_i = states_[:, i * self.number_states:(i + 1) * self.number_states]

            self.agents_networks[i].local_learn(state_i, action_i, reward_l_i, state_next_i, done)

        current_actions = []
        for i in range(self.number_agents):
            self.agents_networks[i].actor.train()
            self.agents_networks[i].critic.eval()
            state_i = states[:, i * self.number_states:(i + 1) * self.number_states]
            current_actions.append(self.agents_networks[i].actor.forward(state_i))

        for i in range(self.number_agents):
            self.agents_networks[i].actor.optimizer.zero_grad()

            joint_actions = []
            for j in range(self.number_agents):
                if i == j:
                    joint_actions.append(current_actions[j])
                else:
                    joint_actions.append(current_actions[j].detach())

            joint_actions_tensor = T.cat(joint_actions, dim=1).to(self.global_critic1.device)

            actor_global_loss = -self.global_critic1.forward(states, joint_actions_tensor).mean()

            state_i = states[:, i * self.number_states:(i + 1) * self.number_states].to(
                self.agents_networks[i].actor.device)
            actor_local_loss = -self.agents_networks[i].critic.forward(state_i, current_actions[i]).mean()

            total_actor_loss = actor_global_loss + actor_local_loss

            total_actor_loss.backward(retain_graph=True)
            self.agents_networks[i].actor.optimizer.step()

            self.agents_networks[i].update_network_parameters()

    def update_global_network_parameters(self, tau=None):

        if tau is None:
            tau = self.tau

        global_critic_1_params = self.global_critic1.named_parameters()
        global_critic_2_params = self.global_critic2.named_parameters()
        global_target_critic_1_params = self.global_target_critic1.named_parameters()
        global_target_critic_2_params = self.global_target_critic2.named_parameters()

        global_critic_1_state_dict = dict(global_critic_1_params)
        global_critic_2_state_dict = dict(global_critic_2_params)
        global_target_critic_1_state_dict = dict(global_target_critic_1_params)
        global_target_critic_2_state_dict = dict(global_target_critic_2_params)

        for name in global_critic_1_state_dict:
            global_critic_1_state_dict[name] = tau * global_critic_1_state_dict[name].clone() + \
                                               (1 - tau) * global_target_critic_1_state_dict[name].clone()

        for name in global_critic_2_state_dict:
            global_critic_2_state_dict[name] = tau * global_critic_2_state_dict[name].clone() + \
                                               (1 - tau) * global_target_critic_2_state_dict[name].clone()

        self.global_target_critic1.load_state_dict(global_critic_1_state_dict)
        self.global_target_critic2.load_state_dict(global_critic_2_state_dict)