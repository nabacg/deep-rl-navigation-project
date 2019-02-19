# coding: utf-8
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from model import QNetwork
from replaybuffers import PrioretizedReplayBuffer



BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters               
LR = 1e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, learning_rate=LR, update_every=UPDATE_EVERY, discount_factor=GAMMA):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        #saving hyperparams
        self.update_every = update_every
        self.discount_factor = discount_factor

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, 64, 128).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, 64, 128).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = PrioretizedReplayBuffer( BUFFER_SIZE, BATCH_SIZE, seed, device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.loss_track = []

    def eval_action_values(self, state, qnetwork):
        """ Helper method to evaluate model on given state and return action state values

        Params
        ==== 
            state (Torch tensor) - current env state
            model (QNetwork) - one of the Q networks (qnetwork_local, qnetwork_target)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        qnetwork.eval() # setting  model to inference 
        with torch.no_grad():
            action_values = qnetwork(state)
        qnetwork.train() # setting model back to training
        return action_values
    
    def load_model_weights(self, weights_file):
        state_dict = torch.load(weights_file)
        self.qnetwork_local.load_state_dict(state_dict)

    def step(self, state, action, reward, next_state, done):

        # calculate TD error in order to save the experience with correct priority into PrioritiedReplayBuffer
        Q_target_vals = self.eval_action_values(state, self.qnetwork_target).numpy()
        Q_vals = self.eval_action_values(state, self.qnetwork_local).numpy()[0]
        td_error = reward + GAMMA*np.max(Q_target_vals) - Q_vals[action] if done != 0 else reward - Q_vals[action]

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, td_error)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.discount_factor)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
#         loss_fn = self.loss_fn

        ## this is required as we're not learning qnetwork_targets weights
#         with torch.no_grad():
#             Q_target = rewards + gamma * (torch.max(self.qnetwork_target(next_states), dim=1)[0].view(64,1))*(1 - dones)
#             Q_target[dones == True] = rewards[dones == True]
#         Q_pred = torch.max(self.qnetwork_local(states), dim=1)[0].view(64,1)

        ## Double Q-Learning implementation 
        # Find action with highest value using Q network under training (argmax on qnetwork_local) for each S'
        best_actions_by_local_nn = torch.max(self.qnetwork_local(next_states).detach(), dim=1)[1].unsqueeze(1)
        # Then use Target Q-network (one not trained atm) to predict Q values for each (S', best_action) pair, which hopefully should be less noisy than Qnetwork_local would predict
        action_values_by_target_nn = self.qnetwork_target(next_states).detach().gather(1, best_actions_by_local_nn)
        # once action_values are predicted using  
        Q_target = rewards + gamma * action_values_by_target_nn * (1 - dones)

        Q_pred = self.qnetwork_local(states).gather(1, actions)

        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_pred, Q_target)

        self.loss_track.append(loss.item())

        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



def train_agent(agent, env, output_weights, target_mean_score=13.0, n_episodes = 2000, eps_decay=0.995, eps_end=0.01, input_weights = None):
    eps = 1.0
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    if input_weights:
        print(input_weights)
        agent.load_model_weights(input_weights)
    
    scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1,n_episodes+1):

        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        state = env_info.vector_observations[0]             # get the current state
        score = 0                                           # initialize the score
        done = False                                          
        while not(done):                                   # exit loop if episode finished
            action = agent.act(state, eps)                 # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished

            agent.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step

        scores_window.append(score)
        scores.append(score)

        eps = max(eps_end, eps_decay*eps)
        mean_score = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
        if mean_score >= target_mean_score:
            print("Target mean score of {:.2f} achived at {:.2f} after {} episodes.".format(target_mean_score, mean_score, i_episode))
                #     print("Score: {}".format(score))
            print("Saving model weights to {}".format(output_weights))
            torch.save(agent.qnetwork_local.state_dict(), output_weights)
            break
    return scores

def test_agent(agent, env, input_weights, n_episodes):
    brain_name = env.brain_names[0]
    agent.load_model_weights(input_weights)

    scores = [] 
    for i_episode in range(1,n_episodes+1):

        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        state = env_info.vector_observations[0]             # get the current state
        score = 0                                           # initialize the score
        done = False                                          
        while not(done):                                   # exit loop if episode finished
            action = agent.act(state)                      # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
        scores.append(score)
        print('\rEpisode {}\ Score: {:.2f}'.format(i_episode, score))
    return scores
