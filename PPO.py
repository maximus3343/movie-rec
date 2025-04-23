import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.distributions import Categorical
import numpy as np

# Custom implementations.
from StateRep import StateRep
from Buffer import Buffer, Experience

class ActorCritic(nn.Module):
    '''
        Actor-Critic architecture for reinforcement learning.

        This class implements an **Actor-Critic model**, which consists of two main components:
        - The **Actor**, which is responsible for selecting actions based on the current policy.
        - The **Critic**, which evaluates the action taken by estimating the value of the current state.

        The Actor outputs a probability distribution over actions, while the Critic outputs a single value representing the expected return from the current state. This architecture is commonly used in policy gradient methods.
    '''

    def __init__(self, in_features: int, out_features: int, hidden_size: int):
        '''
            Initializes the Actor-Critic model.

            Sets up the neural networks for both the Actor and the Critic. Both networks
            are simple feedforward networks with two hidden layers and ReLU activation functions.
            The Actor's output layer uses a Softmax activation to produce a probability
            distribution over the action space. The Critic's output layer is a single
            linear unit predicting the state value.

            Parameters:
            - in_features (int): The number of input features representing the state space.
            - out_features (int): The number of possible actions (output features), representing the action space size.
            - hidden_size (int): The number of neurons in the hidden layers of both the Actor and Critic networks.
        '''
        super(ActorCritic, self).__init__()
        self._in_features = in_features
        self._out_features = out_features

        self.actor_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features),
            nn.Softmax(dim=-1)  # Applied on last dim.
        )  # Outputs stochastic policy over set of movies.

        self.critic_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )  # Outputs state-value V(S).

    def act(self, state, top_k: int = 0):
        '''
            Selects an action from the action space based on the current state and the Actor's policy.

            This method takes a single state as input (batch size 1), passes it through
            the Actor network to get action probabilities, samples an action from the
            resulting categorical distribution, and calculates the log probability of
            the sampled action. It also passes the state through the Critic network
            to get the state value.

            Optionally supports sampling from the top-k actions with the highest probabilities.

            Parameters:
            - state (torch.Tensor): The current state, expected to be a single batch tensor
                                    of shape `(1, in_features)`.
            - top_k (int, optional): If greater than 0 and less than the total number of actions,
                                     sampling is restricted to the `top_k` actions with the
                                     highest probabilities. Defaults to 0 (sample from all actions).

            Returns:
            - action (torch.Tensor): The sampled action (an integer tensor). Detached from the computation graph.
            - action_logprob (torch.Tensor): The log probability of the sampled action under the current policy. Detached.
            - state_val (torch.Tensor): The estimated value of the input state. Detached.
        '''
        assert isinstance(state, torch.Tensor)
        # The state is single batch input.
        assert state.shape == (1, self._in_features)

        action_probs = self.actor_net(state).squeeze(0)
        assert action_probs.shape == (self._out_features,)
        dist = Categorical(action_probs)

        # Ensure top-k is valid.
        if top_k > 0 and top_k < action_probs.size(-1):

            top_k_probs, top_k_indices = torch.topk(action_probs, top_k)

            # Create a new distribution.
            top_k_dist = Categorical(probs=top_k_probs / top_k_probs.sum())

            action = top_k_dist.sample()  # int b/w 0 and top_k-1.
            # Get the actual action index from the original action space using the index sampled from the top-K distribution.
            action = top_k_indices.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        else:
            action = dist.sample()

        action_logprob = dist.log_prob(action)
        state_val = self.critic_net(state).squeeze(0)

        assert action.ndim == 0
        assert action_logprob.ndim == 0
        assert state_val.shape == (1,)

        # Returns action, with associated state value, and probability.
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, states, action):
        '''
            Computes the log probability of a given action under the current policy,
            the estimated value of the state, and the entropy of the action distribution.

            This method is typically used during the training phase to evaluate actions
            that were previously sampled from the environment. It can handle batched input states.

            Parameters:
            - state (torch.Tensor): The input state(s). Expected shape `(batch_size, in_features)`.
            - action (torch.Tensor): The action(s) taken in the given state(s). Expected shape `(batch_size,)`.

            Returns:
            - action_logprobs (torch.Tensor): The log probability of the input action(s) under the current policy. Shape `(batch_size,)`.
            - state_values (torch.Tensor): The estimated value of the input state(s). Shape `(batch_size, 1)`.
            - dist_entropy (torch.Tensor): The entropy of the action distribution for the input state(s). Shape `(batch_size,)`.
        '''

        # Check for errors or explosive gradient values.
        if torch.isnan(states).any():
            print("Input state contains NaN values!")
        if torch.isinf(states).any():
            print("Input state contains infinite values!")

        if states.ndim < 2:
            states.unsqueeze(0)
            print(f'Strange behavior, inside evaluate() function, we have only 1 dim!')

        episode_len = states.shape[0]

        action_probs = self.actor_net(states)
        assert action_probs.shape == (episode_len, self._out_features)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        assert action_logprobs.shape == (episode_len,)
        dist_entropy = dist.entropy()
        state_values = self.critic_net(states)
        assert state_values.shape == (episode_len, 1)

        return action_logprobs, state_values, dist_entropy


class PPO:
    '''
        Proximal Policy Optimization (PPO) agent implementation.

        This class combines a State Representation module (`StateRep`) and an Actor-Critic
        network (`ActorCritic`) to implement the PPO algorithm. The `StateRep` processes
        raw environment observations (sequences of movie IDs) into a fixed-size feature
        vector, which is then used by the `ActorCritic` network to select actions (movie
        recommendations) and estimate state values.

        The agent uses a buffer to collect experiences from an episode and updates the
        policy and value function using the collected data over multiple epochs,
        employing the PPO clipping mechanism. It supports both Monte Carlo and
        Generalized Advantage Estimation (GAE) for calculating returns and advantages.
    '''

    def __init__(self, in_features: int, out_features: int, hidden_size: int, embedding_dim: int, lr_actor: float, lr_critic: float,
                 lr_sr: float, gamma: float, K_epochs: int, eps_clip: float, N: int, movie_id_list: list, top_k: int = 0,
                 algo='monte_carlo'):
        '''
        PPO agent consists of two main modules: StateRep and ActorCritic. StateRep is responsible to process the
        input features to create a suitable representation for the ActorCritic module. ActorCritic module consists
        of two nets, actor and critic. It is responsible to choose and recommend a proper movie for the user watching.

        Parameters:
            in_features (int): Size of the feature vector used by our agent. Directly affects the size in StateRep.
                The custom environment returns a variable state size, and the state size used by agent is only
                determined by the feature vector returned by StateRep. Hence, StateRep takes the state returned by 
                our custom env, and returns a feature vector of size: in_features.
            out_features (int): Size of the action space of agent. Should be the number of movies.
            hidden_size (int): Size of the hidden layers.
            lr_actor, lr_critic, lr_sr (float): Learning rate of both actor and critic nets as well as StateRep module.
            gamma (float): Discount Factor.
            K_epochs (int): Number of times policy is updated using collected, via episode-based sampling.
            eps_clip (float): Epsilon clipping factor in PPO learning function.
            N (int): In StateRep, there is one weight for each recent movie watched. This hp determines how many
                of the past movies have a weight attached to them. If = 5, then only the 5 past movies are
                considered and computed in the state rep via a weighted sum.
            movie_id_list (list): List of all the unique movie ID's.
            top_k: To face the large action space of the DRL, we can sample from the top-k actions based on 
                the stochastic policy. Speeds up training time, but reduces exploration and increases risk
                of converging to a local optimum.
            algo (str): Which algorithm is used to compute the discounted rewards in PPO learning function.
                Either 'monte_carlo' or 'gae'.

        '''
        self.gamma = gamma  # Discount factor.
        self.eps_clip = eps_clip  # Epsilon clipping factor.
        self.K_epochs = K_epochs  # Num of times policy is updated using collected data, via
        # episode-based sampling.
        self.in_features = in_features
        self.out_features = out_features

        self.state_rep = StateRep(movie_id_list=movie_id_list, embedding_dim=embedding_dim,
                                  hidden_dim=hidden_size, output_dim_k=in_features)
        self.policy = ActorCritic(in_features, out_features, hidden_size)
        self.optimizer = torch.optim.Adam([
            {'params': self.state_rep.parameters(), 'lr': lr_sr},
            {'params': self.policy.actor_net.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_net.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(in_features, out_features, hidden_size)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.state_rep_old = StateRep(
            movie_id_list=movie_id_list, embedding_dim=embedding_dim, hidden_dim=hidden_size, output_dim_k=in_features)
        self.state_rep_old.load_state_dict(self.state_rep.state_dict())

        self.buffer = Buffer()

        if algo == 'monte_carlo' or algo == 'gae':
            # Algorithm used to calculate the discounted rewards & advantages.
            self.algo = algo
        else:
            raise ValueError(f"Unknown algorithm for PPO: {algo}")

    def select_action(self, state: list) -> (int, torch.Tensor, torch.Tensor):
        '''
            Selects an action (movie recommendation) given the current state (list of recent movies).

            This method uses the `state_rep_old` module to convert the list of movie IDs
            into a state feature vector and then uses the `policy_old` (Actor network)
            to sample an action based on the current policy. It also returns the estimated
            state value and the log probability of the sampled action from the `policy_old`.

            Parameters:
                state (list): A list of strings representing the movie IDs of the recently watched movies.

            Returns:
                tuple: A tuple containing:
                       - action (int): The index of the selected movie action.
                       - state_value (torch.Tensor): The estimated value of the input state (scalar tensor).
                       - action_logprob (torch.Tensor): The log probability of the selected action (scalar tensor).
        '''

        state = [state]  # Acts as batch input to the state_rep module.
        # Converts list of movies to a proper representation for PPO nets.
        state = self.state_rep_old(state)

        with torch.no_grad():
            action, action_logprob, state_value = self.policy_old.act(state)

        assert action.ndim == 0
        assert action_logprob.ndim == 0
        assert state_value.shape == (1,)

        # Returns action.
        return action.item(), state_value, action_logprob

    def add_experience(self, experience: Experience):
        '''
            Adds a single experience tuple to the buffer.

            Parameters:
                experience (Experience): An `Experience` named tuple containing (state, action, reward,
                                         next_state, done, state_value, log_prob).
        '''
        if isinstance(experience, Experience):
            self.buffer.push(experience)
        else:
            raise ValueError(
                "In add_experience(), the experience added is not an Experience tuple.")

    def _compute_gae(self, rewards: list, state_values: torch.Tensor, dones: list, gamma: float = 0.99, lambda_: float = 0.95) -> (torch.Tensor, torch.Tensor):
        '''
            Computes Generalized Advantage Estimation (GAE) and discounted returns.

            This function calculates the advantages and returns for a sequence of
            experiences, typically representing an episode. It iterates backward
            through the sequence to compute the GAE values, which helps reduce
            the variance of policy gradient estimates.

            Parameters:
            - rewards (list): A list of rewards for each step, provided in reverse order
                              of the episode (Last-In, First-Out - LIFO).
            - state_values (torch.Tensor): A tensor of state values estimated by the critic
                                           for each state, in reverse order of the episode (LIFO).
                                           Shape `(episode_len,)`.
            - dones (list): A list of boolean flags indicating if an episode ended at that step,
                            in reverse order of the episode (LIFO).
            - gamma (float, optional): The discount factor. Defaults to 0.99.
            - lambda_ (float, optional): The GAE parameter. Defaults to 0.95.

            Returns:
            - advantages (torch.Tensor): The computed advantages for each step, in reverse order.
                                         Shape `(episode_len,)`.
            - returns (torch.Tensor): The computed discounted returns (GAE + state_values)
                                      for each step, in reverse order. Shape `(episode_len,)`.
        '''
        episode_len = len(rewards)
        # Initialize tensors for advantages and returns on the same device as state_values.
        advantages = torch.zeros(episode_len)
        returns = torch.zeros(episode_len)

        # Value of the state after the current step (when iterating backward).
        next_state_value = 0
        gae = 0  # Initialize GAE accumulator.

        # Iterate through the experiences in reverse order (LIFO)
        # t=0 corresponds to the last step of the episode
        # t=n-1 corresponds to the first step of the episode
        for t in range(n):
            # If the episode ended at this step (t), the value of the "next" state
            # (which would be the first state of the next episode) is 0.
            if dones[t]:
                next_state_value = 0

            # Calculate the temporal difference (TD) error for step t
            # rewards[t] is R_t (reward received at step t)
            # gamma * next_state_value is gamma * V(S_{t+1}) (discounted value of the state after step t)
            # state_values[t] is V(S_t) (value of the state at step t)
            delta = rewards[t] + gamma * next_state_value - state_values[t]

            # Update the GAE accumulator
            # gae here is the GAE value calculated for the step *after* the current step t (in forward time)
            gae = delta + gamma * lambda_ * gae

            # Store the calculated advantage for step t
            advantages[t] = gae
            # The return for step t is the GAE estimate plus the state value baseline
            returns[t] = gae + state_values[t]

            # Update the next_state_value for the next iteration (which processes the previous step in forward time)
            # The value of the state at step t becomes the "next_state_value" for step t-1
            next_state_value = state_values[t]

        return advantages, returns

    def _compute_gae(self, rewards, state_values, dones, gamma=0.99, lambda_=0.95):
        '''
            Computes the Generalized Advantage Estimation (GAE) and discounted returns.

            This is a helper method used when `self.algo` is set to 'gae'. It calculates
            advantages and returns based on the collected rewards, state values from the
            old policy, and done flags.

            Parameters:
                rewards (list): A list of rewards collected during the episode, in reverse order (LIFO).
                state_values (list): A list of state values estimated by the old critic for each state
                                     in the episode, in reverse order (LIFO).
                dones (list): A list of boolean flags indicating if an episode ended at that step,
                              in reverse order (LIFO).
                gamma (float, optional): The discount factor. Defaults to 0.99.
                lambda_ (float, optional): The GAE parameter. Defaults to 0.95.

            Returns:
                tuple: A tuple containing:
                       - advantages (torch.Tensor): The computed advantages for each step, shape `(episode_len,)`.
                       - returns (torch.Tensor): The computed discounted returns (GAE + state_values) for each step, shape `(episode_len,)`.
        '''
        n = len(rewards)
        advantages = torch.zeros(n)
        returns = torch.zeros(n)

        next_state_value = 0
        gae = 0

        # List of rewards already in reversed.
        for t in range(n):
            if dones[t]:  # If have more than one episode.
                # At 0 when episode is done (no more next states).
                next_state_value = 0

            delta = rewards[t] + gamma*next_state_value - state_values[t]
            gae = delta + gamma*lambda_ * gae

            advantages[t] = gae
            # returns[t] = rewards[t] + gamma * next_state_value
            returns[t] = gae + state_values[t]

            # Update for next iteration.
            next_state_value = state_values[t]

        return advantages, returns

    def learn(self):
        '''
            Updates the policy and value networks using the experiences stored in the buffer.

            This method performs the core PPO learning update. It retrieves data from the
            buffer, calculates discounted rewards (Monte Carlo or GAE) and advantages,
            and then performs multiple epochs (`K_epochs`) of optimization using the
            PPO clipped objective function.

            After the updates, the `policy_old` and `state_rep_old` networks are updated
            to match the current policy, and the buffer is cleared.
        '''

        # Get experiences from the episode, in reverse order (LIFO).
        states, actions, rewards, next_states, dones, old_state_values, old_logprobs = self.buffer.get_fields()
        episode_len = len(rewards)

        if self.algo == 'monte_carlo':
            discounted_reward = 0
            disc_rewards = []  # Stores discounted rewards in FIFO.
            for reward, done in zip(rewards, dones):
                if done:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                # Keep the reversed order.
                disc_rewards.append(discounted_reward)

        # Must be modified.
        if self.algo == 'gae':
            # State_values from old critic network are used, provides stable baseline for estimation.
            # Still in reversed order.
            advantages, disc_rewards = self._compute_gae(
                rewards, old_state_values, dones)

        disc_rewards = torch.tensor(disc_rewards, dtype=torch.float32)
        # Normalize the rewards, if len(episode) > 1.
        if episode_len > 1:
            disc_rewards = (disc_rewards - disc_rewards.mean()
                            ) / (disc_rewards.std() + 1e-7)
        assert disc_rewards.shape == (episode_len,)

        # Convert list of Tensors to a single Tensor of shape [batch, dim].
        actions = torch.stack(actions, dim=0)
        assert actions.shape == (episode_len,)
        old_state_values = torch.cat(old_state_values, dim=0)
        assert old_state_values.shape == (episode_len,)
        old_logprobs = torch.stack(old_logprobs, dim=0)
        assert old_logprobs.shape == (episode_len,)

        # Advantage computed from old policy (critic) estimates.
        advantages = disc_rewards.detach() - old_state_values.detach()
        assert advantages.shape == (episode_len,)

        for _ in range(self.K_epochs):

            # Ensures that state rep used are exactly the same ones that were availablre to the policy_old when data
            # was collected. ALigns more with on-policy nature of PPO.
            state_reps = self.state_rep_old(states)
            # Get a state rep for each state.
            assert state_reps.shape == (episode_len, self.in_features)

            # Get state value, logprobs and entropy from new policy, using states and actions
            # gathered from old policy during episode.
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                state_reps, actions)
            # From [ep_len,1] to [ep_len] to match disc_rewards.shape.
            state_values = state_values.squeeze(dim=1)
            assert state_values.shape == (episode_len,)

            # Detach since do not want to unintentionally update old policy.
            assert logprobs.shape == (episode_len,)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            assert surr1.shape == (episode_len,)
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            assert surr2.shape == (episode_len,)

            loss = -torch.min(surr1, surr2) + 0.5 * \
                F.mse_loss(state_values, disc_rewards) - 0.1 * dist_entropy
            assert loss.shape == (episode_len,)

            # Learn at each epoch.
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Load new updated parameters.
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.state_rep_old.load_state_dict(self.state_rep.state_dict())

        self.buffer.clear()  # Clear episode from buffer.