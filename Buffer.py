from collections import namedtuple, deque
import torch

Experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done',
                                                   'state_value', 'log_prob'])


class Buffer:
    '''
        A simple buffer (stack) implementation used for storing experiences
        (state, action, reward, next_state, done, state_value, log_prob) collected
        during an agent's interaction with the environment.

        This buffer is designed primarily for algorithms like Monte Carlo, where
        experiences from an entire episode are collected before calculating returns
        and updating the policy.

        Each piece of data pushed into the buffer is converted to a PyTorch Tensor,
        except for 'state' and 'next_state', which are stored as lists. When retrieving
        data, the buffer returns lists of Tensors (or lists for states).
    '''

    def __init__(self):
        '''
            Initializes an empty Buffer.

            An internal deque (`_memory`) is used as the underlying data structure
            to store the experiences, functioning as a stack (Last-In, First-Out).
        '''
        # Used as stack for Monte Carlo.
        self._memory = deque()

    def push(self, state, action, reward, next_state, done, state_value, log_prob):
        '''
            Adds a single experience (`Experience` object) to the buffer.

            The fields within the input `Experience` object (action, reward, done,
            state_value, log_prob) are converted to PyTorch Tensors before being
            stored. The 'state' and 'next_state' fields are stored as lists.

            Parameters:
            - exp (Experience): An object containing the experience data
                          (state, action, reward, next_state, done, state_value, log_prob).
        '''
        exp = Experience(
            state=state,  # Kept as list.
            action=torch.tensor(action, dtype=torch.int32),
            reward=torch.tensor(reward, dtype=torch.float32),
            next_state=next_state,  # Kept as list.
            done=torch.tensor(done, dtype=torch.float32),
            state_value=torch.tensor(state_value, dtype=torch.float32),
            log_prob=torch.tensor(log_prob, dtype=torch.float32)
        )
        self._memory.append(exp)

    def push(self, exp: Experience):
        # Convert the fields of the Experience to tensors and store them.
        exp_tensor = Experience(
            state=exp.state,  # Kept as list.
            action=torch.tensor(exp.action, dtype=torch.float32),
            reward=torch.tensor(exp.reward, dtype=torch.float32),
            next_state=exp.next_state,  # Kept as list
            done=torch.tensor(exp.done, dtype=torch.float32),
            state_value=exp.state_value,
            log_prob=exp.log_prob
        )
        self._memory.append(exp_tensor)

    def _whole(self):
        '''
            Retrieves all stored experiences from the buffer in reverse order of insertion (LIFO).
            This method is intended for internal use.

            Returns:
            - list: A list of `Experience` objects stored in the buffer, with the most
                recently added experience first.
        '''
        return list(reversed(self._memory))

    def get_fields(self):
        '''
            Retrieves all stored experiences and separates them into lists for each field.

            The experiences are retrieved in reverse order of insertion (LIFO).
            This method is useful for processing batches of experiences, particularly
            for calculating returns or updating models.

            Returns:
            - tuple: A tuple containing seven lists:
                     - states (list of lists): The states from each experience.
                     - actions (list of Tensors): The actions from each experience.
                     - rewards (list of Tensors): The rewards from each experience.
                     - next_states (list of lists): The next states from each experience.
                     - dones (list of Tensors): The done flags from each experience.
                     - state_values (list of Tensors): The state values from each experience.
                     - log_probs (list of Tensors): The log probabilities from each experience.
        '''
        exps = self._whole()
        states, actions, rewards, next_states, dones, state_values, log_probs = map(
            list, zip(*exps))
        # Returns a list of Tensors.
        return states, actions, rewards, next_states, dones, state_values, log_probs

    def clear(self):
        '''
            Removes all experiences currently stored in the buffer.
            This method is typically called after the experiences from an episode
            have been processed (e.g., after calculating discounted rewards in
            Monte Carlo methods) to prepare the buffer for the next episode.
        '''
        self._memory.clear()
