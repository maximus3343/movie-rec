import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np

class StateRep(nn.Module):
    """
        PyTorch module to represent a variable sequence of movie names as a fixed-size vector.

        This module takes a batch of sequences of movie ID strings (representing
        the history of watched movies), handles padding for variable sequence lengths,
        uses an embedding layer to convert movie IDs to dense vectors, processes
        these sequences using an LSTM, and finally outputs a fixed-size feature
        vector for each sequence in the batch. This fixed-size vector serves as
        a state representation for the reinforcement learning agent.

        Input: A batch of sequences of movie name IDs (string), potentially of
               variable lengths.
        Output: A tensor of fixed-size feature vectors (size K), one for each
                sequence in the batch.
    """

    def __init__(self, movie_id_list: list, embedding_dim: int, hidden_dim: int, output_dim_k,
                 num_layers: int = 1, dropout: float = 0.0):
        """
            Sets up the embedding layer, LSTM layer, and a final linear layer
            to transform variable-length movie sequences into fixed-size state vectors.
            It also creates mappings between movie IDs and integer indices, including
            a special padding token.

            Parameters:
                movie_id_list (list): A list of all unique movie ID strings present in the dataset.
                embedding_dim (int): The dimensionality of the embedding vector for each movie ID.
                hidden_dim (int): The number of features in the hidden state of the LSTM.
                output_dim_k (int): The desired size (K) of the final output feature vector,
                                    representing the dimensionality of the state representation.
                num_layers (int): The number of recurrent layers in the LSTM. Defaults to 1.
                dropout (float): The dropout probability applied to the output of each LSTM layer
                                 except the last one. Defaults to 0.0.
        """
        super().__init__()

        self.output_dim_k = output_dim_k

        # List of unique movie ID's.
        self.padding_token = '<PAD>'
        self.movie_ids = np.append(movie_id_list, self.padding_token)
        # Gives the index of movie_id in the embeddings array.
        self.movie_idx = {movie_id: idx for idx,
                          movie_id in enumerate(self.movie_ids)}

        self.padding_id = self.movie_idx[self.padding_token]

        # The total number of unique movie names in your vocabulary, and padding token.
        vocab_size = len(self.movie_ids)
        # Embedding layer: Converts movie IDs to dense vectors.
        self.embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=self.padding_id)

        # LSTM layer: Processes the sequence of movie embeddings.
        # batch_first=True means input/output tensors are [batch, seq, feature].
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)

        # Linear layer: Maps the final LSTM hidden state to the desired output size K.
        self.fc = nn.Linear(hidden_dim, output_dim_k)

    def forward(self, movie_sequences: list[list]) -> torch.tensor:
        '''
            Processes a batch of variable-length movie ID sequences to produce fixed-size state vectors.

            This method performs the following steps:
            1. Converts movie ID strings in each sequence to integer indices using the internal mapping.
            2. Handles empty input sequences by treating them as a single padding token sequence.
            3. Pads the integer sequences to the maximum length within the batch using the padding ID.
            4. Converts the padded integer sequences and their original lengths to PyTorch tensors.
            5. Passes the integer sequences through the embedding layer.
            6. Packs the padded embedded sequences using `pack_padded_sequence` for efficient LSTM processing.
            7. Feeds the packed sequences into the LSTM layer.
            8. Extracts the final hidden state from the LSTM output for each sequence.
            9. Unsorts the final hidden states to match the original batch order.
            10. Passes the final hidden states through a linear layer to get the fixed-size output vectors.

            Parameters:
            - movie_sequences (list[list]): A list where each element is a list
                                             of movie ID strings representing a sequence
                                             of watched movies for a single user/session.
                                             Example: `[["tt123", "tt456"], ["tt789"]]`

            Returns:
            torch.Tensor: A tensor of shape `(batch_size, output_dim_k)` representing the
                          state vector for each input sequence in the batch.
        '''

        episode_len = len(movie_sequences)

        # Have sequences of index, and the length of each index sequences.
        integer_sequences = []
        sequence_lengths = []
        max_len = 0

        if len(movie_sequences) == 0:
            movie_sequences = [[]]

        processed_movie_sequences = []

        # Check for empty list [] in the movie_sequences.
        for seq in movie_sequences:
            if not seq:
                processed_movie_sequences.append(self.padding_token)
            else:
                processed_movie_sequences.append(seq)

        movie_sequences = processed_movie_sequences
        for seq in movie_sequences:
            int_seq = [self.movie_idx.get(
                movie_id_string, self.padding_id) for movie_id_string in seq]
            integer_sequences.append(int_seq)
            sequence_lengths.append(len(int_seq))
            if len(int_seq) > max_len:
                max_len = len(int_seq)

        # Pad the integer sequences to the maximum length.
        # Padding token defined in __init__ used.
        padded_integer_sequences = []
        for int_seq in integer_sequences:
            # List concat.
            padded_seq = int_seq + [self.padding_id] * (max_len - len(int_seq))
            padded_integer_sequences.append(padded_seq)

        # Convert to tensors.
        movie_ids_tensor = torch.tensor(
            padded_integer_sequences, dtype=torch.long)
        assert movie_ids_tensor.shape == (episode_len, max_len)
        sequence_lengths_tensor = torch.tensor(
            sequence_lengths, dtype=torch.long)
        assert sequence_lengths_tensor.shape == (episode_len,)

        embeds = self.embeddings(movie_ids_tensor)

        # Pack the padded sequences for efficient LSTM processing
        # LSTM requires sequences to be sorted by length in descending order for packing
        sorted_lengths, sorted_indices = torch.sort(
            sequence_lengths_tensor, descending=True)
        sorted_embeds = embeds[sorted_indices]

        # Data transformed from padded tensor into a more memory-efficient format that allows
        # RNN's to process only the actual sequence and ignore the padding.
        packed_embedded = rnn_utils.pack_padded_sequence(
            sorted_embeds, sorted_lengths.cpu(), batch_first=True
        )

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        final_hidden_state = hidden[-1, :, :]

        _, original_indices = torch.sort(sorted_indices)
        final_hidden_state = final_hidden_state[original_indices]

        # 9. Pass the final hidden state through the linear layer
        output_vector = self.fc(final_hidden_state)
        assert output_vector.shape == (episode_len, self.output_dim_k)

        return output_vector


class StateRep_(nn.Module):
    """
        [DEPRECATED] PyTorch module to represent a sequence of movie names as a fixed-size vector
        using learned importance weights and embeddings.

        This is an older version of the state representation module. It represents the state
        as a weighted average of the embeddings of the last N movies, where the weights
        are learned parameters. It does not use a recurrent neural network like LSTM.

        Input: A batch of sequences of movie name IDs (string).
        Output: A tensor of fixed-size feature vectors (size K), one for each sequence.
    """

    def __init__(self, N: int, K: int, movie_id_list: np.ndarray):
        '''
            Sets up learnable embeddings for each unique movie ID and learnable
            importance weights for the last N positions in the movie sequence.

            Parameters:
                N (int): The maximum number of past movie titles considered for the state representation.
                K (int): The desired dimension (size) of the final output feature vector.
                movie_id_list (np.ndarray): A numpy array containing all the unique movie ID strings
                                        present in the dataset.
        '''
        super(StateRep, self).__init__()

        self.num_movies = movie_id_list.shape[0]
        self.movie_ids = movie_id_list
        self.K = K  # Size of feature vector.

        # Have a weight for each of the past N titles clicked or watched by user.
        self.importance_weights = nn.Parameter(torch.rand(N))

        # Shape of [num_movies, K].
        # Creates random embeddings.
        self.embeddings = nn.Parameter(torch.rand(self.num_movies, K))

        # Gives the index of movie_id in the embeddings array.
        self.movie_idx = {movie_id: idx for idx,
                          movie_id in enumerate(self.movie_ids)}

    def forward(self, batch_movie_list: list):
        '''
            Computes the state representation for a batch of movie sequences.

            For each sequence in the batch, it retrieves the embeddings for the movie IDs,
            applies the learned importance weights to the embeddings of the last N movies
            (or fewer if the sequence is shorter than N), and calculates the mean
            of these weighted embeddings to produce a fixed-size feature vector of size K.

            Handles variable-length sequences and empty sequences within the batch.

            Parameters:
                batch_movie_list (list): A list where each element is a list of movie ID strings
                                         representing a sequence of watched movies for a single
                                         user/session. Can also accept a single list of strings
                                         if the batch size is 1. Example: `[["tt123", "tt456"], ["tt789"]]`
                                         or `["tt123", "tt456"]` for a batch size of 1.

            Returns:
            torch.Tensor: A tensor of shape `(batch_size, K)` representing the state vector
                          for each input sequence in the batch. Returns a tensor of zeros
                          for empty sequences.
        '''

        batch_size = len(batch_movie_list)
        state_representations = []

        # If the input is [], treat it as a batch of size 1 with an empty history.
        if not batch_movie_list:
            batch_movie_list = [[]]
        # Adds a 'dimension' to the list, so we always have [[content...]] list.
        if isinstance(batch_movie_list[0], str):
            batch_movie_list = [batch_movie_list]

        for movie_list in batch_movie_list:
            # No movies in the list.
            if not movie_list:
                # Return a zero vector of size K for this item
                state_representations.append(torch.zeros(
                    self.K, device=self.embeddings.device))
                continue  # Move to the next item in the batch

            # Retrieve embedding of the movie.
            indices = []
            try:
                indices = [self.movie_idx[id] for id in movie_list]
            except KeyError as e:
                print(
                    f"Movie ID '{e.args[0]}' not found in movie_idx, in the StateRep module. Please verify that the movie_list input is valid.")
                continue
            embeddings = self.embeddings[indices]

            # Apply importance weights.
            # Use only the first len(movie_list) weights.
            # Shape [len(movie_list), K]
            # Automatically adjust if not enough past movie titles yet.
            weighted_embeddings = embeddings * \
                self.importance_weights[:len(movie_list)].unsqueeze(1)

            # Gets a feature vector of shape = [K].
            # Mean of the weighted embeds.
            # [batch, K].
            # List of tensors.
            state_representations.append(
                torch.mean(weighted_embeddings, dim=0))

        # Convert list of tensors to full tensor rep.
        return torch.stack(state_representations, dim=0)
