import pandas as pd
import numpy as np
import random
import math
from collections import deque, Counter
from gymnasium import Env
from gymnasium.spaces import Discrete
from scipy import stats

class MovieEnv(Env):
    metadata = {'render.modes': ['human', 'logger']}
    id = 'MovieWatch-v0'

    def __init__(self, dataset: pd.DataFrame, N: int, alpha: float = 0.50, beta: float = 0.50, p: float = 0.05, seed: int = 42):
        '''
            This environment simulates a user's movie watching session based on a static
            dataset (Netflix Audience Behaviour). The environment is designed for an agent
            to learn a policy for recommending movies.

            At each step, the environment processes the agent's action (an integer
            representing a movie choice), updates the state based on the last N watched
            movies, and calculates a reward. There is a random chance (`p`) at each step
            that the user session ends.

            The agent's actions are integer indices, which are mapped internally to
            specific movie IDs from the dataset.

            Parameters:
            - dataset (pd.DataFrame): The pandas DataFrame containing the Netflix Audience
                                      Behaviour dataset. Expected to have columns like
                                      'movie_id', 'genres', and 'duration'.
            - N (int): The number of the most recent movies to keep in the environment's
                       state (observation space).
            - alpha (float, optional): The weighting factor for the watch time reward
                                       component. Defaults to 0.50.
            - beta (float, optional): The weighting factor for the genre entropy reward
                                      component. Defaults to 0.50.
            - p (float, optional): The probability at each step that the user's session
                                   ends, terminating the episode. Defaults to 0.05.
            - seed (int, optional): Seed for reproduciblity.

            Attributes:
            - df (pd.DataFrame): The processed dataset DataFrame.
            - num_movies (int): The total number of unique movies in the dataset.
            - _movie_genre_dict (dict): Maps movie IDs to their genre strings.
            - unique_genre_list (list): A list of all unique genres found in the dataset.
            - N (int): The number of last movies kept in the state.
            - last_movies (collections.deque): A deque storing the movie IDs of the last N
                                               watched movies.
            - action_space (gym.spaces.Discrete): The action space, representing the discrete
                                                 choices of movies.
            - _actions (np.ndarray): An internal array of possible integer actions.
            - actions_to_id (dict): Maps integer actions to movie IDs.
            - alpha (float): Weight for watch time reward.
            - beta (float): Weight for entropy reward.
            - p (float): Probability of session termination.
            - _has_reset (bool): Flag to check if reset has been called.
        '''
        super(MovieEnv, self).__init__()

        # Local rng instance.
        self.local_random = random.Random(seed)

        self.df = self._handle_csv(dataset)  # Dataset.

        # Number of unique movies in dataset.
        self.num_movies = len(self.df['movie_id'].unique())

        # Unique list of all the movies along with their genres.
        self._movie_genre_dict = self.df[['movie_id', 'genres']].drop_duplicates(
        ).set_index('movie_id')['genres'].to_dict()
        # List of unique genres among all movies. Conversion and stripping was needed.
        self.unique_genre_list = self.df['genres'].str.split(
            ',').explode().str.strip().unique()

        self.N = N
        # Stores last movies watched by user.
        self.last_movies = deque(maxlen=N)
        self.action_space = Discrete(self.num_movies)  # Public data member.
        # Used for in-class methods.
        self._actions = np.arange(0, self.num_movies)
        # Maps each action to a movie_id.
        self.actions_to_id = self._create_actions_id_mapping()

        # self._kde_dict = {} # Holds the duration probability density function of all movie ID's.
        # self._compute_kde()

        # Used to compute weighted sum for reward.
        self.alpha = alpha
        self.beta = beta

        self.p = p  # Probability of ending watch session at each step.

        self._has_reset = False

    def step(self, action):
        ''' 
        Processes the given action (which corresponds to selecting a movie), updates the list of
            the most recent movies watched (`last_movies`), and computes the reward for the agent. 
            The reward is a weighted combination of two components:
            - **Watch time reward**: This represents the reward based on how much time the 
                user watched the selected movie.
            - **Genre entropy reward**: This represents the diversity of genres among the 
                last `N` watched movies.
        The function also includes a random chance for the user to finish their session, 
            based on the probability `p`.

        Parameters:
        - action (int): The action taken by the agent, representing a movie selected from the action space. 
            It must be an integer between 0 and num_movies - 1.

        Returns:
        - observation (list): The updated list of the last `N` movies watched by the user.
        - reward (float): The weighted sum of the watch time reward and genre entropy reward, 
            representing the total reward for the agent at this step.
        - done (bool): A boolean indicating whether the user has finished their session 
            (based on a random chance with probability `p`).
        - info (None): Additional information (not used in this function, but can be expanded for debugging or logging).

        Example:
        If an agent selects an action (e.g., movie index `3`), the function will:
        - Look up the corresponding `movie_id` for that index.
        - Update the list of recently watched movies.
        - Compute rewards based on watch time and genre entropy.
        - Determine whether the session is complete (based on the random chance `p`).
        - Return the observation (updated list of last `N` movies), the reward, the `done` flag, and `info`.
    '''

        if not self._has_reset:
            raise ResetNeeded(
                "Cannot call env.step() before calling env.reset()")

        if action < 0 or action >= self.num_movies:
            raise ValueError(
                f'The action must be a value between 0 and {self.num_movies}')

        movie_id = self.actions_to_id[action]
        # Add movie_id to list of recently watched movies.
        self.last_movies.append(movie_id)

        # Weighted sum of both reward functions.
        # w_reward = self._get_watch_time(movie_id)
        w_reward = self._get_avg_watch_time(movie_id)
        e_reward = self._compute_entropy()
        reward = self.alpha * w_reward + self.beta * e_reward

        # Since we must simulate a watch session in our gym environment, we'll use a simple
        # random function to handle when user finishes its session.
        return list(self.last_movies), reward, self.local_random.random() < self.p, None, None
        # The last two, truncated and info are not used.

    def reset(self):
        ''' Resets the environment to its initial state for the start of a new episode.

        This involves clearing the history of recently watched movies (`self.last_movies`)
        and setting an internal flag (`_has_reset`) to indicate that the environment
        is ready to begin a new session.

        Returns:
        - observation (list): The initial observation of the environment, which is an
                              empty list representing no movies having been watched yet.
        - info (None): Additional information (not used in our case).
        '''

        self._has_reset = True  # Must be set True at least once.
        # Empty list of recently watched movies.
        self.last_movies.clear()
        return [], None  # Returns empty list of movies watched.

    def _compute_entropy(self):
        ''' 
        Calculates the genre entropy over the last `N` movies watched by the user. 
            The entropy measures the level of diversity or randomness in the genres of the movies watched. 
            Calculates the entropy based on the frequency of genres, giving a higher reward 
            when the genres are uniformly distributed among the last `N` movies.
            The result is a normalized reward value between 0 and 1, 
            with 1 representing maximum diversity (uniform distribution of genres) and 
            0 representing no diversity (all movies from a single genre).

        Returns:
        - A float representing the normalized entropy, which serves as a reward value 
            indicating the diversity of genres.

        Example:
        - If the last `N` movies are `[action, drama, action, action, drama]`, 
            the entropy will be lower because there are fewer genres.
        - If the last `N` movies are `[action, comedy, drama, thriller, horror]`, 
            the entropy will be higher, indicating a more diverse set of genres.

        Attributes:
        - self.last_movies: A deque of the last `N` movies watched by the user.
        - self._movie_genre_dict: A dictionary that maps movie IDs to their genres.
        - self.unique_genre_list: A list of all unique genres in the dataset.
    '''

        genre_list = []

        # Concat all genres among last_movies.
        for movie_id in self.last_movies:
            # Get a list of genres.
            genre = self._movie_genre_dict[movie_id].split(',')
            genre_list.extend(genre)  # Concat.

        # Get a count of the genres.
        genres_count = Counter(genre_list)
        # Number of genres in movie Dataset.
        num_genres = len(self.unique_genre_list)

        entropy = 0
        for count in genres_count.values():
            probability = count/num_genres
            entropy -= probability * math.log(probability, 2)

        # Avoids log(0).
        max_entropy = math.log(num_genres, 2) if num_genres > 0 else 1

        return entropy/max_entropy  # Normalization b/w 0 and 1.

    def _get_avg_watch_time(self, movie_id: str) -> float:
        ''' 
            Gets the average watch time for a specific movie ID from the dataset
            and returns its Z-score normalized value.

            This function calculates the mean watch duration for the given `movie_id`
            across all entries in the dataset. It then normalizes this average
            using the mean and standard deviation of *all* movie durations in the dataset.
            The Z-score normalization provides a measure of how much the movie's average
            watch time deviates from the overall average, scaled by the variability
            of watch times in the dataset.

            Parameters:
            - movie_id (str): The ID of the movie for which to calculate the average watch time.

            Returns:
            - float: The Z-score normalized average watch time for the specified movie.
                     A positive value indicates the movie's average watch time is above the
                     overall dataset average, while a negative value indicates it is below.
        '''

        durations = self.df['duration'].values

        avg_watch_time = self.df.groupby('movie_id')['duration'].mean()[movie_id]

        return (avg_watch_time-durations.mean())/durations.std()

    def _get_watch_time(self, movie_id: str, num_samples: int = 30) -> float:
        ''' 
        [DEPRECATED]
            Calculates the watch time for a movie based on its distribution of watch durations, 
                if available. If a **Kernel Density Estimate (KDE)** exists for the movie, 
                the function samples values from this distribution and returns the mean of the 
                sampled watch times. If no distribution is available 
                (i.e., the watch time is a single constant value), the function simply returns that value. 
            The result is then **Z-score normalized**, which allows the reward to be scaled based on 
                how the watch time compares to the overall dataset's average watch time. 
                The normalization transforms the watch time into a value that reflects how far it is from 
                the mean in terms of standard deviations. A negative reward is given if the watch 
                time is below the mean, and a positive reward if it is above the mean.

        Parameters:
        - movie_id (str): The ID of the movie for which the watch time is being retrieved.
        - num_samples (int, optional): The number of samples to draw from the KDE 
            distribution when calculating the average watch time. Default is 30. 
            This is ignored if a constant value for watch time is used.

        Returns:
        - (float): The Z-score normalized watch time for the given movie. 
        A positive value indicates that the movie's watch time is above average, 
        while a negative value indicates it is below average.

    Example:
    - If the movie has a KDE distribution and the mean of the resampled values is 50 minutes, 
        and the overall average movie duration is 60 minutes with a standard deviation of 15 minutes, 
        the function would return a normalized value indicating how far 50 minutes is from the overall average.
    '''

        watch_time = 0

        # We have a distribution.
        if isinstance(self._kde_dict[movie_id], stats.gaussian_kde):
            # Get mean of the values.
            watch_time = self._kde_dict[movie_id].resample(num_samples).mean()
        else:
            # Single value to be returned.
            watch_time = self._kde_dict[movie_id]

        durations = self.df['duration'].values

        # Z-score normalization of the watch_time.
        # Negative reward if below the mean.
        return (watch_time - durations.mean())/durations.std()

    def _compute_kde(self):
        ''' 
            [DEPRECATED]
            Creates a probability distribution (Kernel Density Estimate) for each movie's watch times.

            The function processes the `duration` values for each movie in the dataset. For each movie:
            - If there are multiple unique watch times, it uses the `gaussian_kde` method from `scipy.stats` 
                to model the distribution of watch times. 
            - If all watch times are identical or one watch exists for the movie, 
                it stores the single value instead of creating a distribution.

            This approach models the variability in how long users watch each movie, 
                providing a richer model of user behavior compared to using a simple average. Some users may stop,
                or watch a movie entirely. Far from perfect, but it provides some variability for the agent.

            **Note**: This operation can be very time-consuming (usually around 5 minutes) 
                depending on the size of the dataset, as it involves processing each movie's 
                watch times to compute KDEs.

            Attributes:
            - `self._kde_dict`: A dictionary that stores the KDE (or single value) 
                for each movie. The movie ID is the key, and the corresponding KDE or duration value is the value.

            Example:
            - For movies with varying watch durations, `self._kde_dict[movie_id]` will hold the KDE object.
            - For movies with identical or single watch duration, `self._kde_dict[movie_id]` 
                will hold the constant watch time value.
        '''
        for movie in self.df['movie_id'].unique():
            movie_data = self.df[self.df['movie_id'] == movie]['duration'].values
            if len(movie_data) > 1:
                # All same values, cannot create distribution.
                if np.all(movie_data == movie_data[0]):
                    self._kde_dict[movie] = movie_data[0]
                else:
                    # Creates distribution based on watch time.
                    self._kde_dict[movie] = stats.gaussian_kde(movie_data)
            else:
                # Stores the single value for that entry.
                self._kde_dict[movie] = movie_data.mean()

    def _handle_csv(self, df: pd.DataFrame):
        ''' 
            Function which specifically handles the Netflix audience behaviour - UK movies dataset
            for our gym environment.

            Parameters:
            - df (pd.DataFrame): The input pandas DataFrame containing the dataset.

            Returns:
            - pd.DataFrame: The cleaned and processed DataFrame.
        '''

        df.dropna(inplace=True)  # Drop rows with missing values.

        # Removes some useless column for the gym environment.
        df.drop('Unnamed: 0', axis=1, inplace=True)
        df.drop('datetime', axis=1, inplace=True)

        return df

    def _create_actions_id_mapping(self) -> dict:
        ''' Creates a dictionary mapping integer actions (0 to num_movies - 1) to their
        corresponding unique movie IDs from the dataset.

        This mapping is necessary because the gym environment's action space is discrete
        (integer indices), but the environment needs to interact with the dataset using
        the actual movie IDs.

        Returns:
        - dict: A dictionary where keys are integer action indices and values are the
                corresponding movie IDs.
        '''
        actions_to_id = {}

        movie_ids = self.df['movie_id'].unique()

        for action, movie_id in zip(self._actions, movie_ids):
            actions_to_id[action] = movie_id

        return actions_to_id