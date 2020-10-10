import numpy as np
from collections import deque
import pickle
import os

"""
This function is used to debug the code via ssh.
"""
def print_experience(obs, action, reward, slot):
    print(obs.tolist())
    print("action: " + str(action) + " reward: ", str(reward) + " Time slot: ", slot)


class EpisodesBufferEntry:
    """
    Entry for episode buffer
    Capacity is the maximum step number in an episode.
    This is used to check whether given position is set or not.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminal = False
        self.init_entry()

    def init_entry(self):
        """
        Initialize the environment
        :return:
        """
        for i in range(self.capacity):
            self.states.append(-1)
            self.actions.append(-1)
            self.rewards.append(-1)

    def update_obs_acts(self, SN, state, action):
        """
        SN value is used to verify the action, state and rewards bare belonging to
        same transmission
        :param SN: Sequence number of the transmitted packet.
        :param state: State and determined reward.
        :param action: action taken for the given state.
        :return:
        """
        SN = SN % self.capacity
        if self.states[SN] is -1:
            self.states[SN] = state.copy()
            self.actions[SN] = action

    def update_rewards(self, SN, reward):
        """
        SN value is used to verify the action, state and rewards bare belonging to
        same transmission
        :param SN: Sequence number of the transmitted packet.
        :param reward: Published reward for the decision.
        :return:
        """
        SN = SN % self.capacity
        if self.rewards[SN] is -1:
            self.rewards[SN] = reward


class EpisodesBuffer:
    """Replay buffer to store a whole episode for all agents
       one entry for one agent
    """
    def __init__(self, capacity):
        self.buffer = {}
        self.capacity = capacity
        self.is_full = False

    def record_step(self, obs, acts, rewards):
        """
        Record transitions (s, a, r, terminal) in a step (based on sequence number)
        Sequence number of the transmitted packet should be controlled due to delayed rewards.
        :param obs: Observation received in one step for all agents.
        :param acts: Taken actions for the observation for all agents.
        :param rewards: Collected delayed rewards.
        :return:
        """
        buffer = self.buffer
        if self.is_full:
            for i in range(len(acts)):
                entry = buffer.get(acts[i+1]) # since we dont have any user with id=0
                if entry is None:
                    continue
                entry.append()

        else:
            for i in obs.keys():
                entry = buffer.get(i)
                if entry is None:
                    if self.is_full:
                        continue
                    else:
                        entry = EpisodesBufferEntry(self.capacity)
                        buffer[i] = entry
                        if len(buffer) >= self.capacity:
                            self.is_full = True

                for k in obs[i].keys():  #Since state and actions are stored at the same time. we can also update taken action
                    entry.update_obs_acts(k, obs[i][k], acts[i][k])
                # This is for asysnchrnozed traffic. For synchronized one I do not need that?,
                # or I need since reward will be delayed compared to obs and act
                for k in rewards[i].keys():
                    entry.update_rewards(k, rewards[i][k])

    def reset(self):
        """ clear replay buffer """
        self.buffer = {}
        self.is_full = False

    def episodes(self):
        """ get episodes """
        return self.buffer.values()


class ReplayBuffer:
    """a circular queue based on numpy array, supporting batch put and batch get"""
    def __init__(self, shape, dtype=np.float32):
        self.buffer = np.empty(shape=shape, dtype=dtype)
        self.head = 0
        self.capacity = len(self.buffer)

    def put(self, data):
        """put data to

        Parameters
        ----------
        data: numpy array
            data to add
        """
        head = self.head
        n = len(data)
        if head + n <= self.capacity:
            self.buffer[head:head+n] = data
            self.head = (self.head + n) % self.capacity
        else:
            split = self.capacity - head
            self.buffer[head:] = data[:split]
            self.buffer[:n - split] = data[split:]
            self.head = split
        return n

    def get(self, index):
        """get items

        Parameters
        ----------
        index: int or numpy array
            it can be any numpy supported index
        """
        return self.buffer[index]

    def clear(self):
        """clear replay buffer"""
        self.head = 0


class Memory():
    """
    Memory class is used for DRQN algorithm to store the observations of the users in the buffer.
    """
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """
        Add the experience to the buffer.
        :param experience:
        :return:
        """
        self.buffer.append(experience)

    def sample(self, batch_size, step_size):
        """
        Sample experience from the buffer for the given batch size and step size.
        :param batch_size:
        :param step_size: Observations should be subsequent for the step size.
        :return:
        """
        idx = np.random.choice(np.arange(len(self.buffer)-step_size),
                               size=batch_size, replace=False)

        res = []

        for i in idx:
            temp_buffer = []
            for j in range(step_size):
                temp_buffer.append(self.buffer[i+j])
            res.append(temp_buffer)
        return res

    def save(self, experiment, time_slot):
        # this function is used to save the experiences to be trained.
        dir_name = 'temp/' + experiment
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        #ex_file = open('temp/experiences/' + experiment + "/" + str(time_slot), 'wb')
        ex_file = open(dir_name + "/"+str(time_slot), 'wb')

        pickle.dump(self.buffer, ex_file)
        ex_file.close()

    def load(self, experiment, time_slot):
        dir_name = 'temp/' + experiment
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
       # ex_file = open(dir_name + "/"+str(time_slot), 'rb+')
        #self.buffer: deque = pickle.load(ex_file)
       # exs = pickle.load(ex_file)
       # type(exs)
        # self.buffer = pickle.load(ex_file)
       # ex_file.close()


#class Memory:
#    def __init__(self, max_size=2000):
#        self.buffer = deque(maxlen=max_size)
#
#    def add(self, experience):
#        self.buffer.append(experience)
#

#    def load(self):
#        ex_file = open('temp/experiences', 'rb')
#        exs = pickle.load(ex_file)
#        ex_file.close()


