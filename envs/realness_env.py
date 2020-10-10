from realness_bridge import RealNeSZmqBridge
import numpy as np
from numpy import linalg as LA
import math
import os
import subprocess
from subprocess import STDOUT

from collections import defaultdict


class RealnessEnv:
    def __init__(self, env_name, **kwargs):
        self.env_name = env_name
        self.port = kwargs.setdefault("port", 5555)
        self.start_sim = kwargs.setdefault("sim_start", False)
        self.sim_seed = kwargs.setdefault("sim_seed", 0)
        self.tcl_script = kwargs.setdefault("tcl_script", 0)
        self.distance_based_reward = kwargs.setdefault("distance_based_reward", False)
        self.reward_design = kwargs.setdefault("reward_design", 4)
        self.state_design = kwargs.setdefault("state_design", 1)
        self.pos_dist = kwargs.setdefault("pos_dist", 2)
        self.state_range = kwargs.setdefault("state_range", 250)
        self.state_bins = kwargs.setdefault("state_bins", 10)
        self.add_reward = kwargs.setdefault("add_reward", False)
        self.add_index = kwargs.setdefault("add_index", False)
        self.realnes = None
        self.realNesZmqBridge = None
        self.action_size = None
        self.state_type = None
        self.obs_size = None  # Received observaiton size from realness simulator.
        self.state_space = None  # This is the total state space including the UE id and previous decision.
        # NOTE: these variables may be required later.
        self.rssi_norm = -97  # lowest value of rssi detected in the simulator.
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        self.realnesZmqBridge = RealNeSZmqBridge(self.port, self.start_sim, self.sim_seed)
        # NOTE: Skip this for now since I will initialize the environment at first.
        #self.realnesZmqBridge.initialize_env()
        # self.realnesZmqBridge.rx_env_state()
        if self.start_sim:
            self.start_realnes()
        self.envDirty = None
        #self.seed()
        self.env = None
        #self.action_space = self.realnesZmqBridge.get_action_space()
        self.last_actions = {}  # store the last actions of the UEs to exclude from the resources.
        self.first_transmissions = {}

    def get_neighbor_dist(self, tx_id, pos_of_neighbors):
        """
        ID !! is shifted -1, to start from zero.
        Returns the distibutional of the other vehicles in vicinity. Assuming that each UE knows the positions of others.
        :param user:  Prepare the state from the given user point of view. We should have the updated position of the user.
        :param pos_of_neighbors:
        :return: binned_observation, distribution of vehicles around
        """
        dist_vector = []
        max_dist = 0
        for rx_id in range(len(pos_of_neighbors)):
            if tx_id == rx_id:
                continue
            if pos_of_neighbors[rx_id]["last_updated"] > 20:  # if we dont receive any message from this ue for the last 2 secs
                continue

            p1 = (pos_of_neighbors[rx_id]["xpos"] , pos_of_neighbors[rx_id]["ypos"])
            p2 = (pos_of_neighbors[tx_id]["xpos"] , pos_of_neighbors[tx_id]["ypos"])
            dist_ , sign = self.dist(p1, p2)
            if dist_ > max_dist:
                max_dist = dist_
            dist_ = dist_ * sign
            dist_vector.append(dist_)

        if len(dist_vector) > 0:
            bins = np.linspace(-1, 1, self.state_bins + 1)
            dist_sorted = sorted(dist_vector)
            norm = LA.norm(dist_vector, np.inf)
            dist_normed = dist_sorted / norm
            binned_observation = np.histogram(dist_normed, bins, weights=dist_normed)[0]
        else:
            binned_observation = np.zeros((self.state_bins,), dtype=int)

        return binned_observation

    def get_neighbor_dist2(self, tx_id, pos_of_neighbors):
        """
        Get the position dist of the given vehicle based on the received positions of the vehicles nearby.
        :param tx_user: Transmitter id
        :param pos_of_neighbors:
        :return: binned_observation, distribution of vehicles around
        :return:
        """
        dist_vector = []
        max_dist = 0
        for rx_id in range(len(pos_of_neighbors)):
            if tx_id == rx_id:
                continue
            if pos_of_neighbors[rx_id]["last_updated"] > 20:  # if we dont receive any message from this ue for the last 2 secs
                continue

            p1 = (pos_of_neighbors[rx_id]["xpos"] , pos_of_neighbors[rx_id]["ypos"])
            p2 = (pos_of_neighbors[tx_id]["xpos"] , pos_of_neighbors[tx_id]["ypos"])
            dist_ , sign = self.dist(p1, p2)
            if dist_ > max_dist:
                max_dist = dist_
            dist_ = dist_ * sign
            dist_vector.append(dist_)

        if len(dist_vector) > 0:
            dist_sorted = sorted(dist_vector)
            binned_observation_2 = np.histogram(dist_sorted, self.state_bins, range=(-self.state_range, self.state_range))[0]
            binned_observation = binned_observation_2 / float(len(dist_sorted))
        else:
            binned_observation = np.zeros((self.state_bins,), dtype=int)

        return binned_observation

    def calculate_distance_based_reward(self, acts, pos):
        """
        Receive the actions and positions of the vehicles and determine the reward for each vehicle.
        We assume that all the vehicle are in coverage?
        acts:  vector array numerated based on ue id.
        pos: vector array
        return: vector for each users
        """
        rews = {}
        for i in range(self.action_size):
            ## for each tti check the actions taken by the vehicles.
            transmitter_ues = []  # ue ids who transmit at the same time.
            reward = None
            for user in range(len(acts)):
                if acts[user] == i:  # if UE transmit at this tti.
                    transmitter_ues.append(user)
            if len(transmitter_ues) == 1:  # if number of ues that transmit at this timestep is high.
                #  rews[transmitter_ues[0]] = 1
                reward = 1
            elif len(transmitter_ues) == 2:
                weight = self.calculate_reward_weight(transmitter_ues, pos)
                # R = weight / float(len(transmitter_ues))
                reward = 2*weight - len(transmitter_ues)
            elif len(transmitter_ues) > 2:
                reward = -len(transmitter_ues)

            for user in transmitter_ues:
                rews[user] = reward

        # TODO convert this dict to list based on the value?
        rewards = rews.values()

        return rewards

    def calculate_reward_weight(self, collided_users, pos):
        """
        Receives the collided users and determine the weight based on the distance among
        Only call this function when two ues transmit at the same time.
        """

        weight = None
        x1 = pos[collided_users[0]]
        x2 = pos[collided_users[1]]
        dist = math.sqrt((x2-x1)**2)

        norm = self.calculate_norm(pos)
        weight = (math.exp(dist) / math.exp(norm))

        return weight

    def calculate_norm(self, pos):
        """
        Calculate the distance between these two users.
        Find the maximum distance among the given dist.
        """
        x_min = 10000
        x_max = -10000
        x_min_user = None
        x_max_user = None
        for user in range(len(pos)):
            if pos[user] < x_min:
                x_min_user = user
                x_min = pos[user]

            if pos[user] > x_max:
                x_max_user = user
                x_max =pos[user]

        # Max distance among the users will be the norm.
        norm = math.sqrt((x_max - x_min)**2)

        return norm

    def dist(self, rx, tx):
        """
        Distance between the two positions
        :param rx: (x1,y1)
        :param tx: (x2,y2)
        :return:
        """
        (x1, y1) = rx
        (x2, y2) = tx
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if x1 - x2 > 0.0:  # if the observed neighbor located right side of the transmitter.
            sign = 1
        else:  # if the observed neighbor located left side of the transmitter.
            sign = -1
        return dist, sign

    def set_last_action(self, user, action):
        """
        Set the last action
        :param user:
        :param action:
        :return:
        """
        self.last_actions[user] = action

    def get_add_reward_flag(self):
        return self.add_reward

    def get_add_index_flag(self):
        return self.add_index

    def start_realnes(self):
        cwd = os.getcwd()
        # TODO:
        run_dir = self.get_run_dir(cwd)
        #run_dir = "/home/gundogan/simulator/5g_d2d_alperen_gundogan/B4G"
        os.chdir(run_dir)
        runString = "./start_debug.sh" + " " + self.tcl_script
        DEVNULL = open(os.devnull, 'wb')
        #self.realnes = subprocess.Popen(runString)
        #self.realnes = subprocess.Popen(runString, shell=False, stdout=subprocess.PIPE, stderr=None)#, universal_newlines = True)
        self.realnes = subprocess.Popen(runString, shell=True, stdout=DEVNULL, stderr=STDOUT)  # , universal_newlines=True)

    def restart_simulation(self):
        """
        Restart the realness simulation from python.
        :return:
        """
        #self.realnesZmqBridge.restart_sockets()
        cwd = os.getcwd()
        print("Current directory " + cwd)
        print("Closing realness...")
        os.system("./kill_amore.sh")
        # Now reset the socket.
        runString = "./start_debug.sh" + " " + self.tcl_script
        self.realnesZmqBridge.restart_sockets()
        DEVNULL = open(os.devnull, 'wb')
        print("Starting realness...")
        self.realnes = subprocess.Popen(runString, shell=True, stdout=DEVNULL,
                                        stderr=STDOUT)  # , universal_newlines=True)

    def get_run_dir(self, cwd):
        """
        Get the running directory of B4G.
        :param cwd:
        :return:
        """
        realnesPath = cwd
        found = False
        os.chdir("../../../../")
        prev_dir = os.getcwd()
        while (not found):
            for fname in os.listdir(prev_dir):
                if fname == "B4G":
                    found = True
                    realnesPath = os.path.join(prev_dir, fname)
                    break

        return realnesPath

    def initialize_env(self):
        """
        Initalize the environment.
        :rtype:
        """
        env = self.realnesZmqBridge.initialize_env()

        self.action_size = self.realnesZmqBridge.get_action_space()
        self.obs_size = self.realnesZmqBridge.get_observation_space()  # this is the observation sent by the simulator
        self.state_type = self.realnesZmqBridge.get_state_type()
        if self.state_design == 1:
            self.state_space = self.action_size + (self.obs_size)
        elif self.state_design == 2:
            self.state_space = self.action_size + self.state_bins

        if self.state_type == 7:  # bins based
            self.state_space = self.action_size + self.state_bins
        if self.add_reward:
            self.state_space += 1
        if self.add_index:
            self.state_space += 1

        # Initialiaze last actions
        num_users = self.realnesZmqBridge.get_total_users()
        for user in range(num_users+1):
            self.last_actions[user] = 1  # initialize with zero since it does not affect
            self.first_transmissions[user] = True

        return env

    def receive_rewards(self):
        """

        :return:
        """
        rewards = self.realnesZmqBridge.receive_rewards().all_rewards
        rews = defaultdict(dict)
        rew_value = []
        for i in range(len(rewards)):
            rews[rewards[i].user_id][rewards[i].SN] = rewards[i].reward
            rew_value.append(rewards[i].reward)
        return rews, rew_value

    def restart_env(self):
        self.realnesZmqBridge.restart_env()

    def get_action_space(self):
        # raise NotImplementedError("get_action_space method is not implemented")
        return self.action_size

    def get_state_space(self):
        # raise NotImplementedError("get_observation_space method is not implemented")
        return self.state_space

    def get_total_users(self):
        return self.realnesZmqBridge.get_total_users()

    def get_observation(self):
        return self.realnesZmqBridge.get_observation()

    def get_observation_syn(self):
        user_id, sn, state, reward = self.realnesZmqBridge.get_observation_syn()
        if self.state_type == 2 or self.state_type == 5 or self.state_type == 6:
            # we use RSSI value as a state, so we need to normalize it.
            state = [state_i - self.rssi_norm for state_i in state]
            state[:] = [x / self.rssi_norm for x in state]
            #state = state * -1
            #act = self.last_actions[user_id]
            #if not self.first_transmissions[user_id]:
        # if len(state)>0:
        #state[act] = 0  # set the transmitted channel to zero due to half duplexity
        # It should be already set to zero from the simulator
        #self.first_transmissions[user_id] = False
        elif self.state_type == 1:
            act = self.last_actions[user_id]
            state[act] = 0
            # Note we set this to one based on the experiments we have it provides stable learning.
            # It should be already set to zero from the simulator

        if reward > 0.9:
            reward = 1.0
        else:
            # reward_i = -1*(1-reward_i)
            reward = -1 * math.exp(1 - reward)

        return user_id, sn, state, reward

    def get_observation_syn_dist(self):
        """
        Special state definition for piggybacked position of neighrbors
        :return:
        """
        user_id, sn, neighbor_table, PRR = self.realnesZmqBridge.get_observation_syn_dist()

        pos_x = neighbor_table[user_id-1]["xpos"]
        # TODO: process the received neighbor pos table to get observation.

        if self.pos_dist == 1:
            state = self.get_neighbor_dist(user_id - 1, neighbor_table)
        elif self.pos_dist == 2:
            state = self.get_neighbor_dist2(user_id - 1, neighbor_table)  # pdf between -1 to +1
        else:
            print("This is not an option!!!")

        if self.reward_design == 4:
            if PRR > 0.95:
                reward = math.exp(PRR)
            else:
                reward = -1*math.exp(1-PRR)
        elif self.reward_design == 3:
            if PRR > 0.95:
                reward = 1
            else:
                reward = -1*math.exp((1 - (PRR)))
        elif self.reward_design == 2:
            if PRR > 0.95:
                reward = 1
            else:
                reward = -1*(1 - (PRR))
        else:
            reward = PRR
            #print("This reward is not defined !!!!")

        return user_id, sn, state, reward, pos_x

    def get_state_type(self):
        """
        Return the state type e.g. detected traffic based, RSSI based, distance based.
        :return:
        """
        return self.realnesZmqBridge.get_state_type()

    def apply_action(self, action):
        """
        Sends the given action to the realness interface to be executed.
        :param action:
        :return:
        """
        self.realnesZmqBridge.send_action(action)

    def obtain_state(self, obs, acts, rewards):
        """
        This function constructs the state to be exploited by the DRQN algorithm.
        :param obs: Vector includes the observation of each user.
        :param acts: Vector includes the action of each user.
        :param rewards: Vector includes the reward of each user.
        :return: State vector as an input to the DRQN algorithm calculated for each user and concatenated.
        """
        input_vector = []

        for user_i in range(len(obs)):
            input_vector_i = self.one_hot(acts[user_i], self.action_size)
            channel_obs_i = obs[user_i]
            input_vector_i = np.append(input_vector_i, channel_obs_i)
            rew_i = rewards[user_i]
            if self.add_reward:
                input_vector_i = np.append(input_vector_i, rew_i)
            if self.add_index:
                input_vector_i = np.append(input_vector_i, user_i+1)
            input_vector.append(input_vector_i)

        return input_vector

    def obtain_state_pos_dist(self, obs, acts, rewards):
        """
        This function constructs the state to be exploited by the DRQN algorithm.
        The difference is between the function above is that we also decide for a reward based on
        the action taken to provide learning where far users choose to transmit at the same resource.
        :param obs: Vector includes the observation of each user.
        :param acts: Vector includes the action of each user.
        :param rewards: Vector includes the reward of each user.
        :return:
        """
        a = 5

    def one_hot(self, num, len):
        """
        It creates a one hot vector of a number as num with size as len
        :param num: index of 1.
        :param len: total length of the vector.
        :return:
        """
        assert num >= 0 and num < len, "error"
        vec = np.zeros([len], np.int32)
        vec[num] = 1
        return vec






