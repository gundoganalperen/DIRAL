from __future__ import division
import numpy as np
import math
from network import Network

class TestEnv:
    def __init__(self, **kwargs):
        """
        This class is used to test the behaviour of the realness simulator for the state space 1.
        :param kwargs:
        """
        self.NUM_USERS = kwargs.setdefault("num_users", 3)  # Number of users in the scenario, if not given starts a simplest scenario.
        self.NUM_CHANNELS = kwargs.setdefault("num_channels", 3)  # Number of available channels that UE can select, defines the action space of the RL agent.
        self.mobility = kwargs.setdefault("mobility", False)
        self.mobility_vary = kwargs.setdefault("mobility_vary", False)
        self.enable_design_topology = kwargs.setdefault("enable_design_topology", False)

        self.highway_length = kwargs.setdefault("highway_length", 200)
        self.enable_fingerprint = kwargs.setdefault("enable_fingerprint", False)
        self.reward_design = kwargs.setdefault("reward_design", 1)
        self.communication_range = kwargs.setdefault("communication_range", 1)
        self.proportional_fair = kwargs.setdefault("proportional_fair", False)
        self.load_positions = kwargs.setdefault("load_positions", False)
        self.bin_range = kwargs.setdefault("bin_range", 500)

        self.state_parameters = kwargs.setdefault("State", False)
        self.state_type = self.state_parameters["type"]
        self.add_reward = self.state_parameters["add_reward"]  # Add the received reward from the prevous packet as a part of the state space.
        self.add_action = self.state_parameters["add_action"]  # Add the received reward from the prevous packet as a part of the state space.
        self.add_index = self.state_parameters["add_index"]  # Add the UE id as a part of the state space.
        self.add_velocity = self.state_parameters["add_velocity"]  # Add the UE id as a part of the state space.
        self.action_index = self.state_parameters["action_index"]
        self.piggybacking = self.state_parameters["piggybacking"]
        self.add_position = self.state_parameters["add_position"]
        self.add_positional_dist = self.state_parameters["add_positional_dist"]
        self.add_positional_dist_piggy = self.state_parameters["add_positional_dist_piggy"]
        self.add_positional_dist_type = self.state_parameters["add_positional_dist_type"]  # if 2 then normalize the vector.
        self.add_positional_velocity = False
        #self.add_positional_velocity = self.state_parameters["add_positional_velocity"]  # if 2 then normalize the vector.
        self.num_bins = self.state_parameters["num_bins"]
        self.add_channel_obs = self.state_parameters["add_channel_obs"]
        self.state_space = 0

        self.action_space = self.NUM_CHANNELS

        self.topology = kwargs.setdefault("topology", "Circle")
        self.radius = kwargs.setdefault("radius", 100)
        self.congestion_test = kwargs.setdefault("congestion_test", False)
        if self.add_action:
            if self.action_index == "binary":
                self.state_space += (self.NUM_CHANNELS)
            elif self.action_index == "real":
                self.state_space += 1  # +1 is the UE agent index.
            else:
                print("Error!!! action index " +  self.action_index + " is not defined!!!")
        if self.add_channel_obs:
            self.state_space += self.NUM_CHANNELS

        if self.add_reward:
            self.state_space += 1
        if self.add_index:
            self.state_space += 1
        if self.add_velocity:
            self.state_space += 1
        if self.add_position:
            self.state_space += 2  # add (x,y) coordinates

        if self.add_positional_dist:  # add the distribution of others
            self.state_space += self.NUM_USERS-1

        if self.piggybacking:
            self.state_space += self.NUM_CHANNELS * (self.NUM_CHANNELS - 1)  # when we combine the observation of the others
            #  we reach at max this value. If there are many collisions we will not receive the observations of the others
            #  Thus, we need to use zero padding.

            # iniatialize the dictionary to be used to construct the state.
            self.prev_obs = {}
            for user in range(self.NUM_USERS):
                self.prev_obs[user] = np.zeros((self.action_space,), dtype=float)

        if self.enable_fingerprint:
            self.state_space += 2  #  Add episode number and epsilon as a part of the state.

        if self.add_positional_dist_piggy:
            self.state_space += self.num_bins

        if self.proportional_fair:
            self.pf_counter = {}  # holds for each user
            self.pf_threshold = 10  # If one user does not transmit succesfully then we apply penalty
            self.pf_penalty = -10
            for user in range(self.NUM_USERS):
                self.pf_counter[user] = 0

        self.action_space_vector = np.arange(self.NUM_CHANNELS)
        self.users_action = np.zeros([self.NUM_USERS],np.int32)
        self.users_observation = np.zeros([self.NUM_USERS],np.int32)

        self.network = Network(congestion_test=self.congestion_test, num_bins=self.num_bins, num_users=self.NUM_USERS, topology=self.topology,
                               radius=self.radius, mobility=self.mobility, mobility_vary=self.mobility_vary,
                               highway_len=self.highway_length, communication_range=self.communication_range,
                               bin_range=self.bin_range, enable_design_topology=self.enable_design_topology)
        if self.load_positions:
            self.load_file_positions = kwargs.setdefault("load_file_pos", " ")
            #self.network.load_x_positions(self.load_file_positions)

        # TODO: call function to initialize the topology.
        a = 5

    def load_saved_positions(self):
        if self.load_positions:
            print("Load the saved positions !!!")
            self.network.load_x_positions(self.load_file_positions)
        else:
            print("Load the saved positions disabled !!!")

    def sample(self):
        """
        Sample random actions for the beginning of the simulation to feed LSTM network.
        :return:
        """
        x =  np.random.choice(self.action_space_vector, size=self.NUM_USERS)
        return x

    def my_step(self, actions, timestep):
        """
        Step function of the environment.
        Calculates the observations and rewards based on the actions taken by the users.
        It calculates the observation of each user and reward. Obs, is later used to construct the state vector.
        :param actions:
        :return:
            obs: channel observations for the taken actions.
            reward: reward of each user for the taken actions
        """
        obs = {}
        piggy_obs = {}
        rews = np.zeros(self.NUM_USERS)
        acts = {}
        if self.add_positional_dist_piggy:
            self.network.periodic_update()

        for user in range(len(actions)):  # since each users takes an action
            acts[user] = np.zeros((self.action_space,), dtype=int)  # Initialize the actions for the users.
            obs[user] = np.zeros((self.action_space,), dtype=float)  # Initialize the observation for the user.
            piggy_obs[user] = np.zeros((self.action_space,), dtype=float)  # Initialize the observation for the user.
            #piggy_obs[user] = []  # Initialize the observation for the user.
            acts[user][actions[user]] = 1
        for i in range(self.action_space):  # For each time step, at each TTI.
            # Check the number of total transmissions at this TTI/time step.
            tot_actions = 0
            transmission = False  # Is there any transmission at this time step?
            #  Transmitters that holds the ue ids.
            transmitters_tti = []
            for user in range(self.NUM_USERS):
                if acts[user][i] == 1:  # which means we have a transmission.
                    tot_actions += 1
                    transmission = True
                    transmitters_tti.append(user)
            # Now set the reward based on the total number of transmission at this timestamp
            if tot_actions == 0:
                reward = 0
            elif tot_actions == 1:
                reward = 1
            else:
                #  TODO: call topology_weight()
                #weights = self.network.calculate_reward_weights(transmitters_tti)
                # if we have multiple transmission at the same time.
                # Rewards of all UEs
                #R = max(weights[0]/float(tot_actions), 1/float(tot_actions+1))
                #R = min(weights[0]/float(tot_actions), 1/float(tot_actions+1))
                if self.reward_design == 1:
                    weights = self.network.calculate_reward_weights(transmitters_tti)
                    R = weights[0]/float(tot_actions)
                    rewards = -1*(1 - (R))
                elif self.reward_design == 2:
                    # only calculate the weight when there are two transmission at the same time.
                    if tot_actions == 2:
                        weights = self.network.calculate_reward_weights(transmitters_tti)
                        rewards = 2 * weights[0] - float(tot_actions)
                    else:
                        weights = 0
                        rewards = weights - float(tot_actions)

                elif self.reward_design == 3:
                    R = 1/float(tot_actions)
                    #rewards = -1*(1 - (math.exp(R)))
                    rewards = -1*math.exp((1 - (R)))
                elif self.reward_design == 4:
                    rewards = 1/float(tot_actions)
                elif self.reward_design == 5:
                    if tot_actions == 2:
                        weights = self.network.calculate_reward_weights(transmitters_tti)
                        if weights[0] == 1:
                            rewards = 0
                        else:
                            rewards = -1
                    else:
                        rewards = -1
                else:
                    print("Such a reward is not defined!!!")
            #rewards = np.log(R)
            #reward = reward[0]  # since weights are the same for now.
            # Prepare the observations of the users.
            for user in range(self.NUM_USERS):
                if acts[user][i] == 1:  # If a user transmit at this timestamp.
                    # NOTE: change this logic above and put 1.
                    obs[user][i] = 0  # it can not detect any other transmission due to half-duplex feature.
                    # piggy_obs[user].append(0)
                    # piggy_obs[user].append(0)
                    piggy_obs[user][i] = 0
                    #obs[user][i] = 1  # we can set this to one.
                    if tot_actions > 1:
                        # NOTE: use like this for now, since all the rewards are the same.
                        #  later, think about that
                        rews[user] = rewards
                        if self.proportional_fair:
                            if self.pf_counter[user] > self.pf_threshold:
                                rews[user] = self.pf_penalty
                            self.pf_counter[user] += 1
                    else:
                        rews[user] = reward
                        if self.proportional_fair:
                            self.pf_counter[user] = 0  # reset again
                # rews[user] = -1*(1 - (weights[user]/float(tot_actions)))
                else:
                    if transmission is True:  # If there exist at least one transmission at this time step we assume
                        if self.state_type == 1:  # then we observe detected traffics.
                            # that UE decodes only one transmission (assuming all the UEs are in coverage.)
                            obs[user][i] = 1
                            # Receive the packet of the closest transmitter, and use the distance as a part of the state.
                            tx_dist, tx_id = self.network.find_closest_tx(transmitters_tti, user)
                            if self.add_positional_dist_piggy:
                                self.network.received_update(rx_id=user, tx_id=tx_id)

                        elif self.state_type == 2:
                            # Receive the packet of the closest transmitter, and use the distance as a part of the state.
                            tx_dist, tx_id = self.network.find_closest_tx(transmitters_tti, user)
                            if self.add_positional_dist_piggy and tx_id is not None:
                                self.network.received_update(rx_id=user, tx_id=tx_id)

                            obs[user][i] = tx_dist
                            if self.piggybacking:
                                # Receive the observaiton of the transmitter at the previous transmission step.
                                tmp_a = self.prev_obs[tx_id]
                                #  tmp_a = np.append(tmp_a, obs[user][i])
                                #  piggy_obs[user].append(tmp_a)
                                piggy_obs[user][i] = tx_dist
                                piggy_obs[user] = np.insert(piggy_obs[user], i, tmp_a)

                    else:
                        if self.piggybacking:
                            # If there is no transmission at this time step and we enabled piggybacking,
                            # We fill with zero in order to provide a fixed obs length for learning.
                            # piggy_obs[user].append(np.zeros((self.action_space,), dtype=float))
                            piggy_obs[user] = np.insert(piggy_obs[user], i, np.zeros((self.action_space,), dtype=float))

                    #  Reward only set if a user decides to transmit

        # Update the positions in every K steps
        self.network.update_mobility(timestep=timestep)
        if self.piggybacking:
            self.prev_obs = obs

        if self.piggybacking:
            return piggy_obs, rews
        else:
            return obs, rews


    def my_step_design(self, actions, timestep):
        obs = {}
        rews = np.zeros(self.NUM_USERS)
        acts = {}
        if self.add_positional_dist_piggy:
            self.network.periodic_update()

        for user in range(len(actions)):  # since each users takes an action
            acts[user] = np.zeros((self.action_space,), dtype=int)  # Initialize the actions for the users.
            obs[user] = np.zeros((self.action_space,), dtype=float)  # Initialize the observation for the user.
            #piggy_obs[user] = np.zeros((self.action_space,), dtype=float)  # Initialize the observation for the user.
            #piggy_obs[user] = []  # Initialize the observation for the user.
            acts[user][actions[user]] = 1

        for i in range(self.action_space):
            tot_actions = 0
            transmission = False  # Is there any transmission at this time step?
            #  Transmitters that holds the ue ids.
            transmitters_tti = []
            for user in range(self.NUM_USERS):
                if acts[user][i] == 1:  # which means we have a transmission.
                    tot_actions += 1
                    transmission = True
                    transmitters_tti.append(user)

            for user in range(self.NUM_USERS):
                if acts[user][i] == 1:  # If a user transmit at this timestamp.
                    obs[user][i] = 0 # it can not detect any other transmission due to half-duplex feature.
                    if tot_actions > 0:
                        if tot_actions == 1:  # then only this user transmit
                            rews[user] = 1
                        else:  # if more than one transmitter uses the same resource, then calculate the reward
                            rews[user] = self.calculate_reward_design(user, transmitters_tti)
                    else:
                        print("This is not possible !!!")
                else:
                    if transmission is True:  # if there is a transmission from other UEs at this resource.
                        obs[user][i] = 1  #  NOTE: I dont use the observation as a part of state now. But this
                                          #  be calculated based on the distance among vehicles.
                        #    Receive the packet of the closest transmitter, and use the distance as a part of the state.
                        tx_dist, tx_id = self.network.find_closest_tx(transmitters_tti, user)
                        if self.add_positional_dist_piggy and tx_id is not None:
                            self.network.received_update(rx_id=user, tx_id=tx_id)

        # Update positions of the vehicles based on the velocity or predefined positions
        self.network.update_mobility(timestep=timestep)

        return obs, rews


    def calculate_reward_design(self, tx_user, transmitters_tti):
        """
        This function receives user, pool, and other tranmission id to determine the reward of the "user"
        :param pool: Resource pool
        :param user: Given user that transmit
        :param transmitters_tti: Other users
        :return:
        """
        comm_range_tx = []
        reward = None
        comm_range_tx.append(tx_user)
        for other_tx in transmitters_tti:
            if tx_user == other_tx:
                continue
            if self.network.is_comm_range(tx_user, other_tx):
                comm_range_tx.append(other_tx)

        if len(comm_range_tx) == 1:  # If there is no one near to the transmitter
            reward = 1
        elif len(comm_range_tx) == 2:  # if there are two users then calculate the reward.
            weights = self.network.calculate_reward_weights_design(tx_user, comm_range_tx)
            if weights == 1:
                reward = 0  # 2 * weights - float(tot_actions)
            else:
                reward = -len(comm_range_tx)
            #reward = 2 * weights - float(len(comm_range_tx))
        else:
            # if number of users that collide are more than 2
            reward = -float(len(comm_range_tx))

        return reward

    def my_step_ch(self, actions, time_step):
        """
        Step function of the environment based on channel model similar to realness.
        Calculates the observations and rewards based on the actions taken by the users.
        It calculates the observation of each user and reward. Obs, is later used to construct the state vector.
        :param actions:
        :return:
            obs: channel observations for the taken actions.
            reward: reward of each user for the taken actions
        """
        obs = {}
        rews = np.zeros(self.NUM_USERS)
        acts = {}
        if self.add_positional_dist_piggy:
            self.network.periodic_update()

        for user in range(len(actions)):  # since each users takes an action
            acts[user] = np.zeros((self.action_space,), dtype=int)  # Initialize the actions for the users.
            obs[user] = np.zeros((self.action_space,), dtype=float)  # Initialize the observation for the user.
            acts[user][actions[user]] = 1
        for i in range(self.action_space):
            # Check the number of total transmissions at this TTI/time step.
            tot_actions = 0
            transmission = False  # Is there any transmission at this time step?

            transmitters_tti = []
            for user in range(self.NUM_USERS):
                if acts[user][i] == 1:  # which means we have a transmission.
                    tot_actions += 1
                    transmission = True
                    transmitters_tti.append(user)

            # If we have more then one transmission we should decide which UEs will receive the messages.
            if tot_actions > 1:
                total_vehicles_received = {}
                total_vehicles_in_range = {}
                rewards = {}
                norm = {}
                for tx in transmitters_tti:
                    total_vehicles_received[tx] = 0
                    total_vehicles_in_range[tx] = 0
                    for rx in range(self.NUM_USERS):
                        if rx in transmitters_tti:  # due to half duplexity
                            continue
                        if not self.network.check_communicaiton_range(tx, rx):
                            continue
                        total_vehicles_in_range[tx] += 1
                        _, nearest_tx = self.network.find_closest_tx(transmitters_tti, rx)
                        if tx == nearest_tx:
                            total_vehicles_received[tx] += 1

                    if total_vehicles_in_range[tx] > 0:
                        rewards[tx] = total_vehicles_received[tx]/total_vehicles_in_range[tx] # divided by the number of the users in the vicinity excluding the tx
                    else:
                        rewards[tx] = 1  # we can assume that UE has choose the right action,

            # Prepare the observations and rewards of the users.
            for user in range(self.NUM_USERS):
                if acts[user][i] == 1:  # If a user transmit at this timestamp.
                    obs[user][i] = 0  # it can not detect any other transmission due to half-duplex feature.
                    if tot_actions > 1:   # if there are other users too transmit at this time step
                        R = rewards[user]
                        if self.reward_design == 3:
                            rews[user] = 1 - math.exp(1-R)
                        elif self.reward_design == 4:
                            rews[user] = -1*math.exp(1-R)
                        elif self.reward_design == 2:
                            rews[user] = -1*(1-R)
                        else:
                            print("This is not defined !!!!!")
                    else:
                        if self.reward_design == 3:
                            rews[user] = 1
                        elif self.reward_design == 4:
                            rews[user] = math.exp(1)
                        elif self.reward_design == 2:
                            rews[user] = 1
                        else:
                            print("This is not defined !!!!!")
                else:
                    if transmission is True:  # If there exist at least one transmission at this time step we assume
                        obs[user][i] = 1
                        # Receive the packet of the closest transmitter, and use the distance as a part of the state.
                        tx_dist, tx_id = self.network.find_closest_tx(transmitters_tti, user)
                        if tx_id is not None:
                            self.network.last_arrival_time[tx_id][user] = time_step
                            if self.add_positional_dist_piggy:
                                # None means that the receiver is out of the coverage of the transmitters.
                                self.network.received_update(rx_id=user, tx_id=tx_id)

        self.network.update_mobility(timestep=time_step)

        return obs, rews

    # def calculate_reward(self, transmitters_tti):
    #     """
    #     This function is called when there are more then one UE that uses the same resource
    #     :param transmitters_tti:
    #     :return:
    #     """
    #     rewards = {}
    #     for tx in transmitters_tti:
    #         rewards[tx] = 0
    #         for rx in range(self.NUM_USERS):
    #             if rx in transmitters_tti:
    #                 continue
    #             if self.nearest_tx(tx, rx, transmitters_tti)
    #
    # def nearest_tx(self, tx, rx, transmitter_tti):
    #     """
    #     Check whether the transmitter is the closest user the receiver.
    #     :param tx:
    #     :param rx:
    #     :param transmitter_tti:
    #     :return: True or False
    #     """
    #     nearest = True
    #     dist_tx_rx = self.network.
    #

    def get_x_pos(self):
        """
        Return the x positions of the users in the topology.
        :return:
        """
        return self.network.get_x_positions()


    def reset_mobility_env(self):
        """
        Resets the mobility environment, initialize again.
        :return:
        """
        self.network.reset_positions()

    def get_total_users(self):
        return self.NUM_USERS

    def get_num_ch(self):
        return self.NUM_CHANNELS

    def get_state_space(self):
        return self.state_space

    def get_action_space(self):
        return self.action_space

    def update_velocity(self):
        """
        Update the velocity of the vehicles if activated.
        :return:
        """
        if self.mobility_vary:
            self.network.update_velocity()

    #generates next-state from action and observation
    def state_generator(self, action, obs):
        """
        This function is enabled for DQN algorithm to construct the state
        based on the received actions ans channel observations.
        :param action:
        :param obs:
        :return:
        """
        input_vector = []
        if action is None:
            print ('None')
            sys.exit()
        for user_i in range(action.size):
            input_vector_i = self.one_hot(action[user_i], self.action_space)
            channel_alloc = obs[-1]
            input_vector_i = np.append(input_vector_i, channel_alloc)
            input_vector_i = np.append(input_vector_i, int(obs[user_i][0]))    #ACK
            input_vector.append(input_vector_i)
        return input_vector

    def obtain_state(self, obs, acts, rewards, episode_number=0, epsilon=1):
        """
        This function constructs the state to be exploited by the DRQN algorithm.
        :param obs: Vector includes the observation of each user.
        :param acts: Vector includes the action of each user.
        :param rewards: Vector includes the reward of each user.
        :return: State vector as an input to the DRQN algorithm calculated for each user and concatenated.
        """
        input_vector = []

        for user_i in range(len(obs)):
            input_vector_i = []
            if self.add_action:
                if self.action_index == "binary":
                    input_vector_i = self.one_hot(acts[user_i], self.action_space)
                elif self.action_index == "real":
                    input_vector_i = acts[user_i]
                else:
                    print("Error at def obtain_state!!!")
            channel_obs_i = obs[user_i]
            if self.add_channel_obs:
                input_vector_i = np.append(input_vector_i, channel_obs_i)
            if self.add_positional_dist:
                # We assume that each UE knows the positions of others in the vicinity.
                # TODO: this function now returns the positional distribution of its own object.
                pos_dis_i = self.network.get_positional_dist(user_i)
                input_vector_i = np.append(input_vector_i, pos_dis_i)
            if self.add_positional_dist_piggy:
                if self.add_positional_dist_type == 1:
                    pos_dis_i_piggy = self.network.get_positional_dist_piggy(user_i)
                elif self.add_positional_dist_type == 2:
                    pos_dis_i_piggy = self.network.get_positional_dist_2_piggy(user_i)
                else:
                    print("Error; this is not defined")

                input_vector_i = np.append(input_vector_i, pos_dis_i_piggy)

            if self.add_positional_velocity:
                pos_vel_i = self.network.get_positional_velocity(user_i)
                input_vector_i = np.append(input_vector_i, pos_vel_i)

            if self.add_reward:
                rew_i = rewards[user_i]
                input_vector_i = np.append(input_vector_i, rew_i)
            if self.add_index:
                input_vector_i = np.append(input_vector_i, user_i+1)
            if self.add_position:
                input_vector_i = np.append(input_vector_i, self.network.get_position(user_i))
            if self.add_velocity:
                input_vector_i = np.append(input_vector_i, self.network.get_velocity(user_i))
            if self.enable_fingerprint:
                input_vector_i = np.append(input_vector_i, episode_number)
                input_vector_i = np.append(input_vector_i, epsilon)

            input_vector.append(input_vector_i)

        return input_vector

    def one_hot(self, num, len):
        """
        It creates a one hot vector of a number as num with size as len
        :param num: index of 1.
        :param len: total length of the vector.
        :return:
        """
        assert num >=0 and num < len ,"error"
        vec = np.zeros([len],np.int32)
        vec[num] = 1
        return vec
