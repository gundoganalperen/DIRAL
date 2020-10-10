import numpy as np
import math
import random
from itertools import combinations
from numpy import linalg as LA
import matplotlib.pyplot as plt
from collections import defaultdict

from vehicle import Vehicle

class Network:
    """
    This class is exploited by the test_env
    """
    def __init__(self, congestion_test, num_users, num_bins=20, topology="circle", radius=5, avg_distance=True,
                 mobility=True, mobility_vary=False, highway_len=200, communication_range=350,
                 bin_range=500, enable_design_topology=False):
        self.num_users = num_users
        self.topology = topology
        self.radius = radius  #  Coverage radius of the network
        self.pos_of_nodes = []
        self.norm = math.sqrt(2*(radius**2))
        self.avg_distance = avg_distance
        self.communication_range = communication_range

        self.mobility = mobility  # do we have mobility in the environment.
        self.mobility_vary = mobility_vary  # do we change the mobility in every episode?
        self.vehicles = []
        self.v_max = 10  # max velocity that a vehicle can have 10m/sn, 1m/100ms
        self.highway_length = highway_len  # length of the highway, user return back the beginning position.
        self.highway_height = 2 # length of the highway, user return back the beginning position.
        self.update_position_interval = 10
        self.update_position_counter = 0  # every (self.update_position_interval*100ms) seconds we update the positions of the UEs.
        self.enable_design_topology_net = enable_design_topology
        self.load_positions_enable = False
        self.toy_example = congestion_test

        # this variable holds the last arrival time of the packet between each tx-rx pair
        self.last_arrival_time = defaultdict(dict)
        for tx_id in range(self.num_users):
            for rx_id in range(self.num_users):
                self.last_arrival_time[tx_id][rx_id] = -1

        self.bin_range = None
        # if self.communication_range*2 > self.highway_length:
        #     self.bin_range = self.highway_length
        # else:
        #     self.bin_range = self.communication_range*2

        self.bin_range = bin_range
        print("Bin Range: " + str(self.bin_range))
        self.bins = num_bins

        if self.mobility and not self.enable_design_topology_net:
            if congestion_test and 0:
                self.initialize_mobility_topology_fixed()  # initialize toy example.
            else:
                self.initialize_mobility_topology(num_users, self.highway_length, self.highway_height)
        if self.enable_design_topology_net:
            self.initialize_mobility_topology_design_test()
        if congestion_test:
            a = 5
            # self.initialize_fixed()  # This is used to test the algorithm. (5UE, 4 resources)
        else:
            a = 5
            #self.initialize_topology()    # Initialize the topology randomly based on the number of users.
        #self.plot_fc()

    def initialize_mobility_topology_design_test(self):
        """
        Initialize a tpopology for design test in longer scnearios.
        :return:
        """
        self.vehicles.append(Vehicle(num_users=6, id=0, pos_x=0, pos_y=1, velocity=1.0, start_direction="right"))
        self.vehicles.append(Vehicle(num_users=6, id=1, pos_x=195, pos_y=1, velocity=1.0, start_direction="right"))
        self.vehicles.append(Vehicle(num_users=6, id=2, pos_x=390, pos_y=2, velocity=1.0, start_direction="right"))
        self.vehicles.append(Vehicle(num_users=6, id=3, pos_x=585, pos_y=2, velocity=1.0, start_direction="right"))
        self.vehicles.append(Vehicle(num_users=6, id=4, pos_x=780, pos_y=2, velocity=1.0, start_direction="right"))
        self.vehicles.append(Vehicle(num_users=6, id=5, pos_x=975, pos_y=2, velocity=1.0, start_direction="right"))

    def initialize_mobility_topology_fixed(self):
        """
        Initialize the fixed mobility topology to be used for all experiments for fair comparision
        :param num_users: 4 USERS 3 channels
        :return:
        """
        self.vehicles.append(Vehicle(num_users=4, id=0, pos_x=3, pos_y=1, velocity=0.5, start_direction="right"))
        self.vehicles.append(Vehicle(num_users=4, id=1, pos_x=5, pos_y=1, velocity=1.0, start_direction="right"))
        self.vehicles.append(Vehicle(num_users=4, id=2, pos_x=3, pos_y=2, velocity=1.25, start_direction="right"))
        self.vehicles.append(Vehicle(num_users=4, id=3, pos_x=5, pos_y=2, velocity=1.5, start_direction="right"))

    def initialize_mobility_topology(self, num_users, highway_length, highway_height):
        """
        In this function we define a highway with two directions i.e. "l" or "r"
        :param num_users:
        :param highway_length:
        :param highway_height:
        :return:
        """
        for user in range(num_users):
            if user % 1 == 0: ## only one direction
                # Then user is located at the bottom of the road and it moves to right with determined velocity.
                pos_x = np.random.randint(0, highway_length)
                pos_y = np.random.randint(0, highway_height/2)
                if self.mobility_vary:
                    # If this is activated then we set the velocity to a fixed point and either increase or reduce
                    # to introduce randomness.
                    velocity = 1.7
                else:
                    velocity = random.uniform(1.1, 2.7) # 20-100kmph1m/100ms, 10m/s, 36km/h np.random.randint(1, self.v_max)  # Velocity should not 1
                direction = "right"
                self.vehicles.append(Vehicle(num_users=num_users, id=user, pos_x=pos_x, pos_y=pos_y, velocity=velocity, start_direction=direction))
            else:
                # Then user is located upper side of the road and it moves to left with determined velocity.
                pos_x = np.random.randint(0, highway_length)
                pos_y = np.random.randint(highway_height/2, highway_height)
                velocity = np.random.randint(1, self.v_max)
                direction = "left"
                self.vehicles.append(Vehicle(num_users=num_users, id=user, pos_x=pos_x, pos_y=pos_y, velocity=velocity, start_direction=direction))


    def is_comm_range(self, tx_user, other_tx):
        """
        This function checks whether the transmitters are in the coverage age of each other?
        :param tx_user:
        :param other_tx:
        :return:
        """
        dist_tx_rx = self.dist(tx_user, other_tx)
        if dist_tx_rx < 2*self.communication_range:
            return True
        else:
            return False

    def calculate_reward_weights_design(self, tx_user, collided_users):
        """
        This function calculates the weight based on the positions of collided users.
        Calculated weight will be used for all collided users at this timestamp.
        :param collided_users: id of the collided users.
        :return: weight of each collided users
        """
        weights = []
        # tuple of coordinates i.e. m = (m_x, m_y)

        if self.avg_distance:
            m = self.calculate_avg_distance(collided_users)
            # norm = self.calculate_norm()
            #norm = self.calculate_norm_pool(tx_user)

            if m > self.communication_range*2:
                weights = 1
            else:
                weights = 0
            #weights.append(math.exp(m)/math.exp(norm))
            # return weights

        return weights


    def get_x_positions(self):
        """
        return the x coordinates of the vehicles.
        :return:
        """
        x_coordinates = []
        for user in range(self.num_users):
            x_coordinates.append(self.vehicles[user].get_x_pos())

        return x_coordinates

    def load_x_positions(self, filename):
        """
        Load the positions that are obtained from the RealNess
        :param filename:
        :return:
        """
        self.x_positions = np.array(np.load(filename))
        self.load_positions_enable = True


    def reset_positions(self):
        """
        Initialize the position that is assinged at first.
        :return:
        """
        self.vehicles = []
        self.initialize_mobility_topology_fixed()

    def update_positions(self, timestep):
        """
        Update the positions of the cars in the topology.
        :return:
        """
        if self.load_positions_enable:
            for user in range(self.num_users):
                lent = len(self.x_positions)
                timestep = timestep % lent
                self.vehicles[user].pos_x = self.x_positions[timestep][user]
                self.vehicles[user].pos[0] = self.x_positions[timestep][user]
        else:
            for user in range(self.num_users):
                if self.vehicles[user].direction == "right":
                    self.vehicles[user].pos_x = (self.vehicles[user].pos_x + (self.vehicles[user].velocity) + self.highway_length) % self.highway_length
                else:
                    self.vehicles[user].pos_x = (self.vehicles[user].pos_x - (self.vehicles[user].velocity) + self.highway_length) % self.highway_length
                self.vehicles[user].pos[0] = self.vehicles[user].pos_x

    def update_velocity(self):
        """
        Update the velocity of the each vehicle after every episode.
        :return:
        """
        for user in range(self.num_users):
            rand = random.randrange(1, 4)
            if rand == 1:
                self.vehicles[user].velocity += 0.55  # +20kmph
                if self.vehicles[user].velocity > 2.77:  # max 100
                    self.vehicles[user].velocity = 2.77
            elif rand == 2:
                self.vehicles[user].velocity -= 0.55  #-20kmph
                if self.vehicles[user].velocity < 1.1:  # min 40
                    self.vehicles[user].velocity = 1.1
            # Else do not change the velocity

    def calculate_norm(self):
        """
        Norm is the largest distance among vehicles which should be calculated every time we update the topology.
        :return: norm, value is used for reward calculations.
        """
        # Find the users that are far away from each other.
        x_min = self.highway_length+1
        x_max = -self.highway_length-1
        x_min_user = None
        x_max_user = None
        for user in range(self.num_users):
            if self.vehicles[user].pos_x < x_min:
                x_min = self.vehicles[user].pos_x
                x_min_user = user
            if self.vehicles[user].pos_x > x_max:
                x_max = self.vehicles[user].pos_x
                x_max_user = user

        # Max distance among the users will be the norm.
        self.norm = self.dist(x_min_user, x_max_user)

        return self.norm

    def initialize_topology(self):
        """
        Initialize the topology randomly within a radius
        without mobility
        :return:
        """
        for i in range((self.num_users)):
            x = random.randint(0, self.radius)
            y = random.randint(0, self.radius)
            pnt = (x, y)
            self.pos_of_nodes.append(pnt)

    def initialize_fixed(self):
        """
        Test case for 5 UEs with determined locations, used to test for congestion_12ue_10r control.
        :return:
        """
        # Points are selected from the x:[0,16] y: [0,4]
        pnt_s = [(1,2), (6,3), (7,1), (8,3), (15,2)]
        for i in range(len(pnt_s)):
            self.pos_of_nodes.append(pnt_s[i])
        self.norm = 14  # longest distance between nodes #math.sqrt(16 + 16**2)


    #  This logic can be used also later to give individual rewards basen on the distributions of location.
    def calculate_reward_weights(self, collided_users):
        """
        This function calculates the weight based on the positions of collided users.
        Calculated weight will be used for all collided users at this timestamp.
        Call this function only when there are two users that use the same resources.
        :param collided_users: id of the collided users.
        :return: weight of each collided users
        """
        weights = []
        # tuple of coordinates i.e. m = (m_x, m_y)
        m = self.calculate_avg_distance(collided_users)
        if self.toy_example:
            norm = self.calculate_norm()
            if m == norm:
    #            weights.append(math.exp(m) / math.exp(norm))
                weights.append(1)
            else:
                weights.append(0)
        else:
            if m > self.communication_range:
                weights.append(1)
            else:
                weights.append(0)
          #  norm = self.calculate_norm()
          #  weights.append(math.exp(m)/math.exp(norm))
            # return weights

        return weights

    def update_mobility(self, timestep):
        # Update the positions after the transmission.
        if self.mobility:  #  if we enable the mobility.
            self.update_positions(timestep)

    def calculate_avg_distance(self, collided_users):
        """
        Calculates the averages distance among collided users.
        This value is later used to calculate the reward.
        :param collided_users:
        :return: avg_distance
        """
        distances = [self.dist(p1, p2) for p1, p2 in combinations(collided_users, 2)]
        avg_distance = sum(distances) / len(distances)
        return avg_distance

    def dist(self, p1, p2):
        """
        Distance information among users.
        :param p1: Location(x, y) of user 1
        :param p2: Location(x, y) of user 2
        :return:
        """
        if 1: #self.mobility:
            (x1, y1) = self.vehicles[p1].pos_x, self.vehicles[p1].pos_y
            (x2, y2) = self.vehicles[p2].pos_x, self.vehicles[p2].pos_y
        else:
           # (x1, y1), (x2, y2) = self.pos_of_nodes[p1], self.pos_of_nodes[p2]
            (x1, y1), (x2, y2) = self.pos_of_nodes[p1], self.pos_of_nodes[p2]

        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def dist_sign(self, user, tx_user):
        """
        Return the sign of the distance difference aamong user.s
        :param user:
        :param tx_user:
        :return: 1 or -1
        """
        if self.mobility:
            (x1, y1), (x2, y2) = self.vehicles[user].pos, self.vehicles[tx_user].pos,
        else:
            (x1, y1), (x2, y2) = self.pos_of_nodes[user], self.pos_of_nodes[tx_user]

        if x1 - x2 > 0.0:
            return 1
        else:
            return -1

    def calculate_center(self, collided_users):
        """
        Calculates the center of the collided users.
        :param collided_users:
        :return: m center coordication point.
        """
        n = len(collided_users)
        mx = 0
        my = 0
        for user in collided_users:
            mx += self.pos_of_nodes[user][0]
            my += self.pos_of_nodes[user][1]
        mx = mx/n
        my = my/n
        return (mx, my)

    def calculate_weight_user(self, user, m):
        """
        Calculcates the weight of the given user based on
        :param user:
        :param m:  center of the collided users.
        :return: weight_i
        """
        user_pos = self.pos_of_nodes[user]
        weight = ((user_pos[0]-m[0])**2 + (user_pos[1]-m[1])**2) ** (1/2.0)
        return weight

    def find_closest_tx(self, tx_ids, rx_id):
        """
        This function receives the number of transmitters at the time slot t, and calculates the distance between the closest one.
        :param tx_ids: list os tx ids
        :param rx_id:  receives id
        :return: distance between rx and tx.
        """
        min_dist = 100000
        min_tx_id = None  # ID of the transmitter that rx accepted its message, this is used for piggybacking.
        for tx in tx_ids:
            dist_tx_rx = self.dist(tx, rx_id)
            if dist_tx_rx < self.communication_range:
                if dist_tx_rx < min_dist:
                    min_dist = dist_tx_rx
                    min_tx_id = tx
            else:
                self.last_arrival_time[tx][rx_id] = -1  # represent out of coverage

        #min_dist_norm = min_dist/self.norm
        # I dont use this norm for now 16.01
        return min_dist, min_tx_id

    def get_velocity(self, user):
        return self.vehicles[user].velocity

    def get_position(self, user):
        pos = self.vehicles[user].pos
        pos = [pos[0]/self.highway_length, pos[1]/self.highway_height]  # normalize the position

        return pos

    def get_positional_dist(self, tx_user):
        """
        Returns the distibutional of the other vehicles in vicinity. Assuming that each UE knows the positions of others.

        :param tx_user: Transmitter UE
        :return: distirutional of vehicles.
        """
        dist_vector = []
        max_dist = 0
        for user in range(self.num_users):
            if user == tx_user:
                continue
            dist_ = self.dist(user, tx_user)
            if dist_ > max_dist:
                max_dist = dist_
            dist_ = dist_ * self.dist_sign(user, tx_user)
            dist_vector.append(dist_)

        # Normalize the vector based on the max distance between the receiver.
        dist_sorted = sorted(dist_vector)
        dist_vector = np.array(dist_sorted) / max_dist
        return dist_vector

    def get_positional_dist_piggy(self, tx_user):
        """
        Get the position dist of the given vehicle based on the received positions of the vehicles nearby.
        :param tx_user: Transmitter id
        :return:
        """
        dist_vector = []
        max_dist = 0
        for user in range(self.num_users):
            if user == tx_user:
                continue
            success, dist_, sign = self.dist_piggy(user, tx_user)
            if success:
                if dist_ > max_dist:
                    max_dist = dist_
                dist_ = dist_ * sign
                dist_vector.append(dist_)

        # Bin magic
        #bins = np.linspace(-self.highway_length, self.highway_length, self.bins+1)
        if len(dist_vector)>0:
            bins = np.linspace(-1, 1, self.bins+1)
            dist_sorted = sorted(dist_vector)

            norm = LA.norm(dist_vector, np.inf)
            dist_normed = dist_sorted / norm
            #dist_normed = minmax_scale(dist_sorted)
            #binned_observation = np.histogram(dist_sorted, bins, weights=dist_sorted)[0]
            binned_observation = np.histogram(dist_normed, bins, weights=dist_normed)[0]
        # binned_observation_norm = binned_observation / max_dist
        else:
            binned_observation = np.zeros((self.bins,), dtype=int)

        #digitized = np.digitize(dist_sorted, bins)
        #bin_means = binned_statistic(dist_sorted, dist_sorted, bins=self.bins-1)
        #obs = np.zeros((self.bins,), dtype=int)
        #for obs_i in range(len(bin_means[2])):
        #    obs[bin_means[2][obs_i]] = 1  # set this to one

        return binned_observation

    def get_positional_dist_2_piggy(self, tx_user):
        """
        Get the position dist of the given vehicle based on the received positions of the vehicles nearby.
        :param tx_user: Transmitter id
        :return:
        """
        dist_vector = []
        max_dist = 0
        for user in range(self.num_users):
            if user == tx_user:
                continue
            success, dist_, sign = self.dist_piggy(user, tx_user)
            if success:
                #if dist_ < self.communication_range*2:  # consider the observation only in the communication range*2
                if dist_ < self.bin_range:  # consider the observation only in the observation range.
                    if dist_ > max_dist:
                        max_dist = dist_
                    dist_ = dist_ * sign
                    dist_vector.append(dist_)

        # Bin magic
        #bins = np.linspace(-self.highway_length, self.highway_length, self.bins+1)
        if len(dist_vector)>0:
            #bins = np.linspace(-1, 1, self.bins+1)
            # bins = np.linspace(-200, 200, self.bins+1)
            dist_sorted = sorted(dist_vector)
            # binned_observation_1 = np.histogram(dist_sorted, self.bins, range=(-200, 200), weights=dist_sorted)
            binned_observation_2 = np.histogram(dist_sorted, self.bins, range=(-self.bin_range, self.bin_range))[0]
            binned_observation = binned_observation_2 / float(len(dist_sorted))
            #binned_observation = np.histogram(dist_normed, bins, weights=dist_normed)[0]
        # binned_observation_norm = binned_observation / max_dist
        else:
            binned_observation = np.zeros((self.bins,), dtype=int)

        #digitized = np.digitize(dist_sorted, bins)
        #bin_means = binned_statistic(dist_sorted, dist_sorted, bins=self.bins-1)
        #obs = np.zeros((self.bins,), dtype=int)
        #for obs_i in range(len(bin_means[2])):
        #    obs[bin_means[2][obs_i]] = 1  # set this to one

        return binned_observation

    def get_positional_velocity(self, tx_user):
        """
        Create positional velocity vector.
        :param tx_user:
        :return:
        """
        dist_vector = []
        dist_vector_users = []
        max_dist = 0
        for user in range(self.num_users):
            if user == tx_user:
                continue
            success, dist_, sign = self.dist_piggy(user, tx_user)
            if success:
                #if dist_ < self.communication_range*2:  # consider the observation only in the communication range*2
                if dist_ < self.bin_range:  # consider the observation only in the observation range.
                    if dist_ > max_dist:
                        max_dist = dist_
                    dist_ = dist_ * sign
                    dist_vector.append(dist_)
                    dist_vector_users.append(tx_user)


    def dist_piggy(self, rx_id, tx_id):
        """
        Distance between received and transmistter, determined base on the neighrboring table of the transmitter
        :param rx_id:
        :param tx_id:
        :return: success, dist, sign
        """
        if self.mobility or self.enable_design_topology_net:
            # if transmitter has the updated position of the receiver, so that we hold updated positions of the other vehicles.
            if self.vehicles[tx_id].pos_of_neighbors[rx_id]["last_updated"] < 20:
                # If we receive a position update of this user via piggybacking or direct observation.
                (x1, y1) = self.vehicles[tx_id].pos_of_neighbors[rx_id]["xpos"], self.vehicles[tx_id].pos_of_neighbors[rx_id]["ypos"]
                (x2, y2) = self.vehicles[tx_id].pos_x, self.vehicles[tx_id].pos_y
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if x1 - x2 > 0.0: # if the observed neighbor located right side of the transmitter.
                    sign = 1
                else:  # if the observed neighbor located left side of the transmitter.
                    sign = -1
                return 1, dist, sign
            else:
                return 0, None, None

    def get_information_age(self, timestep):
        """
        This function is called after each episode?
        param: time_step
        :return:
        """
        ia_vs_timestep = [0 for _ in range(100)]
        for tx_id in range(self.num_users):
            for rx_id in range(self.num_users):
                if tx_id != rx_id:
                    if self.last_arrival_time[tx_id][rx_id] != -1: # so if in coverage of each other
                        ia = timestep - self.last_arrival_time[tx_id][rx_id]
                        if ia < 100:
                            ia_vs_timestep[ia] += 1
        return ia_vs_timestep

    def received_update(self, rx_id, tx_id):
        """
        Update the positions table of the vehicle based on the transmitted positions of the vehicles.
        :param rx_id:
        :param tx_id:
        :return:
        """
        piggybacked_positions = self.vehicles[tx_id].get_piggybacked_positions()

        self.vehicles[rx_id].received_update(piggybacked_positions)

    def periodic_update(self):
        """
        Update the position tables every time we transferred a message.
        :return:
        """
        for user in range(self.num_users):
            self.vehicles[user].periodic_update()

    def check_communicaiton_range(self, tx, rx):
        """
        Check whether tx and rx are in communication range.
        If they are return true, ptherwise false
        :param tx:
        :param rx:
        :return:
        """
        dist_tx_rx = self.dist(tx, rx)
        if dist_tx_rx < self.communication_range:
            return True
        else:
            return False

    def plot_fc(self):
        """
        Plot the network graph.
        :return:
        """
        print(self.pos_of_nodes)
        N = self.num_users
        x = np.zeros(N)
        y = np.zeros(N)
        for i in range(self.num_users):
            x[i] = self.pos_of_nodes[i][0]
            y[i] = self.pos_of_nodes[i][1]

        colors = np.random.rand(N)
        area = (self.radius * 2) ** 2  # 0 to 15 point radii

        for i in self.pos_of_nodes:
            plt.text(i[0], i[1], i)
            for j in self.pos_of_nodes:
                plt.plot([i[0], j[0]], [i[1], j[1]], 'black', lw='0.01')

        # print(PosOfNodes[2],PosOfNodes[3])
        # plt.plot([PosOfNodes[2][0], PosOfNodes[3][0]], [PosOfNodes[2][1], PosOfNodes[3][1]])
        plt.scatter(x, y, s=area, c=colors, alpha=0.5)
        plt.show()

