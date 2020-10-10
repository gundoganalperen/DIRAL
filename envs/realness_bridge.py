import sys
import zmq
import numpy as np
# from marl_agent import ma_messages_pb2 as pb
import ma_messages_pb2 as pb
import time
from collections import defaultdict


class RealNeSZmqBridge(object):
    """
    Python interface base class for the simulator. Includes information about the port numbers for communication with
    the simulator.
    """
    def __init__(self, port=0, start_sim=False, sim_seed=0):
        super(RealNeSZmqBridge, self).__init__()
        port = int(port)
        self.port = int(port)
        self.start_sim = start_sim  # This is left for future work.
        self.sim_seed = sim_seed  # Left for future work
        self.env_stopped = False
        self.state_received_time_ms = 0
        self.action_send_time_ms = 0
        self.total_received = 0
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        try:
            if port == 0 and self.start_sim:
                port = self.socket.bind_to_random_port('tcp://*', min_port=5001, max_port=10000, max_tries=100)
                print("Got new port for RealNeS interface: ", port)

            elif port == 0 and not self.start_sim:
                print("Cannot use port %s to bind" % str(port))
                print("Please specify correct port")
                sys.exit()

            else:
                self.socket.bind("tcp://*:%s" % str(self.port))

        except Exception as e:
            print("Cannot bind to tcp://*:%s as port is already in use" % str(port))
            print("Please specify different port or use 0 to get free port")
            sys.exit()

        if start_sim and sim_seed == 0:
            max_seed = np.iinfo(np.uint32).max
            sim_seed = np.random.randint(0, max_seed)
            self.simSeed = sim_seed

        self.force_env_stop_v = False
        self._total_users = None
        self._action_space = None
        self._observation_space = None
        self._state_space_type = None
        self.next_state = None
        self.init_state = None  # This state will be used when the simulation is reset.
        self.reward = 0
        self.done = None
        self.extraInfo = None
        self.next_state_rx = None
        self.is_first_step = True  # Set this to true to trigger the first step

        # Create a socket to collect rewards from the RewardCollector.
        #self.context = zmq.Context()
        #self.context_reward = zmq.Context()
        self.socket_rewards = self.context.socket(zmq.REQ)
        #self.socket_rewards.connect("tcp://*:%s" % str(self.port+2))  # use 5557 for requesting the rewards.
        self.socket_rewards.connect("tcp://localhost:5557")  # use 5557 for requesting the rewards.

        # Enable reward collector.
        #p = Process(target=RewardCollector, args=(self.port+1, ))
        #p.start()
        #t = threading.Thread(target=RewardCollector, args=(self.port+1, ))
        #t.start()

        #p.join()

    def initialize_env(self):
        """
        Initialize the environment; At first the simulator should be started in order to send the initialization
        message which includes total user, action space, state type etc. Those information is than later will be used to
        setup the RL agent e.g. neural networks, policy etc.
        :return:
        """
        request = self.socket.recv()
        simInitMsg = pb.MA_SimInitMsg()
        simInitMsg.ParseFromString(request)
        self._total_users = simInitMsg.total_users- 1 # since we disable one of the users.
        self._state_space_type = simInitMsg.state_space_type
        self._action_space = simInitMsg.action_space
        self._observation_space = simInitMsg.state_space

        reply = pb.MA_SimInitAck()
        reply.done = False
        reply.stopSimReq = False
        reply_msg = reply.SerializeToString()
        self.socket.send(reply_msg)

    def restart_sockets(self):
        """
        Restarts the sockets, this is used for restarting the simulation.
        :return:
        """
        self.socket.close()
        self.socket_rewards.close()
        time.sleep(1.0)
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:%s" % str(self.port))
        self.socket_rewards = self.context.socket(zmq.REQ)
        self.socket_rewards.connect("tcp://localhost:5557")

    def get_total_users(self):
        return self._total_users

    def get_action_space(self):
        return self._action_space

    def get_observation_space(self):
        return self._observation_space

    def get_state_type(self):
        return self._state_space_type

    def send_action(self, action):
        """
        Sends the given action to the realness simulator via the determined socket.
        :param action:
        :return:
        """
        reply = pb.MA_SchedulingGrant()
        reply.time_stamp = action
        reply.stop_simulation = False

        reply_msg = reply.SerializeToString()
        self.socket.send(reply_msg)
        return True

    def get_observation(self):
        """
        Gets the new observation, sequence number(sn) for each user.
        This is not activated for DRQN agent.
        :return:
        """
        request = self.socket.recv()
        env_state_msg = pb.MA_SchedulingRequest()
        env_state_msg.ParseFromString(request)
        user_id = env_state_msg.user_id
        sn = env_state_msg.SN
        state = np.array(env_state_msg.state)
        return user_id, sn, state
        #obs[user_id][sn] = state

    def get_observation_syn(self):
        """
        Receive observation for synchronized message for each user, this includes also reward.
        :return:
        """
        request = self.socket.recv()
        env_state_msg = pb.MA_SchedulingRequestSyn()
        env_state_msg.ParseFromString(request)
        user_id = env_state_msg.user_id
        sn = env_state_msg.SN
        state = np.array(env_state_msg.state)
        reward = env_state_msg.reward
        return user_id, sn, state, reward
        #obs[user_id][sn] = state

    def get_observation_syn_dist(self):
        """
           Receive observation for synchronized message for each user, this includes also reward.
            This function is separate since we have positional dist of others.(neighr table)
           :return:
        """
        request = self.socket.recv()
        env_state_msg = pb.MA_SchedulingRequestSynDist()
        env_state_msg.ParseFromString(request)
        user_id = env_state_msg.user_id
        sn = env_state_msg.SN
        nb_table = env_state_msg.neighbor
        pos_of_neighbors = defaultdict(dict)
        for user in range(len(nb_table)):
            neighbor_entry = nb_table[user]
            pos_of_neighbors[user]["xpos"] = neighbor_entry.pos_x
            pos_of_neighbors[user]["ypos"] = neighbor_entry.pos_y
            pos_of_neighbors[user]["seq_number"] = neighbor_entry.seq_num
            pos_of_neighbors[user]["last_updated"] = neighbor_entry.last_update

        reward = env_state_msg.reward

        # retuwn a dict.
        return user_id, sn, pos_of_neighbors, reward



    def get_observation_syn_sps(self):
        """
        Receive observation for synchronized message for each user, this includes also reward.
        SPS algorithms sends float instead of int.
        :return:
        """
        request = self.socket.recv()
        env_state_msg = pb.SPS_SchedulingRequestSyn()
        env_state_msg.ParseFromString(request)
        user_id = env_state_msg.user_id
        sn = env_state_msg.SN
        state = np.array(env_state_msg.state)
        reward = env_state_msg.reward
        return user_id, sn, state, reward

    def receive_rewards(self):
        """
        Receives the observed reward from the reward collector.
        Reward collecter is a subscriber socket which subscribes the published rewards from the simulator beacon agent.
        This is not activated for DRQN agent since it already receives the reward from the simulator including with
        state information.
        :return:
        """
        self.socket_rewards.send(b"Send my rewards")
        message = self.socket_rewards.recv()
        reward_received_msg = pb.MA_RewardSentAll()
        reward_received_msg.ParseFromString(message)
        # print("Received reply [ %s ]" % (message))
        return reward_received_msg

    def get_reward(self):
        return self.reward

    def restart_env(self):
        """
        This function is used to reset the observation space of the users.
        :return:
        """
        request = self.socket.recv()  # first receive a scheduling request

        reply = pb.MA_SchedulingGrant()
        reply.time_stamp = -1  # this will indicate that we should restart the script.
        reply.stop_simulation = True

        reply_msg = reply.SerializeToString()
        self.socket.send(reply_msg)