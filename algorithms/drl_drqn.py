"""
This file includes the algorithm structure that is used for resource allocation.
Features:
    Adding LSTM layer as an input of the DQN network.
    Adding reward or agent index as a part of the state which can be adjusted from config file.
    Training with different batch size and number are part of the config file.
"""
import tensorflow as tf
import sys, os
import numpy as np
from policies import BoltzmanPolicy, SoftmaxPolicy, EpsilonGreedy, GreedyPolicy, RandomPolicy
from collections import deque
#tf.config.optimizer.set_jit(True) # Enable XLA.
#tf.debugging.set_log_device_placement(True)

class DRQN:
    """
    Deep Recurrent Q Learning algorithm for distributed and dynamic resource allocation problem.
    This class can be used both for realness and test simulator.
    """
    def __init__(self, env, name="DeepRQN", total_episodes=4000, **kwargs):
        """
        Initialize the required variables and also the network.
        :param env: Either realness or test environment.
        :param name: Name of the scenario to test
        :param kwargs: Additional parameters.
        """
        tf.reset_default_graph()
        self.name = name  # Name of the experiment
        self.num_users = env.get_total_users()  # Number of users in the test scenario
        #========================== Add parameters of agent =============================#
        self.learning_rate = kwargs.setdefault("learning_rate", 1e4)  # Learning rate of the DQN algorithm.
        self.target_update = kwargs.setdefault("target_update", 10)   # every determined time step, we update the target values with the current DQN network parameters(stabilizes learning)
        self.batch_size = kwargs.setdefault("batch_size", 64)   # Train the network with this determined batch_size, changing this value also affect the learning
        self.step_size = kwargs.setdefault("step_size", 5)      # This variable is used for LSTM network, observe not only the last one t but also within the interval [t-step_size, t] as a part of the state.
        self.n_batch = kwargs.setdefault("n_batch", 2)          # Number of times we train the our Q network.
        self.hysteretic = kwargs.setdefault("hysteretic", False) # This is a special technique that I wanted to test. Decreasing the learning rate when TD error is lowe can stabilizes the performance.
        self.state_size = env.get_state_space()                 # How big the state depends on the num users and and add reward and index.
        self.action_size = env.get_action_space()               # Number of available channels that UE can select defines the action space
        self.beta = kwargs.setdefault("beta", 1)                # variable is used for boltzman training
        self.explore_start = kwargs.setdefault("explore_start", 4)  # variable is used for boltzman training
        self.explore_stop = kwargs.setdefault("explore_stop", 4)  # variable is used for boltzman training
        self.decay_rate = kwargs.setdefault("decay_rate", 4)  # variable is used for boltzman training
        self.alpha = kwargs.setdefault("alpha", 1)  # variable is used for eps greedy policy
        self.eps = kwargs.setdefault("eps_init", 1)  # variable is used for eps greedy policy
        self.eps_decay = kwargs.setdefault("eps_decay", 0.9999)
        self.temperature = kwargs.setdefault("temperature", 0.001)

        self.network_param = kwargs.setdefault("network", False)
        self.use_lstm_input = self.network_param["use_lstm_input"]  # Use the lstm layer as an input to the DQN
        self.use_dueling = self.network_param["use_dueling"]        # An approach to improve the performance
        self.use_double = self.network_param["use_double"]          # Another approach to improve the performance
        self.nn_layers = self.network_param["layers"]               # Num of deep layers in the network.
        if 1: #with tf.device("cpu"):
        # ========================== Build Network ============================= #
            if self.use_lstm_input:
                self.inputs_ = tf.placeholder(tf.float32, [None, self.step_size, self.state_size], name='inputs_')
            else:
                self.inputs_ = tf.placeholder(tf.float32, [None, self.state_size], name='inputs_')
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            self.one_hot_actions = tf.one_hot(self.actions_, self.action_size)
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

        # Create the network and training parameters below.
            with tf.variable_scope(name):
                with tf.variable_scope("eval_net_scope"):
                    self.eval_scope_name = tf.get_variable_scope().name
                    self.qvalues = self._create_network()

                with tf.variable_scope("target_net_scope"):
                    self.target_scope_name = tf.get_variable_scope().name
                    self.target_qvalues = self._create_network()

            # ============== LOSS ============== #
                self.gamma = kwargs.setdefault("gamma", 0.99)
                self.Q = tf.reduce_sum(tf.multiply(self.qvalues, self.one_hot_actions), axis=1)
                self.h_loss = self.Q - self.targetQs_  # Loss value is calculated by Hysteretic theorem.
                if self.hysteretic:
                    self.h_loss = tf.where(self.h_loss<0, self.h_loss/10, self.h_loss)
                self.loss = tf.reduce_mean(tf.square(self.h_loss))
                self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # target network update op
            self.update_target_op = []
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.target_scope_name)
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.eval_scope_name)
            for i in range(len(t_params)):
                self.update_target_op.append(tf.assign(t_params[i], e_params[i]))

        # init tensorflow session
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

        #################### Create Policy ######################
            self.policy_name = kwargs.setdefault("policy", "")
            if self.policy_name == "softmax":
                self.policy = SoftmaxPolicy(nA=self.action_size, temperature=self.temperature, episodes=total_episodes)
            elif self.policy_name == "boltzman":
                self.policy = BoltzmanPolicy(self.action_size, beta=self.beta, explore_start=self.explore_start,
                                        explore_stop=self.explore_stop, decay_rate=self.decay_rate, alpha=self.alpha)
            elif self.policy_name == "eps_greedy":
                self.policy = EpsilonGreedy(eps_init=self.eps, eps_decay=self.eps_decay, nA=self.action_size, episodes=total_episodes, explore_stop=self.explore_stop)
            else:
                self.policy = GreedyPolicy(nA=self.action_size)


    def _create_network(self):
        """
        Creates a deep neural network as a function approximation of DQN algorithm.
        :return: Model for networks
        """
        hidden_size = list(self.nn_layers.values())
        #hidden_size = hidden_size[0]
        if self.use_lstm_input:
            lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size[0])
            #lstm = tf.contrib.rnn.GRUCell(hidden_size)
            lstm_out, state = tf.nn.dynamic_rnn(lstm, self.inputs_, dtype=tf.float32)
            reduced_out = lstm_out[:,-1,:]
            reduced_out = tf.reshape(reduced_out, shape=[-1, hidden_size[0]])

        else:
            w1 = tf.Variable(tf.random_uniform([self.state_size, hidden_size[0]]))
            b1 = tf.Variable(tf.constant(0.1, shape=[hidden_size[0]]))
            h1 = tf.matmul(self.inputs_, w1) + b1
            h1 = tf.nn.relu(h1)
            h1 = tf.contrib.layers.layer_norm(h1)

        w2 = tf.Variable(tf.random_uniform([hidden_size[0], hidden_size[1]]))
        b2 = tf.Variable(tf.constant(0.1, shape=[hidden_size[1]]))
        if self.use_lstm_input:
            h2 = tf.matmul(reduced_out, w2) + b2
        else:
            h2 = tf.matmul(h1, w2) + b2
        h2 = tf.nn.relu(h2)
        h2 = tf.contrib.layers.layer_norm(h2)

        if len(hidden_size) == 3:
            w3 = tf.Variable(tf.random_uniform([hidden_size[1], hidden_size[2]]))
            b3 = tf.Variable(tf.constant(0.1, shape=[hidden_size[2]]))
            h3 = tf.matmul(h2, w3) + b3
            h3 = tf.nn.relu(h3)
            h3 = tf.contrib.layers.layer_norm(h3)

            w4 = tf.Variable(tf.random_uniform([hidden_size[2], self.action_size]))
            b4 = tf.Variable(tf.constant(0, 1, shape=[self.action_size]))
            output = tf.matmul(h3, w4) + b4

        elif len(hidden_size) == 2:
            w3 = tf.Variable(tf.random_uniform([hidden_size[1], self.action_size]))
            b3 = tf.Variable(tf.constant(0, 1, shape=[self.action_size]))
            output = tf.matmul(h2, w3) + b3

        return output

    def infer_action(self, user, state_vector, episode, policy="boltzman"):
        """
        This function is used to take the action based on input.
        :param user: Id os the user
        :param state_vector: State of the user
        :param time_slot: Given time that the UE takes the decision.
        :param policy: Which policy to be applied for learning
        :return: action determined by the policy, a
        """
        #feeding the input-history-sequence of (t-1) slot for each user seperately
        if policy == "explore":
            action = np.random.randint(self.action_size)
            return action

        if self.use_lstm_input:
            feed = {self.inputs_:state_vector[:, user].reshape(1, self.step_size, self.state_size)}
        else:
            feed = {self.inputs_:state_vector[-1:,user].reshape(1, self.state_size)}
        Qs = self.sess.run(self.qvalues, feed_dict=feed)

        if policy =="greedy":
            action = np.argmax(Qs, axis=1)
        else:
            action = self.policy.action(Qs, episode)

        return action

    def set_eps(self, eps):
        """
        Used when we load the model, then we can start from a specific eps.
        :param eps:
        :return:
        """
        self.policy.set_epsilon(eps)

    def get_eps(self):
        """
        Get the eps value of the model
        :return: self.eps
        """
        return self.policy.get_epsilon()

    def train(self, sample_buffer, time_step):
        """
        Train the model based on the collected experience. Update the target network time to time.
        :param sample_buffer:
        :param time_step:
        :return:
        """
        # TODO: add training freq.
        n_batches = self.n_batch
        for k in range(n_batches):

            #  sampling a batch from memory buffer for training
            if self.use_lstm_input:
                batch = sample_buffer.sample(self.batch_size, self.step_size)
            else:
                batch = sample_buffer.sample(self.batch_size, 1)

            #   matrix of rank 4
            #   shape [NUM_USERS,batch_size,step_size,state_size]
            states = self.get_states_user(batch)

            #   matrix of rank 3
            #   shape [NUM_USERS,batch_size,step_size]
            actions = self.get_actions_user(batch)

            #   matrix of rank 3
            #   shape [NUM_USERS,batch_size,step_size]
            rewards = self.get_rewards_user(batch)

            #   matrix of rank 4
            #   shape [NUM_USERS,batch_size,step_size,state_size]
            next_states = self.get_next_states_user(batch)

            #   Converting [NUM_USERS,batch_size]  ->   [NUM_USERS * batch_size]
            #   first two axis are converted into first axis
            if self.use_lstm_input:
                states = np.reshape(states, [-1, states.shape[2], states.shape[3]])
                actions = np.reshape(actions, [-1, actions.shape[2]])
                rewards = np.reshape(rewards, [-1, rewards.shape[2]])
                next_states = np.reshape(next_states, [-1, next_states.shape[2], next_states.shape[3]])
            else:
                states = np.reshape(states, [-1, states.shape[3]])
                actions = np.reshape(actions, [-1])
                rewards = np.reshape(rewards, [-1])
                next_states = np.reshape(next_states, [-1, next_states.shape[3]])

            # creating target vector (possible best action)
            # target_Qs = self.sess.run(self.target_qvalues, feed_dict={self.inputs_:next_states})
            #  creating target vector (possible best action)
            #target_Qs = self.sess.run(self.qvalues, feed_dict={self.inputs_: next_states})

            #  Q_target =  reward + gamma * Q_next
            # targets = rewards[:,-1] + self.gamma * np.max(target_Qs, axis=1)
            targets = self._calc_target(rewards, next_states)
            #  calculating loss and train using Adam  optimizer
            if self.use_lstm_input:
                acts = actions[:,-1]
            else:
                acts = actions
            loss, _ = self.sess.run([self.loss, self.opt],
                                        feed_dict={self.inputs_:states,
                                        self.targetQs_:targets,
                                        self.actions_:acts})

        if (time_step+1) % self.target_update == 0:
            # print("Target Q update ct " + str(time_step))
            self.sess.run(self.update_target_op)

    def _calc_target(self, rewards, next_states):
        """
        Calculates the TD(temporal difference) target to be used for training.
        :param rewards: matrix with size (num_users*batch_size, step size)
        :param next_states: matrix with size (num_users*batch_size, step_size, state_space)
        :return:
        """
        n = len(rewards)
        if self.use_double:
            t_qvalues, qvalues = self.sess.run([self.target_qvalues, self.qvalues],
                                               feed_dict={self.inputs_: next_states})
            act = np.argmax(qvalues, axis=1)

            next_value = t_qvalues[np.arange(n), act]

        else:
            t_qvalues = self.sess.run(self.target_qvalues, feed_dict={self.inputs_: next_states})
            next_value = np.max(t_qvalues, axis=1)

        #  Q_target =  reward + gamma * Q_next
        if self.use_lstm_input:
            target = rewards[:, -1] + self.gamma * next_value
        else:
            target = rewards + self.gamma * next_value

        return target

    def get_states_user(self, batch):
        """
        Receives a batch and returns the collected states of each user
        :param batch: (batch_size, step_size, (states_step, actions_step, rewards_step, next_state_step))
        :return:
        """
        states = []
        for user in range(self.num_users):
            states_per_user = []
            for each in batch:
                states_per_batch = []
                for step_i in each:
                    try:
                        states_per_step = step_i[0][user]

                    except IndexError:
                        print (step_i)
                        print ("-----------")

                        print ("error")

                        '''for i in batch:
                            print i
                            print "**********"'''
                        sys.exit()
                    states_per_batch.append(states_per_step)
                states_per_user.append(states_per_batch)
            states.append(states_per_user)
        #print len(states)
        return np.array(states)

    def get_actions_user(self, batch):
        """
        Receives a batch and returns the collected actions of each user
        :param batch: (batch_size, step_size, (states_step, actions_step, rewards_step, next_state_step))
        :return: collected actions
        """
        actions = []
        for user in range(self.num_users):
            actions_per_user = []
            for each in batch:
                actions_per_batch = []
                for step_i in each:
                    actions_per_step = step_i[1][user]
                    actions_per_batch.append(actions_per_step)
                actions_per_user.append(actions_per_batch)
            actions.append(actions_per_user)
        return np.array(actions)

    def get_rewards_user(self, batch):
        """
        Receives a batch and returns the collected rewards of each user
        :param batch: (batch_size, step_size, (states_step, actions_step, rewards_step, next_state_step))
        :return: collected rewards
        """
        rewards = []
        for user in range(self.num_users):
            rewards_per_user = []
            for each in batch:
                rewards_per_batch = []
                for step_i in each:
                    rewards_per_step = step_i[2][user]
                    rewards_per_batch.append(rewards_per_step)
                rewards_per_user.append(rewards_per_batch)
            rewards.append(rewards_per_user)
        return np.array(rewards)

    def get_next_states_user(self, batch):
        """
        Receives a batch and returns the the next states of each user
        :param batch: (batch_size, step_size, (states_step, actions_step, rewards_step, next_state_step))
        :return: collected next states.
        """
        next_states = []
        for user in range(self.num_users):
            next_states_per_user = []
            for each in batch:
                next_states_per_batch = []
                for step_i in each:
                    next_states_per_step = step_i[3][user]
                    next_states_per_batch.append(next_states_per_step)
                next_states_per_user.append(next_states_per_batch)
            next_states.append(next_states_per_user)
        return np.array(next_states)

    def sample(self):
        """
        Sample action randomly to be used for initilization phase.
        """
        action_sampled = np.random.randint(self.action_size)
        return action_sampled

    def save_model(self, dir_name, slot, simulation):
        """
        Save model to dir
        :param dir_name: str
            Name of the directory
        :param epoch: int
        :return:
        """
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = os.path.join(dir_name, self.name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        saver = tf.train.Saver(model_vars)
        saver.save(self.sess, os.path.join(dir_name, ("sim_%d_%d") % (simulation, slot)))

    def load_model(self, dir_name, epoch=0, name=None):
        """
        load model from dir
        :param dir_name: str
            name of the directory
        :param epoch:
        :param name:
        :return:
        """
        if name is None or name == self.name:  # the name of saved model is the same as ours
            dir_name = os.path.join(dir_name, self.name)
            model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            #model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            saver = tf.train.Saver(model_vars)
            # dir_name = dir_name + "/sim_0"
            saver.restore(self.sess, os.path.join(dir_name, ("sim_0_%d") % epoch))
        else:  # load a checkpoint with different name
            backup_graph = tf.get_default_graph()
            kv_dict = {}

            # load checkpoint from another saved graph
            with tf.Graph().as_default(), tf.Session() as sess:
                tf.train.import_meta_graph(os.path.join(dir_name, name, (self.subclass_name + "_%d") % epoch + ".meta"))
                dir_name = os.path.join(dir_name, name)
                model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(model_vars)
                saver.restore(sess, os.path.join(dir_name, (self.subclass_name + "_%d") % epoch))
                for item in tf.global_variables():
                    kv_dict[item.name] = sess.run(item)

            # assign to now graph
            backup_graph.as_default()
            model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            for item in model_vars:
                old_name = item.name.replace(self.name, name)
                self.sess.run(tf.assign(item, kv_dict[old_name]))
