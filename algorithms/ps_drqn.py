"""Deep Recurrent Q network with parameter sharing(PS) for multi-agent reinforcement learning"""
import time
import os
import collections

import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

from base import TFBaseModel


class DeepRecurrentQNetwork(TFBaseModel):
    def __init__(self, env, name, **kwargs):
        """
        Init DRQN
        :param env: Environment
        :param name: str
            name of this model
        :param kwargs:
            other parameters including batch_size, learning_rate, reward_decay,
            train_freq, target_updaate, memory size, eval_obs, use_duelling,
            use_double, use_conv, num_gpu etc.
        """
        TFBaseModel.__init__(self, env, name, "tfdrqn")
        # ======================== set config  ========================
        self.env = env
        self.state_space = env.get_state_space()
        self.num_actions = env.get_action_space()
        self.num_users = env.get_total_users()

        self.batch_size = kwargs.setdefault("batch_size", 64)
        self.unroll_step = kwargs.setdefault("unroll_step", 8)
        self.learning_rate = kwargs.setdefault("learning_rate", 1e4)
        self.train_freq = kwargs.setdefault("training_freq", 1)
        self.target_update = kwargs.setdefault("target_update", 1000)
        self.eval_obs = kwargs.setdefault("eval_obs", None)

        self.network_param = kwargs.setdefault("network", False)
        self.use_dueling = self.network_param["use_dueling"]
        self.use_double = self.network_param["use_double"]
        self.skip_error = self.network_param["skip_error"]
        self.num_gpu = self.network_param["num_gpu"]
        self.use_conv = self.network_param["use_conv"]

        self.nn_layers = self.network_param["layers"]
        self.train_ct = 0
        self.agent_states = {}

        # ======================= build network =======================
        # input place holder
        self.target = tf.placeholder(tf.float32, [None], name="target")

        self.input_state = tf.placeholder(tf.float32, [None, self.state_space], name="input_state")
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.mask = tf.placeholder(tf.float32, [None], name="mask")

        self.batch_size_ph = tf.placeholder(tf.int32, [])
        self.unroll_step_ph = tf.placeholder(tf.int32, [])

        # build a graph
        with tf.variable_scope(self.name):
            with tf.variable_scope("eval_net_scope"):
                self.eval_scope_name = tf.get_variable_scope().name
                self.qvalues, self.state_in, self.rnn_state = \
                    self._create_network(self.input_state)
            with tf.variable_scope("target_net_scope"):
                self.target_scope_name = tf.get_variable_scope().name
                self.target_qvalues, self.target_state_in, self.target_rnn_state = \
                    self._create_network(self.input_state)

        # loss
        self.gamma = kwargs.setdefault("reward_decay", 0.99)
        self.actions_onehot = tf.one_hot(self.action, self.num_actions)
        self.td_error = tf.square(
            self.target - tf.reduce_sum(tf.multiply(self.actions_onehot, self.qvalues), axis=1))
        self.loss = tf.reduce_sum(self.td_error * self.mask) / tf.reduce_sum(self.mask)
        self.loss_summary = tf.summary.scalar(name="Loss_summary", tensor=self.loss)

        # train op (clip gradient)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name="ADAM")
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        # target network update op
        self.update_target_op = []
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.target_scope_name)
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.eval_scope_name)
        for i in range(len(t_params)):
            self.update_target_op.append(tf.assign(t_params[i], e_params[i]))

        # Initialize the tensor board
        if not os.path.exists("summaries"):
            os.mkdir("summaries")
        if not os.path.exists(os.path.join("summaries", "first")):
            os.mkdir(os.path.join("summaries", "first"))

        # Init tensorflow session
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sum_writer = tf.summary.FileWriter(os.path.join("summaries", "first"))
        self.sess.run(tf.global_variables_initializer())

        # Init memory buffers
        #self.memory_size = int(kwargs.setdefault("memory_size", 2**10))
        self.memory_size = self.num_users  # Set the memory size to number of total users.
        self.replay_buffer_lens = collections.deque(maxlen=self.memory_size)
        self.replay_buffer = collections.deque(maxlen=self.memory_size)
        # item format [observation, action, reward, terminals, masks, len]

        # init training buffer
        self.state_buf = np.empty((1, self.state_space))
        self.action_buf, self.reward_buf = np.empty(1, dtype=np.int32), np.empty(1)
        self.terminal_buf = np.empty(1, dtype=np.bool)

    def _create_network(self, input_state, reuse=None):
        """
        Define computation graph of network
        :param input_state: tf.tensor
        :param reuse: bool
        :return:
        """
        kernel_num = [32, 32]
        hidden_size = self.nn_layers.values()
        if len(hidden_size) is 1:
            print("1 Layer DRQN NN with " + str(hidden_size[0]) + " neurons")
            dense = tf.layers.dense(input_state,  units=hidden_size[0], activation=tf.nn.relu,
                                    name="dense", reuse=reuse)
            state_size = hidden_size[0]
        elif len(hidden_size) is 2:
            print("2 Layers DRQN NN with " + str(hidden_size[0]) + " and " + str(hidden_size[1]) + " neurons")
            # TODO: think how to do it better.
            h_state = tf.layers.dense(input_state, units=hidden_size[0], activation=tf.nn.relu,
                                      name="h_state", reuse=reuse)
            # TODO: I am not sure whether I need to use one NN or multiple NN.
            dense = tf.layers.dense(h_state, units=hidden_size[1], activation=tf.nn.relu,
                                    name="dense", reuse=reuse)
            state_size = hidden_size[1]

        # RNN

        # TODO: try also with LSTM?
        rnn_cell = tf.contrib.rnn.GRUCell(num_units=state_size)
        rnn_in = tf.reshape(dense, shape=[self.batch_size_ph, self.unroll_step_ph, state_size])
        state_in = rnn_cell.zero_state(self.batch_size_ph, tf.float32)
        rnn, rnn_state = tf.nn.dynamic_rnn(
            cell=rnn_cell, inputs=rnn_in, dtype=tf.float32, initial_state=state_in
        )

        rnn = tf.reshape(rnn, shape=[-1, state_size])

        if self.use_dueling:
            value = tf.layers.dense(dense, units=1, name="dense_value", reuse=reuse)
            advantage = tf.layers.dense(dense, units=self.num_actions, use_bias=False,
                                        name="dense_advantage", reuse=reuse)

            qvalues = value + advantage - tf.reduce_sum(advantage, axis=1, keep_dims=True)
        else:
            qvalues = tf.layers.dense(rnn, units=self.num_actions)

        self.state_size = state_size

        return qvalues, state_in, rnn_state

    def _get_agent_states(self, user_id):
        """
        Get hidden state of the given agent.
        :param user_id: int
            user id of the vehicle
        :return:
        """
        n = 1  # since there is only one user.
        states = np.empty([n, self.state_size])
        default = np.zeros([n, self.state_size])
        # NOTE: I am not sure about this part.
        states = self.agent_states.get(user_id, default)

        return states

    def _set_agent_states(self, user_id, state):
        """
        Set hidden state of the given user
        :param user_id:
        :param state:
        :return:
        """
        # TODO: I think, I do not need such thing since the users in my scenario are fixed.
        #if len(user_id) <= len(self.agent_states) * 0.5:
        #    self.agent_states = {}
        self.agent_states[user_id] = state

    def infer_action(self, user_id, sn, obs, policy="e-greedy", eps=0):
        """
        Infer action for the given agent.
        :param user_id: int
            id of the user
        :param sn:
            Sequence number of the packet that will be transmitted.
        :param obs:
        :param policy:
            can be eps-greedy or greedy.
        :param eps: float
            used when policy is eps-greedy.
        :return:
        """

        n = 1 #len(user_id)
        # which is one for now.

        state_in = self._get_agent_states(user_id)
        # state_in = np.reshape(state_in, [n, state_in.shape[0]])
        qvalues, states = self.sess.run([self.qvalues, self.rnn_state], feed_dict={
            self.input_state: obs,
            self.state_in: state_in,
            self.batch_size_ph: n,
            self.unroll_step_ph: 1,
        })
        self._set_agent_states(user_id, states)
        best_actions = np.argmax(qvalues, axis=1)

        if policy == 'e_greedy':
            random = np.random.randint(self.num_actions, size=(n,))
            cond = np.random.uniform(0, 1, size=(n,)) < eps
            ret = np.where(cond, random, best_actions)
        elif policy == 'greedy':
            ret = best_actions

        return ret.astype(np.int32)

    def _calc_target(self, next_obs, rewards, terminal, batch_size, unroll_step):
        """
        Calculate target value
        :param next_obs: next observation of the user.
        :param reward: rewards of the previous action
        :param terminal:
        :param batch_size:
        :param unroll_step:
        :return:
        """
        n = len(rewards)
        if self.use_double:
            t_qvalues, qvalues = self.sess.run([self.target_qvalues, self.qvalues], feed_dict={
                self.input_state: next_obs,
                # self.state_in:        state_in,
                # self.target_state_in: state_in,
                self.batch_size_ph: batch_size,
                self.unroll_step_ph: unroll_step
            })
            # ignore the first value (the first value is for computing correct hidden state)
            # t_qvalues = t_qvalues.reshape([-1, unroll_step, self.num_actions])
            # t_qvalues = t_qvalues[:, 1:, :].reshape([-1, self.num_actions])
            # qvalues = qvalues.reshape([-1, unroll_step, self.num_actions])
            # qvalues = qvalues[:, 1:, :].reshape([-1, self.num_actions])
            next_value = t_qvalues[np.arange(n), np.argmax(qvalues, axis=1)]
        else:
            t_qvalues = self.sess.run(self.target_qvalues, feed_dict={
                self.input_state: next_obs,
                # self.target_state_in: state_in,
                self.batch_size_ph: batch_size,
                self.unroll_step_ph: unroll_step})
            # t_qvalues = t_qvalues.reshape([-1, unroll_step, self.num_actions])
            # t_qvalues = t_qvalues[:,1:,:].reshape([-1, self.num_actions])

            next_value = np.max(t_qvalues, axis=1)

        target = np.where(terminal, rewards, rewards + self.gamma * next_value)

        return target

    def _add_to_replay_buffer(self, sample_buffer):
        """
        add samples in sample buffer to replay buffer
        :param sample_buffer:
        :return:
        """
        n = 0
        for episode in sample_buffer.episodes(): # Each user has its own episode.
            s, a, r = [], [], []
            for step in range(len(episode.states)):  # Step represent the sequence number of the transmitted packet
                if (episode.states[step] is not -1)and (episode.actions[step] is not -1)and (episode.rewards[step]is not -1):
                    # This part is required to make sure we synchronize the s,a and reward.
                    # in order words, to alleviate the effect of delayed rewards.
                    s.append(episode.states[step])
                    a.append(episode.actions[step])
                    r.append(episode.rewards[step])

            m = len(r)
            if m is 0:
                continue
            mask = np.ones((m,))
            terminal = np.zeros((m,), dtype=np.bool)
            if episode.terminal:
                terminal[-1] = True
            else:
                mask[-1] = 0

            item = [s, a, r, terminal, mask, m]
            self.replay_buffer_lens.append(m)
            self.replay_buffer.append(item)

            n += m
        return n

    def train(self, sample_buffer, print_every=1000, **kwargs):
        """
        add new samples in sample_buffer to replay buffer and train
        do not keep hidden state (split episode into short sequences)
        --------------
        :param sample_buffer: memory.EpisodesBuffer
        :param print_every: int
            print log every print_every batches
        :param kwargs:
        :return:
        -------------
        loss: float
            bellman residual loss
        value: float
            estimated state value
        """
        add_num = self._add_to_replay_buffer(sample_buffer)

        batch_size = self.batch_size
        unroll_step = self.unroll_step

        # Calc sample weight of episodes (i.e. their lengths)
        replay_buffer = self.replay_buffer
        replay_lens_sum = np.sum(self.replay_buffer_lens)
        weight = np.array(self.replay_buffer_lens, dtype=np.float32) / replay_lens_sum

        n_batches = self.train_freq * add_num / (batch_size * (unroll_step - self.skip_error))
        if n_batches == 0:
            return 0, 0

        max_ = batch_size * unroll_step
        batch_obs = np.zeros((max_+1,  self.state_space), dtype=np.float32)
        batch_action = np.zeros((max_,), dtype=np.float32)
        batch_reward = np.zeros((max_,), dtype=np.float32)
        batch_terminal = np.zeros((max_,), dtype=np.float32)
        batch_mask = np.zeros((max_,), dtype=np.float32)

        # calc batch number
        n_batches = int(self.train_freq * add_num / (batch_size * (unroll_step - self.skip_error)))
        print("batches: %d  add: %d  replay_len: %d/%d" %
              (n_batches, add_num, len(self.replay_buffer), self.memory_size))

        ct = 0
        total_loss = 0
        start_time = time.time()
        # train batches
        for i in range(n_batches):
            indexes = np.random.choice(len(replay_buffer), self.batch_size, p=weight)

            batch_mask[:] = 0

            for j in range(batch_size):
                item = replay_buffer[indexes[j]]
                s, a, r, t, = item[0], item[1], item[2], item[3]
                length = len(s)

                start = np.random.randint(length)
                real_step = min(length - start, unroll_step)

                beg = j * unroll_step
                batch_obs[beg:beg+real_step] = s[start:start+real_step]
                batch_action[beg:beg+real_step] = a[start:start+real_step]
                batch_reward[beg:beg+real_step] = r[start:start+real_step]
                batch_terminal[beg:beg+real_step] = t[start:start+real_step]
                batch_mask[beg:beg+real_step] = 1.0

                if not t[start+real_step-1]:
                    batch_mask[beg+real_step-1] = 0

            # collect trajectories from different IDs to a single buffer
            target = self._calc_target(batch_obs[1:], batch_reward, batch_terminal, batch_size,
                                       unroll_step)

            ret = self.sess.run([self.train_op, self.loss], feed_dict={
                self.input_state: batch_obs[:-1],
                self.action: batch_action,
                self.target: target,
                self.mask: batch_mask,
                self.batch_size_ph: batch_size,
                self.unroll_step_ph: unroll_step,
            })
            loss = ret[1]
            total_loss += loss

            if ct % self.target_update == 0:
                print("Target Q update ct " + str(ct))
                self.sess.run(self.update_target_op)
            if ct % print_every == 0:
                print("batch %5d, loss %.6f, qvalue %.6f" % (ct, loss, self._eval(target)))

            ct += 1
            self.train_ct += 1

        total_time = time.time() - start_time
        step_average = total_time / max(1.0, (ct / 1000.0))

        print("batches: %d,  total time: %.2f,  1k average: %.2f" % (ct, total_time, step_average))

        return total_loss / ct if ct != 0 else 0, self._eval(target)

    # TODO: check this function later.
    #def train_keep_hidden()

    @staticmethod
    def _div_round(x, divisor):
        """round up to nearest integer that are divisible by divisor"""
        return (x + divisor - 1) / divisor * divisor

    def _eval(self, target):
        """
        evaluate estimated q value
        :param target:
        :return:
        """
        if self.eval_obs is None:
            return np.mean(target)
        else:
            return np.mean(self.sess.run(self.target_qvalues, feed_dict={
                self.input_state: self.eval_obs[0],
                self.batch_size_ph: self.eval_obs[0].shape[0],
                self.unroll_step_ph: 1
            }))

    def get_info(self):
        """ get information of model """
        return "tfdrqn train_time: %d" % (self.train_ct)





