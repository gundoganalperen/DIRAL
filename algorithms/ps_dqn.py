"""Deep Q network with parameter sharing(PS) for multi-agent reinforcement learning"""
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

from base import TFBaseModel
#from marl_agent.utils.memory import ReplayBuffer
from utils.memory import ReplayBuffer
#from marl_agent.utils.policies import BoltzmanPolicy
from utils.policies import BoltzmanPolicy
import time
import os

class DeepQNetwork(TFBaseModel):
    def __init__(self, env, name, **kwargs):
                 #batch_size=64, learning_rate=1e-4, reward_decay=0.99,
                 #train_freq=1, target_update=2000, memory_size=2 ** 10, eval_obs=None,
                 #use_dueling=True, use_double=True, use_conv=False,
                 #custom_state_space=None, num_gpu=1, infer_batch_size=8192, network_type=0):
        """
        Init DQN
        :param env: Environment
            environment
        :param name: str
            name of this model
        :param batch_size: int
        :param learning_rate: float
        :param reward_decay: float
            reward_decay in TD
        :param train_freq: int
            mean training times of a sample
        :param target_update: int
            target will update every target_update batches
        :param memory_size: int
            weight of entropy loss in total loss
        :param eval_obs: numpy array
            evaluation set of observation
        :param use_dueling: bool
            whether use dueling q network
        :param use_double: bool
            whether use double q network
        :param use_conv: bool
            use convolution or fully connected layer as state encoder
        :param custom_state_space: tuple
        :param num_gpu: int
            number of gpu
        :param infer_batch_size: int
            batch size while inferring actions
        :param network_type:
        """
        TFBaseModel.__init__(self, env, name, "tfdqn")
        # ======================== set config  ========================
        self.env = env
        self.state_space = env.get_state_space()
        self.num_actions = env.get_action_space()

        self.batch_size = kwargs.setdefault("batch_size", 64)
        self.learning_rate = kwargs.setdefault("learning_rate", 1e4)
        self.training_freq = kwargs.setdefault("training_freq", 1)  # train time of every sample (s,a,r,s')
        self.target_update = kwargs.setdefault("target_update", 1000)   # target network update frequency
        self.eval_obs = kwargs.setdefault("eval_obs", None)
       # self.infer_batch_size = kwargs.setdefault("infer_batch_size", 8192)  # maximum batch size when infer actions,
       # change this to fit your GPU memory if you meet a OOM

        self.network_param = kwargs.setdefault("network", False)
        self.use_dueling = self.network_param["use_dueling"]
        self.use_double = self.network_param["use_double"]
        self.num_gpu = self.network_param["num_gpu"]
        self.use_conv = self.network_param["use_conv"]

        self.nn_layers = self.network_param["layers"]
        self.activation = self.network_param["activation"]
        self.train_ct = 0

        # ======================= build network =======================
        tf.reset_default_graph()
        # input place holder
        self.target = tf.placeholder(tf.float32, [None], name="target")
        self.weight = tf.placeholder(tf.float32, [None], name="weight")

        self.input_state = tf.placeholder(tf.float32, [None, self.state_space], name="input_state")
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.mask = tf.placeholder(tf.float32, [None], name="mask")
        self.eps = tf.placeholder(tf.float32, name="eps")  # e-greedy

        # build a graph
        with tf.variable_scope(self.name):
            with tf.variable_scope("eval_net_scope"):
                self.eval_scope_name = tf.get_variable_scope().name
                self.qvalues = self._create_network(self.input_state, self.use_conv)

            if self.num_gpu > 1: # build inference graph for multiple gpus
                self._build_multi_gpu_infer(self.num_gpu)

            with tf.variable_scope("target_net_scope"):
                self.target_scope_name = tf.get_variable_scope().name
                self.target_qvalues = self._create_network(self.input_state, self.use_conv)

        # loss
        self.gamma = kwargs.setdefault("reward_decay", 0.99)
        self.actions_onehot = tf.one_hot(self.action, self.num_actions)
        td_error = tf.square(self.target - tf.reduce_sum(tf.multiply(self.actions_onehot, self.qvalues), axis=1))
        self.loss = tf.reduce_sum(td_error * self.mask) / tf.reduce_sum(self.mask)
        self.loss_summary = tf.summary.scalar(name='Loss_summary', tensor=self.loss)

        # train op(clip gradient)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name="ADAM")
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables), name="train_op")
#        self.train_summary = tf.summary.scalar(name='Train_summary', tensor=self.train_op)

        # output action
        def out_action(qvalues):
            best_action = tf.argmax(qvalues, axis=1)
            best_action = tf.to_int32(best_action)
            random_action = tf.random_uniform(tf.shape(best_action), 0, self.num_actions, tf.int32, name="random_action")
            should_explore = tf.random_uniform(tf.shape(best_action), 0, 1) < self.eps
            return tf.where(should_explore, random_action, best_action)

        self.output_action = out_action(self.qvalues)
        if self.num_gpu > 1:
            self.infer_out_action = [out_action(qvalue) for qvalue in self.infer_qvalues]

        # target network update op
        self.update_target_op = []
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.target_scope_name)
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.eval_scope_name)
        for i in range(len(t_params)):
            self.update_target_op.append(tf.assign(t_params[i], e_params[i]))

        # Initialize the tensor board
        if not os.path.exists('summaries'):
            os.mkdir('summaries')
        if not os.path.exists(os.path.join('summaries', 'first')):
            os.mkdir(os.path.join('summaries', 'first'))

        # init tensorflow session
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.summ_writer = tf.summary.FileWriter(os.path.join('summaries', 'first'), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        # init replay buffers
        self.replay_buffer_len = 0
        self.memory_size = int(kwargs.setdefault("memory_size", 2**10))
        print("Memory size ", self.memory_size)
        self.replay_buf_state = ReplayBuffer(shape=(self.memory_size, self.state_space))
        self.replay_buf_action = ReplayBuffer(shape=(self.memory_size,), dtype=np.int32)
        self.replay_buf_reward = ReplayBuffer(shape=(self.memory_size,))
        self.replay_buf_terminal = ReplayBuffer(shape=(self.memory_size,), dtype=np.bool)
        self.replay_buf_mask = ReplayBuffer(shape=(self.memory_size,))
        # if mask[i] == 0, then the item is used for padding, not for training
        self.policy = BoltzmanPolicy(action_space=self.num_actions)

    def _create_network(self, input_state, use_conv=False, reuse=None):
        """
        Define computation graph of network
        :param input_state: tf.tensor
        :param use_conv: bool
        :param reuse: bool
        :return:
        """
        kernel_num = [32, 32]
        hidden_size = self.nn_layers.values()
        if len(hidden_size) is 1:
            if self.activation == "Linear":
                print("1 NN with Linear activation, " + str(hidden_size[0]) + " neurons")
                h_state = tf.layers.dense(input_state,  units=hidden_size[0], activation=None,
                                    name="h_state", reuse=reuse)
            else:
                print("1 NN with RELU activation, " + str(hidden_size[0]) + " neurons")
                h_state = tf.layers.dense(input_state,  units=hidden_size[0], activation=tf.nn.relu,
                                    name="h_state", reuse=reuse)
        elif len(hidden_size) is 2:
            print("2 Layers NN with " + str(hidden_size[0]) + " and " + str(hidden_size[1]) + " neurons")
            activation = None
            if self.activation != "Linear":
                activation = tf.nn.relu

            h_state_0 = tf.layers.dense(input_state,  units=hidden_size[0], activation=activation,
                                    name="h_state_0", reuse=reuse)

            h_state = tf.layers.dense(h_state_0,  units=hidden_size[1], activation=activation,
                                    name="h_state", reuse=reuse)

        if self.use_dueling:
            value = tf.layers.dense(h_state, units=1, name="value", reuse=reuse)
            advantage = tf.layers.dense(h_state, units=self.num_actions, use_bias=False,
                                        name="advantage", reuse=reuse)

            qvalues = value + advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True)
        else:
            qvalues = tf.layers.dense(h_state, units=self.num_actions, name="value", reuse=reuse)

        return qvalues

    def infer_action(self, user_id, sn, obs, step, policy="e_greedy", eps=0):
        """
        infer action for the given agent.
        :param raw_obs:
        :param user_id: int
            id of the user
        :param policy:
            can be eps-greedy or greedy
        :param eps: float
            used when policy is eps-greedy
        :return:
        """

        if policy == 'e_greedy':
            eps = eps
        elif policy == 'greedy':
            eps = 0

#        if self.num_gpu > 1 and n > batch_size:  # infer by multi gpu in parallel
#            ret = self._infer_multi_gpu(view, feature, ids, eps)
        qvalues = self.sess.run(self.qvalues, feed_dict={self.input_state: obs})
        best_actions = np.argmax(qvalues, axis=1)

        n = 1  # Since we take an action only for 1 user.
        random = np.random.randint(self.num_actions, size=(n,))
        cond = np.random.uniform(0, 1, size=(n,)) < eps
        ret = np.where(cond, random, best_actions)
        action = ret.astype(np.int32)
        # TODO: enable this later.
        #action = self.policy.take_action(qvalues, step)
        #actions.append(action)
        # action = self.sess.run(self.output_action, feed_dict={
        #     self.input_state: obs,
        #     self.eps: eps
        # })
        return action

    def _calc_target(self, next_state, reward, terminal):
        """
        Calculate target value
        :param next_state: next_state of the user.
        :param reward: reward of the previous action
        :param terminal:
        :return:
        """
        n = len(reward)
        if self.use_double:
            t_qvalues, qvalues = self.sess.run([self.target_qvalues, self.qvalues],
                                               feed_dict={self.input_state: next_state})
            next_value = t_qvalues[np.arange(n), np.argmax(qvalues, axis=1)]
        else:
            t_qvalues = self.sess.run(self.target_qvalues, feed_dict={self.input_state: next_state})
            next_value = np.max(t_qvalues, axis=1)

        target = np.where(terminal, reward, reward + self.gamma * next_value)

        return target

    def _add_to_replay_buffer(self, sample_buffer):
        """
        Add stored episode buffers to replay buffer.
        :param sample_buffer:
        :return:
        """
        n = 0
        for episode in sample_buffer.episodes(): # Each user has its own episode.
            s, a, r = [], [], []
            for step in range(len(episode.states)):  # Step represent the sequence number of the transmitted packet.
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

            self.replay_buf_state.put(s)
            self.replay_buf_action.put(a)
            self.replay_buf_reward.put(r)
            self.replay_buf_terminal.put(terminal)
            self.replay_buf_mask.put(mask)

            n += m

        self.replay_buffer_len = min(self.memory_size, self.replay_buffer_len + n)
        return n

    def train(self, sample_buffer, print_every=1000, **kwargs):
        """
        Add new samples in sample buffer to replay buffer and train
        Parameters
        ----------
        :param sample_buffer: memory.EpisodesBuffer
        :param print_every: int
            print log every print_every batches
        :param kwargs:
        :return:
        loss: float
            bellman residual loss
        value: float
            estimated state value
        """
        add_num = self._add_to_replay_buffer(sample_buffer)
        batch_size = self.batch_size
        total_loss = 0

        n_batches = int(self.training_freq * add_num / batch_size)
        if n_batches == 0:
            return 0, 0

        print("batch number: %d  add: %d  batch_size: %d training_freq: %d  replay_len: %d/%d" %
              (n_batches, add_num, batch_size, self.training_freq, self.replay_buffer_len, self.memory_size))

        start_time = time.time()
        ct = 0
        for i in range(n_batches):
            # fetch a batch
            index = np.random.choice(self.replay_buffer_len - 1, batch_size)

            batch_state = self.replay_buf_state.get(index)
            batch_action = self.replay_buf_action.get(index)
            batch_reward = self.replay_buf_reward.get(index)
            batch_terminal = self.replay_buf_terminal.get(index)
            batch_mask = self.replay_buf_mask.get(index)

            batch_next_state = self.replay_buf_state.get(index+1)

            batch_target = self._calc_target(batch_next_state, batch_reward, batch_terminal)

            ret = self.sess.run([self.train_op, self.loss], feed_dict = {
                self.input_state: batch_state,
                self.action: batch_action,
                self.target: batch_target,
                self.mask: batch_mask
            })
            loss = ret[1]
            total_loss += loss

            if ct % self.target_update == 0:
                print("Target Q update ct " + str(ct))
                self.sess.run(self.update_target_op)

            if ct % print_every == 0:
                print("batch %5d,  loss %.6f, eval %.6f" % (ct, loss, self._eval(batch_target)))

            ct += 1
            self.train_ct += 1

        total_time = time.time() - start_time
        step_average = total_time / max(1.0, (ct / 1000.0))
        print("batches: %d,  total time: %.2f,  1k average: %.2f" % (ct, total_time, step_average))

        return total_loss / ct if ct != 0 else 0, self._eval(batch_target)

    def _eval(self, target):
        """ Evaluate estimated q value"""
        if self.eval_obs is None:
            return np.mean(target)
        else:
            return np.mean(self.sess.run([self.qvalues], feed_dict = {
                self.input_state: self.eval_obs[0]
            }))

    def clean_buffer(self):
        """ Clean replay buffer """
        self.replay_buf_len = 0
        self.replay_buf_view.clear()
        self.replay_buf_feature.clear()
        self.replay_buf_action.clear()
        self.replay_buf_reward.clear()
        self.replay_buf_terminal.clear()
        self.replay_buf_mask.clear()