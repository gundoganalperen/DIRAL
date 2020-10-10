import numpy as np
import tensorflow as tf


class PPO(object):
    def __init__(self, env, name="PPO", **kwargs):
        tf.reset_default_graph()
        self.name = name
        self.s_dim = env.get_state_space()
        self.a_dim = env.get_action_space()
        self.gamma = kwargs.setdefault("gamma", 0.9)
        self.a_lr = kwargs.setdefault("a_lr", 0.0001)
        self.c_lr = kwargs.setdefault("c_lr", 0.0001)
        self.update_step = kwargs.setdefault("update_step", 2)
        self.eps_clip = kwargs.setdefault("eps_clip", 0.2)
        self.batch = kwargs.setdefault("batch_size", 32)
        self.step_size = kwargs.setdefault("step_size", 5)
        self.entropy_coef = kwargs.setdefault("entropy_coef", 0.1)

        self.network_param = kwargs.setdefault("network", False)
        self.use_lstm = self.network_param["use_lstm_input"]
        self.nn_layers = self.network_param["layers"]
        hidden_size = list(self.nn_layers.values())
        self.hidden_size = hidden_size[0]

        self.sess = tf.Session()
        if self.use_lstm:
            # NOTE: use seperate LSTM layer for actor and critic.
            # ACTOR LSTM
            with tf.variable_scope('lstm_actor'):
                self.inputs_ = tf.placeholder(tf.float32, [None, self.step_size, self.s_dim], name='inputs_')
                lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
                #lstm = tf.cont rib.rnn.GRUCell(self.hidden_size)
                lstm_out, state = tf.nn.dynamic_rnn(lstm, self.inputs_, dtype=tf.float32)
                reduced_out = lstm_out[:,-1,:]
                self.reduced_out = tf.reshape(reduced_out, shape=[-1, self.hidden_size])
            # CRITIC LSTM
            with tf.variable_scope('lstm_critic'):
                self.inputs_c = tf.placeholder(tf.float32, [None, self.step_size, self.s_dim], name='inputs_c')
                lstm_c = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
                #lstm = tf.cont rib.rnn.GRUCell(self.hidden_size)
                lstm_out_c, state_c = tf.nn.dynamic_rnn(lstm_c, self.inputs_c, dtype=tf.float32)
                reduced_out_c = lstm_out_c[:,-1,:]
                self.reduced_out_c = tf.reshape(reduced_out_c, shape=[-1, self.hidden_size])
        else:
            self.inputs_ = tf.placeholder(tf.float32, [None, self.s_dim], 'state')

        # critic
        w_init = tf.random_normal_initializer(0., .1)
        if self.use_lstm:
            lc = tf.layers.dense(self.reduced_out_c, self.hidden_size, tf.nn.relu, kernel_initializer=w_init, name='lc')
        else:
            lc = tf.layers.dense(self.inputs_, self.hidden_size, tf.nn.relu, kernel_initializer=w_init, name='lc')
        self.v = tf.layers.dense(lc, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(self.c_lr).minimize(self.closs)

        # actor
        self.pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
        pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)   # shape=(None, )
        oldpi_prob = tf.gather_nd(params=oldpi, indices=a_indices)  # shape=(None, )
        ratio = pi_prob/oldpi_prob
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - self.eps_clip, 1. + self.eps_clip) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(self.a_lr).minimize(self.aloss)


        # Test with different PPO architecture.
        #self.entropy_tmp = -tf.reduce_sum(self.pi * tf.log(self.pi), axis=1)
        #self.entropy = tf.reduce_mean(self.entropy_tmp)
        e_coef = 0.01
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=-1)
        self.entropy_loss = -tf.reduce_sum(tf.reduce_mean(entropy, axis=-1)) * e_coef
        self.loss = self.aloss + 0.5 * self.closs + self.entropy_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(self.loss)
        #self.grads = self.optimizer.compute_gradients(self.loss)
        #self.update_batch = self.optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def update(self, e):
        data = np.vstack(e)
        self.sess.run(self.update_oldpi_op)
        s, a, r = data[:, :self.s_dim], data[:, self.s_dim: self.s_dim + 1].ravel(), data[:, -1:]
        adv = self.sess.run(self.advantage, {self.inputs_: s, self.tfdc_r: r})
        # update actor and critic in a update loop
        #[self.sess.run(self.atrain_op, {self.inputs_: s, self.tfa: a, self.tfadv: adv}) for _ in range(self.update_step)]
        #[self.sess.run(self.ctrain_op, {self.inputs_: s, self.tfdc_r: r}) for _ in range(self.update_step)]

        for _ in range(self.update_step):
            #value_loss = self.sess.run(self.closs, {self.inputs_: s, self.tfdc_r: r})
            #actor_loss = self.sess.run(self.aloss, {self.inputs_: s, self.tfa: a, self.tfadv: adv})
            #ent_loss = self.sess.run(self.entropy_loss, {self.inputs_ :s})
            self.sess.run(self.optimizer, {self.inputs_: s, self.tfa: a, self.tfdc_r: r, self.tfadv: adv})

    def update_lstm(self, memory, user):
        s, a, r = self.obtain_update_var(memory, user)
        r = r.reshape(len(r), 1)
        adv = self.sess.run(self.advantage, {self.inputs_c: s, self.tfdc_r: r})
        # update actor and critic in a update loop
        [self.sess.run(self.atrain_op, {self.inputs_: s, self.tfa: a, self.tfadv: adv}) for _ in range(self.update_step)]
        [self.sess.run(self.ctrain_op, {self.inputs_c: s, self.tfdc_r: r}) for _ in range(self.update_step)]

    def obtain_update_var(self, memory, user):
        s, a, r = [], [],[]
        for i in range(self.batch):
            s_tmp = []
            for step in range(self.step_size):
                element = memory[i+step]
                s_tmp.append(element[0][user])
            s.append(s_tmp)
            a.append(memory[i][1][user])
            r.append(memory[i][2][user])
        return np.asarray(s), np.asarray(a), np.asarray(r)

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            if self.use_lstm:
                l_a = tf.layers.dense(self.reduced_out, self.hidden_size, tf.nn.relu, trainable=trainable)
            else:
                l_a = tf.layers.dense(self.inputs_, self.hidden_size, tf.nn.relu, trainable=trainable)
            a_prob = tf.layers.dense(l_a, self.a_dim, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_prob, params

    def choose_action(self, s):  # run by a local
        prob_weights = self.sess.run(self.pi, feed_dict={self.inputs_: s[None, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        if self.use_lstm:
            s = np.expand_dims(s, axis=0)
        return self.sess.run(self.v, {self.inputs_: s})[0, 0]
