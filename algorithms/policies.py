import numpy as np


class RandomPolicy(object):
    def __init__(self, nA = 10):
        self.nA = nA
        self.actions_onehot = np.eye(self.nA)

    def random_action(self):
        action_idx = np.random.randint(self.nA)
        return action_idx

    def action(self, Qs=None):
        return self.random_action()

    def __call__(self, params=None):
        return self.action(params)


class GreedyPolicy(RandomPolicy):
    def __init__(self, nA=10):
        super(GreedyPolicy, self).__init__(nA)

    def greedy_action(self, Qs):
        """
        takes a greedy action
        :param Qs:
        :return:
        """
        action = np.argmax(Qs, axis=1)
        return action

    def action(self, Qs=None):
        action_idx = self.greedy_action(Qs)
        return action_idx


class EpsilonGreedy(GreedyPolicy):
    def __init__(self, nA = 10, eps_init=0.99, eps_decay=0.999, episodes=4000, explore_stop=0.05):
        super(EpsilonGreedy, self).__init__(nA)
        self.eps = eps_init
        self.eps_decay = eps_decay
        self.episode = 0

    def action(self, Qs=None, episode=None):
        if episode > self.episode:  # for each episode we update the epsilon value.
            self.update_eps()
            self.episode = episode
        draw = np.random.random()
        if draw > self.eps:
            action_idx = self.greedy_action(Qs=Qs)
        else:
            action_idx = self.random_action()
        return action_idx

    def update_eps(self):
        """
        Update the epsilon value by multiplying with eps_decay parameter.
        :return:
        """
        self.eps = self.eps * self.eps_decay
        if self.eps < 0.001:
            self.eps = 0.001

    def get_epsilon(self):
        """
        :return: return the eps value.
        """
        return self.eps

    def set_epsilon(self, eps):
        """
        Set the eps to a specific value
        :param eps:
        :return:
        """
        self.eps = eps

class SoftmaxPolicy(GreedyPolicy):
    def __init__(self, nA=10, temperature=0.05, episodes=1000):
        super(SoftmaxPolicy,self).__init__(nA)
        self.temperature = temperature
        self.nA = nA
        self.tmp = None
        try:
            self.temperature_list = np.concatenate((np.geomspace(1.0, temperature, int(episodes*2.0/3)),
                                                    np.repeat(temperature, episodes - int(episodes*2.0/3))))
        except:
            self.temperature_list = np.concatenate((np.logspace(np.log10(1.0), np.log10(temperature), int(episodes*2.0/3)),
                                                    np.repeat(temperature, episodes - int(episodes*2.0/3))))

    def action(self, Qs=None, episode=None):
        try:
            self.tmp = self.temperature_list[episode]
        except:
            self.tmp = self.temperature

        app_values = Qs/self.tmp
        prob = self.softmax(app_values)
        action_idx = np.random.choice(np.arange(self.nA), p=prob[0])
        return action_idx

    def softmax(self, x, axis=None):
        """
        X is the qvalues
        :param x:
        :param axis:
        :return:
        """
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def get_epsilon(self):
        """
        :return: return the eps value.
        """
        return self.tmp

class Policy(GreedyPolicy):
    def __init__(self, policy="eps_greedy", nA=10, episodes=1000, **kwargs):
        super(Policy, self).__init__(nA=nA)

        if policy == 'greedy':
            self.policy = GreedyPolicy(nA=nA)
        if policy == 'eps_greedy':
            self.policy = EpsilonGreedy(nA=nA, **kwargs)
        if policy == 'softmax':
            self.policy = SoftmaxPolicy(nA=nA, **kwargs)

    def action(self, Qs=None):
        return self.policy.action(Qs=Qs)


class BoltzmanPolicy():
    def __init__(self, action_space, beta=1, explore_start=0.02, explore_stop=0.01, decay_rate=0.0001, alpha=0):
        self.beta = beta                     # Annealing constant for Monte - Carlo
        self.explore_start = explore_start   # initial exploration rate
        self.explore_stop = explore_stop     # final exploration rate
        self.decay_rate =decay_rate          # rate of exponential decay of exploration
        self.alpha = alpha                   # co-operative fairness constant
        self.nA = action_space

    def take_action(self, Qs, time_slot, axis=None):
        """
        :param Qs: Q values for each action.
        :param beta:
        :param alpha:
        :param axis:
        :return:
        """

        # changing beta at every 50 time-slots
        if time_slot % 50 == 0:
            if time_slot < 5000:
                self.beta -= 0.001

        # curent exploration probability
        explore_p = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * time_slot)

        # Exploration
        if explore_p > np.random.rand():
            # random action sampling, take action only for 1 agent.
            action = np.random.choice(self.nA, size=1)
            print("explored explore_p " + str(explore_p))
            return action

        # Exploitation
        else:
            #nA = len(Qs)  # Number of possible actions
            prob1 = (1 - self.alpha) * np.exp(self.beta * Qs)
            prob = prob1 / np.sum(np.exp(self.beta * Qs)) + self.alpha/self.nA

            #  choosing action with max probability
            action = np.argmax(prob, axis=1)
            #action = np.argmax(prob, axis=None)

            return action