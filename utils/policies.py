import numpy as np
import tensorflow as tf

class SoftmaxPolicy():
	"""
	Softmax policy is used to select an action for the given Q values.
	Details: http://www.incompleteideas.net/book/ebook/node17.html
	"""
	def __init__(self, nA, temperature=0.0001, episodes=10000):
		self.temperature = temperature
		self.nA = nA
		try:
			self.temperature_list = np.concatenate((np.geomspace(1.0, temperature, int(episodes*2.0/3)),
											np.repeat(temperature, episodes - int(episodes*2.0/3))))
		except:
			self.temperature_list = np.concatenate((np.logspace(np.log10(1.0), np.log10(temperature), int(episodes*2.0/3)),
											np.repeat(temperature, episodes - int(episodes*2.0/3))))

	def action(self, Qs, episode=None):
		"""
		Returns the action
		:param Qs: Q values for each possible action.
		:param episode: determines the temperature parameters, high temp give equal probable actions, lowe values give
		greedy action, for higher values user receives greedy actions.
		:return: action
		"""
		try:
			temp = self.temperature_list[episode]
		except:
			temp = self.temperature

		app_values = Qs/temp
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
			print ("explored")

		# Exploitation
		else:
			nA = len(Qs)  # Number of possible actions
			prob1 = (1 - self.alpha) * np.exp(self.beta * Qs)
			prob = prob1 / np.sum(np.exp(self.beta * Qs)) + self.alpha/self.nA

			#  choosing action with max probability
			action = np.argmax(prob, axis=1)
			return action


def normalize_with_moments(x, axes=[0], epsilon=1e-8):
	mean, variance = tf.nn.moments(x, axes=axes)
	x_normed = (x - mean) / tf.sqrt(variance + epsilon)
	return x_normed

