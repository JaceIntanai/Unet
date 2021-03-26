# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback

class LearningRateDecay:
	def plot(self, epochs, title="Learning Rate Schedule"):
		# compute the set of learning rates for each corresponding
		# epoch
		lrs = [self(i) for i in epochs]

		# the learning rate schedule
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(epochs, lrs)
		plt.title(title)
		plt.xlabel("Epoch #")
		plt.ylabel("Learning Rate")

class StepDecay(LearningRateDecay):
	def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
		# store the base initial learning rate, drop factor, and
		# epochs to drop every
		self.initAlpha = initAlpha
		self.factor = factor
		self.dropEvery = dropEvery

	def __call__(self, epoch):
		# compute the learning rate for the current epoch
		exp = np.floor((1 + epoch) / self.dropEvery)
		alpha = self.initAlpha * (self.factor ** exp)

		# return the learning rate
		return float(alpha)

class PolynomialDecay(LearningRateDecay):
	def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
		# store the maximum number of epochs, base learning rate,
		# and power of the polynomial
		self.maxEpochs = maxEpochs
		self.initAlpha = initAlpha
		self.power = power

	def __call__(self, epoch):
		# compute the new learning rate based on polynomial decay
		decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
		alpha = self.initAlpha * decay

		# return the new learning rate
		return float(alpha)
    
class CyclicalLearningRateDecay(LearningRateDecay):
    def __init__(self, max_lr=1e-3, min_lr=1e-4, step_size=10, step_decay=0.2, decay_mode='linear', scale_mode='cycle'):
      # store the base initial learning rate, drop factor, and
      # epochs to drop every
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_size = step_size
        self.decay_mode = decay_mode
        self.scale_mode = scale_mode
        self.step_decay = step_decay

    def __call__(self, step):
        cycle = np.floor(1 + step / (self.step_size))
        float_step = (((step - 1) % self.step_size) / (self.step_size - 1)) * np.pi/2
        x = np.abs(step / self.step_size - cycle + 1)

        mode_step = cycle if self.scale_mode == "cycle" else step
        decay_step = 1-x if self.decay_mode == "linear" else np.cos(float_step)

        alpha = self.min_lr + (self.max_lr-self.min_lr) * decay_step * ((1-self.step_decay)**mode_step)
        return float(alpha)

if __name__ == "__main__":
	# plot a step-based decay which drops by a factor of 0.5 every
	# 25 epochs
	sd = StepDecay(initAlpha=0.01, factor=0.5, dropEvery=25)
	sd.plot(range(0, 100), title="Step-based Decay")
	plt.show()

	# plot a linear decay by using a power of 1
	pd = PolynomialDecay(power=1)
	pd.plot(range(0, 100), title="Linear Decay (p=1)")
	plt.show()

	# show a polynomial decay with a steeper drop by increasing the
	# power value
	pd = PolynomialDecay(power=5)
	pd.plot(range(0, 100), title="Polynomial Decay (p=5)")
	plt.show()