import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def UpdateGrad(self, grad, learning_rate, param, name):
        self.learning_rate = learning_rate
        param -= self.learning_rate * grad


class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = {}

    def UpdateGrad(self, grad, learning_rate, param, name):
        self.learning_rate = learning_rate

        if name not in self.v.keys():
            self.v[name] = np.zeros_like(param)

        self.v[name] = self.momentum * self.v[name] - self.learning_rate * grad
        param += self.v[name]


class Nesterov:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = {}

    def UpdateGrad(self, grad, learning_rate, param, name):
        self.learning_rate = learning_rate

        if name not in self.v.keys():
            self.v[name] = np.zeros_like(param)

        self.v[name] *= self.momentum
        self.v[name] -= self.learning_rate * grad
        param += self.momentum * self.momentum + self.v[name]
        param -= (1 + self.momentum) * self.learning_rate * grad


class AdaGrad:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.h = {}
        self.epsilon = 1e-7

    def UpdateGrad(self, grad, learning_rate, param, name):
        self.learning_rate = learning_rate

        if name not in self.h.keys():
            self.h[name] = np.zeros_like(param)

        self.h[name] += grad * grad
        param -= self.learning_rate * grad / (np.sqrt(self.h[name]) + self.epsilon)


class RMSprop:
    def __init__(self, learning_rate=0.01, decay_rate=0.99):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.h = {}
        self.epsilon = 1e-7

    def UpdateGrad(self, grad, learning_rate, param, name):
        self.learning_rate = learning_rate

        if name not in self.h.keys():
            self.h[name] = np.zeros_like(param)

        self.h[name] *= self.decay_rate
        self.h[name] += (1 - self.decay_rate) * grad * grad
        param -= self.learning_rate * grad / (np.sqrt(self.h[name]) + self.epsilon)


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = {}
        self.v = {}
        self.epsilon = 1e-7

    def UpdateGrad(self, grad, learning_rate, param, name):
        self.learning_rate = learning_rate

        if name not in self.m.keys():
            self.m[name] = np.zeros_like(param)
        if name not in self.v.keys():
            self.v[name] = np.zeros_like(param)

        self.iter += 1
        lr_t = self.learning_rate * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        self.m[name] += (1 - self.beta1) * (grad - self.m[name])
        self.v[name] += (1 - self.beta2) * (grad ** 2 - self.v[name])

        param -= lr_t * self.m[name] / (np.sqrt(self.v[name]) + self.epsilon)







