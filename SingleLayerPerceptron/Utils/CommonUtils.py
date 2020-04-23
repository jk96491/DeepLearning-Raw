import numpy as np
import time
import enum


class CommonUtils:
    def __init__(self):
        np.random.seed(time.time())

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def sigmoid(x):
        return np.exp(-CommonUtils.relu(-x)) / (1.0 + np.exp(-np.abs(x)))

    @staticmethod
    def sigmoid_derivative(x, y):
        return y * (1 - y)

    @staticmethod
    def sigmoid_cross_entropy_with_logits(z, x):
        return CommonUtils.relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))

    @staticmethod
    def sigmoid_cross_entropy_with_logits_derivative(z, x):
        return -z + CommonUtils.sigmoid(x)

    @staticmethod
    def softmax(x):
        max_elem = np.max(x, axis=1)
        diff = (x.transpose() - max_elem).transpose()
        exp = np.exp(diff)
        sum_exp = np.sum(exp, axis=1)
        probs = (exp.transpose() / sum_exp).transpose()
        return probs

    @staticmethod
    def softmax_derivative(x, y):
        mb_size, nom_size = x.shape

        derv = np.ndarray([mb_size, nom_size, nom_size])
        for n in range(mb_size):
            for i in range(nom_size):
                for j in range(nom_size):
                    derv[n, i, j] = -y[n, i] * y[n, j]
                derv[n, i, i] += y[n, i]
        return derv

    @staticmethod
    def softmax_cross_entropy_with_logits(labels, logits):
        probs = CommonUtils.softmax(logits)
        return -np.sum(labels * np.log(probs + 1.0e-10), axis=1)

    @staticmethod
    def softmax_cross_entropy_with_logits_derivative(labels, logits):
        return CommonUtils.softmax(logits) - labels


class ErrorType(enum.Enum):
    MSE = 0
    ENTROPY = 1


class ActivationFuncType(enum.Enum):
    NONE = -1
    SIGMOID = 0
    RELU = 1
    SOFTMAX = 2