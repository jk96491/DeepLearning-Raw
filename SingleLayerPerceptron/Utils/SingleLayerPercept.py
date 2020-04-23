import numpy as np

from SingleLayerPerceptron.Utils.CommonUtils import CommonUtils
from SingleLayerPerceptron.Utils.CommonUtils import ActivationFuncType
from SingleLayerPerceptron.Utils.CommonUtils import ErrorType


class SLP:
    def __init__(self, input_count, output_count, epoch_count, mb_size, report, ErrorType, ActivationFuncType):
        self.RAND_MEAN = 0
        self.RAND_STANDARD = 0.0030
        self.LEARNING_RATE = 0.001

        self.weight = None
        self.bias = None
        self.input_count = input_count
        self.output_count = output_count

        self.epoch_count = epoch_count
        self.mb_size = mb_size
        self.report = report

        self.shuffle_map = None
        self.test_begin_idx = 0
        self.data = None

        self.ErrorType = ErrorType
        self.ActivationFuncType = ActivationFuncType

        np.random.seed(1234)

        self.init_model()

    def init_model(self):
        self.weight = np.random.normal(self.RAND_MEAN, self.RAND_STANDARD, [self.input_count, self.output_count])
        self.bias = np.zeros([self.output_count])

    def Train(self, data):
        self.data = data
        step_count = self.shuffle_Data(data)
        test_input, test_output = self.get_test_data()

        for epoch in range(self.epoch_count):
            losses, accuracies = [], []

            for num in range(step_count):
                train_input, train_output = self.get_train_data(num)
                loss, accuracy = self.run_train(train_input, train_output)
                losses.append(loss)
                accuracies.append(accuracy)

            if self.report > 0 and (epoch + 1) % self.report == 0:
                acc = self.run_test(test_input, test_output)
                print('Epoch {}: loss={:5.3f}, accuracy={:5.3f}/{:5.3f}'. \
                      format(epoch + 1, np.mean(losses), np.mean(accuracies), acc))

        final_accuracy = self.run_test(test_input, test_output)
        print('\nFinal Test: final accuracy = {:5.3f}'.format(final_accuracy))

    def shuffle_Data(self, data):
        self.shuffle_map = np.arange(data.shape[0])
        np.random.shuffle(self.shuffle_map)
        step_count = int(data.shape[0] * 0.8) // self.mb_size
        self.test_begin_idx = step_count * self.mb_size
        return step_count

    def get_test_data(self):
        test_data = self.data[self.shuffle_map[self.test_begin_idx:]]

        testInput = test_data[:, :-self.output_count]
        testOutput = test_data[:, -self.output_count:]

        return testInput, testOutput

    def get_train_data(self, nth):
        if nth == 0:
            np.random.shuffle(self.shuffle_map[:self.test_begin_idx])
        train_data = self.data[self.shuffle_map[self.mb_size * nth:self.mb_size * (nth + 1)]]

        trainInput = train_data[:, :-self.output_count]
        trainOutput = train_data[:, -self.output_count:]

        return trainInput, trainOutput

    def run_train(self, input, output):
        result, aux_nn = self.forward(input)
        loss, aux_pp = self.get_forward_lossInfo(result, output)
        accuracy = self.eval_accuracy(result, output)

        G_loss = 1.0
        Gradient = self.get_backprop_Gradient(G_loss, aux_pp)
        self.backpropagation(Gradient, aux_nn)

        return loss, accuracy

    def run_test(self, input, output):
        result, _ = self.forward(input)
        accuracy = self.eval_accuracy(result, output)
        return accuracy

    def forward(self, input):
        output = np.matmul(input, self.weight) + self.bias
        return output, input

    def get_forward_lossInfo(self, result, output):
        loss = 0
        aux_pp = None

        if self.ErrorType == ErrorType.ENTROPY:
            entropy = None

            if self.ActivationFuncType == ActivationFuncType.SIGMOID:
                entropy = CommonUtils.sigmoid_cross_entropy_with_logits(output, result)
            elif self.ActivationFuncType == ActivationFuncType.SOFTMAX:
                entropy = CommonUtils.softmax_cross_entropy_with_logits(output, result)

            loss = np.mean(entropy)
            aux_pp = [output, result, entropy]

        elif self.ErrorType == ErrorType.MSE:
            diff = result - output
            square = np.square(diff)
            loss = np.mean(square)
            aux_pp = diff

        return loss, aux_pp

    def eval_accuracy(self, result, output):
        accuracy = None

        if self.ActivationFuncType == ActivationFuncType.NONE:
            accuracy = 1 - np.mean(np.abs((result - output) / output))
        elif self.ActivationFuncType == ActivationFuncType.SIGMOID:
            estimate = np.greater(result, 0)
            answer = np.greater(output, 0.5)
            correct = np.equal(estimate, answer)
            accuracy = np.mean(correct)
        elif self.ActivationFuncType == ActivationFuncType.SOFTMAX:
            estimate = np.argmax(result, axis=1)
            answer = np.argmax(output, axis=1)
            correct = np.equal(estimate, answer)
            accuracy = np.mean(correct)

        return accuracy

    def get_backprop_Gradient(self, G_loss, aux):
        Gradient = None

        if self.ErrorType == ErrorType.ENTROPY:
            Gradient = self.GetEntropyGradient(G_loss, aux)

        elif self.ErrorType.MSE:
            shape = aux.shape
            Gradient = self.GetMseGrdient(G_loss, aux)

        return Gradient

    def GetMseGrdient(self, G_loss, aux):
        shape = aux.shape

        g_loss_square = np.ones(shape) / np.prod(shape)
        g_square_diff = 2 * aux
        g_diff_output = 1

        G_square = g_loss_square * G_loss
        G_diff = g_square_diff * G_square
        Gradient = g_diff_output * G_diff
        return Gradient

    def GetEntropyGradient(self, G_loss, aux):
        y, output, entropy = aux

        g_loss_entropy = 1.0 / np.prod(entropy.shape)

        g_entropy_output = None

        if self.ActivationFuncType == ActivationFuncType.SIGMOID:
            g_entropy_output = CommonUtils.sigmoid_cross_entropy_with_logits_derv(y, output)
        elif self.ActivationFuncType == ActivationFuncType.SOFTMAX:
            g_entropy_output = CommonUtils.softmax_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        Gradient = g_entropy_output * G_entropy

        return Gradient

    def backpropagation(self, Gradient, input):
        g_output_w = input.transpose()

        G_w = np.matmul(g_output_w, Gradient)
        G_b = np.sum(Gradient, axis=0)

        self.weight -= self.LEARNING_RATE * G_w
        self.bias -= self.LEARNING_RATE * G_b