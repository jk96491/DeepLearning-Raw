import numpy as np
import csv
from SingleLayerPerceptron.Utils.SingleLayerPercept import SLP
from SingleLayerPerceptron.Utils.CommonUtils import ActivationFuncType
from SingleLayerPerceptron.Utils.CommonUtils import ErrorType

epoch_count = 10
mb_size = 10
report = 1

input_count = 10
output_count = 1


def run():
    model = SLP(input_count, output_count, epoch_count, mb_size, report, ErrorType.MSE, ActivationFuncType.NONE)
    data = load_data()
    model.Train(data)


def load_data():
    with open('abalone.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)

        data = np.zeros([len(rows), input_count + output_count])

        for n, row in enumerate(rows):
            if rows[0] == 'I':
                data[n, 0] = 1
            if rows[0] == 'M':
                data[n, 1] = 1
            if rows[0] == 'F':
                data[n, 2] = 1

            data[n, 3:] = row[1:]

    return data


if __name__ == '__main__':
    run()
