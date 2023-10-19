import numpy as np
import argparse
import csv

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eta', '--eta')
    parser.add_argument('--data', '--data')
    parser.add_argument('--iterations', '--iterations')
    args = parser.parse_args()
    return args


def InitializeWeights():
    np.random.seed(1)
    Weights_Hidden = np.array([[-0.30000, -0.10000, 0.20000], [0.40000, -0.40000, 0.1000]])
    Weights_Output = np.array([[0.10000, 0.30000, -0.40000]])
    Weights_Output = Weights_Output.reshape(-1, 1)
    Bias_Hidden = np.array([0.20000, -0.50000, 0.30000])
    Bias_Output = -0.10000

    return  Weights_Hidden, Bias_Hidden, Weights_Output, Bias_Output

def flattenedList(lists):
    flattened = []
    for list in lists:
        for val in list:
            flattened.append(val)
    return flattened

def readCsv(file):
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = np.array(list(reader)).astype(float)
    X = data[:, :-1]
    Y = data[:, -1]
    Y = Y.reshape(-1,1)

    return X, Y

def FindSigmoid(n):
    sig = 1 / (1 + np.exp(-n))
    return sig

def SigDer(z):
    return z * (1 - z)



if __name__ == '__main__':
    args = parseArgs()
    features, labels = readCsv(args.data)
   
    iterations = int(args.iterations)
    learning_rate = float(args.eta)


    header2 = ["-","-","-","-","-","-","-","-","-","-","-",   0.20000,  -0.30000,   0.40000,  -0.50000,  -0.10000,  -0.40000,   0.30000,   0.20000,   0.10000,  -0.10000,   0.10000,   0.30000,  -0.40000]

    print(*header2, sep=',   ')

    weights_hidden, bias_hidden, weights_output, bias_output = InitializeWeights()

    for epoch in range(0, iterations):
        for X, Y in zip(features, labels):
            curr_list = []

            curr_list.append(list(X))

            X = X.reshape(1, -1).astype(float)

            H = FindSigmoid(np.dot(X, weights_hidden) + bias_hidden)
            curr_list.append(flattenedList(H.tolist()))

            O = FindSigmoid(np.dot(H, weights_output) + bias_output)

            curr_list.append(flattenedList(O))
            curr_list.append(list(Y))

            delta_O = (Y - O) * SigDer(O)

            delta_H = SigDer(H) * (weights_output.T * delta_O)

            curr_list.append(flattenedList(delta_H))

            curr_list.append(flattenedList(delta_O))

            bias_hidden = bias_hidden + (learning_rate * delta_H * 1)

            weights_hidden = weights_hidden + (learning_rate * X.T.dot(delta_H))

            for i in range(0, bias_hidden.shape[1]):
                curr_list.append(flattenedList(bias_hidden[:, i:i + 1]))
                curr_list.append(flattenedList(weights_hidden[:, i:i + 1]))

            bias_output += (learning_rate * delta_O * 1)
            curr_list.append(flattenedList(bias_output))
            weights_output += (learning_rate * H.T.dot(delta_O))

            curr_list.append(flattenedList(weights_output))

            print(",   ".join(repr(round(e, 5)) for e in flattenedList(curr_list)))
            curr_list.clear()