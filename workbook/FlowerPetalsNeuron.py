import numpy as np
from matplotlib import pyplot as plt

SOURCE_DATA = [[3,   1.5,  1],
               [2,   1,    0],
               [4,   1.5,  1],
               [3,   1,    0],
               [3.5, 0.5,  1],
               [2,   0.5,  0],
               [5.5, 1,    1],
               [1,   1,    0]]


MYSTERY_DATA = [4.5, 1]
LEARNING_STEP = 0.2


def NN(m1, m2, w1, w2, b):
    val = m1 * w1 + m2 * w2 + b
    return sigmoid(val)


def sigmoid(val):
    return 1/(1 + np.exp(-val))


def sigmoid_d(val):
    return sigmoid(val) * (1 - sigmoid(val))


def vis_data():
    plt.grid()
    for i in range(len(SOURCE_DATA)):
        c = 'r'
        if SOURCE_DATA[i][2] == 0:
            c = 'b'
        plt.scatter([SOURCE_DATA[i][0]], [SOURCE_DATA[i][1]], c=c)

    plt.scatter([MYSTERY_DATA[0]], [MYSTERY_DATA[1]], c='gray')
    plt.show()


costs = []
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

for i in range(10000):
    item = SOURCE_DATA[np.random.randint(len(SOURCE_DATA))]

    neural_val = item[0] * w1 + item[1] * w2 + b
    predict = sigmoid(neural_val)
    target = item[2]

    # Main cost of current function
    cost = np.square(predict - target)
    # Derivative of current cost
    d_cost = 2 * (predict - target)
    # Derivative of current predict
    d_predict = sigmoid_d(neural_val)

    # Multiplication of Neural Wights (w1, w2, b)
    n_dw1 = item[0]
    n_dw2 = item[0]
    n_db = 1

    # Main function for calculation derivative of each
    # data points using [Der. Cost of neuron WITH Der.
    # Prediction neuron WITH data value]
    d_cost_w1 = d_cost * d_predict * n_dw1
    d_cost_w2 = d_cost * d_predict * n_dw2
    d_cost_b = d_cost * d_predict * n_db

    # Calculate new more efficient Neuron Weights!
    w1 = w1 - d_cost_w1 * LEARNING_STEP
    w2 = w2 - d_cost_w2 * LEARNING_STEP
    b = b - d_cost_b * LEARNING_STEP

    # print the cost over all data points every 1k iters
    if i % 100 == 0:
        c = 0
        for j in range(len(SOURCE_DATA)):
            p = SOURCE_DATA[j]
            p_pred = sigmoid(w1 * p[0] + w2 * p[1] + b)
            c += np.square(p_pred - p[2])
        costs.append(c)

for x in np.linspace(0, 6, 20):
    for y in np.linspace(0, 3, 20):
        pred = sigmoid(w1 * x + w2 * y + b)
        c = 'b'
        if pred > 0.5:
            c = 'r'
        plt.scatter([x],[y],c=c, alpha=.2)
vis_data()


# print NN(3, 1.5, w1, w2, b)
# print NN(2, 1, w1, w2, b)
