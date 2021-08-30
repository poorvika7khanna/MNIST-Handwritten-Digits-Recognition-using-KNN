import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('seaborn')


def drawImg(sample):
    img = sample.reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()


def dist(x1, x2):
    return np.sqrt(sum((x1 - x2)**2))


def knn(X, Y, queryPoint, k=5):
    vals = []
    for i in range(X.shape[0]):
        distance = dist(queryPoint, X[i])
        vals.append((distance, Y[i]))
    vals = sorted(vals)
    vals = np.array(vals[:k], dtype=int)
    newval = np.unique(vals[:, 1], return_counts=True)
    ind = newval[1].argmax()
    pred = newval[0][ind]
    return pred


def accuracy(X, Y):
    cnt = 0
    for i in range(X.shape[0]):
        pred = knn(x, y, X[i])
        if pred == Y[i]:
            cnt += 100
    return cnt/X.shape[0]


train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")
data = train.values
x = data[:, 1:]
y = data[:, 0]
tx = test.values[:, 1:]
ty = test.values[:, 0]
# print(accuracy(tx, ty))             # to find the accuracy of the algorithm
