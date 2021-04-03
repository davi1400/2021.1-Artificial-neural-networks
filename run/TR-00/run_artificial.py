from mlfwk.readWrite import load_mock
import matplotlib.pyplot as plt
from numpy import where

if __name__ == '__main__':
    X, Y = load_mock(type='LOGICAL_AND')

    pos = X[where(Y == 1)[0]]
    neg = X[where(Y == -1)[0]]
    plt.plot(pos[:, 0], pos[:, 1], 'go')
    plt.plot(neg[:, 0], neg[:, 1], 'ro')
    plt.show()