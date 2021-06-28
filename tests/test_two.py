# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 20:36:08 2018

@author: davi Leão
Classificao da iris com rede perceptron usando a função degrau como função de ativação
"""
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


def predicao(Y):
    y = np.zeros((Y.shape[0], 1))
    for j in range(Y.shape[0]):
        i = np.where(Y[j, :] == Y[j, :].max())[0][0]
        y[j] = i
    return y


# tratamento dos dados
# ------------------------------------------------------------
iris = load_iris()
X = np.array((iris.data))
# Xnorm = (X - X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
Xnorm = (X - X.mean(axis=0)) / (1.0 * np.std(X, axis=0))
X = np.append(-1 * np.ones((X.shape[0], 1)), Xnorm, 1)
Y = np.array((iris.target), ndmin=2).T
Mat_Y = np.zeros((Y.shape[0], Y.max() + 1))

for i in range(len(Y)):
    if Y[i] == 0:
        Mat_Y[i, 0] = 1
    elif Y[i] == 1:
        Mat_Y[i, 1] = 1
    else:
        Mat_Y[i, 2] = 1

# -----------------------------------------------------------
pesos = (0.4 + (0.6 - (0.4)) * np.random.rand(3, 5))
Taxas_acerto = []
lr = 0.01
k = 0  # contador
# /////////////////////////////////////////////////////////////////////////////////////

# np.random.seed(1)
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Mat_Y, test_size=0.2)

    for j in range(10000):
        r = np.random.permutation(X_train.shape[0])  # Função equivalente ao randperm() do matlab
        for k in range(len(r)):
            exemploX = np.array((X_train[r[k]]), ndmin=2)
            exemploY = np.array((Y_train[r[k]]), ndmin=2)
            h_train = (pesos.dot(exemploX.T))  # Pesos(3xn) * X(nx1)
            g_train = np.heaviside(h_train, 0)
            error_train = exemploY.T - g_train
            pesos += lr * 1.0 * (error_train).dot(exemploX)

    # teste

    h_test = np.heaviside((X_test.dot(pesos.T)), 0)
    h_test = predicao(h_test)
    y_test = predicao(Y_test)
    Taxas_acerto.append((h_test == y_test).sum() / (1.0 * len(y_test)))
    print((h_test == y_test).sum() / (1.0 * len(y_test)))

# //////////////////////////////////////////////////////////////////////////////
# Teste final
g_teste = predicao(np.heaviside((pesos.dot(X_test.T)).T, 0))
Y_teste = predicao(Y_test)
Matrix_Confusao_teste = confusion_matrix(g_teste, Y_teste)

print('Acuracia: ', np.mean(Taxas_acerto))
print('Desvio padrao das taxas de acerto: %f') % (np.std(Taxas_acerto))
print('Matrix de confusão Teste: ')
print(Matrix_Confusao_teste)
plt.matshow(Matrix_Confusao_teste)
plt.colorbar()
# plt.savefig("Grafico3Art5")
plt.show()