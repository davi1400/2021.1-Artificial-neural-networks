# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:23:52 2018

@author: davi Leão
Rede perceptron usando função degrau para ativação
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# from matplotlib.colors import  ListedColormap
import time


def u(x):
    for i in range(len(x)):
        if x[i] > 0:
            x[i] = 1
        else:
            x[i] = 0
    return x


def predicao(Y):
    y = np.zeros((Y.shape[0], 1))
    for j in range(Y.shape[0]):
        i = np.where(Y[j, :] == Y[j, :].max())[0][0]
        y[j] = i
    return y


# //////////////////////////////////////////////////////////////////////////
# geração dos dados
inicio = time.time()
t = np.linspace(0, 2 * np.pi, 50)
r = np.random.rand((50)) / 1.38
data1 = np.c_[np.array((4 + r * np.cos(t))), np.array((4 + r * np.sin(t)))]
data2 = np.c_[np.array((5 + r * np.cos(t))), np.array((2 + r * np.sin(t)))]
data3 = np.c_[np.array((6 + r * np.cos(t))), np.array((4 + r * np.sin(t)))]

X = np.c_[data1.T, data2.T, data3.T].T
Xnorm = (X - X.mean(axis=0)) / (1.0 * np.std(X, axis=0))
X = np.c_[-1 * np.ones((150, 1)), Xnorm]
Y = np.zeros((150, 1))
for i in range(50, len(Y)):
    if (i < 100):
        Y[i] = 1
    else:
        Y[i] = 2

Mat_Y = np.zeros((Y.shape[0], (Y.max() + 1).astype(int)))
for i in range(len(Y)):
    if Y[i] == 0:
        Mat_Y[i, 0] = 1
    elif Y[i] == 1:
        Mat_Y[i, 1] = 1
    else:
        Mat_Y[i, 2] = 1

classe0 = X[np.where(Y == 0)[0]]
classe1 = X[np.where(Y == 1)[0]]
classe2 = X[np.where(Y == 2)[0]]

plt.plot(classe0[:, 1], classe0[:, 2], '+');
plt.plot(classe1[:, 1], classe1[:, 2], 'g*');
plt.plot(classe2[:, 1], classe2[:, 2], 'ro');
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig("Grafico3Art2")
plt.close()
# /////////////////////////////////////////////////////////////////////////
# Treinamento
m, n = X.shape
lr = 0.1
'''
cada linha dos pesos indica os pesos de associados a cada neuronio de saida. 
Foi usada a função degrau como função de ativação.

'''
pesos = np.random.rand((Y.max() + 1).astype(int), n)
Taxas_acerto = []
k = 0  # contador

for i in range(1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Mat_Y, test_size=0.2)
    r = np.random.permutation(X_train.shape[0])  # Função equivalente ao randperm() do matlab
    for j in range(10000):
        r = np.random.permutation(X_train.shape[0])  # Função equivalente ao randperm() do matlab
        for k in range(len(r)):
            exemploX = np.array((X_train[r[k]]), ndmin=2)
            exemploY = np.array((Y_train[r[k]]), ndmin=2)
            h_train = pesos.dot(exemploX.T)
            g_train = np.heaviside(h_train, 0)
            error_train = exemploY.T - g_train
            pesos += lr * (error_train).dot(exemploX)

    # Teste
    h_test = (pesos.dot(X_test.T)).T
    g_test = predicao(np.heaviside(h_test, 0))
    y_test = predicao(Y_test)
    Taxas_acerto.append((y_test == g_test).sum() / (1.0 * len(y_test)))

# //////////////////////////////////////////////////////////////////////////////
# Teste final

g_teste = predicao(np.heaviside((pesos.dot(X_test.T)).T, 0))
Y_test = predicao(Y_test)
Matrix_Confusao = confusion_matrix(g_teste, Y_test)

Classe0 = X_test[np.where(Y_test == 0)[0]]
Classe1 = X_test[np.where(Y_test == 1)[0]]
Classe2 = X_test[np.where(Y_test == 2)[0]]

plt.plot(Classe0[:, 1], Classe0[:, 2], '+');
plt.plot(Classe1[:, 1], Classe1[:, 2], 'g*');
plt.plot(Classe2[:, 1], Classe2[:, 2], 'ro');

x1 = np.array((X[:, 1]))
Reta0 = -(pesos[0, 1] * x1 - pesos[0, 0]) / (1.0 * pesos[0, 2])
Reta1 = -(pesos[1, 1] * x1 - pesos[1, 0]) / (1.0 * pesos[1, 2])
Reta2 = -(pesos[2, 1] * x1 - pesos[2, 0]) / (1.0 * pesos[2, 2])
plt.plot(X[:, 1], Reta0)
plt.plot(X[:, 1], Reta1)
plt.plot(X[:, 1], Reta2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Superficie de decisao")
plt.axis([-2, 2, -2, 2])
# plt.savefig('Grafico3Art3')
plt.show()
plt.close()

h = .02
x_min, x_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
y_min, y_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
new = np.c_[xx.ravel(), yy.ravel()]
Z = (np.heaviside((pesos.dot(np.c_[-1 * np.ones(new.shape[0]), new].T)).T, 0))
Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
for i in range(Z.shape[0]):
    for j in range(Z[0].shape[0]):
        ind = np.where(Z[i][j] == Z[i][j].max())[0][0]
        Z[i][j] = 0
        Z[i][j][ind] = 1
plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")
plt.plot(Classe0[:, 1], Classe0[:, 2], 'ro', marker='s', markeredgecolor='w');
plt.plot(Classe1[:, 1], Classe1[:, 2], 'go', marker='D', markeredgecolor='w');
plt.plot(Classe2[:, 1], Classe2[:, 2], 'bo', markeredgecolor='w');
plt.show()
plt.close()

# //////////////////////////////////////////////////////////////////////////////
# plot
Y_train = predicao(Y_train)
Classe0 = X_train[np.where(Y_train == 0)[0]]
Classe1 = X_train[np.where(Y_train == 1)[0]]
Classe2 = X_train[np.where(Y_train == 2)[0]]

plt.plot(Classe0[:, 1], Classe0[:, 2], '+');
plt.plot(Classe1[:, 1], Classe1[:, 2], 'g*');
plt.plot(Classe2[:, 1], Classe2[:, 2], 'ro');

x1 = np.array((X[:, 1]))
Reta0 = -(pesos[0, 1] * x1 - pesos[0, 0]) / (1.0 * pesos[0, 2])
Reta1 = -(pesos[1, 1] * x1 - pesos[1, 0]) / (1.0 * pesos[1, 2])
Reta2 = -(pesos[2, 1] * x1 - pesos[2, 0]) / (1.0 * pesos[2, 2])
plt.plot(X[:, 1], Reta0)
plt.plot(X[:, 1], Reta1)
plt.plot(X[:, 1], Reta2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Superficie de decisao")
plt.axis([-2, 2, -2, 2])
# plt.savefig('Grafico3Art1')
plt.show()
plt.close()

h = .02
x_min, x_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
y_min, y_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
new = np.c_[xx.ravel(), yy.ravel()]
Z = (np.heaviside((pesos.dot(np.c_[-1 * np.ones(new.shape[0]), new].T)).T, 0))
Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
for i in range(Z.shape[0]):
    for j in range(Z[0].shape[0]):
        ind = np.where(Z[i][j] == Z[i][j].max())[0][0]
        Z[i][j] = 0
        Z[i][j][ind] = 1
plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")
plt.plot(Classe0[:, 1], Classe0[:, 2], 'ro', marker='s', markeredgecolor='w');
plt.plot(Classe1[:, 1], Classe1[:, 2], 'go', marker='D', markeredgecolor='w');
plt.plot(Classe2[:, 1], Classe2[:, 2], 'bo', markeredgecolor='w');
plt.show()
plt.close()

print('Acuracia: %f') % (np.mean(Taxas_acerto))
print('Desvio padrão das taxas de acerto: %f') % (np.std(Taxas_acerto))
print('Matrix de confusão: ')
print(Matrix_Confusao)
plt.matshow(Matrix_Confusao)
plt.colorbar()
# plt.savefig("Grafico3Art4")
plt.show()

fim = time.time()
print(fim - inicio)