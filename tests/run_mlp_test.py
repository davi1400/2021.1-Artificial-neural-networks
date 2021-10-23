#  - * - coding: utf - 8 - *-
"""
Created on Mon Aug 27 20:11:16 2018
@author: davi Leão
"""
from mlfwk.models.MULTILAYER_PERCEPTRON_NET import mlp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
import numpy as np
from matplotlib.colors import ListedColormap

#  tratamento dos dados
# ------------------------------------------------------------
iris = load_iris()
X = np.array((iris.data))
Xnorm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
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

Taxas_de_acerto = []

Vetor_Neuronios = 3 * np.arange(1, 5)
Matrizes = []
# -----------------------------------------------------------

for realizacoes in range(20):
    # Validacao
    # K- Fold com K = 10
    AcuraciasVal = []
    X_train, X_test, Y_train, Y_test = train_test_split(X, Mat_Y, test_size=0.2)
    for Neuronios_Ocultos in range(len(Vetor_Neuronios)):
        K = 10
        Taxas_de_acertoVal = []
        for esimo in range(1, K + 1):
            L = int(X_train.shape[0] / K)
            X_trainVal = (np.c_[X_train[:L * esimo - L, :].T, X_train[esimo * L:, :].T]).T
            X_testVal = (X_train[L * esimo - L:esimo * L, :])
            Y_trainVal = (np.c_[Y_train[:L * esimo - L, :].T, Y_train[esimo * L:, :].T]).T
            Y_testVal = (Y_train[L * esimo - L:esimo * L, :])

            RedeVal = mlp.MultiLayerPerceptron(X_trainVal.shape[1], Vetor_Neuronios[Neuronios_Ocultos], Y.max() + 1, 0.15,
                                           False)
            RedeVal.InicializacaoPesos()
            RedeVal.Train(X_trainVal, Y_trainVal, 100)

            G_SaidaVal = RedeVal.Saida(X_testVal)
            Y_SaidaVal = RedeVal.predicao(Y_testVal)
            Taxas_de_acertoVal.append(((G_SaidaVal == Y_SaidaVal).sum()) / (1.0 * len(Y_SaidaVal)))
        AcuraciasVal.append(np.mean(Taxas_de_acertoVal))

    # Treino
    Neuronios_ocultos = np.where(AcuraciasVal == np.max(AcuraciasVal))[0][0]
    print(AcuraciasVal)
    print("Quantidade de neuronios ocultos", Vetor_Neuronios[Neuronios_ocultos])
    Rede = mlp.MultiLayerPerceptron(X_train.shape[1], Vetor_Neuronios[Neuronios_ocultos], Y.max() + 1, 0.15, False);
    Rede.InicializacaoPesos()
    Rede.Train(X_train, Y_train, 100)

    G_SaidaTest = RedeVal.Saida(X_test)
    Y_SaidaTest = RedeVal.predicao(Y_test)
    Taxas_de_acerto.append(((G_SaidaTest == Y_SaidaTest).sum()) / (1.0 * len(Y_SaidaTest)))
    Matrix_Confusao_test = confusion_matrix(G_SaidaTest, Y_SaidaTest)
    Matrizes.append(Matrix_Confusao_test)
    print(((G_SaidaTest == Y_SaidaTest).sum()) / (1.0 * len(Y_SaidaTest)))
    print(Matrix_Confusao_test)

'''
#---------------------------------------------------------------------------------------------------
X2 = X[:,3:].copy()
X2 = np.append(-1*np.ones((X.shape[0],1)),X2, 1)
Rede =  MultiLayerPerceptron(X2.shape[1],6,Y.max()+1,0.15);
for realizacoes in range(5):
    X_train,X_test,Y_train,Y_test = train_test_split(X2,Mat_Y,test_size=0.2);
    Rede.InicializacaoPesos()
    Rede.Train(X_train,Y_train,100)

    G_Saida = Rede.Saida(X_test)
    Y_Saida  = Rede.predicao(Y_test)
    Taxas_de_acerto.append(((G_Saida==Y_Saida).sum())/(1.0*len(Y_Saida)))
    print((G_Saida==Y_Saida).sum()/(1.0*len(Y_Saida)))

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
Rede.plotar(X_test,Y_test,cmap_light)  
X1 = X[:,:3].copy()
Rede =  MultiLayerPerceptron(X1.shape[1],6,Y.max()+1,0.15);
for realizacoes in range(5):
    X_train,X_test,Y_train,Y_test = train_test_split(X1,Mat_Y,test_size=0.2);
    Rede.InicializacaoPesos()
    Rede.Train(X_train,Y_train,100)

    G_Saida = Rede.Saida(X_test)
    Y_Saida  = Rede.predicao(Y_test)
    Taxas_de_acerto.append(((G_Saida==Y_Saida).sum())/(1.0*len(Y_Saida)))
    print((G_Saida==Y_Saida).sum()/(1.0*len(Y_Saida)))

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
Rede.plotar(X_test,Y_test,cmap_light)     
#---------------------------------------------------------------------------------------------------
'''
