# -*- coding: utf-8 -*-
"""
Created on Sat Sep 08 12:50:58 2018

@author: davi Leão
"""

import numpy  as np
import pandas as pd
from sklearn.preprocessing import Imputer
from MLP import MultiLayerPerceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = pd.read_csv('breast-cancer-wisconsin.data.txt',names=[0,1,2,3,4,5,6,7,8,9,10])

data[6] = data[6].replace('?','NaN')
imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
data[[6]] = imputer.fit_transform(data[[6]])


Y = np.array(data[10],ndmin=2).T # Classes 2 para begino e 4 para maligino
data = data.drop(10,axis=1)

X = np.zeros((data.shape))
for i in range(X.shape[1]):
    X.T[i] = data[i]
    
Xnorm = (X - X.min(axis=0))/(X.max(axis=0)-X.min(axis=0)) 
X = np.append(-1*np.ones((X.shape[0],1)),Xnorm, 1)

Mat_Y = np.zeros((Y.shape[0],2))
for i in range(len(Y)):
    if Y[i]==2: # Benigno
        Mat_Y[i,0] = 1
    elif Y[i] == 4: #Maligno
        Mat_Y[i,1] = 1
        
#------------------------------------------------------------------------------
Vetor_Neuronios = 2*np.arange(1,10)
Matrizes = []
Taxas_de_acerto = []
for realizacoes in range(20):
    #Validacao
    # K- Fold com K = 10
    AcuraciasVal = []
    X_train,X_test,Y_train,Y_test = train_test_split(X,Mat_Y,test_size=0.2);
    for Neuronios_Ocultos in range(len(Vetor_Neuronios)):
        K = 10
        Taxas_de_acertoVal = []
        for esimo in range(1,K+1):
            L = int(X_train.shape[0]/K)
            X_trainVal = (np.c_[X_train[:L*esimo-L,:].T,X_train[esimo*L:,:].T]).T
            X_testVal = (X_train[L*esimo-L:esimo*L,:])
            Y_trainVal = (np.c_[Y_train[:L*esimo-L,:].T,Y_train[esimo*L:,:].T]).T
            Y_testVal = (Y_train[L*esimo-L:esimo*L,:])
            
            RedeVal =  MultiLayerPerceptron(X_trainVal.shape[1],Vetor_Neuronios[Neuronios_Ocultos],2,0.15,False);
            RedeVal.InicializacaoPesos()
            RedeVal.Train(X_trainVal,Y_trainVal,500)
            
            G_SaidaVal  = RedeVal.Saida(X_testVal)
            Y_SaidaVal  = RedeVal.predicao(Y_testVal)
            Taxas_de_acertoVal.append(((G_SaidaVal==Y_SaidaVal).sum())/(1.0*len(Y_SaidaVal)))  
        AcuraciasVal.append(np.mean(Taxas_de_acertoVal))
    
    #Treino
    Neuronios_ocultos = np.where(AcuraciasVal == np.max(AcuraciasVal))[0][0]
    print(AcuraciasVal)
    print("Quantidade de neuronios ocultos %f") %(Vetor_Neuronios[Neuronios_ocultos])
    Rede =  MultiLayerPerceptron(X_train.shape[1],Vetor_Neuronios[Neuronios_ocultos],2,0.15,False);
    Rede.InicializacaoPesos()
    Rede.Train(X_train,Y_train,500)
    
    G_SaidaTest = RedeVal.Saida(X_test)
    Y_SaidaTest = RedeVal.predicao(Y_test)
    Taxas_de_acerto.append(((G_SaidaTest==Y_SaidaTest).sum())/(1.0*len(Y_SaidaTest)))  
    Matrix_Confusao_test = confusion_matrix(G_SaidaTest,Y_SaidaTest)
    Matrizes.append(Matrix_Confusao_test)
    print(((G_SaidaTest==Y_SaidaTest).sum())/(1.0*len(Y_SaidaTest)))
    print(Matrix_Confusao_test)