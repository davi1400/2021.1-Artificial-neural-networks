# -*- coding: utf-8 -*-
"""
Created on Sat Sep 01 16:26:34 2018

@author: davi Leão
"""
from MLP import MultiLayerPerceptron
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from random import uniform
#---------------------------------------------------------------------------------
#Criação do conjunto de dados artificial(3xSin(X) + 1)
X1 = np.linspace(-10,10,500)
Barulho = (-1 + (1 - (-1)) *np.random.rand(500))*2.0
X2 = (3.0*np.sin(X1) + 1)  + Barulho



X1 = np.array((X1),ndmin=2).T
X2 = np.array((X2),ndmin=2).T
#X1 = (X1 - X1.min(axis=0))/(X1.max(axis=0)-X1.min(axis=0)) 
X1 = (X1 - X1.mean(axis=0))/(1.0*np.std(X1,axis=0)) 
X = np.concatenate((-1*np.ones((500,1)),X1),axis=1)
Y = X2.copy()

plt.plot(X[:,1:],Y,'go')
#plt.savefig('Grafico2Art1')
plt.show()

Vetor_Neuronios = 3*np.arange(1,12)
Error = []
Matrizes = []
Neuronios_realizacao = []

#-------------------------------------------------------------------------------------

for realizacoes in range(20):
    #Validacao
    # K- Fold com K = 10
    MSEVal = []
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2);
    for Neuronios_Ocultos in range(len(Vetor_Neuronios)):
        K = 5
        ErrorVal = []
        for esimo in range(1,K+1):
            L = int(X_train.shape[0]/K)
            X_trainVal = (np.c_[X_train[:L*esimo-L,:].T,X_train[esimo*L:,:].T]).T
            X_testVal = (X_train[L*esimo-L:esimo*L,:])
            Y_trainVal = (np.c_[Y_train[:L*esimo-L,:].T,Y_train[esimo*L:,:].T]).T
            Y_testVal = (Y_train[L*esimo-L:esimo*L,:])
            
             
            RedeVal =  MultiLayerPerceptron(X_trainVal.shape[1],Vetor_Neuronios[Neuronios_Ocultos],1,0.15,True);
            RedeVal.InicializacaoPesos()
            RedeVal.Train(X_trainVal,Y_trainVal,1000)
            
            #MSE e RSME
            G_SaidaVal  =   RedeVal.Saida(X_testVal)
            Y_SaidaVal  =   (Y_testVal).copy()
            ErrorVal.append(np.sum(np.power(Y_SaidaVal - G_SaidaVal,2))/(len(Y_SaidaVal)*1.0))
        MSEVal.append(np.mean(ErrorVal))
        
    
    Neuronios_ocultos = np.where(MSEVal == np.min(MSEVal))[0][0]
    Neuronios_realizacao.append(Vetor_Neuronios[Neuronios_ocultos])
    print(MSEVal)
    print("Quantidade de neuronios ocultos %f") %(Vetor_Neuronios[Neuronios_ocultos])
    Rede =  MultiLayerPerceptron(X_train.shape[1],Vetor_Neuronios[Neuronios_ocultos],1,0.15,True);
    Rede.InicializacaoPesos()
    Rede.Train(X_train,Y_train,1000)
    
    G_Saida  =   Rede.Saida(X_test)
    Y_Saida  =   (Y_test).copy()
    Error.append(np.sum(np.power(Y_Saida - G_Saida,2))/(len(Y_Saida)*1.0))
    print(np.sum(np.power(Y_Saida - G_Saida,2))/(len(Y_Saida)*1.0))
     

plt.plot(X[:,1:],Rede.Saida(X))
plt.plot(X[:,1:],Y,'go')
    
    
    
    
    
    