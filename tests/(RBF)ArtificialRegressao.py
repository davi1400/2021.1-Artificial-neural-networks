# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:46:30 2018

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
X1 = (X1 - X1.min(axis=0))/(X1.max(axis=0)-X1.min(axis=0)) 
#X1 = (X1 - X1.mean(axis=0))/(1.0*np.std(X1,axis=0)) 
X = np.concatenate((-1*np.ones((500,1)),X1),axis=1)
Y = X2.copy()

plt.plot(X[:,1:],Y,'go')
#plt.savefig('Grafico2Art1')
plt.show()

Vetor_Neuronios = 2*np.arange(1,12)
Error = []
Matrizes = []
Neuronios_realizacao = []

#-------------------------------------------------------------------------------------


N_centros = 2*np.arange(1,50)
Abertura =  np.linspace(0.1,50.0,50)
Taxas_de_acerto = []
Total_de_aberturas = []
Total_de_NumCentros = []

for realizacoes in range(20):
    i=0
    #Validacao
    # K- Fold com K = 5
    MSEVal = np.zeros((len(N_centros),len(Abertura)))
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2);
    for centros in range(len(N_centros)):
        
        for i in range(len(Abertura)):
            K = 5
            ErrorVal = []
            for esimo in range(1,K+1):
                L = int(X_train.shape[0]/K)
                X_trainVal = (np.c_[X_train[:L*esimo-L,:].T,X_train[esimo*L:,:].T]).T
                X_testVal = (X_train[L*esimo-L:esimo*L,:])
                Y_trainVal = (np.c_[Y_train[:L*esimo-L,:].T,Y_train[esimo*L:,:].T]).T
                Y_testVal = (Y_train[L*esimo-L:esimo*L,:])
                
                alpha =  Abertura[i]
                Num_centros = N_centros[centros]
                indices = np.arange(Num_centros)
                np.random.shuffle(indices)
                Centroides = X_trainVal[indices]
                
                
                
                
                # Calcular a matriz H , oculta
                H_trainVal = np.zeros((Num_centros,X_trainVal.shape[0]))
                for j in range(Num_centros):
                    r = np.sqrt((X_trainVal[:,1] - Centroides[j][1])**2 + (X_trainVal[:,0]-Centroides[j][0])**2)
                    U_j = np.exp(-np.power(alpha*r,2))
                    #U_j = np.exp(-Distancia/(2.0*(alpha**2)))
                    H_trainVal[j] = U_j
                    
                H_trainVal = H_trainVal.T
                
                H_trainVal = np.c_[-1*np.ones((H_trainVal.shape[0],1)),H_trainVal]
                eye = uniform(0,1)*np.eye(H_trainVal.shape[1])
                Pesos= (((np.linalg.inv( (H_trainVal.T.dot(H_trainVal)) + eye )).dot(H_trainVal.T)).dot(Y_trainVal))
               
                # Teste
                H_testVal = np.zeros((Num_centros,X_testVal.shape[0]))
                for j in range(Num_centros):
                    r = np.sqrt((X_testVal[:,1] - Centroides[j][1])**2 + (X_testVal[:,0]-Centroides[j][0])**2)
                    U_j = np.exp(-np.power(alpha*r,2))
                    #U_j = np.exp(-Distancia/(2.0*(alpha**2)))
                    H_testVal[j] = U_j
                
                H_testVal = H_testVal.T
                H_testVal = np.c_[-1*np.ones((H_testVal.shape[0])),H_testVal]
                
                G_SaidaVal = H_testVal.dot(Pesos)
                Y_SaidaVal = Y_testVal.copy()
                ErrorVal.append(np.sum(np.power(Y_SaidaVal - G_SaidaVal,2))/(len(Y_SaidaVal)*1.0))
                
            MSEVal[centros][i] = (np.mean(ErrorVal))
    
    Centro = np.where(MSEVal==(MSEVal).min())[0][0]
    i = np.where(MSEVal==(MSEVal).min())[1][0]
    alpha =  Abertura[i]
    Num_centros = N_centros[Centro]
    indices = np.arange(Num_centros)
    np.random.shuffle(indices)
    Centroides = X_train[indices]
    
    
    # Calcular a matriz H , oculta
    H_train = np.zeros((Num_centros,X_train.shape[0]))
    for j in range(Num_centros):
        r = np.sqrt((X_train[:,1] - Centroides[j][1])**2 + (X_train[:,0]-Centroides[j][0])**2)
        U_j = np.exp(-np.power(alpha*r,2))
        #U_j = np.exp(-Distancia/(2.0*(alpha**2)))
        H_train[j] = U_j
                    
    H_train = H_train.T
    H_train = np.c_[-1*np.ones((H_train.shape[0],1)),H_train]
    eye = uniform(0,1)*np.eye(H_train.shape[1])
    Pesos= (((np.linalg.inv( (H_train.T.dot(H_train)) + eye )).dot(H_train.T)).dot(Y_train))
        
    # Teste
    H_test = np.zeros((Num_centros,X_test.shape[0]))
    for j in range(Num_centros):
        r = np.sqrt((X_test[:,1] - Centroides[j][1])**2 + (X_test[:,0]-Centroides[j][0])**2)
        U_j = np.exp(-np.power(alpha*r,2))
        #U_j = np.exp(-Distancia/(2.0*(alpha**2)))
        H_test[j] = U_j
                
    H_test = H_test.T
    H_test = np.c_[-1*np.ones((H_test.shape[0])),H_test]
        
    G_Saida = H_test.dot(Pesos)
    Y_Saida = Y_test.copy()
    print("MSE da %f Realizacao = ") %(realizacoes)
    print((np.sum(np.power(Y_SaidaVal - G_SaidaVal,2))/(len(Y_SaidaVal)*1.0))) 
    Taxas_de_acerto.append(np.sum(np.power(Y_SaidaVal - G_SaidaVal,2))/(len(Y_SaidaVal)*1.0))     
    print("Abertura = %f") %(alpha)
    Total_de_aberturas.append(alpha)
    print("Numero de Centros = %f") %(Num_centros)        
    Total_de_NumCentros.append(Num_centros)