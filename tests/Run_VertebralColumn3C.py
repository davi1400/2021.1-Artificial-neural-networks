# -*- coding: utf-8 -*-
"""
Created on Fri Sep 07 14:50:26 2018

@author: davi Leão
"""
import numpy as np
from scipy.io import arff
from MLP import MultiLayerPerceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#-----------------------------------------------------------------------------
#Tratamento dos dados
data,meta = arff.loadarff('column_3C_weka.arff')
from scipy.io import arff

X = np.zeros((int(data.shape[0]),len(data[0])-1))

for i in range(int(data.shape[0])):
    for j in range(len(data[0])-1):
        X[i][j] = data[i][j]
        
Y = np.zeros((int(data.shape[0]),1))

for i in range(int(data.shape[0])):
    if(data[i][6]=='Hernia'):
        Y[i][0] = 1
    if(data[i][6]=='Spondylolisthesis'):
        Y[i][0] = 2
    
        
Xnorm = (X - X.min(axis=0))/(X.max(axis=0)-X.min(axis=0)) 
X = np.append(-1*np.ones((X.shape[0],1)),Xnorm, 1)

Mat_Y = np.zeros((Y.shape[0],(Y.max()+1).astype(int)))
for i in range(len(Y)):
    if Y[i]==0:
        Mat_Y[i,0] = 1
    elif Y[i] == 1:
        Mat_Y[i,1] = 1
    elif Y[i] == 2:
        Mat_Y[i,2] = 1
        
#------------------------------------------------------------------------------
Vetor_Neuronios = 2*np.arange(1,11) #4*arange(1,5)
Matrizes = []
Taxas_de_acerto = []
for realizacoes in range(20):
    #Validacao
    # K- Fold com K = 10
    AcuraciasVal = []
    X_train,X_test,Y_train,Y_test = train_test_split(X,Mat_Y,test_size=0.2);
    for Neuronios_Ocultos in range(len(Vetor_Neuronios)):
        K = 5
        Taxas_de_acertoVal = []
        for esimo in range(1,K+1):
            L = int(X_train.shape[0]/K)
            X_trainVal = (np.c_[X_train[:L*esimo-L,:].T,X_train[esimo*L:,:].T]).T
            X_testVal = (X_train[L*esimo-L:esimo*L,:])
            Y_trainVal = (np.c_[Y_train[:L*esimo-L,:].T,Y_train[esimo*L:,:].T]).T
            Y_testVal = (Y_train[L*esimo-L:esimo*L,:])
            
            RedeVal =  MultiLayerPerceptron(X_trainVal.shape[1],Vetor_Neuronios[Neuronios_Ocultos],Y.max()+1,0.15,False);
            RedeVal.InicializacaoPesos()
            RedeVal.Train(X_trainVal,Y_trainVal,1000)
            
            G_SaidaVal  = RedeVal.Saida(X_testVal)
            Y_SaidaVal  = RedeVal.predicao(Y_testVal)
            Taxas_de_acertoVal.append(((G_SaidaVal==Y_SaidaVal).sum())/(1.0*len(Y_SaidaVal)))  
        AcuraciasVal.append(np.mean(Taxas_de_acertoVal))
    
    #Treino
    Neuronios_ocultos = np.where(AcuraciasVal == np.max(AcuraciasVal))[0][0]
    print(AcuraciasVal)
    print("Quantidade de neuronios ocultos %f") %(Vetor_Neuronios[Neuronios_ocultos])
    Rede =  MultiLayerPerceptron(X_train.shape[1],Vetor_Neuronios[Neuronios_ocultos],Y.max()+1,0.15,False);
    Rede.InicializacaoPesos()
    Rede.Train(X_train,Y_train,1000)
    
    G_SaidaTest = RedeVal.Saida(X_test)
    Y_SaidaTest = RedeVal.predicao(Y_test)
    Taxas_de_acerto.append(((G_SaidaTest==Y_SaidaTest).sum())/(1.0*len(Y_SaidaTest)))  
    Matrix_Confusao_test = confusion_matrix(G_SaidaTest,Y_SaidaTest)
    Matrizes.append(Matrix_Confusao_test)
    print(((G_SaidaTest==Y_SaidaTest).sum())/(1.0*len(Y_SaidaTest)))
    print(Matrix_Confusao_test)