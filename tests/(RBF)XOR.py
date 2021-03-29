# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:10:26 2018

@author: davi Leão
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.colors import ListedColormap
from random import randint
from scipy.special import expit
from random import uniform


def Sigmoid(h):
        return expit(h)
    
def predicao(Y):
        y = np.zeros((Y.shape[0],1))
        for j in range(Y.shape[0]):
            i = np.where(Y[j,:]==Y[j,:].max())[0][0]
            y[j] = i
        return y

#Criação dos dados
t = np.linspace(0,2*np.pi,50)
r = np.random.rand((50))/5.0
data1 = np.c_[np.array((0 +  r*np.cos(t))) ,np.array((0 +  r*np.sin(t)))]
data2 = np.c_[np.array((0 +  r*np.cos(t))) ,np.array((1 +  r*np.sin(t)))]
data3 = np.c_[np.array((1 +  r*np.cos(t))) ,np.array((0 +  r*np.sin(t)))]
data4 = np.c_[np.array((1 +  r*np.cos(t))) ,np.array((1 +  r*np.sin(t)))]

X = np.c_[data1.T,data2.T,data3.T,data4.T].T
Xnorm = (X - X.min(axis=0))/(X.max(axis=0)-X.min(axis=0)) 
X = Xnorm.copy()
Y = np.ones((X.shape[0],1))
for i in range(len(Y)):
    if(i<50):
        Y[i] = 0
    if(i>=150):
        Y[i] = 0
Mat_Y = np.zeros((Y.shape[0],(Y.max()+1).astype(int)))
for i in range(len(Y)):
    if Y[i]==0:
        Mat_Y[i,0] = 1
    elif Y[i] == 1:
        Mat_Y[i,1] = 1


#--------------------------------------------------------------------------------------------------------
        
N_centros = 2*np.arange(1,50)
Abertura =  np.linspace(0.1,50.0,50)
Taxas_de_acerto = []
Total_de_aberturas = []
Total_de_NumCentros = []

for realizacoes in range(20):
    i=0
    #Validacao
    # K- Fold com K = 5
    AcuraciasVal = np.zeros((len(N_centros),len(Abertura)))
    X_train,X_test,Y_train,Y_test = train_test_split(X,Mat_Y,test_size=0.2);
    for centros in range(len(N_centros)):
        
        for i in range(len(Abertura)):
            K = 5
            Taxas_de_acertoVal = []
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
                    H_trainVal[j] = np.exp(-np.power(alpha*r,2))
                    #U_j = np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)
                    
                    
                H_trainVal = H_trainVal.T 
                
                H_trainVal = np.c_[-1*np.ones((H_trainVal.shape[0],1)),H_trainVal]
                eye = uniform(0,1)*np.eye(H_trainVal.shape[1])
                Pesos= (((np.linalg.inv( (H_trainVal.T.dot(H_trainVal)) + eye )).dot(H_trainVal.T)).dot(Y_trainVal))
                
                # Teste
                H_testVal = np.zeros((Num_centros,X_testVal.shape[0]))
                
                for j in range(Num_centros):
                    r = np.sqrt((X_testVal[:,1] - Centroides[j][1])**2 + (X_testVal[:,0]-Centroides[j][0])**2)
                    #U_j = np.exp(-Distancia/(2.0*(alpha**2)))
                    H_testVal[j]= np.exp(-np.power(alpha*r,2))
                    
                   
                
                H_testVal = H_testVal.T 
                H_testVal = np.c_[-1*np.ones((H_testVal.shape[0])),H_testVal]
                
                G_SaidaVal = predicao(Sigmoid(H_testVal.dot(Pesos)))
                Y_SaidaVal = predicao(Y_testVal)
                Taxas_de_acertoVal.append(((G_SaidaVal==Y_SaidaVal).sum())/(1.0*len(Y_SaidaVal)))
                
            AcuraciasVal[centros][i] = (np.mean(Taxas_de_acertoVal))
    
    Centro = np.where(AcuraciasVal==AcuraciasVal.max())[0][0]
    i = np.where(AcuraciasVal==AcuraciasVal.max())[1][0]
    alpha =  Abertura[i]
    Num_centros = N_centros[Centro]
    indices = np.arange(Num_centros)
    np.random.shuffle(indices)
    Centroides = X_train[indices]
    
    
    # Calcular a matriz H , oculta
    H_train = np.zeros((Num_centros,X_train.shape[0]))
   
    for j in range(Num_centros):
        r = np.sqrt((X_train[:,1] - Centroides[j][1])**2 + (X_train[:,0]-Centroides[j][0])**2)
        #r = np.linalg.norm(X_train-Centroides[j])
        H_train[j] = np.exp(-np.power(alpha*r,2))
        #U_j = np.exp(-Distancia/(2.0*(alpha**2)))
        
                    
    H_train = H_train.T 
    H_train = np.c_[-1*np.ones((H_train.shape[0],1)),H_train]
    eye = uniform(0,1)*np.eye(H_train.shape[1])
    Pesos= (((np.linalg.inv( (H_train.T.dot(H_train)) + eye)).dot(H_train.T)).dot(Y_train))
        
    # Teste
    H_test = np.zeros((Num_centros,X_test.shape[0]))
   
    for j in range(Num_centros):
        r = np.sqrt((X_test[:,1] - Centroides[j][1])**2 + (X_test[:,0]-Centroides[j][0])**2)
        #r = np.linalg.norm(X_test-Centroides[j])
        H_test[j] = np.exp(-np.power(alpha*r,2))
        
                
    H_test = H_test.T
    H_test = np.c_[-1*np.ones((H_test.shape[0])),H_test]
        
    G_Saida = predicao(Sigmoid(H_test.dot(Pesos)))
    Y_Saida = predicao(Y_test)
    print("Acuracia da %f Realizacao = ") %(realizacoes)
    print(((G_Saida==Y_Saida).sum())/(1.0*len(Y_Saida))) 
    Taxas_de_acerto.append(((G_Saida==Y_Saida).sum())/(1.0*len(Y_Saida)))     
    print("Abertura = %f") %(alpha)
    Total_de_aberturas.append(alpha)
    print("Numero de Centros = %f") %(Num_centros)        
    Total_de_NumCentros.append(Num_centros)
                    
                    
                    
                    
                
                
#-----------------------------------------------------------------------------------------------------
#Plot    
Mapa_Cor = ListedColormap(['#FFAAAA','#AAAAFF'])
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                                 np.arange(y_min, y_max, .02))
        
new = np.c_[xx.ravel(), yy.ravel()]
H_test = np.zeros((Num_centros,new.shape[0]))
for j in range(Num_centros):
    r = np.sqrt((new[:,1] - Centroides[j][1])**2 + (new[:,0]-Centroides[j][0])**2)
    #r = np.linalg.norm(X_test-Centroides[j])
    H_test[j] = np.exp(-np.power(alpha*r,2))


Classe0 = X_test[np.where(predicao(Y_test)==0)[0]]
Classe1 = X_test[np.where(predicao(Y_test)==1)[0]]

H_test = H_test.T
H_test = np.c_[-1*np.ones((H_test.shape[0])),H_test]


Z = predicao(Sigmoid(H_test.dot(Pesos)))
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=Mapa_Cor)
plt.plot(Classe0[:,0],Classe0[:,1],'ro',marker='s',markeredgecolor='w');
plt.plot(Classe1[:,0],Classe1[:,1],'bo',marker='D',markeredgecolor='w');   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

