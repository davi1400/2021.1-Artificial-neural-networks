# -*- coding: utf-8 -*-
"""
Created on Sat Sep 01 15:07:10 2018

@author: davi Leão

XOR usando Multi Layer Perceptron
"""
from MLP import MultiLayerPerceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.colors import ListedColormap
#Criação dos dados
t = np.linspace(0,2*np.pi,50)
r = np.random.rand((50))/5.0
data1 = np.c_[np.array((0 +  r*np.cos(t))) ,np.array((0 +  r*np.sin(t)))]
data2 = np.c_[np.array((0 +  r*np.cos(t))) ,np.array((1 +  r*np.sin(t)))]
data3 = np.c_[np.array((1 +  r*np.cos(t))) ,np.array((0 +  r*np.sin(t)))]
data4 = np.c_[np.array((1 +  r*np.cos(t))) ,np.array((1 +  r*np.sin(t)))]

X = np.c_[data1.T,data2.T,data3.T,data4.T].T
Xnorm = (X - X.min(axis=0))/(X.max(axis=0)-X.min(axis=0)) 
X = np.append(-1*np.ones((X.shape[0],1)),Xnorm, 1)
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



Vetor_Neuronios = 2*np.arange(1,5)
Taxas_de_acerto = []
Matrizes = []
Neuronios_realizacao = []
#--------------------------------------------------------------------------------------

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
                    
            RedeVal =  MultiLayerPerceptron(X_trainVal.shape[1],Vetor_Neuronios[Neuronios_Ocultos],Y.max()+1,0.15,False);
            RedeVal.InicializacaoPesos()
            RedeVal.Train(X_trainVal,Y_trainVal,500)
            
            G_SaidaVal = RedeVal.Saida(X_testVal)
            Y_SaidaVal  = RedeVal.predicao(Y_testVal)
            Taxas_de_acertoVal.append(((G_SaidaVal==Y_SaidaVal).sum())/(1.0*len(Y_SaidaVal)))  
        AcuraciasVal.append(np.mean(Taxas_de_acertoVal))
     
    # Treino
    Neuronios_ocultos = np.where(AcuraciasVal == np.max(AcuraciasVal))[0][0]
    Neuronios_realizacao.append(Vetor_Neuronios[Neuronios_ocultos])
    print(AcuraciasVal)
    print("Quantidade de neuronios ocultos %f") %(Vetor_Neuronios[Neuronios_ocultos])
    Rede =  MultiLayerPerceptron(X_train.shape[1],Vetor_Neuronios[Neuronios_ocultos],Y.max()+1,0.15,False);
    Rede.InicializacaoPesos()
    Rede.Train(X_train,Y_train,500)
    
    G_Saida = RedeVal.Saida(X_test)
    Y_Saida = RedeVal.predicao(Y_test)
    Taxas_de_acerto.append(((G_Saida==Y_Saida).sum())/(1.0*len(Y_Saida)))
    Matrix_Confusao_test = confusion_matrix(G_Saida,Y_Saida)
    Matrizes.append(Matrix_Confusao_test)
    print(((G_Saida==Y_Saida).sum())/(1.0*len(Y_Saida)))
    print(Matrix_Confusao_test)
    
#-----------------------------------------------------------------------------------------------
# Acuracia, desvio padrão e Matriz de confusão
print("Acuracia = %f") %(np.mean(Taxas_de_acerto))
print("Desvio Padrão das Taxas de acerto = %f ") %(np.std(Taxas_de_acerto))
print("Matriz de confusão: ")
print(Matrizes[np.where(Taxas_de_acerto == np.max(Taxas_de_acerto))[0][0]])
plt.matshow(Matrizes[np.where(Taxas_de_acerto == np.max(Taxas_de_acerto))[0][0]])
plt.colorbar()
plt.show()
plt.close()
#------------------------------------------------------------------------------------------------
#Plot
h = .02
Mapa_Cor = ListedColormap(['#FFAAAA','#AAAAFF'])
x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
new = np.c_[xx.ravel(), yy.ravel()]
H = (Rede.Sigmoid(Rede.Pesos_ocultos.dot(np.c_[-1*np.ones(new.shape[0]),new].T)).T)
Z = (Rede.Sigmoid(Rede.Pesos_saida.dot((np.c_[-1*np.ones(H.shape[0]),H]).T)))
pos = X_test[np.where(Rede.predicao(Y_test)==1)[0]]
neg = X_test[np.where(Rede.predicao(Y_test)==0)[0]] 
Z = Rede.predicao(Z.T)
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=Mapa_Cor)
plt.plot(pos[:,1],pos[:,2],'bo',marker='s',markeredgecolor='w')
plt.plot(neg[:,1],neg[:,2],'ro',marker='s',markeredgecolor='w') 
plt.xlabel("X1")
plt.ylabel("X2") 
#plt.savefig('Grafico7Art8')  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    