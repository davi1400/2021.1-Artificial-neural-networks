# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 20:36:08 2018

@author: davi Leão
Classificao da iris com rede perceptron usando a função degrau como função de ativação
"""
import sys
from pathlib import Path
print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from pandas import DataFrame
from matplotlib.colors import ListedColormap

from numpy import where, append, ones, array, zeros, mean, argmax, linspace, concatenate, c_, std
from mlfwk.metrics import metric
from mlfwk.readWrite import load_mock
from mlfwk.utils import split_random, get_project_root, normalization, out_of_c_to_label
from mlfwk.models import simple_perceptron_network
from mlfwk.readWrite import load_base
from mlfwk.visualization import generate_space, coloring

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from mlfwk.readWrite import load_base


def predicao(Y):
    y = np.zeros((Y.shape[0], 1))
    for j in range(Y.shape[0]):
        i = np.where(Y[j, :] == Y[j, :].max())[0][0]
        y[j] = i
    return y


# tratamento dos dados
# ------------------------------------------------------------

base = load_base(path='column_2C_weka.arff', type='arff')

# features
features = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius',
            'degree_spondylolisthesis']

print(base.info())

# ----------------------------- Clean the data ----------------------------------------------------------------

# -------------------------- Normalization ------------------------------------------------------------------

# normalizar a base
base[features] = normalization(base[features], type='min-max')

# ------------------------------------------------------------------------------------------------------------

y_out_of_c = pd.get_dummies(base['class'])

base = base.drop(['class'], axis=1)
base = concatenate([base[features], y_out_of_c], axis=1)

# -----------------------------------------------------------
# pesos = (0.4 + (0.6 - (0.4)) * np.random.rand(2, 6))
pesos = np.array(np.random.rand(2, 6), ndmin=2)
Taxas_acerto = []
lr = 0.01
k = 0  # contador
# /////////////////////////////////////////////////////////////////////////////////////

# np.random.seed(1)
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(base[:, :6], base[:, 6:], test_size=0.2)

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
print('Desvio padrao das taxas de acerto: %f', np.std(Taxas_acerto))
print('Matrix de confusão Teste: ')
print(Matrix_Confusao_teste)
plt.matshow(Matrix_Confusao_teste)
plt.colorbar()
# plt.savefig("Grafico3Art5")
plt.show()