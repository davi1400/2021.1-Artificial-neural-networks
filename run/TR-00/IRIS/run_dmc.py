import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import seaborn as sn
import matplotlib.pyplot as plt
from numpy import where, array, mean, std
from pandas.core.frame import DataFrame
from mlfwk.readWrite import load_base
from mlfwk.utils import split_random, normalization, get_project_root
from mlfwk.models import dmc
from mlfwk.metrics import metric

if __name__ == '__main__':

    versus = ['S_vs_OT', 'Vc_vs_OT', 'Vg_vs_OT']
    final_result = {
        'versus': [],
        'ACCURACY': [],
        'std ACCURACY': [],
        'f1_score': [],
        'std f1_score': [],
        'precision': [],
        'std precision': [],
        'recall': [],
        'std recall': [],
        'best_cf': []
    }

    for one_versus_others in versus:

        # --------------------- load base ------------------------------------------------------------ #

        # carregar a base
        iris_base = load_base(path='iris.data', type='csv')

        # normalizar a base
        iris_base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = normalization(
            iris_base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], type='min-max')

        setosa_ind = where(iris_base['Species'] == 'Iris-setosa')[0]
        versicolor_ind = where(iris_base['Species'] == 'Iris-versicolor')[0]
        virginica_ind = where(iris_base['Species'] == 'Iris-virginica')[0]

        if one_versus_others == 'S_vs_OT':
            # setosa versus others

            iris_base['Species'].iloc[setosa_ind] = 1
            iris_base['Species'].iloc[versicolor_ind] = 0
            iris_base['Species'].iloc[virginica_ind] = 0

        elif one_versus_others == 'Vc_vs_OT':
            # versicolor versus others

            iris_base['Species'].iloc[setosa_ind] = 0
            iris_base['Species'].iloc[versicolor_ind] = 1
            iris_base['Species'].iloc[virginica_ind] = 0

        elif one_versus_others == 'Vg_vs_OT':
            # virginica versus others

            iris_base['Species'].iloc[setosa_ind] = 0
            iris_base['Species'].iloc[versicolor_ind] = 0
            iris_base['Species'].iloc[virginica_ind] = 1
        # ----------------------------------------------------------------------------------------------- #

        accuracys = []
        results = {
            'versus': [],
            'realization': [],
            'ACCURACY': [],
            'f1_score': [],
            'precision': [],
            'recall': [],
            'cf': []
        }
        C = len(iris_base['Species'].unique())

        for realization in range(20):
            train, test = split_random(iris_base)

            x_train = train.drop(['Species'], axis=1)
            y_train = train['Species']

            x_test = test.drop(['Species'], axis=1)
            y_test = test['Species']

            classifier_dmc = dmc(x_train.to_numpy(), y_train.to_numpy())
            y_out_dmc = classifier_dmc.predict(x_test.to_numpy(), [0, 1])

            metrics_calculator = metric(list(y_test), y_out_dmc, types=['ACCURACY', 'precision',
                                                                  'recall', 'f1_score'])
            metric_results = metrics_calculator.calculate(average='macro')

            results['cf'].append((metric_results['ACCURACY'],
                                  metrics_calculator.confusion_matrix(list(y_test), y_out_dmc, labels=range(C)),
                                  classifier_dmc
                                  ))

            results['realization'].append(realization)
            for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
                results[type].append(metric_results[type])


        final_result['versus'].append(one_versus_others)
        for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
            final_result[type].append(mean(results[type]))
            final_result['std ' + type].append(std(results[type]))

        results['cf'].sort(key=lambda x: x[0], reverse=True)
        final_result['best_cf'].append(results['cf'][0][1])
        best_acc_clf = results['cf'][0][2]
        best_acc = results['cf'][0][0]


        df_cm = DataFrame(results['cf'][0][1], index=[i for i in range(C)],
                              columns=[i for i in range(C)])
        sn.heatmap(df_cm, annot=True)
        plt.title('Matriz de connfusão do DMC com acurácia de ' + str(round(best_acc, 2) * 100) + "%")
        plt.xlabel('Valor Esperado')
        plt.ylabel('Valor Encontrado')

        path = get_project_root() + '/run/TR-00/IRIS/results/'
        plt.savefig(path + one_versus_others + "-conf_result_dmc.jpg")
        plt.show()

    DataFrame(final_result).to_csv(get_project_root() + '/run/TR-00/IRIS/results/' + 'result_dmc.csv')
    print(DataFrame(final_result))