import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import seaborn as sn
import matplotlib.pyplot as plt
from numpy import where, mean, std, c_, array
from pandas.core.frame import DataFrame
from mlfwk.utils import split_random, get_project_root
from collections import Counter
from mlfwk.models import knn
from mlfwk.metrics import metric
from mlfwk.visualization import generate_space, coloring
from matplotlib.colors import ListedColormap
from mlfwk.readWrite import load_base
from mlfwk.utils import normalization

if __name__ == '__main__':
    print("run coluna 3 classes")

    # carregar a base
    base = load_base(path='column_3C_weka.arff', type='arff')

    # features
    features = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius',
                'degree_spondylolisthesis']

    print(base.info())

    # ----------------------------- Clean the data ----------------------------------------------------------------

    # -------------------------- Normalization ------------------------------------------------------------------

    # normalizar a base
    base[features] = normalization(base[features], type='min-max')

    # ------------------------------------------------------------------------------------------------------------
    base['class'][base['class'] == b'Abnormal'] = 1
    base['class'][base['class'] == b'Normal'] = 0

    for one_versus_others in ['Hernia_vs_OT', 'Spondylolisthesis_vs_OT', 'Normal_vs_OT']:
        final_result = {
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

        results = {
            'realization': [],
            'ACCURACY': [],
            'f1_score': [],
            'precision': [],
            'recall': [],
            'cf': []
        }

        if one_versus_others == 'Hernia_vs_OT':

            base['class'][base['class'] == b'Hernia'] = 1
            base['class'][base['class'] == b'Spondylolisthesis'] = 0
            base['class'][base['class'] == b'Normal'] = 0

        elif one_versus_others == 'Spondylolisthesis_vs_OT':

            base['class'][base['class'] == b'Hernia'] = 0
            base['class'][base['class'] == b'Spondylolisthesis'] = 1
            base['class'][base['class'] == b'Normal'] = 0

        elif one_versus_others == 'Normal_vs_OT':
            # virginica versus others

            base['class'][base['class'] == b'Hernia'] = 0
            base['class'][base['class'] == b'Spondylolisthesis'] = 0
            base['class'][base['class'] == b'Normal'] = 1


        N, M = base.shape
        C = len(base['class'].unique())

        for realization in range(20):
            train, test = split_random(base, train_percentage=.8)
            train = train.to_numpy()
            test = test.to_numpy()


            x_train = train[:, :len(features)]
            y_train = train[:, len(features):]

            x_test = test[:, :len(features)]
            y_test = test[:, len(features):]

            classifier_knn = knn(x_train, y_train)
            y_out_knn = classifier_knn.predict(x_test)

            metrics_calculator = metric(list(y_test.reshape(y_test.shape[0])), y_out_knn,
                                        types=['ACCURACY', 'AUC', 'precision',
                                        'recall', 'f1_score'])
            metric_results = metrics_calculator.calculate(average='macro')

            results['cf'].append((metric_results['ACCURACY'],
                                  metrics_calculator.confusion_matrix(list(y_test.reshape(y_test.shape[0])), y_out_knn,
                                                                      labels=range(C)),
                                  classifier_knn
                                  ))

            results['realization'].append(realization)
            for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
                results[type].append(metric_results[type])

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
        plt.title('Matriz de connfusão do KNN com acurácia de ' + str(round(best_acc, 2) * 100) + "%")
        plt.xlabel('Valor Esperado')
        plt.ylabel('Valor Encontrado')

        path = get_project_root() + '/run/ML-00/COLUNA_3C/results/'
        plt.savefig(path + one_versus_others + "-conf_result_knn.jpg")
        plt.show()

        print(DataFrame(final_result))
        DataFrame(final_result).to_csv(get_project_root() + '/run/ML-00/COLUNA_3C/results/' + one_versus_others + '_result_knn.csv')

