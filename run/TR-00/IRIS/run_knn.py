import sys
from pathlib import Path
print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


from numpy import where, array, mean, std
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from mlfwk.readWrite import load_base
from mlfwk.utils import split_random, normalization, get_project_root
from mlfwk.models import knn, dmc
from mlfwk.metrics import metric

if __name__ == '__main__':

    versus = ['S_vs_OT', 'Vc_vs_OT', 'Vg_vs_OT']
    final_result = {
        'versus': [],
        'K': [],
        'ACCURACY': [],
        'std ACCURACY': [],
        'AUC': [],
        'std AUC': [],
        'MCC': [],
        'std MCC': [],
        'f1_score': [],
        'std f1_score': [],
        'precision': [],
        'std precision': [],
        'recall': [],
        'std recall': []
    }


    for one_versus_others in versus:

        # --------------------- load base ------------------------------------------------------------ #

        # carregar a base
        iris_base = load_base(path='iris.data', type='csv')

        # normalizar a base
        iris_base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = normalization(
            iris_base[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], type='z-score')

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
            'AUC': [],
            'MCC': [],
            'f1_score': [],
            'precision': [],
            'recall': []
        }
        for realization in range(20):
            train, test = split_random(iris_base)

            x_train = train.drop(['Species'], axis=1)
            y_train = train['Species']

            x_test = test.drop(['Species'], axis=1)
            y_test = test['Species']

            classifier_knn = knn(x_train.to_numpy(), y_train.to_numpy(), k=3, class_column_name='Species')
            y_out_knn = classifier_knn.predict(x_test.to_numpy())

            metrics_calculator = metric(list(y_test), y_out_knn, types=['ACCURACY', 'AUC', 'precision',
                                                                  'recall', 'f1_score', 'MCC'])
            metric_results = metrics_calculator.calculate()

            results['realization'].append(realization)
            for type in ['ACCURACY', 'AUC', 'precision', 'recall', 'f1_score', 'MCC']:
                results[type].append(metric_results[type])


        final_result['versus'].append(one_versus_others)
        final_result['K'].append(3)

        for type in ['ACCURACY', 'AUC', 'precision', 'recall', 'f1_score', 'MCC']:
            final_result[type].append(mean(results[type]))
            final_result['std ' + type].append(std(results[type]))

    DataFrame(final_result).to_csv(get_project_root() + '/run/TR-00/IRIS/results/' + 'result_knn.csv')
