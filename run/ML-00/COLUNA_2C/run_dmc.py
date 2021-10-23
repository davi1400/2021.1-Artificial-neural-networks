import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
print(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


import matplotlib.pyplot as plt
from numpy import where, mean, std, c_, array
from pandas.core.frame import DataFrame
from mlfwk.utils import split_random, get_project_root
from collections import Counter
from mlfwk.models import dmc
from mlfwk.metrics import metric
from mlfwk.visualization import generate_space, coloring
from matplotlib.colors import ListedColormap
from mlfwk.readWrite import load_base
from mlfwk.utils import normalization

if __name__ == '__main__':
    print("run coluna 2 classes")
    final_result = {
        'ACCURACY': [],
        'std ACCURACY': [],
        # 'MCC': [],
        # 'std MCC': [],
        'f1_score': [],
        'std f1_score': [],
        'precision': [],
        'std precision': [],
        'recall': [],
        'std recall': []
    }

    results = {
        'realization': [],
        'ACCURACY': [],
        # 'MCC': [],
        'f1_score': [],
        'precision': [],
        'recall': [],
        'cf': []
    }

    # carregar a base
    base = load_base(path='column_2C_weka.arff', type='arff')

    # features
    features = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius',
                'degree_spondylolisthesis']

    print(base.info())

    # ----------------------------- Clean the data ----------------------------------------------------------------

    # -------------------------- Normalization ------------------------------------------------------------------

    # normalizar a base
    base[features] = normalization(base[features], type='min-max')
    base['class'][base['class'] == b'Abnormal'] = 1
    base['class'][base['class'] == b'Normal'] = 0


    # ------------------------------------------------------------------------------------------------------------

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

        classifier_dmc = dmc(x_train, y_train)
        y_out_dmc = classifier_dmc.predict(x_test, [0, 1])

        metrics_calculator = metric(list(y_test.reshape(y_test.shape[0])), y_out_dmc,
                                    types=['ACCURACY', 'AUC', 'precision',
                                    'recall', 'f1_score'])
        metric_results = metrics_calculator.calculate(average='micro')

        results['realization'].append(realization)
        for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
            results[type].append(metric_results[type])

    for type in ['ACCURACY', 'precision', 'recall', 'f1_score']:
        final_result[type].append(mean(results[type]))
        final_result['std ' + type].append(std(results[type]))

    print(DataFrame(final_result))
    DataFrame(final_result).to_csv(get_project_root() + '/run/ML-00/COLUNA_2C/results/' + 'result_dmc.csv')

