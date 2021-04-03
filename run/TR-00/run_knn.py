from numpy import where, array, mean
from pandas.core.frame import DataFrame
from mlfwk.readWrite import load_base
from mlfwk.utils import split_random, normalization, calculate_metrics
from mlfwk.models import knn


if __name__ == '__main__':

    versus = ['S_vs_OT', 'Vc_vs_OT', 'Vg_vs_OT']
    results = {
        'versus': [],
        'accuracys': []
    }

    for one_versus_others in versus:

        # --------------------- load base ------------------------------------------------------------ #

        # carregar a base
        iris_base = load_base(path='iris.data', column_names=['1', '2', '3', '4', 'class'], type='csv')

        # normalizar a base
        iris_base[['1', '2', '3', '4']] = normalization(iris_base[['1', '2', '3', '4']], type='z-score')

        setosa_ind = where(iris_base['class'] == 'Iris-setosa')[0]
        versicolor_ind = where(iris_base['class'] == 'Iris-versicolor')[0]
        virginica_ind = where(iris_base['class'] == 'Iris-virginica')[0]

        if one_versus_others == 'S_vs_OT':
            # setosa versus others

            iris_base['class'].iloc[setosa_ind] = 1
            iris_base['class'].iloc[versicolor_ind] = 0
            iris_base['class'].iloc[virginica_ind] = 0

        elif one_versus_others == 'Vc_vs_OT':
            # versicolor versus others

            iris_base['class'].iloc[setosa_ind] = 0
            iris_base['class'].iloc[versicolor_ind] = 1
            iris_base['class'].iloc[virginica_ind] = 0

        elif one_versus_others == 'Vg_vs_OT':
            # virginica versus others

            iris_base['class'].iloc[setosa_ind] = 0
            iris_base['class'].iloc[versicolor_ind] = 0
            iris_base['class'].iloc[virginica_ind] = 1


        # ----------------------------------------------------------------------------------------------- #

        accuracys = []
        for realization in range(20):
            train, test = split_random(iris_base)

            x_train = train.drop(['class'], axis=1)
            y_train = train['class']

            x_test = test.drop(['class'], axis=1)
            y_test = test['class']

            classifier_knn = knn(x_train.to_numpy(), y_train.to_numpy(), class_column_name='class')
            y_out = classifier_knn.predict(x_test.to_numpy())

            accuracys.append(calculate_metrics(y_out, list(y_test), metrics=['ACCURACY']))

        results['versus'].append(one_versus_others)
        results['accuracys'].append(mean(accuracys))
    print(DataFrame(results))