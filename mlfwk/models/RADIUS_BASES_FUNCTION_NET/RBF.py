from mlfwk.metrics.metrics import accuracy_score
from mlfwk.models.KMEANS.kmeans import kmeans


class RadialBasisFunction:
    def __init__(self, epochs=1000, number_of_neurons=2, learning_rate=.01, activation_function='sigmoid logistic',
                 case='max'):
        """

            Performs a neural network with jsut one layer(one layer of neurons + input layer), using in default sigmoid
        logistic function.

        @param epochs: Number of train epochs
        @param number_of_neurons: number of neurons in output layer
        @param learning_rate: lambda
        @param activation_function: the type of activate funtion in each neuron
        @param max:
        """

        self.epochs = epochs
        self.number_of_neurons = number_of_neurons
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.case = case

    def add_bias(self, x):
        """
        Add bias function

        @param x:
        @return:
        """
        return concatenate([ones((x.shape[0], 1)), x], axis=1)

    def find_centroides(self, x, y):
        # using k - mean to find the centroids
        clf_kmeans = kmeans(k=self.number_of_neurons)
        clf_kmeans.fit(x, y, validation=False)

        return clf_kmeans.centroids

    def fit(self, x, y, x_train_val=None, y_train_val=None, alphas=None, hidden=None, batch=False, validation=True, bias=True):

        """
        Fit function, training the model.

        @param hidden:
        @param x: features train set
        @param y: objectives train set
        @param x_train_val: features validation set
        @param y_train_val: objectives valition set
        @param alphas: alphas used to find the optimal hyparameter
        @param batch: batch version, if false use the SGD
        @param validation: if true perform the k-fold cross valition with grid search
        @param bias: if true add the bias column

        @return:
        """
        if validation:
            self.k_fold_cross_validate(x_train_val, y_train_val, alphas=alphas, hidden=hidden, bias=True)

        self.cetroids = self.find_centroides(x_train_val, y_train_val)

        if batch:
            # TODO
            return self.batch_train()
        else:
            if bias:
                x = self.add_bias(x)
            return self.online_train(x, y)

    def k_fold_cross_validate(self, x, y, alphas, hidden, bias=True):
        """

        @param x:
        @param y:
        @param alphas:
        @param hidden:
        @param bias:
        @return:
        """
        if bias:
            x = self.add_bias(x)

        K = 10
        N, M = x.shape
        validation_metrics = np.zeros((len(hidden), len(alphas)))

        y = np.array(y, ndmin=2)
        if y.shape[0] == 1:
            y = y.T

        i = 0
        j = 0
        for hidden_layer in hidden:
            for alpha in alphas:
                k_validation_metrics = []
                for esimo in range(1, K + 1):
                    L = int(x.shape[0] / K)
                    x_train_val = (c_[x[:L * esimo - L, :].T, x[esimo * L:, :].T]).T
                    x_test_val = (x[L * esimo - L:esimo * L, :])
                    y_train_val = (c_[y[:L * esimo - L, :].T, y[esimo * L:, :].T]).T
                    y_test_val = (y[L * esimo - L:esimo * L, :])

                    N, M = x_train_val.shape

                    classifier = RadialBasisFunction(M, self.N_Classes,
                                                      hidden_layer_neurons=hidden_layer,
                                                      learning_rate=alpha,
                                                      epochs=500,
                                                      Regressao=self.key)

                    classifier.fit(x_train_val, y_train_val, validation=False, bias=False)
                    y_out_val = classifier.predict(x_test_val)

                    if self.case == 'classification':
                        y_test_val = classifier.predicao(y_test_val)

                    calculate_metrics = metric(y_test_val, y_out_val, types=[self.types[self.case]])
                    metric_results = calculate_metrics.calculate(average='micro')
                    k_validation_metrics.append(metric_results[self.types[self.case]])

                validation_metrics[i][j] = mean(k_validation_metrics)
                j += 1
            i += 1
            j = 0

        if self.case == 'classification':
            hidden_indice, alpha_indice = np.unravel_index(np.argmax(validation_metrics, axis=None),
                                                           validation_metrics.shape)
        elif self.case == 'regression':
            hidden_indice, alpha_indice = np.unravel_index(np.argmin(validation_metrics, axis=None),
                                                           validation_metrics.shape)

        self.learning_rate = alphas[alpha_indice]
        self.N_Neruronios = hidden[hidden_indice]

    def predict(self, x, test=True):
        """

        @param test: if true add bias
        @param x: features test set
        @return: predicted values
        """

        if test:
            x = self.add_bias(X)

        H = self.H(x)
        if self.N_Classes > 1:
            return self.greater_prob(self.Sigmoid(H.dot(self.Pesos)))
        else:
            return H.dot(self.Pesos)

    def greater_prob(self, Y):
        """
        This case is only when we have more than one neuron in output layer and just one hidden layer
        @param Y:
        @return:
        """
        y = np.zeros((Y.shape[0], 1))
        for j in range(Y.shape[0]):
            i = np.argmax(Y[j])
            y[j] = i
        return y

    def foward(self, x):
        """

        Propagate the information throgth the neuron...
        @param x:
        @return:
        """
        hidden = np.zeros((self.N_centros, x.shape[0]))
        for j in range(self.N_centros):
            dist = np.linalg.norm(x - self.Centroides[j], axis=1).T
            hidden[j] = np.exp(-(dist ** 2) / (2.0 * (self.alpha ** 2)))

        hidden = hidden.T
        hidden = self.add_bias(hidden)
        return hidden

    def backward(self, error, x, y):
        """
        Adjust the wheights

        @param error:
        @param x:
        @param y:
        @return:
        """


    def online_train(self, x, y):

        """
        Stochastic variation of train gradient descent

        @param x:
        @param y:
        @return:
        """
        k = 0
        N, M = x.shape
        r = permutation(N)

        self.errors_per_epoch = []
        for epoch in range(self.epochs):
            h_output = self.foward(x[r[k]])


            neurons_error = self.calculate_error(array(y[r[k]], ndmin=2).T, y_predict)
            # error = array(y[r[k]], ndmin=2).T - y_output

            self.backward(neurons_error, x[r[k]], y_output)
            k += 1

            if k >= r.shape[0]:
                """
                    Verificar se já passou por todos os exemplos se sim, 
                    fazer novamente randperm() e colocar o contador no 0
                """
                r = permutation(x.shape[0])
                k = 0

