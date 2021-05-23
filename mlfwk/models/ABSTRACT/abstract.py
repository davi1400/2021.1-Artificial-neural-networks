"""
    This will be an  abstract class to perform the duplicated functions in each class
"""


class abstract_network:
    def __init__(self):
        pass

    def add_bias(self, x):
        """

        add the bias to the input

        @param x: input
        @return:
        """
        return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)

    def fit(self, x, y, x_val=None, y_val=None, alphas=None, batch=False, validation=True, bias=True):
        pass

    # TODO CHANGE
    def k_fold_cross_validate(self, x, y, alphas):
        pass

    def predict(self, x, test=True):
        """

        @param test:
        @param x:
        @return:
        """
        pass

    def foward(self, x):
        indice = 0
        outputs = {}
        # outputs = zeros((x.shape[0], 1))
        for neuron_key in self.network.keys():
            outputs.update({
                neuron_key: self.network[neuron_key].foward(array(x, ndmin=2))
            })

        return outputs

    def backward(self, error, x):
        pass

    def calculate_error(self, y, y_output):
        pass

    def online_train(self, x, y):
       pass