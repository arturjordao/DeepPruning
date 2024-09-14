import copy

import numpy as np
from keras.layers import *
from keras.models import Model
from numpy.linalg import matrix_rank


class CKA:
    __name__ = 'CKA'

    def __init__(self):
        pass

    @staticmethod
    def _debiased_dot_product_similarity_helper(xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x,
                                                squared_norm_y, n):
        return (xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y) + squared_norm_x * squared_norm_y / (
                (n - 1) * (n - 2)))

    def feature_space_linear_cka(self, features_x, features_y, debiased=False):
        features_x = features_x - np.mean(features_x, 0, keepdims=True)
        features_y = features_y - np.mean(features_y, 0, keepdims=True)

        dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
        normalization_x = np.linalg.norm(features_x.T.dot(features_x))
        normalization_y = np.linalg.norm(features_y.T.dot(features_y))

        if debiased:
            n = features_x.shape[0]
            # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
            sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
            sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
            squared_norm_x = np.sum(sum_squared_rows_x)
            squared_norm_y = np.sum(sum_squared_rows_y)

            dot_product_similarity = self._debiased_dot_product_similarity_helper(
                dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
                squared_norm_x, squared_norm_y, n)
            normalization_x = np.sqrt(self._debiased_dot_product_similarity_helper(
                normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
                squared_norm_x, squared_norm_x, n))
            normalization_y = np.sqrt(self._debiased_dot_product_similarity_helper(
                normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
                squared_norm_y, squared_norm_y, n))

        return dot_product_similarity / (normalization_x * normalization_y)

    def scores(self, model, x_train=None, y_train=None, allowed_layers=[], n_samples=None):
        scores = []

        if n_samples is not None:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                            np.unique(y_)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:
            sub_sampling = np.arange(x_train.shape[0])

        F = Model(model.input, model.get_layer(index=-2).output)  # Flatten
        features_F = F.predict(x_train[sub_sampling], verbose=0)

        for layer_idx in allowed_layers:

            idx_dense_and_multihead = []

            # Given a layer_normalization (allowed_layers), we need to find
            # the first and second add layers and zeroed-out the dense and multi head attention
            i = layer_idx
            while len(idx_dense_and_multihead) < 2:
                if isinstance(model.get_layer(index=i), Add):
                    idx_dense_and_multihead.append(i - 1)
                i = i + 1

            multi_head = model.get_layer(index=idx_dense_and_multihead[0])
            w1 = multi_head.get_weights()
            original_w1 = copy.deepcopy(w1)

            dense = model.get_layer(index=idx_dense_and_multihead[1])
            w2 = dense.get_weights()
            original_w2 = copy.deepcopy(w2)

            for i in range(0, len(w1)):
                w1[i] = np.zeros(w1[i].shape)

            for i in range(0, len(w2)):
                w2[i] = np.zeros(w2[i].shape)

            F_line = Model(model.input, model.get_layer(index=-2).output)
            features_line = F_line.predict(x_train, verbose=0)

            model.get_layer(index=idx_dense_and_multihead[0]).set_weights(original_w1)
            model.get_layer(index=idx_dense_and_multihead[1]).set_weights(original_w2)

            score = self.feature_space_linear_cka(features_F, features_line)
            scores.append((layer_idx, 1 - score))

        return scores


class Random:
    def __init__(self, ):
        pass

    @staticmethod
    def scores(model, X_train=None, y_train=None, allowed_layers=[]):
        output = [(x, np.random.rand()) for x in allowed_layers]
        return output


def criteria(method='random'):
    n_components = 2
    # if method == 'PLS+VIP':
    #     return PLSVIP(n_components, preprocess_input)
    #
    # if method == 'infFS' or method == 'infFSU' or method == 'ilFS':
    #     return infFSFramework(n_components, method, preprocess_input)
    #
    # if method == 'rank':
    #     return rank(n_components, preprocess_input)
    #
    # if method == 'expectedABS':
    #     return expectedABS(n_components, preprocess_input)
    #
    # if method == 'klDivergence':
    #     return klDivergence()

    if method == 'random':
        return Random()

    if method == 'CKA':
        return CKA()
