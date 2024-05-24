import numpy as np
import copy
import time
from keras.layers import *
from keras.models import Model
import os.path
import sys

from numpy.linalg import matrix_rank
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.extmath import softmax
from sklearn.utils import gen_batches


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
        output = []

        if n_samples is not None:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in
                            np.unique(y_)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:
            sub_sampling = np.arange(x_train.shape[0])

        F = Model(model.input, model.get_layer(index=-2).output)
        F_features = F.predict(x_train[sub_sampling], verbose=0)

        for i in range(len(allowed_layers)):
            scores = []

            layer = model.get_layer(index=allowed_layers[i])

            n_heads = layer._num_heads
            for h in range(n_heads):
                weights = layer.get_weights()
                original_weights = copy.deepcopy(weights)

                # Zeroed-out process for transformer-like architecture
                for k in range(0, 6):
                    if k % 2 == 0:
                        weights[k][:, h, :] = 0
                    else:
                        weights[k][h, :] = 0

                weights[-2][h] = 0

                layer.set_weights(weights)

                F_line = Model(model.input, model.get_layer(index=-2).output)
                F_line_features = F_line.predict(x_train[sub_sampling], verbose=0)

                layer.set_weights(original_weights)

                score = self.feature_space_linear_cka(F_features, F_line_features)
                scores.append(1 - score)

            output.append((allowed_layers[i], scores))

        return output


class Random:
    __name__ = 'Random Pruning'

    def __init__(self):
        pass

    @staticmethod
    def scores(model, x_train=None, y_train=None, allowed_layers=[]):
        output = []

        for idx in allowed_layers:
            num_heads = model.get_layer(index=idx)._num_heads
            scores = np.random.rand(num_heads)
            output.append((idx, scores))

        return output


def criteria(method):
    # if method == 'L1':
    #     return L1(model, percentage_discard=p)
    #
    # if method == 'ilFS':
    #     return ilFSPruning(model=model, representation='max', percentage_discard=p)
    #
    # if method == 'infFS':
    #     return infFSPruning(model=model, representation='max', percentage_discard=p)
    #
    # if method == 'infFSU':
    #     return infFSUPruning(model=model, representation='max', percentage_discard=p)
    #
    # if method == 'PLS+VIP':
    #     return PLSVIP(model=model, representation='max', percentage_discard=p)
    #
    # if method == 'rank':
    #     return rank(model=model, percentage_discard=p)
    #
    # if method == 'klDivergence':
    #     return klDivergence(model=model, percentage_discard=p)
    #
    # if method == 'expectedABS':
    #     return expectedABS(model=model, percentage_discard=p)
    #
    if method == 'random':
        return Random()

    if method == 'CKA':
        return CKA()
