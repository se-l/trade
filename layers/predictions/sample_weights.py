import numpy as np
import pandas as pd
import copy

from common.modules.logger import logger
from collections import Counter
from scipy.stats import gmean
from sklearn.cluster import MiniBatchKMeans


def check_weights(func):
    def inner(*args, **kwargs):
        self = args[0]
        arr = func(*args, **kwargs)
        assert max(arr) == 1
        self.weight_arrays[func.__name__] = arr
        return self
    return inner


class SampleWeights:
    def __init__(self, ps_label: pd.Series, df: pd.DataFrame):
        self.ps_label = ps_label
        self.df = df
        self.weight_arrays = {}

    @check_weights
    def uniqueness_sample_weights(self):
        n = len(self.ps_label)
        self.weight_arrays['uniqueness'] = None

    @check_weights
    def return_attribution_sample_weights(self, return_amplifier=10):
        """
        instance with high absolute return change should get a much higher weight. ignore the noise.
        return of 1 -> weight of 1.
        """
        arr = copy.deepcopy(self.ps_label).values
        arr = np.abs((arr - arr.mean()) / arr.std())
        arr = arr / arr.max()
        return arr

    @check_weights
    def cluster_sample_weight(self, n_clusters: int):
        """
        kmeans = MiniBatchKMeans(n_clusters=2,
        #                          random_state=0,
        #                          batch_size=6)
        # kmeans = kmeans.partial_fit(self.df.values)
        # # kmeans = kmeans.partial_fit(X[6:12, :])
        # kmeans.cluster_centers_
        #
        # kmeans.predict(self.df.values)
        :param n_clusters:
        :return:
        """
        arr = self.df.values
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, max_iter=1000).fit(arr)
        cnt_clusters = dict(Counter(kmeans.labels_))
        cluster_weights = {label: 1 / count for label, count in cnt_clusters.items()}
        weights = pd.Series(kmeans.labels_).map(cluster_weights)
        return (weights / max(weights)).values

    def time_decay_sample_weights(self): pass

    @check_weights
    def label_sample_weight(self):
        """
        Ensure short/long get significant representation long/short to avoid getting low error scores by just prediction flat.
        Presume int labels starting from 0.
        """
        n = len(self.ps_label)
        classes = sorted(self.ps_label.unique())
        n_classes = len(classes)
        if n_classes == 3:  # 1 means flat
            cnt_classes = dict(Counter(self.ps_label))
            class_weights = {1: 0.5 / cnt_classes[1]}  # n_flat * weight_flat = 0.5
            class_weights[2] = 0.5 / (2 * cnt_classes[2])
            class_weights[0] = cnt_classes[2] * class_weights[2] / cnt_classes[0]
            logger.info(f'Label Sample weights: {class_weights}')
        else:
            raise NotImplementedError
        return (self.ps_label.map(class_weights).values * 1000).values

    def geometric_mean(self) -> np.array:
        res = gmean(np.stack(self.weight_arrays.values()))
        return res / max(res)

    def arithmetic_mean(self) -> np.array:
        res = np.stack(self.weight_arrays.values()).mean(axis=0)
        return res / max(res)

