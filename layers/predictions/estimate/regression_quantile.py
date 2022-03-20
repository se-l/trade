import pickle
import os
import datetime
import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import Counter, defaultdict

from common.interfaces.iestimate import IEstimate
from common.interfaces.iload_xy import ILoadXY
from common.modules.logger import logger
from itertools import product
from sklearn.model_selection import KFold
from sklearn.cluster import MiniBatchKMeans
from common.modules.assets import Assets
from common.modules.exchange import Exchange
from common.utils.estimations import compute_quantile_loss
from common.utils.util_func import ex
from common.utils.window_aggregator import WindowAggregator
from connector.influxdb.influxdb_wrapper import influx
from common.paths import Paths
from layers.predictions.load_xy import LoadXY
from layers.predictions.sample_weights import SampleWeights


class EstimateQuantiles(IEstimate):  # inherit some interface
    """Estimate side by:
    - Loading label ranges from inlux
    - Samples are events where series diverges from expectation: load from inlux
    - Weights: Less unique sample -> lower weight
    - CV. Embargo area
    - Store in Experiment
    - Generate feature importance plot
    - ToInflux: Signed estimates
    """

    def __init__(self, sym: Assets, load_xy=None, quantiles=None):
        self.load_xy: ILoadXY = load_xy
        self.quantiles = quantiles if quantiles else [0.1, 0.5, 0.9]
        self.window_aggregator_window = [int(2**i) for i in range(20)]
        self.window_aggregator_func = ['sum']
        self.window_aggregators = [WindowAggregator(window, func) for (window, func) in product(self.window_aggregator_window, self.window_aggregator_func)]
        self.tags = {}
        self.ex = ex(sym)
        logger.info(self.ex)
        self.df = None
        self.boosters = defaultdict(list)
        self.pred_label_val = {}
        self.pred_label_ho = {}

    def load_inputs(self):
        self.load_xy.assemble_frames()
        # apply sampling and purging separately perhaps
        self.df = self.load_xy.df
        self.ps_label = self.load_xy.ps_label

    @staticmethod
    def purge_overlap(train_index, test_index, i=250):
        """
        In each iteration remove i periods from around test.
        i should be derived somewhat intelligently ...
        """
        if test_index[0] == 0:  # test on left side
            return train_index[i:]
        elif test_index[0] > train_index[-1]:  # test on right side
            return train_index[:-i]
        else:
            i_begin = test_index[0] - 1
            i_end = test_index[-1] + 1
            return np.array(train_index[:(train_index.tolist().index(i_begin) - i)].tolist() + train_index[(train_index.tolist().index(i_end) + i):].tolist())

    def split_ho(self, ho_share=0.3):
        n_ho = int(len(self.df) * ho_share // 1)
        n_cv = len(self.df) - n_ho
        df = self.df.iloc[:n_cv]
        ps_label = self.ps_label.iloc[:n_cv]
        df_ho = self.df.iloc[-n_ho:]
        ps_label_ho = self.ps_label.iloc[-n_ho:]
        self.df = df
        self.ps_label = ps_label
        self.df_ho = df_ho
        self.ps_label_ho = ps_label_ho

    def train(self):
        self.split_ho()
        kf = KFold(n_splits=5, shuffle=False)

        arr_weight = SampleWeights(ps_label=self.ps_label, df=self.df).\
            return_attribution_sample_weights().\
            cluster_sample_weight(50).geometric_mean()

        quantile_booster_scores = defaultdict(list)
        pred_label_val = defaultdict(list)
        pred_label_ho = defaultdict(list)
        for quantile in self.quantiles:
            logger.info(f'Quantile {quantile}')
            estimator_params = {
                'objective': 'quantile',
                'alpha': quantile,
                'verbosity': 0,
                'learning_rate': 0.1,
                'early_stopping_round': 20,
                'boosting_type': 'gbdt',
            }
            preds_val = []
            preds_ho = []
            for train_index, test_index in kf.split(self.df.index):
                train_index = self.purge_overlap(train_index, test_index)
                x_train, x_test = self.df.iloc[train_index], self.df.iloc[test_index]
                y_train, y_test = self.ps_label.iloc[train_index], self.ps_label.iloc[test_index]
                dataset_train = lgb.Dataset(x_train, label=y_train, weight=arr_weight[train_index])
                lgb_booster = lgb.train(estimator_params,
                                        train_set=dataset_train,
                                        valid_sets=[lgb.Dataset(x_test, label=y_test, weight=arr_weight[test_index]), dataset_train],
                                        valid_names=['valid_0', 'valid_train'],
                                        )
                self.boosters[quantile].append(lgb_booster)
                quantile_booster_scores[quantile].append(lgb_booster.best_score['valid_0'])
                preds_val.append(pd.DataFrame(lgb_booster.predict(self.df.iloc[test_index]), index=self.df.iloc[test_index].index))
                preds_ho.append(pd.DataFrame(lgb_booster.predict(self.df_ho), index=self.df_ho.index))

            preds_val = pd.concat(preds_val, axis=0).sort_index().groupby(level=0).mean()
            preds_ho = pd.concat(preds_ho, axis=0).sort_index().groupby(level=0).mean()
            self.pred_label_val[quantile] = preds_val.merge(self.ps_label, how='inner', right_index=True, left_index=True)
            self.pred_label_ho[quantile] = preds_ho.merge(self.ps_label_ho, how='inner', right_index=True, left_index=True)

    def evaluate_quantiles(self, estimator_params, pred_label_val, pred_label_ho):
        ho_quantile_mae = {}
        ho_quantile_rmse = {}
        quantiles = list(range(1, 10))
        for quantile in quantiles:
            threshold = pred_label_ho['label'].quantile(quantile/10)
            if threshold >= 1:
                ix = np.where(pred_label_ho.iloc[:, 0] > threshold)[0]
            else:
                ix = np.where(pred_label_ho.iloc[:, 0] < threshold)[0]
            if len(ix) == 0:
                ho_quantile_mae[quantile] = ho_quantile_rmse[quantile] = None
            else:
                ho_quantile_mae[quantile] = mean_absolute_error(pred_label_ho.values[ix, 0], pred_label_ho.values[ix, 1])
                ho_quantile_rmse[quantile] = mean_squared_error(pred_label_ho.values[ix, 0], pred_label_ho.values[ix, 1])
        plt.plot(quantiles, ho_quantile_mae.values())
        plt.xlabel('Quantile MAE, RMSE of K')
        plt.ylabel('MAE')
        plt.title('Quantile')
        plt.show()

        ho_quantile_mae = {}
        ho_quantile_rmse = {}
        quantiles = list(range(1, 10))
        for quantile in quantiles:
            threshold = pred_label_ho['label'].quantile(quantile / 10)
            if threshold >= 1:
                ix = np.where(pred_label_ho.iloc[:, 0] > threshold)[0]
            else:
                ix = np.where(pred_label_ho.iloc[:, 0] < threshold)[0]
            if len(ix) == 0:
                ho_quantile_mae[quantile] = ho_quantile_rmse[quantile] = None
            else:
                ho_quantile_mae[quantile] = mean_absolute_error(np.ones(len(ix)), pred_label_ho.values[ix, 1])
                ho_quantile_rmse[quantile] = mean_squared_error(np.ones(len(ix)), pred_label_ho.values[ix, 1])
        plt.plot(quantiles, ho_quantile_mae.values())
        plt.xlabel('Quantile MAE guessing return == 1')
        plt.ylabel('MAE')
        plt.title('Quantile')
        plt.show()

    def save(self):
        """
        Models
        Feat importance !!!!!!!!!!!!!!!!!
        Influx
        """
        try:
            os.mkdir(os.path.join(Paths.trade_model, self.ex))
        except FileExistsError:
            pass
        with open(os.path.join(Paths.trade_model, self.ex, 'boosters.p'), 'wb') as f:
            pickle.dump(self.boosters, f)
        with open(os.path.join(Paths.trade_model, self.ex, 'pred_label_val.p'), 'wb') as f:
            pickle.dump(self.pred_label_val, f)
        with open(os.path.join(Paths.trade_model, self.ex, 'pred_label_ho.p'), 'wb') as f:
            pickle.dump(self.pred_label_ho, f)

    def to_influx(self):
        assert len(self.preds_val.index.unique()) == len(self.preds_val), 'Timestamp is not unique. Group By time first before uploading to influx.'
        self.preds_val = self.preds_val.rename(columns={0: 'predictions'})
        influx.write(
            record=self.preds_val,
            data_frame_measurement_name='predictions',
            data_frame_tag_columns={**{
                'exchange': self.exchange.name,
                'asset': self.sym.name,
                'information': 'CV',
                'ex': self.ex
            }, **self.tags},
        )
        self.preds_ho = self.preds_ho.rename(columns={0: 'predictions'})
        influx.write(
            record=self.preds_ho,
            data_frame_measurement_name='predictions',
            data_frame_tag_columns={**{
                'exchange': self.exchange.name,
                'asset': self.sym.name,
                'information': 'HO',
                'ex': self.ex
            }, **self.tags},
        )

    def best_k_elbow(self, k_max: int):
        logger.info('Find optimal # k cluster using Elbow method')
        sum_squared_distances = []
        k = list(range(2, k_max))
        for i, num_clusters in enumerate(k):
            kmeans = MiniBatchKMeans(n_clusters=num_clusters,
                                 # random_state=0,
                                 # batch_size=6,
                                 max_iter=1000).fit(self.df.values)
            sum_squared_distances.append(kmeans.inertia_)
        res = pd.Series(dict(zip(k, sum_squared_distances))).plot()
        plt.xlabel('Values of K')
        plt.ylabel('Sum of squared distances/Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    def best_k_silhouette(self, k_max: int, k_min: int = 2):
        logger.info('Find optimal # k cluster using Silhouette score')

        def silhouette_score(values, cluster_labels):
            scores = []
            map_label2vec_internal = {label: values[np.where(cluster_labels == label)[0], :] for label in np.unique(cluster_labels)}
            map_label2vec_external = {label: values[np.where(cluster_labels != label)[0], :] for label in np.unique(cluster_labels)}
            for i, label in enumerate(cluster_labels):
                internal_distance = np.linalg.norm(map_label2vec_internal[label] - values[i], axis=1).mean()
                external_distance = np.linalg.norm(map_label2vec_external[label] - values[i], axis=1).mean()
                scores.append((external_distance - internal_distance) / max(internal_distance, external_distance))
            return np.mean(scores)
        k = list(range(k_min, k_max))
        silhouette_avg = []
        for i, num_clusters in enumerate(k):
            if i % 10 == 0:
                print(i)
            # initialise kmeans
            kmeans = MiniBatchKMeans(n_clusters=num_clusters, max_iter=1000).fit(self.df.values)
            # silhouette score
            silhouette_avg.append(silhouette_score(self.df.values, kmeans.labels_))
        plt.plot(k, silhouette_avg)
        plt.xlabel('Values of K')
        plt.ylabel('Silhouette score')
        plt.title('Silhouette analysis For Optimal k')
        plt.show()
        # Plot Dispersity
        dct = dict(Counter(kmeans.labels_))
        tup_lst = [(k, v) for k, v in dct.items()]
        tup_lst = sorted(tup_lst, key=lambda tup: tup[1], reverse=True)
        plt.bar(list(range(len(tup_lst))), [tup[1] for tup in tup_lst])
        plt.xlabel('Cluster Label')
        plt.ylabel('Count states')
        plt.title('Points per cluster')
        plt.show()

        return k[silhouette_avg.index(pd.Series(silhouette_avg).bfill().ffill().min())]


if __name__ == '__main__':
    exchange = Exchange.bitfinex
    sym = Assets.ethusd
    start = datetime.datetime(2022, 2, 7)
    end = datetime.datetime(2022, 3, 2)

    inst = EstimateQuantiles(
        sym=sym,
        load_xy=LoadXY(exchange=exchange, sym=sym, start=start, end=end, labels=None, signals=None, features=None)
    )
    inst.load_inputs()
    # inst.best_k_elbow(100)
    # inst.best_k_silhouette(50, 50)
    # k ==50 is okay. rather have more than fewer. only useful if actually disperse. means some k have
    # much larger count than others... need dispersity measure for each k like median cnt?
    inst.train()
    inst.save()
    # inst.to_influx()
    logger.info(f'Done. Ex: {inst.ex}')
