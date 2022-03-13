import pickle
import os
import datetime
from collections import Counter

import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import gmean
from common.modules.logger import logger
from itertools import product
from functools import reduce
from sklearn.model_selection import KFold
from sklearn.cluster import MiniBatchKMeans
from common.modules.assets import Assets
from common.modules.exchange import Exchange
from common.utils.util_func import ex
from common.utils.window_aggregator import WindowAggregator
from connector.influxdb.influxdb_wrapper import influx
from common.paths import Paths
from layers.features.upsampler import Upsampler
from layers.predictions.event_extractor import EventExtractor
from common.utils.util_func import is_stationary
from layers.predictions.sample_weights import SampleWeights

"""
Continue picking samples from parts where a series has values beyond expectation. say 2 sigma away.
Todo:
- Given any Influx raw data set with high granularity:
    - aggregate on a range of time frames: 
    - use multiple aggregation functions, either from Influx or custom
    - what's beyond expectation: gaussian and sigma? Moving averages with bands? smoothing + range: what's range: smoothing would be a function of the aggregation time window
        what if it's trending? run the stationarity test before. if fails, exclude that series!
"""


class EstimateSide:
    """Estimate side by:
    - Loading label ranges from inlux
    - Samples are events where series diverges from expectation: load from inlux
    - Weights: Less unique sample -> lower weight
    - CV. Embargo area
    - Store in Experiment
    - Generate feature importance plot
    - ToInflux: Signed estimates
    """

    def __init__(self, exchange: Exchange, sym, start: datetime, end: datetime, labels=None, signals=None, features=None):
        self.exchange = exchange
        self.sym = sym
        self.start = start
        self.end = end
        self.labels = labels
        self.signals = signals
        self.features = features
        self.window_aggregator_window = [int(2**i) for i in range(20)]
        self.window_aggregator_func = ['sum']
        self.window_aggregators = [WindowAggregator(window, func) for (window, func) in product(self.window_aggregator_window, self.window_aggregator_func)]
        self.boosters = []
        self.tags = {}
        self.ex = ex(sym)
        logger.info(self.ex)
        self.df = None

    def load_label(self, df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        df_label = influx.query(query=influx.build_query(predicates={'_measurement': 'label', 'exchange': self.exchange.name, 'asset': self.sym.name,
                                                                     # 'expiration_window': '180min', '_field': 'label'},
                                                                     'ewm_span': '60min', '_field': 'forward_return_ewm'},
                                                         start=self.start,
                                                         end=self.end),
                                return_more_tables=False
                                )
        df_label.columns = ['label']
        df_m = df.merge(df_label, how='outer', left_index=True, right_index=True).sort_index()
        df_m = self.curtail_nnan_front_end(df_m).ffill()
        df_m = df_m.iloc[df_m.isna().sum().max():]
        assert df_m.isna().sum().sum() == 0
        # return df_m[[c for c in df_m.columns if c != 'label']], df_m['label'] + 1  # multi class label
        return df_m[[c for c in df_m.columns if c != 'label']], df_m['label']  # return label

    @staticmethod
    def curtail_nnan_front_end(df):
        iloc_begin = np.argmax((~np.isnan(df.values)), axis=0)
        for i, iloc in enumerate(iloc_begin):
            if iloc == 0 and np.isnan(df.values[0, i]):
                iloc_begin[i] = np.isnan(df.values[:, i]).sum()
        iloc_begin = iloc_begin.max()
        iloc_end = len(df) - np.argmax((~np.isnan(df.values[::-1])), axis=0).max()
        return df.iloc[iloc_begin:iloc_end]

    def load_sample_from_signals(self, df) -> pd.Index:
        """
        Order book imbalance > thresh.
        Sample space limiting vs non-sample sapce limiting
        """
        # for ... in self.window_aggregators
        return pd.Index(reduce(lambda res, item: res.union(set(item)), (EventExtractor(thresholds=[2, 2])(df[col], 500) for col in df.columns), set()))

    def apply_embargo(self):
        pass

    def calc_weights(self):
        pass

    def load_features(self) -> pd.DataFrame:
        """order book imbalance, tick imbalance, sequence etc."""
        logger.info(f'Loading features. Book...')
        df_book = influx.query(query=influx.build_query(predicates={'_measurement': 'order book', 'exchange': self.exchange.name, 'asset': self.sym.name},
                                                                   start=self.start,
                                                                   end=self.end),
                                          return_more_tables=True
                                          )
        logger.info('Influx Order book queried')
        df_book = self.exclude_non_stationary(df_book)
        df_book = pd.concat([Upsampler(df_book[c]).upsample(aggregate_window.window, aggregate_window.aggregator) for (c, aggregate_window) in product(df_book.columns, self.window_aggregators)],
                            sort=True, axis=1)
        logger.info('Order book done')
        df_trade_imbalance = influx.query(query=influx.build_query(predicates={'_measurement': 'trade bars',
                                                                               'exchange': self.exchange.name,
                                                                               'asset': self.sym.name,
                                                                               'information': 'imbalance'
                                                                               },
                                                                    start=self.start,
                                                                    end=self.end),
                                           return_more_tables=True
                                )
        df_trade_imbalance = self.exclude_non_stationary(df_trade_imbalance)
        df_trade_imbalance = pd.concat([Upsampler(df_trade_imbalance[c]).upsample(aggregate_window.window, aggregate_window.aggregator) for (c, aggregate_window) in product(df_trade_imbalance.columns, self.window_aggregators)],
                               sort=True, axis=1)
        df_trade_sequence = influx.query(query=influx.build_query(predicates={'_measurement': 'trade bars', 'exchange': self.exchange.name, 'asset': self.sym.name,
                                                                              'information': 'sequence'},
                                                                   start=self.start,
                                                                   end=self.end),
                                          return_more_tables=True
                                          )
        df_trade_sequence = self.exclude_non_stationary(df_trade_sequence)
        logger.info('Influx Trade book queried')
        df_trade_sequence = pd.concat(
            [Upsampler(df_trade_sequence[c]).upsample(aggregate_window.window, aggregate_window.aggregator) for (c, aggregate_window) in product(df_trade_sequence.columns, self.window_aggregators)],
            sort=True, axis=1)
        logger.info('Trade book done')
        df = pd.concat((df_book, df_trade_imbalance, df_trade_sequence), sort=True, axis=1)
        df = self.exclude_too_little_data(df)
        # ffill after curtailing to avoid arriving at incorrect states for timeframes where information has simply not been loaded
        df = self.curtail_nnan_front_end(df).ffill()
        df = self.exclude_too_little_data(df)
        df = df.iloc[df.isna().sum().max():]
        assert df.isna().sum().sum() == 0
        return df

    @staticmethod
    def exclude_too_little_data(df) -> pd.DataFrame:
        iloc_begin = np.argmax((~np.isnan(df.values)), axis=0)
        for i, iloc in enumerate(iloc_begin):
            if iloc == 0:
                iloc_begin[i] = np.isnan(df.values[:, i]).sum()
        iloc_end = len(df) - np.argmax((~np.isnan(df.values[::-1])), axis=0)
        # exclude whose range is too small. series with high resolution will have small data cnt, still large range
        mean_range = (iloc_end-iloc_begin).mean()
        col_range = dict(zip(list(df.columns), (iloc_end-iloc_begin) > mean_range))

        # ps_cnt_nonna = df.isna().sum().to_dict()
        # mean_cnt_non_na = np.mean(list(ps_cnt_nonna.values()))
        # ex_cols = r"\n".join([c for c, cnt in ps_cnt_nonna.items() if cnt > mean_cnt_non_na])
        logger.info(f'Removing columns having range <{mean_range} NANs`: {[c for c in col_range.keys() if not col_range[c]]}')
        return df[[c for c in col_range.keys() if col_range[c]]]

    @staticmethod
    def exclude_non_stationary(df) -> pd.DataFrame:
        def f(ps: pd.Series):
            res = is_stationary(ps[ps.notna()].values)
            if not res:
                logger.warning(f'{ps.name} is not stationary. Excluding!')
            return res
        return df[[c for c in df.columns if f(df[c])]]

    def assemble_frame(self):
        if False:
            self.df = self.load_features()
            with open(os.path.join(Paths.data, 'df.p'), 'wb') as f:
                pickle.dump(self.df, f)
            # with open(os.path.join(Paths.data, 'label.p'), 'wb') as f:
            #     pickle.dump(self.ps_label, f)
        else:
            with open(os.path.join(Paths.data, 'df.p'), 'rb') as f:
                self.df = pickle.load(f)
            # with open(os.path.join(Paths.data, 'label.p'), 'rb') as f:
            #     self.ps_label = pickle.load(f)
        self.df, self.ps_label = self.load_label(self.df)
        # ts_sample = self.load_sample_from_signals(self.df)  # samples
        ts_sample = pd.Index(reduce(lambda res, item: res.union(set(item)), (EventExtractor(thresholds=[1, 1])(self.df[col], len(self.df) / 10) for col in self.df.columns), set()))
        logger.info(f'len ts_sample: {len(ts_sample)} - {100 * len(ts_sample) / len(self.df)} of df')
        if not ts_sample.empty:
            ix_sample = ts_sample.intersection(self.ps_label.index)
            self.df, self.ps_label = self.df.loc[ix_sample].sort_index(), self.ps_label.loc[ix_sample].sort_index()
        logger.info(f'DF Shape: {self.df.shape}')

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
        estimator_params = {
            'objective': 'huber',
            # 'objective': 'multiclass',
            'verbosity': 0,
            'learning_rate': 0.1,
            # 'num_class': 3,
            'early_stopping_round': 20,
        }
        preds_val = []
        preds_ho = []
        arr_weight = SampleWeights(ps_label=self.ps_label, df=self.df).\
            return_attribution_sample_weights().\
            cluster_sample_weight(50).geometric_mean()

        scores = []
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
            self.boosters.append(lgb_booster)
            scores.append(lgb_booster.best_score['valid_0'])
            preds_val.append(pd.DataFrame(lgb_booster.predict(self.df.iloc[test_index]), index=self.df.iloc[test_index].index))
            preds_ho.append(pd.DataFrame(lgb_booster.predict(self.df_ho), index=self.df_ho.index))
        if estimator_params.get('num_class'):
            logger.info(f'Scores: Mean: {np.mean([s["multi_logloss"] for s in scores])}  {scores}')
            self.preds_val = pd.concat(preds_val, axis=0).sort_index().groupby(level=0).mean()
            self.preds_val.columns = ['short', 'flat', 'long']
            self.preds_ho = pd.concat(preds_ho, axis=0).sort_index().groupby(level=0).mean()
            self.preds_ho.columns = ['short', 'flat', 'long']
        else:
            self.preds_val = pd.concat(preds_val, axis=0).sort_index().groupby(level=0).mean()
            self.preds_ho = pd.concat(preds_ho, axis=0).sort_index().groupby(level=0).mean()
        pred_label_val = self.preds_val.merge(self.ps_label, how='inner', right_index=True, left_index=True)
        pred_label_ho = self.preds_ho.merge(self.ps_label_ho, how='inner', right_index=True, left_index=True)

        if estimator_params.get('num_class'):
            from sklearn.metrics import f1_score
            print('VALIDATION')
            for i, side in enumerate(['short', 'flat', 'long']):
                print(f"{side}: {f1_score(np.where(pred_label_val['label'] == i, 1, 0), pred_label_val[side].round().values)}")
            print('HOLDOUT')
            for i, side in enumerate(['short', 'flat', 'long']):
                print(f"{side}: {f1_score(np.where(pred_label_ho['label'] == i, 1, 0), pred_label_ho[side].round().values)}")
        else:
            logger.info(f"VALIDATION: MAE: {mean_absolute_error(pred_label_val.iloc[:, 0], pred_label_val['label'])} MSE: {mean_squared_error(pred_label_val.iloc[:, 0], pred_label_val['label'])}")
            logger.info(f"HOLDOUT: MAE: {mean_absolute_error(pred_label_ho.iloc[:, 0], pred_label_ho['label'])} MSE: {mean_squared_error(pred_label_ho.iloc[:, 0], pred_label_ho['label'])}")
            logger.info(f"RETURN == 1: MAE: {mean_absolute_error(np.ones(len(pred_label_ho)), pred_label_ho['label'])} MSE: {mean_squared_error(np.ones(len(pred_label_ho)), pred_label_ho['label'])}")

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


        from common.utils.util_func import get_model_fscore
        importances = [get_model_fscore(booster) for booster in self.boosters]
        res = pd.DataFrame(importances).mean(axis=0).sort_values(ascending=False)
        logger.info(res)

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
        with open(os.path.join(Paths.trade_model, self.ex, 'preds.p'), 'wb') as f:
            pickle.dump(self.preds_val, f)
        with open(os.path.join(Paths.trade_model, self.ex, 'label.p'), 'wb') as f:
            pickle.dump(self.ps_label, f)
        self.to_influx()

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

    inst = EstimateSide(
        exchange=exchange,
        sym=sym,
        start=datetime.datetime(2022, 2, 7),
        end=datetime.datetime(2022, 3, 2),
    )
    inst.assemble_frame()
    # inst.best_k_elbow(100)
    # inst.best_k_silhouette(50, 50)
    # k ==50 is okay. rather have more than fewer. only useful if actually disperse. means some k have
    # much larger count than others... need dispersity measure for each k like median cnt?
    inst.train()
    inst.save()
    logger.info(f'Done. Ex: {inst.ex}')
