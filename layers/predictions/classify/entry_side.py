import pickle
import os
import datetime
from collections import Counter

import pandas as pd
import lightgbm as lgb
import numpy as np

from common.modules.logger import logger
from itertools import product
from functools import reduce
from sklearn.model_selection import KFold
from common.modules.assets import Assets
from common.modules.exchange import Exchange
from common.utils.util_func import ex
from common.utils.window_aggregator import WindowAggregator
from connector.influxdb.influxdb_wrapper import influx
from common.paths import Paths
from layers.features.upsampler import Upsampler
from layers.predictions.event_extractor import EventExtractor
from common.utils.util_func import is_stationary

"""
Continue picking samples from parts where a series has values beyond expectation. say 2 sigma away.
Todo:
- Given any Influx raw data set with high granularity:
    - aggregate on a range of time frames: 
    - use multiple aggregation functions, either from Influx or custom
    - what's beyond expectation: gaussian and sigma? Moving averages with bands? smoothing + range: what's range: smoothing would be a function of the aggregation time window
        what if it's trending? run the stationarity test before. if fails, exclude that series!
"""


class SampleWeights:
    def __init__(self, ps_label: pd.Series, df: pd.DataFrame):
        self.ps_label = ps_label
        self.df = df
        self.weight_arrays = {}
        # weight array weight

    def uniqueness_sample_weights(self):
        n = len(self.ps_label)
        self.weight_arrays['uniqueness'] = None
        # assert round(self.weight_arrays['uniqueness'].sum(), 3) == 1000
        return self

    def return_attribution_sample_weights(self): pass
    def time_decay_sample_weights(self): pass

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
            # n_short * weight_short + n_long * weight_long = 0.5
            # n_short * weight_short = n_long * weight_long
            # weight_short = n_long * weight_long / n_short
            # n_short * ( n_long * weight_long / n_short ) + n_long * weight_long = 0.5
            # n_long * weight_long + n_long * weight_long = 0.5
            # weight_long = 0.5 / (2 * n_long)
            class_weights[2] = 0.5 / (2 * cnt_classes[2])
            class_weights[0] = cnt_classes[2] * class_weights[2] / cnt_classes[0]
            logger.info(f'Label Sample weights: {class_weights}')
        else:
            raise NotImplementedError

        self.weight_arrays['label_classes'] = self.ps_label.map(class_weights).values * 1000  # counter floating point issues
        assert round(self.weight_arrays['label_classes'].sum(), 3) == 1000
        return self

    def combine(self): pass

    def __call__(self) -> np.array:
        res = np.array(list(self.weight_arrays.values())).mean(axis=0)  # weight the classes
        assert len(res) == len(self.ps_label)
        return res


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
                                                                     'expiration_window': '180min', '_field': 'label'},
                                                         start=self.start,
                                                         end=self.end),
                                return_more_tables=False
                                )
        df_label.columns = ['label']
        df_m = df.merge(df_label, how='outer', left_index=True, right_index=True).sort_index()
        df_m = self.curtail_nnan_front_end(df_m).ffill()
        df_m = df_m.iloc[df_m.isna().sum().max():]
        assert df_m.isna().sum().sum() == 0
        return df_m[[c for c in df_m.columns if c != 'label']], df_m['label'] + 1

    @staticmethod
    def curtail_nnan_front_end(df):
        iloc_begin = np.argmax((~np.isnan(df.values)), axis=0).max()
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
        df = pd.concat((df_book, df_trade_imbalance, df_trade_sequence), sort=False, axis=1)
        df = self.exclude_too_little_data(df)
        # ffill after curtailing to avoid arriving at incorrect states for timeframes where information has simply not been loaded
        df = self.curtail_nnan_front_end(df).ffill()
        df = df.iloc[df.isna().sum().max():]
        assert df.isna().sum().sum() == 0
        return df

    @staticmethod
    def exclude_too_little_data(df) -> pd.DataFrame:
        iloc_begin = np.argmax((~np.isnan(df.values)), axis=0)
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
        if True:
            self.df = self.load_features()
            with open(os.path.join(Paths.lib_path, 'df.p'), 'wb') as f:
                pickle.dump(self.df, f)
            # with open(os.path.join(Paths.lib_path, 'label.p'), 'wb') as f:
            #     pickle.dump(self.ps_label, f)
        else:
            with open(os.path.join(Paths.lib_path, 'df.p'), 'rb') as f:
                self.df = pickle.load(f)
            # with open(os.path.join(Paths.lib_path, 'label.p'), 'rb') as f:
            #     self.ps_label = pickle.load(f)
        self.df, self.ps_label = self.load_label(self.df)
        # ts_sample = self.load_sample_from_signals(self.df)  # samples
        ts_sample = pd.Index(reduce(lambda res, item: res.union(set(item)), (EventExtractor(thresholds=[4, 4])(self.df[col], len(self.df) / 10) for col in self.df.columns), set()))
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

    def split_ho(self, ho_share=0.7):
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
            'verbosity': 0,
            'learning_rate': 0.1,
            'objective': 'multiclass',
            'num_class': 3,
            'early_stopping_round': 20,
        }
        preds_val = []
        preds_ho = []
        arr_weight = SampleWeights(ps_label=self.ps_label, df=self.df).label_sample_weight()()
        # arr_weight = np.ones(len(self.ps_label))
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
        logger.info(f'Scores: Mean: {np.mean([s["multi_logloss"] for s in scores])}  {scores}')
        self.preds_val = pd.concat(preds_val, axis=0).sort_index().groupby(level=0).mean()
        self.preds_val.columns = ['short', 'flat', 'long']
        self.preds_ho = pd.concat(preds_ho, axis=0).sort_index().groupby(level=0).mean()
        self.preds_ho.columns = ['short', 'flat', 'long']
        f1_val = self.preds_val.merge(self.ps_label, how='inner', right_index=True, left_index=True)
        f1_ho = self.preds_ho.merge(self.ps_label_ho, how='inner', right_index=True, left_index=True)

        from sklearn.metrics import f1_score
        print('VALIDATION')
        for i, side in enumerate(['short', 'flat', 'long']):
            print(f"{side}: {f1_score(np.where(f1_val['label'] == i, 1, 0), f1_val[side].round().values)}")
        print('HOLDOUT')
        for i, side in enumerate(['short', 'flat', 'long']):
            print(f"{side}: {f1_score(np.where(f1_ho['label'] == i, 1, 0), f1_ho[side].round().values)}")

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
        # self.to_influx()

    def to_influx(self):
        assert len(self.preds_val.index.unique()) == len(self.preds_val), 'Timestamp is not unique. Group By time first before uploading to influx.'
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
    inst.train()
    inst.save()
