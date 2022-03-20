import pickle
import os
import datetime
import pandas as pd
import numpy as np

from common.interfaces.iload_xy import ILoadXY
from common.modules.logger import logger
from itertools import product
from functools import reduce
from common.modules.assets import Assets
from common.modules.exchange import Exchange
from common.utils.util_func import ex
from common.utils.window_aggregator import WindowAggregator
from connector.influxdb.influxdb_wrapper import influx
from common.paths import Paths
from layers.features.upsampler import Upsampler
from layers.predictions.event_extractor import EventExtractor
from common.utils.util_func import is_stationary

map_re_information2aggregator = {
    '^imbalance': ['sum'],
    '^volume': ['sum'],
    '^sequence': ['sum'],
    'bid_buy_size_imbalance_ratio': ['last', 'max', 'min'],
    'bid_buy_count_imbalance_ratio': ['last', 'max', 'min'],
    'bid_buy_size_imbalance_net': ['last', 'max', 'min'],
    'bid_buy_count_imbalance_net': ['last', 'max', 'min'],
}


class LoadXY(ILoadXY):
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
        self.window_aggregator_window = [int(2**i) for i in range(15)]
        self.dflt_window_aggregator_func = ['sum']
        self.window_aggregators = [WindowAggregator(window, func) for (window, func) in product(self.window_aggregator_window, self.dflt_window_aggregator_func)]
        self.book_window_aggregators = [WindowAggregator(window, func) for (window, func) in product(self.window_aggregator_window, ['last', 'max', 'min'])]
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

    def apply_embargo(self):
        pass

    def calc_weights(self):
        pass

    def load_features(self) -> pd.DataFrame:
        """order book imbalance, tick imbalance, sequence etc."""
        logger.info(f'Loading features. Book...')
        df_book = influx.query(query=influx.build_query(predicates={'_measurement': 'order book', 'exchange': self.exchange.name,
                                                                    # 'asset': self.sym.name
                                                                    },
                                                                   start=self.start,
                                                                   end=self.end),
                                          return_more_tables=True
                                          )
        logger.info('Influx Order book queried')
        df_book = self.exclude_non_stationary(df_book)
        df_book = pd.concat([Upsampler(df_book[c]).upsample(aggregate_window.window, aggregate_window.aggregator) for (c, aggregate_window) in product(df_book.columns, self.book_window_aggregators)],
                            sort=True, axis=1)
        logger.info('Order book done')
        logger.info('Fetching Trade volume')
        df_trade_volume = influx.query(query=influx.build_query(predicates={'_measurement': 'trade bars',
                                                                               'exchange': self.exchange.name,
                                                                               # 'asset': self.sym.name,
                                                                               'information': 'volume'
                                                                               },
                                                                   start=self.start,
                                                                   end=self.end),
                                          return_more_tables=True
                                          )
        df_trade_volume = self.exclude_non_stationary(df_trade_volume)
        df_trade_volume = pd.concat(
            [Upsampler(df_trade_volume[c]).upsample(aggregate_window.window, aggregate_window.aggregator) for (c, aggregate_window) in product(df_trade_volume.columns, self.window_aggregators)],
            sort=True, axis=1)

        df_trade_imbalance = influx.query(query=influx.build_query(predicates={'_measurement': 'trade bars',
                                                                               'exchange': self.exchange.name,
                                                                               # 'asset': self.sym.name,
                                                                               'information': 'imbalance'
                                                                               },
                                                                    start=self.start,
                                                                    end=self.end),
                                           return_more_tables=True
                                )
        df_trade_imbalance = self.exclude_non_stationary(df_trade_imbalance)
        df_trade_imbalance = pd.concat([Upsampler(df_trade_imbalance[c]).upsample(aggregate_window.window, aggregate_window.aggregator) for (c, aggregate_window) in product(df_trade_imbalance.columns, self.window_aggregators)],
                               sort=True, axis=1)
        df_trade_sequence = influx.query(query=influx.build_query(predicates={'_measurement': 'trade bars', 'exchange': self.exchange.name,
                                                                              # 'asset': self.sym.name,
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
        df = pd.concat((df_book, df_trade_imbalance, df_trade_sequence, df_trade_volume), sort=True, axis=1)
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

    def assemble_frames(self):
        if True:
            self.df = self.load_features()
            with open(os.path.join(Paths.data, 'df_loadxy.p'), 'wb') as f:
                pickle.dump(self.df, f)
        else:
            with open(os.path.join(Paths.data, 'df_loadxy.p'), 'rb') as f:
                self.df = pickle.load(f)
        self.df, self.ps_label = self.load_label(self.df)
        # ts_sample = self.load_sample_from_signals(self.df)  # samples
        ts_sample = pd.Index(reduce(lambda res, item: res.union(set(item)), (EventExtractor(thresholds=[1.5, 1.5])(self.df[col], len(self.df) / 10) for col in self.df.columns), set()))
        logger.info(f'len ts_sample: {len(ts_sample)} - {100 * len(ts_sample) / len(self.df)} of df')
        if not ts_sample.empty:
            ix_sample = ts_sample.intersection(self.ps_label.index)
            self.df, self.ps_label = self.df.loc[ix_sample].sort_index(), self.ps_label.loc[ix_sample].sort_index()
        logger.info(f'DF Shape: {self.df.shape}')


if __name__ == '__main__':
    exchange = Exchange.bitfinex
    sym = Assets.ethusd

    inst = LoadXY(
        exchange=exchange,
        sym=sym,
        # start=datetime.datetime(2022, 2, 7),
        # end=datetime.datetime(2022, 3, 13),
        start=datetime.datetime(2022, 2, 14),
        end=datetime.datetime(2022, 2, 16),
        # end=datetime.datetime(2022, 3, 1),
    )
    inst.assemble_frames()
