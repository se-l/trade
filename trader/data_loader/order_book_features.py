import itertools
import importlib
import os
import click
import numpy as np
import pandas as pd

from functools import partial
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from common.modules import ctx
from common.modules import dotdict
from common.paths import Paths
from common.refdata import tick_size
from common.utils.util_func import rolling_window, reduce_to_intersect_ts
from common.utils.normalize import Normalize
from connector.influxdb.influxdb_wrapper import InfluxClientWrapper as Influx
from trader.data_loader.raw_data_fetcher.crypto.bitmex.convert_bitmex_raw_to_qc import ConvertBitmexToQC
from common.modules.logger import logger


class OrderBookFeatures:
    def __init__(s, params, normalize=None):
        s.params = params
        s.ts_start = params.data_start
        s.ts_end = params.data_end
        s.req_feats = None
        s.influx = Influx()
        s.normalize = normalize or Normalize(False, False)
        s.schema = ['asset', 'ts_start', 'ts_end', 'price_level', 'prev_price', 'next_price', 'tick', 'period_length',
                    'mean_ask_size', 'mean_bid_size', 'count',
                    'high_ask_size', 'high_bid_size',
                    'low_ask_size', 'low_bid_size',
                    'open_ask_size', 'close_ask_size',
                    'open_bid_size', 'close_bid_size',
                    'max_ask_div_bid_ratio', 'min_ask_div_bid_ratio']
        s.dir_quotes = os.path.join(Paths.bitmex_raw, 'quote')
        s.dir_trades = os.path.join(Paths.bitmex_raw, 'trade')

    @staticmethod
    def get_order_book_feat_names(params) -> list:
        features = []
        time_increases = params.order_book_feats_n
        time_increase_power = params.order_book_tick_increase_power
        max_delta_time = [int(1 + i ** time_increase_power) for i in range(time_increases)][-1]
        for ab in ['ask', 'bid']:
            features += list(itertools.chain(*[[
                f'{ab}_size_update_add_sum_{delta_time}',
                f'{ab}_size_update_remove_sum_{delta_time}',
                f'{ab}_size_update_add_cnt_sum_{delta_time}',
                f'{ab}_size_update_remove_cnt_sum_{delta_time}',
            ] for delta_time in [int(1 + i ** time_increase_power) for i in range(time_increases)]]))
        logger.info(f'Order Book Features: {len(features)}')
        return features

    @staticmethod
    def get_trade_metric_feat_names(params) -> list:
        features = []
        time_increases = params.trade_metric_feats_n
        time_increase_power = params.trade_metric_tick_increase_power
        max_delta_time = [int(1 + i ** time_increase_power) for i in range(time_increases)][-1]
        features = list(itertools.chain(*[[
            f'trade_volume_buy_sum_{delta_time}',
            f'trade_volume_sell_sum_{delta_time}',
            f'trade_count_sell_sum_{delta_time}',
            f'trade_count_buy_sum_{delta_time}',
        ] for delta_time in [int(1 + i ** time_increase_power) for i in range(time_increases)]]))
        logger.info(f'Trade Activity Features: {len(features)}')
        return features

    def get_order_book_features(s):
        s.quote = s.compute_quotes()
        s.trade = s.compute_trades()
        x_tv = s.merge_quote_trade(s.quote, s.trade) if True else s.quote
        x_tv = s.normalize.normalize_scale01_ndarr(x_tv)
        cnt_nan = x_tv.isna().sum().sum()
        if cnt_nan > 0:
            print(f'Found {cnt_nan} nan')
            x_tv.fillna(0, inplace=True)
        x_tv = s.normalize.float_to_int_approx(x_tv, digits=5)

    def compute_quotes(s):
        quote = s.load_from_qt_file(s.dir_quotes)

        quote = s.add_quote_feature_b4_resampling(quote)
        # first passing then resampling. as a consequence preventing OHLC from 1 bar into next memberwise.
        # so first resample or later resample and fill with close?
        # rather having close completely
        # quote_bid = s.resample_quote_ticks(quote, 'bid')
        # quote_ask = s.resample_quote_ticks(quote, 'ask')
        quote_spread = s.resample_quote_spead(quote)
        quote = pd.concat([quote_spread, quote_ask, quote_bid], axis=1)
        quote = s.add_quote_features(quote)
        return quote

    def compute_trades(s):
        raw_trades = s.load_from_qt_file(s.dir_trades)

        trade = s.add_trade_feature_b4_resampling(trade)
        # trade = s.resample_trade_ticks(trade)
        trade = s.add_trade_features(trade)
        return trade

    @staticmethod
    def add_quote_features(df):
        # ['spread_ticks_close', 'spread_ticks_max', 'ask_price_open',
        #  'ask_price_high', 'ask_price_low', 'ask_price_close', 'ask_size_open',
        #  'ask_size_high', 'ask_size_low', 'ask_size_close', 'ask_size_mean',
        #  'bool_ask_size_update', 'bid_price_open', 'bid_price_high',
        #  'bid_price_low', 'bid_price_close', 'bid_size_open', 'bid_size_high',
        #  'bid_size_low', 'bid_size_close', 'bid_size_mean',
        #  'bool_bid_size_update']
        # cum sum of size updates, cum count of size update bool(is a count of updates)
        # missing sum(size reduction), sum(size in increase) per side. maybe max-min. so got it per sec only. can do better
        # fast way: just add MOM _price(in ticks!!)
        mid_price = np.subtract(
            df['ask_price_close'],
            np.divide(np.subtract(df['ask_price_close'], df['bid_price_close']), 2)
        )
        for delta_time in [int(1 + i**2.5) for i in range(8)]:
            if delta_time == 1:
                continue
            df[f'mid_price_delta_{delta_time}'] = 0
            df[f'mid_price_delta_{delta_time}'].iloc[delta_time:] = np.subtract(mid_price[delta_time:], mid_price[:-delta_time])
            for ab in ['ask', 'bid']:
                df[f'size_{ab}_add_cumsum_{delta_time}'] = 0
                df[f'size_{ab}_remove_cumsum_{delta_time}'] = 0
                df[f'bool_{ab}_size_update_cumsum_{delta_time}'] = 0
                df[f'size_{ab}_add_cumsum_{delta_time}'].iloc[delta_time-1:] = \
                    np.sum(rolling_window(df[f'size_{ab}_add_sum'], delta_time), axis=1)
                df[f'size_{ab}_remove_cumsum_{delta_time}'].iloc[delta_time-1:] = \
                    np.sum(rolling_window(df[f'size_{ab}_remove_sum'], delta_time), axis=1)
                df[f'bool_{ab}_size_update_cumsum_{delta_time}'].iloc[delta_time-1:] = \
                    np.sum(rolling_window(df[f'bool_{ab}_size_update'], delta_time), axis=1)
        return df

    @staticmethod
    def add_trade_features(df):
        # ['side_buy', 'side_sell', 'trade_size_total', 'trade_size_mean', 'trade_size_buy', 'trade_size_sell']
        # - cnt(buys), cnt(sells), sum(buys), sum(sells), sum(volume), cnt(trades), net(cnt_trades), net(vol_trades)
        # 	- cumulative over various timeframes
        for delta_time in [int(1 + i ** 2.5) for i in range(8)]:
            if delta_time == 1:
                continue
            df['trade_cnt_total'] = 0
            df['trade_cnt_total'].iloc[delta_time - 1:] = \
                np.add(
                    np.sum(rolling_window(df[f'side_sell'], delta_time), axis=1),
                    np.sum(rolling_window(df[f'side_buy'], delta_time), axis=1)
                )
            df['trade_cnt_net'] = 0
            df['trade_cnt_net'].iloc[delta_time - 1:] = \
                np.subtract(
                    np.sum(rolling_window(df[f'side_buy'], delta_time), axis=1),
                    np.sum(rolling_window(df[f'side_sell'], delta_time), axis=1)
                )
            df['trade_size_net'] = 0
            df['trade_size_net'].iloc[delta_time - 1:] = \
                np.subtract(
                    np.sum(rolling_window(df[f'trade_size_buy'], delta_time), axis=1),
                    np.sum(rolling_window(df[f'trade_size_sell'], delta_time), axis=1)
                )
            for ab in ['buy', 'sell']:
                # buy sell volume
                df[f'trade_size_{ab}_cumsum_{delta_time}'] = 0
                df.iloc[delta_time-1:, df.columns.get_loc(f'trade_size_{ab}_cumsum_{delta_time}')] = \
                    np.sum(rolling_window(df[f'trade_size_{ab}'], delta_time), axis=1)
                # buy sell trade count
                df[f'trade_side_{ab}_cumsum_{delta_time}'] = 0
                df[f'trade_side_{ab}_cumsum_{delta_time}'].iloc[delta_time - 1:] = \
                    np.sum(rolling_window(df[f'side_{ab}'], delta_time), axis=1)
        return df

    def load_from_qt_file(s, qt_dir):
        raw_bitmex_loader = ConvertBitmexToQC(s.params.asset, s.params.resolution)
        raw_bitmex_loader.req_start = s.ts_start
        raw_bitmex_loader.req_end = s.ts_end
        return raw_bitmex_loader.load_ex_bitmex_data(qt_dir)

    def add_quote_feature_b4_resampling(s, df):
        """spread, bool_bid_size_update, bool_ask_size_update"""
        df['spread_ticks'] = np.rint(np.divide(df['spread'], tick_size[s.params.asset])).astype(int)
        df = s.add_feature_size_ab_sum(df, 'bid')
        df = s.add_feature_size_ab_sum(df, 'ask')
        return df

    @staticmethod
    def add_trade_feature_b4_resampling(df):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(df['side'])
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        enc = OneHotEncoder(sparse=False)
        onehot_encoded = enc.fit_transform(integer_encoded).astype(int)
        new_feats = ['side_' + c.lower() for c in label_encoder.classes_]
        for i in range(len(new_feats)):
            df[new_feats[i]] = onehot_encoded[:, i]
        for side in ['buy', 'sell']:
            df[f'size_{side}'] = np.multiply(df['size'], df[f'side_{side}'])
        df.drop('side', inplace=True, axis=1)
        return df

    @staticmethod
    def add_feature_bool_ab_update(df, ab):
        df[f'bool_{ab}_size_update'] = None
        df.iloc[0, df.columns.get_loc(f'bool_{ab}_size_update')] = False
        df.iloc[:, df.columns.get_loc(f'bool_{ab}_size_update')] = np.sum(df.loc[:, [f'size_{ab}_add', f'size_{ab}_remove']], axis=1) > 0
        return df

    @staticmethod
    def add_feature_size_ab_sum(df, ab):
        df[f'size_{ab}_add'] = 0
        df[f'size_{ab}_remove'] = 0
        delta_size = np.subtract(df.iloc[1:, df.columns.get_loc(f'{ab}_size')].values, df.iloc[:-1, df.columns.get_loc(f'{ab}_size')].values)
        df.iloc[1:, df.columns.get_loc(f'size_{ab}_add')] = np.where(delta_size > 0, delta_size, 0)
        df.iloc[1:, df.columns.get_loc(f'size_{ab}_remove')] = np.abs(np.where(delta_size < 0, delta_size, 0))
        return df

    @staticmethod
    def resample_quote_spead(df, time_period='1S'):
        df = pd.concat([
            df['spread_ticks'].resample(rule=time_period).last(),
            df['spread_ticks'].resample(rule=time_period).max()
            ], axis=1)
        df.columns = ['spread_ticks_close', 'spread_ticks_max']
        # padding is correct because midPrice changes would trigger tick updates. can assume spread is maintaned
        df = df.fillna(method='pad')
        df = df.astype(int)
        return df

    def merge_quote_trade(s, quote, trade):
        return pd.concat(
            reduce_to_intersect_ts(quote, trade),
            axis=1)

    def load_qt_db(s, params):
        s.quote = s.load_qt_single_db('quote', db, params)
        s.trade = s.load_qt_single_db('trade', db, params)
        assert len(s.quote) == len(s.trade), 'Length of quote and trade not equal. Probably missing entries in db'
        return pd.concat([s.quote, s.trade], axis=1)

    def load_raw_preds_db(s, ts_start, ts_end):
        print('Loadings tickforecast raw preds from db...')
        data = s.load_qt_single_db('preds', db, dotdict(dict(data_start=ts_start, data_end=ts_end)))
        s.raw_preds = data
        return data

    @staticmethod
    def load_qt_single_db(qt, db, params):
        data_schema = [el[0] for el in db.fetchall(
            f'''SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS where table_name='tickforecast_{qt}'; ''')]
        sql = '''select `{3}` from trade.tickforecast_{2} where
                        ts >= '{0}' and ts <= '{1}' '''.format(
            params.data_start,
            params.data_end,
            qt,
            '`,`'.join(data_schema)
        )
        data = db.fetchall(sql)
        data = pd.DataFrame(data, columns=data_schema)
        data.set_index(keys='ts', drop=True, inplace=True)
        # data = s.normalize.normalize_scale01_ndarr(data)
        # data = float_to_int_approx(data, digits=5)
        return data

    def store_quote(s):
        values = s.quote
        values['ts'] = values.index
        target_cols = values.columns
        values = list(map(tuple, values.values))
        sql = '''insert into trade.tickforecast_quote (`{0}`) 
                    values ({1}) 
                    on duplicate key update {2};'''.format(
            '`,`'.join(target_cols),
            s.db.perc_s_string(values),
            ','.join(['`{0}`=values(`{0}`)'.format(col) for col in target_cols])
        )
        print(f'Inserting {len(values)} records in trade.tickforecast_quote...')
        for _ in map(partial(s.db.db_exec_commit_many, sql=sql, db=s.db), s.db.slice_over_nda(nda=values)):
            pass

    def store_trade(s):
        values = s.trade
        values['ts'] = values.index
        target_cols = values.columns
        values = list(map(tuple, values.values))
        sql = '''insert into trade.tickforecast_trade (`{0}`) 
                            values ({1}) 
                            on duplicate key update {2};'''.format(
            '`,`'.join(target_cols),
            s.db.perc_s_string(values),
            ','.join(['`{0}`=values(`{0}`)'.format(col) for col in target_cols])
        )
        print(f'Inserting {len(values)} records in trade.tickforecast_trade...')
        for _ in map(partial(s.db.db_exec_commit_many, sql=sql, db=s.db), s.db.slice_over_nda(nda=values)):
            pass

    @staticmethod
    def fill_nans(x_tv):
        cnt_nan = x_tv.isna().sum().sum()
        if cnt_nan > 0:
            print(f'Found {cnt_nan} nan')
            x_tv.fillna(0, inplace=True)
        return x_tv


@click.command('train')
@click.pass_context
def run(ctx: ctx):
    params = importlib.import_module('{}.{}'.format(Paths.path_config_supervised, ctx.obj.fn_params)).Params()
    settings = importlib.import_module('{}.{}'.format(Paths.path_config_supervised, ctx.obj.fn_settings))
    inst = OrderBookFeatures(params, settings)
    inst.get_order_book_features()


@click.command()
@click.pass_context
def main(ctx):
    ctx.obj = dotdict(dict(
        fn_params='ethusd_train',
        fn_settings='settings'
    ))
    ctx.forward(run)


if __name__ == '__main__':
    main()
