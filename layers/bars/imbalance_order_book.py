from typing import List

import yaml
import numpy as np
import datetime
import pandas as pd

from functools import lru_cache
from itertools import product

from numba import jit

from common.paths import Paths
from connector.influxdb.influxdb_wrapper import influx
from common.modules.logger import logger


def apply_best_bid_ask(price: np.ndarray, side: np.ndarray, size: np.ndarray):
    active_bids = set()
    active_asks = set()
    n = len(price)
    bbid = price[np.argmax(side == 1)]
    bask = price[np.argmax(side == -1)]
    best_bids = np.empty(n)
    best_asks = np.empty(n)
    for i in range(n):
        p = price[i]
        c = side[i]
        if c == 0:
            if size[i] > 0:
                if p in active_bids:
                    active_bids.remove(p)
                if p > bbid:
                    if active_bids:
                        bbid = max(active_bids)
                    else:
                        bbid = 0

            elif size[i] < 0:
                if p in active_asks:
                    active_asks.remove(p)

                if p < bask:
                    if active_asks:
                        bask = min(active_asks)
                    else:
                        bask = 99999
        else:
            if size[i] > 0:
                active_bids.add(p)
                if p >= bbid:
                    bbid = p
            elif size[i] < 0:
                active_asks.add(p)
                if p <= bask:
                    bask = p

        if bbid >= bask:  # something went wrong. make it consistent
            if size[i] > 0:  # is bid
                for n in [n for n in active_asks if n <= bbid]:
                    active_asks.remove(n)
                bask = min(active_asks) if active_asks else 99999
            if size[i] < 0:  # is ask
                for n in [n for n in active_bids if n >= bask]:
                    active_bids.remove(n)
                bbid = max(active_bids) if active_bids else 0

        best_bids[i] = bbid
        best_asks[i] = bask
        if i % 1000000 == 0:
            print(i)
    return best_bids, best_asks


class OrderBook:
    """
    - Multi dimensional frame with dimensions: time, side, level    ( potentially another to get size of each bid/ask within level)!
        created from a stream of ticks
    - Best bid ask frame.
    """
    def __init__(self, df_quotes, level_from_price_pct):
        self.df_quotes = df_quotes
        self.level_from_price_pct = level_from_price_pct

    @property
    @lru_cache()
    def level_distance(self):
        res = (self.df_quotes['price'].shift(1) - self.df_quotes['price']).abs()
        return res[res != 0].min()

    def create_order_book(self):
        df = self.df_quotes
        assert sorted(df['side'].unique()) == [-1, 1], 'Side is not fully determined. Infer from BBAB and price'
        # Count may only be availbe with Bitfinex at the moment. Enrich if necessary. Emptied levels, count 0, mean to be ignore for best bid determination, but
        # need information to accurately encode when levels where emptied and filled
        a, b = apply_best_bid_ask(df['price'].values, np.where(df['count'] == 0, 0, df['side']), df['size'].values)
        df['best_bid'] = a
        df['best_ask'] = b

        df['best_bid'] = df['best_bid'].ffill()
        df.loc[df.index[df['best_bid'] == 0], 'best_bid'] = None

        df['best_ask'] = df['best_ask'].ffill()
        df.loc[df.index[df['best_ask'] == 99999], 'best_ask'] = None
        df['best_ask'] = df['best_ask'].bfill()
        df['best_bid'] = df['best_bid'].bfill()
        # df = df.iloc[1000:].reset_index(drop=True)
        df = df[~df['price'].isna()]
        df = df[~df['size'].isna()]
        df = self.impute_nan_count(df)
        assert df.isna().sum().sum() == 0, 'NANs at this step. why?'
        assert (df['best_ask'] < df['best_bid']).sum() == 0
        logger.info(f'Null price rows: {df["price"].isna().sum()}')
        # Add Level

        ix_drop_no_level_land = df.index[np.where((df['best_bid'] < df['price']) & (df['price'] < df['best_ask']))]
        if len(ix_drop_no_level_land) > 0:
            logger.info(f'''Dropping {len(ix_drop_no_level_land)} order book levels that are in between best ask and best bid.''')
            df = df.drop(ix_drop_no_level_land).reset_index(drop=True)

        # Assign side to emptied levels
        ix_level_empty = df.index[df['count'] == 0]
        df.loc[ix_level_empty, 'size'] = 0
        assert df['size'][df['count'] == 0].sum() == 0, 'Size should be 0 whenever emtpy order book level / count is 0.'

        # Might be better to drop these rather than reassigning. Could be due to data coming in wrong sequence.
        ix_wrong_side_ask = df.index[(df['side'] == -1) & (df['price'] < df['best_ask'])]
        ix_wrong_side_bid = df.index[(df['side'] == 1) & (df['price'] > df['best_bid'])]
        if len(ix_wrong_side_bid) + len(ix_wrong_side_ask) > 0:
            logger.info(f'''Reassigning Inconsistent / Wrong side level BID: {len(ix_wrong_side_bid)} - ASK: {len(ix_wrong_side_ask)}. Investigate if inconsistency count is high ''')
            df.loc[ix_wrong_side_bid, 'side'] *= -1
            df.loc[ix_wrong_side_ask, 'side'] *= -1

        # Filled Levels
        ix_ask = df.index[df['side'] == -1]
        ix_bid = df.index[df['side'] == 1]
        df['level'] = None
        df.loc[ix_ask, 'level'] = ((df.loc[ix_ask, 'price'] - df.loc[ix_ask, 'best_ask']) / self.level_distance + 1).astype(int)
        df.loc[ix_bid, 'level'] = ((df.loc[ix_bid, 'price'] - df.loc[ix_bid, 'best_bid']) / self.level_distance - 1).astype(int)

        assert df.loc[ix_ask, 'level'].min() >= 1
        assert df.loc[ix_bid, 'level'].max() <= 0
        assert df['level'].isna().sum() == 0

        df['level'] = df['level'].astype(int).abs()
        self.df_quotes = df = df.set_index(['timestamp', 'side', 'level'])

    @staticmethod
    def impute_nan_count(df: pd.DataFrame) -> pd.DataFrame:
        if df.isna().sum()['count'] > 0:
            ix_zero = df.index[(df['count'].isna()) & (df['size'].abs() == 1)]
            df.loc[ix_zero, 'count'] = 0
            ix_non_zero = df.index[(df['count'].isna()) & (df['size'].abs() != 1)]
            df.loc[ix_non_zero, 'count'] = 1
            logger.info(f'Imputed {len(ix_zero) + len(ix_non_zero)} count values.')
            return df
        else:
            return df

    @property
    @lru_cache()
    def level(self) -> int:
        return min(int((self.df_quotes['price'] * self.level_from_price_pct / 100).max() / self.level_distance), 300)

    def derive_events(self):
        df = self.df_quotes
        df = df.loc[(slice(None), slice(None), list(range(self.level))), :]
        df = df.groupby(['timestamp', 'side', 'level']).last()
        # cumulative race through all rows. final state must include every single level
        ix_ts = df.index.get_level_values('timestamp').unique()
        book = pd.DataFrame(None, index=pd.MultiIndex.from_product([
                                ix_ts,
                                [1, -1],  # side
                                range(self.level)
                            ], names=["timestamp", "side", 'level']),
                            columns=['size', 'count'], dtype=float)
        # book2 = np.empty(shape=(len(df.index.get_level_values('timestamp').unique()), 2, self.level, 2))
        logger.info('Inserting values into empty book frame')
        book.loc[df.index] = df[['size', 'count']]
        df = None
        arr = book.values.reshape(len(ix_ts), 2, self.level, 2)
        book = None
        ix_valid = 0
        logger.info('FFill each level')
        for side, level in product([0, 1], range(self.level)):
            # print(f'Side: {side} - Level: {level}')
            arr[:, side, level, 0] = ffill(arr[:, side, level, 0])
            ix_valid = max(np.argmin(np.isnan(arr[:, side, level, 0])), ix_valid)
            arr[:, side, level, 1] = ffill(arr[:, side, level, 1])
            ix_valid = max(np.argmin(np.isnan(arr[:, side, level, 1])), ix_valid)
        arr, ts = arr[ix_valid:], ix_ts[ix_valid:]
        arr = np.nan_to_num(arr)
        return arr, ts


def ffill(arr: np.ndarray) -> np.ndarray:
    # same technique - ffill across all levels and sides
    return arr[np.maximum.accumulate(np.where(~np.isnan(arr), np.arange(arr.shape[0]), 0))]


# @jit(nopython=False)
def ix_every_delta(arr: np.array, delta: float) -> (List, List):
    cumsum = np.array([0] + (arr[:-1] - arr[1:]).cumsum().tolist())
    ix_events = []
    actual_deltas = []
    while True:
        ix_event = np.argmax(np.abs(cumsum) >= delta)
        if ix_event == 0:
            break
        else:
            actual_delta = cumsum[ix_event]
            ix_events.append(ix_event)
            actual_deltas.append(actual_delta)
            cumsum -= actual_delta
            cumsum[:ix_event] = 0
    return ix_events, actual_deltas


def invert_lt_zero_ratio(ps: pd.Series) -> pd.Series:
    ix_ask_greater = ps.index[ps < 1]  # i
    ps.loc[ix_ask_greater] = 1 / ps.loc[ix_ask_greater]
    return ps


if __name__ == '__main__':
    with open(Paths.layer_settings, "r") as stream:
        settings = yaml.safe_load(stream)
    for exchange in settings.keys():
        for asset, params in settings[exchange].items():
            if asset not in ['solusd']:  # continue from march 8
            # if asset not in ['xrpusd']:
                continue

            logger.info(f'Loading order book for {exchange} - {asset}')
            params = params.get('order book', {})
            delta_size_ratio = params.get('delta_size_ratio')
            from layers.bitfinex_reader import BitfinexReader
            start = datetime.datetime(2022, 2, 7)
            end = datetime.datetime(2022, 3, 13)
            for i in range(1 + (end-start).days):
                dt = start + datetime.timedelta(days=i)
                logger.info(f'Running {dt}')
                df = BitfinexReader.load_quotes(sym=asset, start=dt, end=dt)
                ob = OrderBook(df, level_from_price_pct=params.get('level_from_price_pct'))
                level = ob.level
                logger.info(f'{asset} - Summarizing {level} order book levels')
                ob.create_order_book()
                arr, ix_ts = ob.derive_events()

                alpha = 2/(level + 1)
                weights = np.array([(1-alpha)**i for i in range(level)]).reshape(1, 1, level, 1)

                arr = (arr * weights).sum(axis=2)
                size, count = arr[:, :, 0], arr[:, :, 1]

                count_ratio = (count[:, 0] / np.abs(count[:, 1]))
                size_ratio = (size[:, 0] / np.abs(size[:, 1]))  # bid|buy / ask|sell
                ps_count_ratio = invert_lt_zero_ratio(pd.Series(count_ratio, index=ix_ts, name='count_ratio'))
                ps_size_ratio = invert_lt_zero_ratio(pd.Series(size_ratio, index=ix_ts, name='size_ratio'))
                ps_size_net = pd.Series(size[:, 0] + size[:, 1], index=ix_ts, name='size_net')
                ps_count_net = pd.Series(count[:, 0] - count[:, 1], index=ix_ts, name='count_net')  # bid - ask

                ix_events, actual_deltas = ix_every_delta(ps_size_ratio.values, delta=delta_size_ratio)
                ts_events = ix_ts[ix_events].unique()
                logger.info(f'Ingesting Size, Count Ratios and Net for {len(ix_events)} unique: {len(ts_events)}')

                for information, ps in {
                    'bid_buy_size_imbalance_ratio': ps_size_ratio,
                    'bid_buy_count_imbalance_ratio': ps_count_ratio,
                    'bid_buy_size_imbalance_net': ps_size_net,
                    'bid_buy_count_imbalance_net': ps_count_net,
                }.items():
                    influx.write(
                        record=ps.loc[ts_events].groupby(level=0).last(),
                        data_frame_measurement_name='order book',
                        data_frame_tag_columns={**{
                            'exchange': exchange,
                            'asset': asset,
                            'information': information,
                            'unit': 'size_ewm_sum',
                            'levels': level,
                            'alpha': alpha,
                            'delta_size_ratio': delta_size_ratio
                        }},
                    )
    logger.info('Done')

