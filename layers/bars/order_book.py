import numpy as np
import datetime
import pandas as pd

from itertools import product
from common.modules.assets import Assets
from common.modules.exchange import Exchange
from connector.influxdb.influxdb_wrapper import influx
from common.modules.logger import logger


# @numba.jit(nopython=True)
from layers.bars.base_bar import BaseBar


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
    level_distance = 0.1

    def __init__(self, df_quotes: pd.DataFrame):
        self.df_quotes = df_quotes

    def create_order_book(self):
        df = self.df_quotes
        a, b = apply_best_bid_ask(df['price'].values, df['side'].values, df['size'].values)
        df['best_bid'] = a
        df['best_ask'] = b

        df['best_bid'] = df['best_bid'].ffill()
        df.loc[df.index[df['best_bid'] == 0], 'best_bid'] = None

        df['best_ask'] = df['best_ask'].ffill()
        df.loc[df.index[df['best_ask'] == 99999], 'best_ask'] = None
        df['best_ask'] = df['best_ask'].bfill()
        df['best_bid'] = df['best_bid'].bfill()
        df = df.iloc[1000:].reset_index(drop=True)
        assert (df['best_ask'] < df['best_bid']).sum() == 0
        logger.info(f'Null price rows: {df["price"].isna().sum()}')
        df = df[~df['price'].isna()]
        # Add Level
        ix_ask = df.index[df['side'] == -1]
        ix_bid = df.index[df['side'] == 1]
        df['level'] = None
        df.loc[ix_ask, 'level'] = ((df.loc[ix_ask, 'price'] - df.loc[ix_ask, 'best_ask']) / self.level_distance + 1).astype(int)
        df.loc[ix_bid, 'level'] = ((df.loc[ix_bid, 'price'] - df.loc[ix_bid, 'best_bid']) / self.level_distance - 1).astype(int)
        assert df.loc[ix_ask, 'level'].min() >= 1
        assert df.loc[ix_bid, 'level'].max() <= 0

        ix_level_empty_bid = df.index[(df['level'].isna()) & (df['size'] > 0)]
        ix_level_empty_ask = df.index[(df['level'].isna()) & (df['size'] < 0)]
        df.loc[ix_level_empty_bid, 'size'] = 0
        df.loc[ix_level_empty_bid, 'side'] = 1
        df.loc[ix_level_empty_ask, 'size'] = 0
        df.loc[ix_level_empty_ask, 'side'] = -1
        df['level'] = df['level'].fillna(0).astype(int).abs()
        self.df_quotes = df = df.set_index(['timestamp', 'side', 'level'])

    def derive_events(self, depth=30):
        df = self.df_quotes
        df = df.loc[(slice(None), slice(None), list(range(depth))), :]
        df = df.groupby(['timestamp', 'side', 'level']).last()
        # cumulative race through all rows. final state must include every single level
        book = pd.DataFrame(None,
                            index=pd.MultiIndex.from_product([
                                df.index.get_level_values('timestamp').unique(),
                                [1, -1],  # side
                                range(depth)
                            ], names=["timestamp", "side", 'level']),
                            columns=['size', 'count'], dtype=float)
        book.loc[df.index] = df[['size', 'count']]
        df = None
        ix_ts = book.index.get_level_values('timestamp').unique()
        arr = book.values.reshape(len(ix_ts), 2, depth, 2)
        ix_valid = 0
        for side, level in product([0, 1], range(depth)):
            print(f'Side: {side} - Level: {level}')
            arr[:, side, level, 0] = ffill(arr[:, side, level, 0])
            ix_valid = max(np.argmin(np.isnan(arr[:, side, level, 0])), ix_valid)
            arr[:, side, level, 1] = ffill(arr[:, side, level, 1])
            ix_valid = max(np.argmin(np.isnan(arr[:, side, level, 1])), ix_valid)
        return arr[ix_valid:], ix_ts[ix_valid:]

    def to_influx(self, df):
        if not self.information:
            raise ValueError('No information tag set.')
        assert len(df.index.unique()) == len(df), 'Timestamp is not unique. Group By time first before uploading to influx.'
        influx.write(
            record=df,
            data_frame_measurement_name='trade bars',
            data_frame_tag_columns={**{
                'exchange': self.exchange.name,
                'asset': self.sym.name,
                'information': self.information,
                'unit': self.unit,
            }, **self.tags},
        )


def ffill(arr: np.ndarray) -> np.ndarray:
    # same technique - ffill across all levels and sides
    return arr[np.maximum.accumulate(np.where(~np.isnan(arr), np.arange(arr.shape[0]), 0))]


if __name__ == '__main__':
    import pickle
    from layers.bitfinex_reader import BitfinexReader

    exchange = Exchange.bitfinex
    asset = Assets.ethusd
    start = datetime.datetime(2022, 2, 7)
    end = datetime.datetime(2022, 2, 16)
    level = 30
    for i in range(1 + (end-start).days):
        dt = start + datetime.timedelta(days=i)
        logger.info(f'Running {dt}')
        df = BitfinexReader.load_quotes(sym=asset.name, start=dt, end=dt)
        ob = OrderBook(df)
        ob.create_order_book()
        arr, ix_ts = ob.derive_events()

        alpha = 2/(level + 1)
        weights = np.array([(1-alpha)**i for i in range(level)]).reshape(1, 1, level, 1)

        arr = (arr * weights).sum(axis=2)
        size, count = arr[:, :, 0], arr[:, :, 1]

        for name in ('size_ratio', 'count_ratio', 'size_net', 'count_net'):
            if name == 'size_ratio':
                size_ratio = (size[:, 0] / np.abs(size[:, 1]))  # bid|buy / ask|sell
                ps = pd.Series(size_ratio, index=ix_ts, name=name)
            elif name == 'count_ratio':
                count_ratio = (count[:, 0] / np.abs(count[:, 1]))
                ps = pd.Series(count_ratio, index=ix_ts, name=name)
            elif name == 'size_net':
                ps = pd.Series(size[:, 0] + size[:, 1], index=ix_ts, name=name)
            elif name == 'count_net':
                ps = pd.Series(count[:, 0] - count[:, 1], index=ix_ts, name=name)  # bid - ask
            else:
                raise ValueError

            if 'ratio' in ps.name:
                ix_ask_greater = ps.index[ps < 1]
                ps.loc[ix_ask_greater] = -1 / ps.loc[ix_ask_greater]
                max_bid = ps.resample('15s').max()
                max_ask = ps.resample('15s').min()
                psc = pd.Series(np.where(max_bid > np.abs(max_ask), max_bid, max_ask), index=max_bid.index, name=f'bid_buy_{name}')
                logger.info(f"Influx load: {f'bid_buy_{name}_imbalance_net_resampled_max'}")
                influx.write(
                    record=psc,
                    data_frame_measurement_name='order book',
                    data_frame_tag_columns={**{
                        'exchange': exchange.name,
                        'asset': asset.name,
                        'information': f'bid_buy_{name}_imbalance_ratio_resampled_max',
                        'unit': 'size_ewm_sum',
                        'levels': level,
                        'alpha': alpha,
                        'unit_size': 15
                    }},
                )
            elif 'net' in ps.name:
                unit_size = 15
                max_bid = ps.resample('15s').max()
                max_ask = ps.resample('15s').min()
                psc = pd.Series(np.where(max_bid > np.abs(max_ask), max_bid, max_ask), index=max_bid.index, name=f'bid_buy_{name}')
                logger.info(f"Influx load: {f'bid_buy_{name}_imbalance_net_resampled_max'}")
                influx.write(
                    record=psc,
                    data_frame_measurement_name='order book',
                    data_frame_tag_columns={**{
                        'exchange': exchange.name,
                        'asset': asset.name,
                        'information': f'bid_buy_{name}_imbalance_net_resampled_max',
                        'unit': 'size_ewm_sum',
                        'levels': level,
                        'alpha': alpha,
                        'unit_size': 15
                    }},
                )
            else:
                pass

            if 'ratio' in ps.name:
                unit_size = 5
                if 'ratio' in ps.name:
                    ix_ask_greater = ps.index[ps < 1]  # i
                    ps.loc[ix_ask_greater] = 1 / ps.loc[ix_ask_greater]
                    pa = (ps - ps.shift(1)).fillna(0).cumsum()
                else:
                    pa = (ps - ps.shift(1)).fillna(0).cumsum()
                bars = []
                while True:
                    iloc_event = np.argmax(pa.abs().values >= unit_size)
                    if iloc_event == 0:
                        break
                    iloc_event = 10 ** 99 if iloc_event == 0 else iloc_event

                    if pa.iloc[iloc_event] > 0:
                        direction = 1
                    else:
                        direction = -1

                    bars.append({'imbalance_direction': direction, 'timestamp': pa.index[iloc_event]})
                    pa = pa.iloc[iloc_event:] - pa.iloc[iloc_event]

                if bars:
                    psc = pd.DataFrame(bars).set_index('timestamp')['imbalance_direction']
                    logger.info(f"Influx load: {f'bid_buy_imbalance_delta_{psc.name}'}")
                    influx.write(
                        record=psc,
                        data_frame_measurement_name='order book',
                        data_frame_tag_columns={**{
                            'exchange': exchange.name,
                            'asset': asset.name,
                            'information': f'bid_buy_imbalance_delta_{psc.name}',
                            'unit': 'size_ewm_sum',
                            'levels': level,
                            'alpha': alpha,
                            'unit_size': unit_size
                        }},
                    )
            else:
                print('Wrong')

            print(f'Done {name}.')
