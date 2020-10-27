import os
import datetime
import zipfile
import shutil
import pandas as pd
import numpy as np

from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from common.globals import OHLCV, OHLC
from common.modules.logger import Logger
from common.paths import Paths
from common.refdata import date_formats
from common.utils.util_func import create_dir, date_day_range, pipe, count_non_zero, sum_lt_zero, count_lt_zero, sum_gt_zero, count_gt_zero

logger = Logger()


# Bitfinex notes. When e.g. July 30 is selected in the calendar, the received file contains trade data
# from July 29. Hence uploading up to today's date. The format for QC is different. QC file of July 30 contains trade
# data from julty 30


class ConvertBitmexToQC:
    def __init__(s, symbol, res):
        s.symbol = symbol if len(symbol) == 6 else symbol + 'usd'
        s.res = res
        s.save_dir = os.path.join(Paths.qc_bitmex_crypto, res, s.c_symbol(s.symbol))
        create_dir(s.save_dir)
        # we want to overwrite the last saved day to resample pre-/post-midnight ticks correctly
        s.req_start: datetime = s.identify_latest_file_date(folder='converted') - datetime.timedelta(days=1)
        s.req_end: datetime = s.identify_latest_file_date(folder='raw')
        # s.req_start = datetime.datetime(2018, 12, 16)
        # s.req_end = datetime.datetime(2019, 7, 15)
        logger.info(f'START DATE: {s.req_start}')
        if s.req_end - s.req_start > datetime.timedelta(days=30):
            logger.info('Reducing the time range to convert to 30 days to avoid memory exhaustion. Run again until all converted.')
            s.req_end = s.req_start + datetime.timedelta(days=30)
        s.range = [s.req_start, s.req_end]
        s.dir_quotes = os.path.join(Paths.bitmex_raw, 'quote')
        s.dir_trades = os.path.join(Paths.bitmex_raw, 'trade')

    def identify_latest_file_date(s, folder=None):
        if folder == 'raw':
            filenames = os.listdir(os.path.join(Paths.bitmex_raw, 'quote'))
        elif folder == 'converted' and s.symbol is not None:
            filenames = os.listdir(os.path.join(Paths.qc_bitmex_crypto, s.res, s.symbol.lower()))
        else:
            raise Exception("Information missing to convert raw data")
        try:
            last_day = max([int(fn[:8]) for fn in filenames])
        except ValueError:  # coz there is not a single file yet
            last_day = (datetime.date.today() - datetime.timedelta(days=700)).strftime(date_formats.Ymd)
        return datetime.datetime.strptime(str(last_day), date_formats.Ymd)

    @staticmethod
    def c_symbol(x: str):
        return 'xbtusd' if x in ['btcusd', 'xbtusd'] else x

    @staticmethod
    def resample_index(df, time_period=None, agg_col=['volume'], price_col=['price'], **kwargs):
        if agg_col is not None:
            vol = df[agg_col].resample(rule=time_period).sum()
        if price_col is not None:
            price = pd.concat([
                df[price_col].resample(rule=time_period).first(),
                df[price_col].resample(rule=time_period).max(),
                df[price_col].resample(rule=time_period).min(),
                df[price_col].resample(rule=time_period).last(),
                # df[priceCol].resample(rule=timePeriod).mean(),
                # df[priceCol].resample(rule=timePeriod).count()
            ],
                axis=1
            )
            price.columns = ['open', 'high', 'low', 'close']  # ,'count']
        return price if agg_col is None else pd.concat([price, vol], axis=1)

    @staticmethod
    def res_p(res):
        return '1T' if res == 'minute' else '1S'

    def load_ex_bitmex_data(s, directory):
        df_lst = []
        for root, dirs, filenames in os.walk(directory):
            for file in filenames:
                fn_date = datetime.datetime.strptime(file[:8], date_formats.Ymd)
                if s.req_start <= fn_date <= s.req_end + datetime.timedelta(days=1):
                    df = pd.read_csv(os.path.join(directory, file), compression='gzip')
                    df = df[df['symbol'] == s.symbol.upper()]
                    df_lst.append(df)
            break
        logger.info(f'Concatenating {len(df_lst)} dataframes ...')
        return pd.concat(df_lst).reset_index(drop=True)

    @staticmethod
    def prepare_bitmex_trades(df, res_p):
        df.index = pd.to_datetime(df['timestamp'], format='%Y-%m-%dD%H:%M:%S.%f')
        df = df.drop(['timestamp', 'symbol'], axis=1)
        df = df.loc[:, ['size', 'price']]
        df = ConvertBitmexToQC.resample_index(df, time_period=res_p, agg_col=['size'], price_col=['price'])
        df = df.dropna(axis=0, how='any')
        return df

    @staticmethod
    def post_process_raw_trade(pdf: pd.DataFrame) -> pd.DataFrame:
        """split size into sell & buy size"""
        pdf = pdf.rename({'timestamp': 'ts'}, axis='columns')
        for side in ['Buy', 'Sell']:
            c = side.lower()
            pdf[f'size_{c}'] = 0
            ix_side = pdf.index[pdf['side'] == side]
            pdf.loc[ix_side, f'size_{c}'] = pdf.loc[ix_side, 'size'].values
        return pdf

    @staticmethod
    def post_process_raw_quote(pdf: pd.DataFrame) -> pd.DataFrame:
        # split into sell & buy amount
        pdf = pdf.rename({'timestamp': 'ts'}, axis='columns')
        for c in ['askPrice', 'bidPrice']:
            pdf[f'delta_{c}'] = pdf[c].diff()
            pdf.iloc[0, pdf.columns.get_loc(f'delta_{c}')] = 0
        # want count bids and count asks separately.
        for c in ['bid', 'ask']:
            # only subtract same price levels from each other
            pdf[f'delta_{c}Size'] = 0
            ix_best_move = pdf.index[pdf[f'delta_{c}Price'] != 0]
            d_size = pdf[f'{c}Size'].diff()
            d_size.iloc[0] = 0
            d_size.loc[ix_best_move] = 0
            pdf[f'delta_{c}Size'] = d_size
            pdf[f'delta_{c}Size'] = pdf[f'delta_{c}Size'].astype(int)
        pdf = pdf.drop([f'delta_{c}' for c in ['askPrice', 'bidPrice']], axis=1)
        return pdf

    @staticmethod
    def insert_volume_tick(pdf: pd.DataFrame, amount: int) -> pd.DataFrame:
        """set volume tick index"""
        pdf['volume_tick'] = pdf['foreignNotional'].cumsum() / amount
        pdf = pdf.set_index('volume_tick', drop=True)
        pdf.index = np.floor(pdf.index).astype(int)
        return pdf

    @staticmethod
    def resample_by_index_trade(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.groupby(pdf.index).agg({
            'ts': 'last',
            'price': ['first', 'max', 'min', 'last', 'count'],
            'size': ['sum', 'mean'],
            'size_buy': ['sum', count_non_zero],
            'size_sell': ['sum', count_non_zero],
            'grossValue': ['sum'],
            'foreignNotional': ['sum']
        })
        pdf.columns = ['ts'] + OHLC + ['count', 'volume', 'volume_mean', 'volume_buy', 'count_buy', 'volume_sell', 'count_sell', 'grossValue', 'foreignNotional']
        return pdf

    @staticmethod
    def bin_by_ts_interval(pdf: pd.DataFrame, pdf_intervals: pd.DataFrame) -> pd.DataFrame:
        pdf['ts'] = pd.to_datetime(pdf['ts'], format=date_formats.pd_datetime_ns)
        pdf['bins'] = pdf['ts'].to_numpy(dtype='datetime64[ns]').astype(float)
        pdf['bins'] = np.digitize(
            pdf['bins'],
            np.unique(pdf_intervals['ts'].to_numpy(dtype='datetime64[ns]')).astype(float),
            right=True)
        # if quotes happened after the last trade (likely), there will be trade_bins + 1    quote_bins.
        # in the same millisecond, can have multiple ticks, when trade hit through at least 2 order book levels
        ts_tick_dic = pdf_intervals.reset_index()[['ts', 'volume_tick']].groupby('ts').max().to_dict()['volume_tick']
        # ensure a zero is present. 0 bin gets 0 vol tick because it's before first ts.
        sorted_max_volume_ticks = sorted(ts_tick_dic.values()) + [None]  # None for anything future of last tick
        bin_tick_dic = {i: sorted_max_volume_ticks[i] for i in range(pdf['bins'].max())}
        pdf['bins'] = pdf['bins'].map(bin_tick_dic)
        pdf = pdf.rename({'bins': 'volume_tick'}, axis='columns')
        pdf = pdf.drop(pdf.index[pdf['volume_tick'].isna()])
        pdf['volume_tick'] = pdf['volume_tick'].astype(int)
        assert pdf['volume_tick'].isna().sum() == 0
        return pdf.set_index('volume_tick', drop=True)

    @staticmethod
    def resample_by_index_quote(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.groupby(pdf.index).agg({
            'ts': 'last',
            'bidPrice': ['first', 'max', 'min', 'last'],
            'bidSize': ['first', 'max', 'min', 'last'],
            'delta_bidSize': [sum_lt_zero, count_lt_zero, sum_gt_zero, count_gt_zero],
            'askPrice': ['first', 'max', 'min', 'last'],
            'askSize': ['first', 'max', 'min', 'last'],
            'delta_askSize': [sum_lt_zero, count_lt_zero, sum_gt_zero, count_gt_zero],
            'spread': ['max', 'min']
        })
        pdf.columns = ['ts'] + \
                      ['bid_' + c for c in OHLC] + \
                      ['bid_size_' + c for c in OHLC] + \
                      ['bid_size_removed', 'bid_size_removed_count', 'bid_size_added', 'bid_size_added_count'] + \
                      ['ask_' + c for c in OHLC] + \
                      ['ask_size_' + c for c in OHLC] + \
                      ['ask_size_removed', 'ask_size_removed_count', 'ask_size_added',  'ask_size_added_count'] + \
                      ['spread_high', 'spread_low']
        for c in ['ask_size_removed', 'bid_size_removed']:
            pdf[c] = pdf[c].abs()
        return pdf

    @staticmethod
    def prepare_bitmex_quotes(df: pd.DataFrame, res_p):
        df.index = pd.to_datetime(df['ts'], format='%Y-%m-%dD%H:%M:%S.%f')
        df = df.drop(['ts', 'symbol'], axis=1)
        df_bid = ConvertBitmexToQC.resample_index(df, time_period=res_p, agg_col=['bidSize'], price_col=['bidPrice'])
        df_bid.columns = ['bid_' + c for c in OHLCV]
        df_ask = ConvertBitmexToQC.resample_index(df, time_period=res_p, agg_col=['askSize'], price_col=['askPrice'])
        df_ask.columns = ['ask_' + c for c in OHLCV]
        df = pd.concat((df_bid, df_ask), axis=1)
        df = df.dropna(axis=0, how='any')
        return df

    @staticmethod
    def pick_idx(df, date):
        df_day = df.iloc[np.where((df.index.day == date.day) & (df.index.month == date.month) & (df.index.year == date.year))[0], :]
        df_day.index = (df_day.index - date).total_seconds()
        df_day.index = pd.to_numeric(df_day.index * 1000, downcast='integer')
        return df_day

    def fn_qt(s, str_date, qt):
        return os.path.join(s.save_dir, f'{str_date}_{s.c_symbol(s.symbol)}_{s.res}_{qt}.csv')

    def zip_qt(s, str_date, qt):
        return os.path.join(s.save_dir, f'{str_date}_{qt}.zip')

    def df_day_qt(s, df, str_date, qt, header=False):
        df.to_csv(s.fn_qt(str_date, qt), header=header)
        # better use io.text buffer instead of actually writing to disk twice.
        s.zip_away(s.fn_qt(str_date, qt), s.zip_qt(str_date, qt))

    @staticmethod
    def zip_away(fn, zip_name):
        with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as myzip:
            myzip.write(fn, os.path.basename(fn))
        os.remove(fn)

    def run(s):
        if s.req_end == s.req_start:
            logger.info('Start end equals end data. No new data is converted')
            return
        if s.res in ['minute', 'second']:
            df_trades = s.prepare_bitmex_trades(s.load_ex_bitmex_data(s.dir_trades), s.res_p(s.res))
            df_quotes = s.prepare_bitmex_quotes(s.load_ex_bitmex_data(s.dir_quotes), s.res_p(s.res))
            for date in date_day_range(s.req_start, s.req_end + datetime.timedelta(days=1)):
                df_quotes_intts = s.pick_idx(df_quotes, date)
                df_trades_intts = s.pick_idx(df_trades, date)
                s.df_day_qt(df_trades_intts, date.strftime(date_formats.Ymd), qt='trade')
                s.df_day_qt(df_quotes_intts, date.strftime(date_formats.Ymd), qt='quote')
        elif s.res in ['tick']:
            # trades first
            # target schema - time (res ??), value, Q, Side, TickDirection
            df = s.load_ex_bitmex_data(s.dir_trades)
            df.index = pd.to_datetime(df['timestamp'], format='%Y-%m-%dD%H:%M:%S.%f')
            df = df.drop(['timestamp', 'symbol'], axis=1)
            df = df.loc[:, ['price', 'size', 'side', 'tickDirection', 'grossValue', 'foreignNotional']]
            df = df.dropna(axis=0, how='any')
            for date in date_day_range(s.req_start, s.req_end + datetime.timedelta(days=1)):
                df_tick_intts = s.pick_idx(df, date)
                s.df_day_qt(df_tick_intts, date.strftime(date_formats.Ymd), qt='trade')
            # check above
            # target schema - time (res ??), bidPrice, bidSize, aksPrice, askSize,
            df = s.load_ex_bitmex_data(s.dir_quotes)
            df.index = pd.to_datetime(df['timestamp'], format='%Y-%m-%dD%H:%M:%S.%f')
            df = df.drop(['timestamp', 'symbol'], axis=1)
            df = df.loc[:, ['bidPrice', 'bidSize', 'askPrice', 'askSize']]
            df = df.dropna(axis=0, how='any')
            for date in date_day_range(s.req_start, s.req_end + datetime.timedelta(days=1)):
                df_tick_intts = s.pick_idx(df, date)
                s.df_day_qt(df_tick_intts, date.strftime(date_formats.Ymd), qt='quote')

        elif 'volume' in s.res:
            volume, ccy, amount = s.res.split('_')

            df_trades = pipe((
                s.load_ex_bitmex_data,
                s.post_process_raw_trade
            ), s.dir_trades)()
            df_trades['ts'] = pd.to_datetime(df_trades['ts'], format=date_formats.pd_datetime_ns)
            df_quotes = pipe((
                s.load_ex_bitmex_data,
                s.post_process_raw_quote,
            ), s.dir_quotes)()
            df_quotes['ts'] = pd.to_datetime(df_quotes['ts'], format=date_formats.pd_datetime_ns)

            ts_insert = np.unique(np.setdiff1d(df_trades['ts'].values, df_quotes['ts'].values))
            pad_quotes = df_trades.set_index('ts').loc[ts_insert, ['symbol', 'price', 'size_buy', 'size_sell']]
            for c in ['bidSize', 'bidPrice', 'askPrice', 'askSize']:
                pad_quotes[c] = None
            pad_quotes = pad_quotes.reset_index()
            ix_sell = pad_quotes.index[pad_quotes['size_sell'] > 0]
            ix_buy = pad_quotes.index[pad_quotes['size_buy'] > 0]
            pad_quotes.loc[ix_sell, 'bidPrice'] = pad_quotes.loc[ix_sell, 'price']
            pad_quotes.loc[ix_buy, 'askPrice'] = pad_quotes.loc[ix_buy, 'price']
            pad_quotes = pad_quotes.drop('price', axis=1)
            pad_quotes = pad_quotes.rename({'size_buy': 'delta_askSize', 'size_sell': 'delta_bidSize'}, axis='columns')
            df_quotes = pd.concat([df_quotes, pad_quotes], axis=0, sort=False)
            df_quotes = df_quotes.sort_values('ts').reset_index()
            for c in ['askPrice', 'bidPrice', 'askSize', 'bidSize']:
                df_quotes[c] = df_quotes[c].fillna(method='ffill')
                # whatever is at beginning
                df_quotes[c] = df_quotes[c].fillna(method='bfill')
            df_quotes['spread'] = df_quotes['askPrice'] - df_quotes['bidPrice']

            df_trades = s.insert_volume_tick(df_trades, int(amount))
            df_trades = s.resample_by_index_trade(df_trades)
            df_trades['ts'] = pd.to_datetime(df_trades['ts'], format=date_formats.pd_datetime_ns)
            df_quotes = s.bin_by_ts_interval(df_quotes, df_trades)
            df_quotes = s.resample_by_index_quote(df_quotes)
            # df_trades = s.resample_by_index_trade(s.load_ex_bitmex_data(s.dir_trades), int(amount))
            # df_trades['ts'] = pd.to_datetime(df_trades['ts'], format=date_formats.pd_datetime_ns)
            # df_quotes = s.resample_by_index_quote(s.load_ex_bitmex_data(s.dir_quotes), df_trades)
            # dont save first days as first tick doesnt include previous pre-midgnight ticks
            # but need to overwrite whatever we saved last, hence 2 days start at last day - 1
            logger.info(f'Saving from {s.req_start + datetime.timedelta(days=1)} to {s.req_end + datetime.timedelta(days=1)}')
            for date in date_day_range(s.req_start + datetime.timedelta(days=1), s.req_end + datetime.timedelta(days=1)):
                # for every day need to set volume_tick of t_day_zero[0] = t_day_zero[0] - t_day_previous[-1]
                df_trades_day = df_trades[(df_trades['ts'] >= date) & (df_trades['ts'] < date + datetime.timedelta(days=1))]
                df_quotes_day = df_quotes[(df_quotes['ts'] >= date) & (df_quotes['ts'] < date + datetime.timedelta(days=1))]
                try:
                    previous_volume_tick = df_trades[df_trades['ts'] < date].index[-1]
                except IndexError:  # means it's the first day
                    previous_volume_tick = 0
                df_trades_day.index -= previous_volume_tick
                df_quotes_day.index -= previous_volume_tick
                s.df_day_qt(df_trades_day, date.strftime(date_formats.Ymd), qt='trade', header=True)
                s.df_day_qt(df_quotes_day, date.strftime(date_formats.Ymd), qt='quote', header=True)


def run(symbol_res):
    symbol, res = symbol_res
    converter = ConvertBitmexToQC(symbol, res)
    converter.run()


def main(assets=None, resolutions=None):
    resolutions = resolutions or ['second', 'minute', 'tick']
    symbol_res = []
    for symbol in assets:
        for res in resolutions:
            symbol_res.append((symbol, res))
    with PoolExecutor(max_workers=min(8, len(symbol_res))) as executor:
        for _ in executor.map(run, symbol_res):
            pass


def copy_xbt_to_btc():
    for res in ['minute', 'second', 'tick']:
        src_root = os.path.join(Paths.qc_bitmex_crypto, r'{}\xbtusd'.format(res))
        src_fn = os.listdir(src_root)
        target_root = os.path.join(Paths.qc_bitmex_crypto, r'{}\btcusd'.format(res))
        target_fn = os.listdir(target_root)
        for fn in src_fn:
            if fn not in target_fn:
                shutil.copy(os.path.join(src_root, fn), os.path.join(target_root, fn))


if __name__ == '__main__':
    for i in range(30):
        # main(assets=['xbtusd'], resolutions=['second', 'minute', 'tick'])
        # main(assets=['ethusd'], resolutions=['tick'])
        main(assets=['xbtusd'], resolutions=['volume_usd_10000'])
