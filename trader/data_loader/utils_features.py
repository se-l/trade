import datetime
import os
import numpy as np
import pandas as pd
import operator

from itertools import groupby
from collections import namedtuple
from common.globals import OHLC
from common.modules.enums import Exchanges, Assets, Series
from common.utils.util_func import reduce_to_intersect_ts, SeriesTickType
from common.modules.logger import Logger
from trader.data_loader.config.config import exchange2asset_class, resolution2folder, ccy2folder, resample_sec2resample_str


class QuoteTradePadIndices:
    def __init__(self, pdf_qt):
        self.pdf_qt = pdf_qt
        self.ix_na = None
        self.ix_na_plus_1 = None
        self.ix_na_minus_1 = None
        self.set_ix_na_pm_1(col='grossValue')
        self.ix_bids_na = pdf_qt.index[pdf_qt['bid_size_open'].isna()]
        self.ix_trade_not_na = pdf_qt.index[pdf_qt['grossValue'].notna()]
        self.ix_trade_sell_no_quote_info = pdf_qt.index[(pdf_qt['volume_sell']) > 0 & (pdf_qt['bid_close'].isna())]
        self.ix_trade_buy_no_quote_info = pdf_qt.index[(pdf_qt['volume_buy'] > 0) & (pdf_qt['ask_close'].isna())]
        self.ix_fill_quote_from_trade = np.intersect1d(self.ix_bids_na, self.ix_trade_not_na)
        self.dict_bba = None
        self.bba_fixed = None
        self.group_na = None
        self.group_na_pm_1 = None

    def mask_n_eq_bba(self, col: str, op):
        # if col.startswith('bid_size'):
        #     snippet = 'bid_size_'
        # elif col.startswith('ask_size'):
        #     snippet = 'ask_size_'
        if col.startswith('bid_'):
            snippet = 'bid_'
        elif col.startswith('ask_'):
            snippet = 'ask_'
        elif any([el in col for el in ['count', 'gross', 'volume', 'foreign']]):
            snippet = ''
        else:
            raise NotImplementedError('Column not handled')
        return op(self.pdf_qt.loc[self.ix_na_minus_1, f'{snippet}close'].values, self.pdf_qt.loc[self.ix_na_plus_1, f'{snippet}open'].values)

    def set_ix_na_pm_1(self, col='grossValue'):
        self.ix_na = self.pdf_qt.index[self.pdf_qt[col].isna()]
        self.ix_na_plus_1 = np.setdiff1d(self.ix_na + 1, self.ix_na)
        self.ix_na_minus_1 = np.setdiff1d(self.ix_na - 1, self.ix_na)

    def set_groups_na(self, col: str, op):
        self.set_ix_na_pm_1(col)
        mask_bba = self.mask_n_eq_bba(col, op)
        ix_na_plus_1_bba = self.ix_na_plus_1[mask_bba]
        ix_na_minus_1_bba = self.ix_na_minus_1[mask_bba]
        self.dict_bba = {ix_na_plus_1_bba[i]: list(range(ix_na_minus_1_bba[i] + 1, ix_na_plus_1_bba[i] + 1)) for i in range(len(ix_na_plus_1_bba))}
        self.bba_fixed = sorted(list(self.dict_bba.keys()))
        try:
            self.group_na = sorted(list(set.union(*[set(range(ix_na_minus_1_bba[i] + 1, ix_na_plus_1_bba[i])) for i in range(len(ix_na_plus_1_bba))])))
            self.group_na_pm_1 = sorted(list(set.union(*[set(range(ix_na_minus_1_bba[i], ix_na_plus_1_bba[i] + 1)) for i in range(len(ix_na_plus_1_bba))])))
            try:
                self.group_na_pm_1.pop(self.group_na_pm_1.index(-1))
            except ValueError:
                pass
        except TypeError:
            self.group_na = []
            self.group_na_pm_1 = []


def _transform_index(pdf, series_tick_type: SeriesTickType):
    if pd.api.types.is_datetime64_dtype(pdf.index):
        pdf['ts'] = pdf.index
        pdf.reset_index(drop=True, inplace=True)
    if 'ts' in pdf.columns:
        pdf['ts'] = pd.to_datetime(pdf['ts'])
    return pdf


def _insert_trade_volume(df_ohlc_mid, params):
    if params.exchange in [Exchanges.bitmex]:
        kwargs = {name: params.__getattribute__(name) for name in ['exchange', 'series_tick_type', 'asset']}
        pdf_trade = get_ohlc(
            start=params.data_start,
            end=params.data_end,
            series=Series.trade,
            **kwargs
        )
        pdf_trade, df_ohlc_mid = reduce_to_intersect_ts(pdf_trade, df_ohlc_mid)
        df_ohlc_mid['volume'] = pdf_trade['volume']
        # only NAs filled w zero. volume from tradebar remains
        df_ohlc_mid['volume'].fillna(0, inplace=True)
    else:
        Logger.info(f'No trade volume data for exchange {params.exchange}. Setting volume to zero.')
        df_ohlc_mid['volume'] = 0
    return df_ohlc_mid


def _get_mid(bid, ask):
    bid, ask = reduce_to_intersect_ts(bid, ask)
    df_ohlc_mid = pd.DataFrame([], columns=ask.columns)
    for col in OHLC:
        df_ohlc_mid[col] = np.divide(np.add(bid[col], ask[col]), 2)
    return df_ohlc_mid[OHLC]


def get_ohlcv_mid_price(params, return_ask_bid=False):
    kwargs = {name: params.__getattribute__(name) for name in ['exchange', 'series_tick_type', 'asset']}
    quote = get_ohlc(
        start=params.data_start,
        end=params.data_end,
        series=Series.quote,
        **kwargs
    )
    add_ts = lambda cols: ['ts'] if 'ts' in cols else []
    ask = quote[[c for c in quote.columns if 'ask_' in c] + add_ts(quote.columns)].rename({c: c.replace('ask_', '') for c in quote.columns}, axis='columns')
    bid = quote[[c for c in quote.columns if 'bid_' in c] + add_ts(quote.columns)].rename({c: c.replace('bid_', '') for c in quote.columns}, axis='columns')

    df_ohlc_mid = pd.concat([_get_mid(bid, ask), ask['ts']], axis=1)
    # if params.exchange in [Exchanges.bitmex]:  #.series_tick_type.type in ['ts', 'int']:
    df_ohlc_mid = _insert_trade_volume(df_ohlc_mid, params)
    # df_ohlc_mid = _transform_index(df_ohlc_mid, params.series_tick_type)
    if return_ask_bid:
        return df_ohlc_mid, ask, bid
    else:
        return df_ohlc_mid


def select_series_type(df, series):
    if series == 'trade':
        return df
    elif series in ['ask', 'ask_size']:
        return df.iloc[:, [0] + list(range(6, 11))]
    elif series in ['bid', 'bid_size']:
        return df.iloc[:, list(range(0, 6))]
    else:
        raise ValueError('Specify trade or which quote side')


def merge_qc_sec(dfd, date, series):
    dfd.columns = ['ts'] + ['bid_' + c for c in OHLC] + ['bid_size'] + ['ask_' + c for c in OHLC] + ['ask_size'] if series == series.quote else \
        ['ts'] + OHLC + ['volume']
    dfd['ts'] = dfd['ts'] // 1000 + (date - datetime.datetime(1970, 1, 1)).total_seconds()
    dfd['ts'] = dfd['ts'].apply(lambda x: datetime.datetime.utcfromtimestamp(x))
    return dfd.set_index('ts', drop=True)


def load_zipped_data(folder, start, end, series: Series, series_tick_type: SeriesTickType):
    qt_snippet = 'trade' if series in [Series.trade, None] else 'quote'
    dfs = []
    for root, dirs, filenames in os.walk(folder):
        for file in filenames:
            if qt_snippet in file:
                date = datetime.datetime.strptime(file[0:8], '%Y%m%d')
                if start <= date <= end:
                    if 'volume' in series_tick_type.type:
                        df = interpolate_volume_ticks(pd.read_csv(os.path.join(folder, file)))
                    else:
                        df = pd.read_csv(os.path.join(folder, file), header=None)
                        df = merge_qc_sec(df, date, series)
                        df = resample_qc_data(df, series, series_tick_type)
                    dfs.append(df)
    if not dfs:
        raise FileNotFoundError('DF is empty. {} No data could be loaded since start date: {}'.format(folder, start))
    if 'volume' in series_tick_type.type:
        # 0 is previous dfs last unit. and make monotonically increasing
        for i in range(1, len(dfs)):
            dfs[i].index += dfs[i-1].index[-1]
        return pd.concat(dfs).reset_index(drop=True)
    else:
        return pd.concat(dfs).sort_index()


def resample_qc_data(df: pd.DataFrame, series: Series, series_tick_type: SeriesTickType):
    # this resample missing seconds in between loaded days +/- 10 sec around midnight
    if series in [series.trade, None]:
        df = df.resample(resample_sec2resample_str(series_tick_type.resample_val)).agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    elif series in [Series.ask, Series.bid, Series.quote]:
        # todo better remode size also into OHLC. now only H
        df = df.resample(resample_sec2resample_str(series_tick_type.resample_val)).agg(
            {'bid_open': 'first', 'bid_high': 'max', 'bid_low': 'min', 'bid_close': 'last', 'bid_size': 'max',
             'ask_open': 'first', 'ask_high': 'max', 'ask_low': 'min', 'ask_close': 'last', 'ask_size': 'max'}
        )
    elif series in ['ask_size', 'bid_size']:
        df = df['volume'].resample(resample_sec2resample_str(series_tick_type.resample_val)).agg(['first', 'max', 'min', 'last', 'mean', 'count'])
    else:
        raise TypeError('series type is unknown. returning None ohlc.')
    return df.sort_index()


def get_qc_folder(exchange, asset: Assets, series_tick_type: SeriesTickType):
    try:
        return os.path.join(exchange2asset_class[exchange], exchange, resolution2folder.get(series_tick_type.folder, series_tick_type.folder), ccy2folder.get(asset.lower(), asset.lower()))
    except FileNotFoundError as e:
        Logger.error(f'{e} - Is the resolution valid and a folder exists for it?')


def interpolate_volume_ticks(pdf):
    pdf = pdf.set_index('volume_tick', drop=True)
    # 0 tick only when actually trades in there.
    # start tick is 0 only when available because previous day's last tick will added to this index.
    new_pdf = pd.DataFrame(None, columns=pdf.columns, index=range(min(pdf.index[0], 1), pdf.index[-1] + 1))
    new_pdf.loc[pdf.index] = pdf
    new_pdf['bfill'] = True
    new_pdf.loc[pdf.index, 'bfill'] = False
    return new_pdf


def pad_volume_trade(pdf):
    pdf.bfill()
    pdf.loc[pdf['bfill'] == True, 'trade_count'] = 1
    return pdf


def pad_volume_quote(pdf_qt):
    pdf_qt = pad_volume_quote_from_trades(pdf_qt, qt_pad_ix=None)
    pdf_qt = pad_volume_quote_bffill_prices_ts(pdf_qt)
    pdf_qt = pad_volume_quote_fill_defaults(pdf_qt)
    pdf_qt = pad_volume_quote_ffill_bba_size(pdf_qt, qt_pad_ix=None)
    pdf_qt = pad_volume_quote_sync_ba_size_delta(pdf_qt)
    pdf_qt = pad_volume_quote_n_eq_bba_gaps(pdf_qt, operator.eq, qt_pad_ix=None)
    pdf_qt = pad_volume_quote_n_eq_bba_gaps(pdf_qt, operator.ne, qt_pad_ix=None)
    pdf_qt = pad_volume_quote_fill_spread(pdf_qt)
    pdf_qt = pad_volume_quote_ffill_bb_sizes(pdf_qt)
    pdf_qt = pad_volume_quote_bfill_trade_prices(pdf_qt)
    pdf_qt = pad_volume_quote_fill_trade_totals(pdf_qt)

    # pd.options.display.max_columns = 20
    # pdf_qt.isna().sum()

    # add feature limit removed by subtracting trade volume from bid_size. and opposite of that represent add. info
    # whether it's hit by order order lob updates only.
    # feature hidden lob present whenever trade volume on 1 side > 0 bid delta bid_size == 0 (not opposite. lob addition
    # might have greater than market/aggressive limit order)
    # check out those hidden. see if it makes any sense...
    return pdf_qt


def pad_volume_quote_fill_trade_totals(pdf_qt):
    ix = pdf_qt.index[pdf_qt['count'].isna()]
    for c in ['count', 'volume']:
        pdf_qt.loc[ix, c] = pdf_qt.loc[ix, f'{c}_buy'] + pdf_qt.loc[ix, f'{c}_sell']
    pdf_qt.loc[ix, 'volume_mean'] = pdf_qt.loc[ix, 'volume'] / pdf_qt.loc[ix, 'count']
    return pdf_qt


def pad_volume_quote_bfill_trade_prices(pdf_qt):
    c_col_defaults = {
        'open': ('high', 'low', 'close'),
        'open': ('high', 'low', 'close')
    }
    for source_col, target_cols in c_col_defaults.items():
        b_filled = pdf_qt[source_col].fillna(method="bfill")
        pdf_qt[source_col] = b_filled
        for c in target_cols:
            pdf_qt[c] = b_filled
        # f_filled = pdf_qt[source_col].fillna(method="ffill")
        # for c in target_cols:
        #     pdf_qt[c] = f_filled
    pdf_qt['ts_trade'] = pdf_qt['ts_trade'].fillna(method="bfill")
    return pdf_qt


def pad_volume_quote_ffill_bb_sizes(pdf_qt, qt_pad_ix=None):
    qt_pad_ix = qt_pad_ix or QuoteTradePadIndices(pdf_qt)
    for neq_op in (operator.eq, operator.ne):
        for col in ['bid_size_close', 'ask_size_close']:
            qt_pad_ix.set_groups_na(col, neq_op)
            if not qt_pad_ix.dict_bba:
                continue
            ffill = pdf_qt.loc[qt_pad_ix.group_na_pm_1, col].fillna(method='ffill')
            replace = 'bid_size_' if col == 'bid_size_close' else 'ask_size_'
            for c in [replace + c for c in OHLC]:
                try:
                    pdf_qt.loc[qt_pad_ix.group_na_pm_1, c] = ffill
                except KeyError:
                    a=1
    return pdf_qt


def pad_volume_quote_fill_spread(pdf_qt):
    ix_na = pdf_qt.index[pdf_qt['spread_high'].isna()]
    pdf_qt.loc[ix_na, 'spread_high'] = pdf_qt.loc[ix_na, 'ask_high'] - pdf_qt.loc[ix_na, 'bid_low']
    pdf_qt.loc[ix_na, 'spread_low'] = pdf_qt.loc[ix_na, 'ask_low'] - pdf_qt.loc[ix_na, 'bid_high']
    return pdf_qt


def pad_volume_quote_ffill_bba_size(pdf_qt, qt_pad_ix=None):
    qt_pad_ix = qt_pad_ix or QuoteTradePadIndices(pdf_qt)
    c_bb_col_defaults = {
        'bid_size_open': ('bid_size_open', 'bid_size_high', 'bid_size_low', 'bid_size_close'),
        'ask_size_open': ('ask_size_open', 'ask_size_high', 'ask_size_low', 'ask_size_close')
    }
    for source_col, target_cols in c_bb_col_defaults.items():
        qt_pad_ix.set_groups_na(source_col, operator.eq)
        b_filled = pdf_qt.loc[qt_pad_ix.group_na_pm_1, source_col].fillna(method="ffill")
        for c in target_cols:
            pdf_qt.loc[b_filled.index, c] = b_filled
    return pdf_qt


def pad_volume_quote_trade_to_quote(pdf_qt, qt_pad_ix=None):
    qt_pad_ix = qt_pad_ix or QuoteTradePadIndices(pdf_qt)
    quote_to_trade = {
        'volume_buy': 'ask_size_removed',
        'count_buy': 'ask_size_removed_count',
        'volume_sell': 'bid_size_removed',
        'count_sell': 'bid_size_removed_count',
    }
    for src_col, target_col in quote_to_trade.items():
        pdf_qt.loc[qt_pad_ix.ix_fill_quote_from_trade, target_col] = pdf_qt.loc[qt_pad_ix.ix_fill_quote_from_trade, src_col]

    quote_to_trade_ffill = ['bid_size_' + c for c in OHLC] + ['ask_size_' + c for c in OHLC]
    ix_all_but_qt_na = np.setdiff1d(pdf_qt.index, qt_pad_ix.ix_trade_na)
    for c in quote_to_trade_ffill:
        pdf_qt.loc[ix_all_but_qt_na, c] = pdf_qt.loc[ix_all_but_qt_na, c].fillna(method='ffill')
    return pdf_qt


def pad_volume_quote_fill_defaults(pdf_qt) -> pd.DataFrame:
    quote_to_trade_defaults = {
        'bid_size_added': 0,
        'bid_size_added_count': 0,
        'ask_size_added': 0,
        'ask_size_added_count': 0
    }
    for c, val in quote_to_trade_defaults.items():
        pdf_qt.loc[pdf_qt.index[pdf_qt[c].isna()], c] = val
    return pdf_qt


def pad_volume_quote_n_eq_bba_gaps(pdf_qt: pd.DataFrame, neq_op, qt_pad_ix=None) -> pd.DataFrame:
    qt_pad_ix = qt_pad_ix or QuoteTradePadIndices(pdf_qt)
    # only_nas = functools.reduce(lambda x, y: x+y, list(dict_eq_bba.values()))
    # assert pdf_qt.loc[only_nas, 'bid_size_added'].isna().sum() == len(only_nas)
    SrcTargetOp = namedtuple('SrcTargetOp', 'src target op group_pm_col')
    interpolate_ops = [
        SrcTargetOp('volume_buy', 'ask_size_removed', 'divide', 'ask_close'),
        SrcTargetOp('count_buy', 'ask_size_removed_count', 'divide_count', 'ask_close'),
        SrcTargetOp('volume_sell', 'bid_size_removed', 'divide', 'bid_close'),
        SrcTargetOp('count_sell', 'bid_size_removed_count', 'divide_count', 'bid_close'),
        SrcTargetOp('volume_buy', 'volume_buy', 'divide', 'close'),
        SrcTargetOp('count_buy', 'count_buy', 'divide_count', 'close'),
        SrcTargetOp('volume_sell', 'volume_sell', 'divide', 'close'),
        SrcTargetOp('count_sell', 'count_sell', 'divide_count', 'close'),
        SrcTargetOp('grossValue', 'grossValue', 'divide', 'close'),
        SrcTargetOp('foreignNotional', 'foreignNotional', 'divide', 'close'),
    ]
    for src, target, op, group_pm_col in interpolate_ops:
        ix_assign = []
        ix_val = []
        qt_pad_ix.set_groups_na(target, neq_op)
        group_key_value = dict(zip(qt_pad_ix.bba_fixed, pdf_qt.loc[qt_pad_ix.bba_fixed, src]))
        for ix, val in group_key_value.items():
            indices_group = qt_pad_ix.dict_bba[ix]
            len_group = len(indices_group)
            ix_assign += indices_group
            if op == 'divide':
                ix_val += [val / len_group] * len_group
            elif op == 'divide_count':
                ix_val += ([max(val / len_group, 1)] if val > 0 else [0]) * len_group
            else:
                raise TypeError('Operation to interpolate column unknown.')
        pdf_qt.loc[ix_assign, target] = ix_val
        pdf_qt.loc[ix_assign, target] = pdf_qt.loc[ix_assign, target].astype(int)
    return pdf_qt


def pad_volume_quote_sync_ba_size_delta(pdf_qt):
    for src, target in [
        ('ask_size_removed', 'bid_size_removed'),
        ('ask_size_removed_count', 'bid_size_removed_count'),
        ('bid_size_removed', 'ask_size_removed'),
        ('bid_size_removed_count', 'ask_size_removed_count'),
    ]:
        pdf_qt[target] = pdf_qt[target].mask(pdf_qt[target].isna() & pdf_qt[src].notna(), 0)
    return pdf_qt


def pad_volume_quote_bffill_prices_ts(pdf: pd.DataFrame) -> pd.DataFrame:
    c_col_defaults = {
        'bid_open': ('bid_open', 'bid_high', 'bid_low', 'bid_close'),
        'ask_open': ('ask_open', 'ask_high', 'ask_low', 'ask_close')
    }
    for source_col, target_cols in c_col_defaults.items():
        b_filled = pdf[source_col].fillna(method="bfill")
        for c in target_cols:
            pdf[c] = b_filled

    c_col_defaults = {
        'bid_close': ('bid_open', 'bid_high', 'bid_low', 'bid_close'),
        'ask_close': ('ask_open', 'ask_high', 'ask_low', 'ask_close')
    }
    for source_col, target_cols in c_col_defaults.items():
        f_filled = pdf[source_col].fillna(method="ffill")
        for c in target_cols:
            pdf[c] = f_filled
    pdf['ts'] = pdf['ts'].fillna(method="bfill")
    pdf['ts'] = pdf['ts'].fillna(method="ffill")
    return pdf


def pad_volume_quote_from_trades(pdf: pd.DataFrame, qt_pad_ix=None) -> pd.DataFrame:
    """FILL ROWS WITH TRADE INFO YES, BUT QUOTE INFO NO"""
    qt_pad_ix = qt_pad_ix or QuoteTradePadIndices(pdf)
    pdf.loc[qt_pad_ix.ix_trade_sell_no_quote_info, ['bid_' + c for c in OHLC] + ['bid_size_removed', 'bid_size_removed_count']] = pdf.loc[qt_pad_ix.ix_trade_sell_no_quote_info, OHLC + ['volume_sell', 'sell_count']]
    pdf.loc[qt_pad_ix.ix_trade_sell_no_quote_info, ['bid_size_' + c for c in OHLC]] = 0
    pdf.loc[qt_pad_ix.ix_trade_sell_no_quote_info, ['bid_size_' + c for c in OHLC]] = 0
    pdf.loc[qt_pad_ix.ix_trade_buy_no_quote_info, ['ask_' + c for c in OHLC] + ['ask_size_removed', 'ask_size_removed_count']] = pdf.loc[qt_pad_ix.ix_trade_buy_no_quote_info, OHLC + ['volume_buy', 'buy_count']]
    pdf.loc[qt_pad_ix.ix_trade_buy_no_quote_info, ['ask_size_' + c for c in OHLC]] = 0
    pdf.loc[qt_pad_ix.ix_trade_buy_no_quote_info, ['ask_size_' + c for c in OHLC]] = 0

    pdf.loc[qt_pad_ix.ix_trade_sell_no_quote_info, 'ts'] = pdf.loc[qt_pad_ix.ix_trade_sell_no_quote_info, 'ts_trade']
    pdf.loc[qt_pad_ix.ix_trade_buy_no_quote_info, 'ts'] = pdf.loc[qt_pad_ix.ix_trade_buy_no_quote_info, 'ts_trade']
    return pdf


def get_ohlc(start, end, asset: Assets, exchange: Exchanges, series: Series, series_tick_type: SeriesTickType, **kwargs) -> pd.DataFrame:
    # need to get data from folder volume_usd_10000 or generally tick type. mix up of resolution and tick type here.
    folder = get_qc_folder(exchange, asset, series_tick_type)
    if 'volume' in series_tick_type.type:
        pdf_quote = load_zipped_data(folder, start=start, end=end, series=Series.ask, series_tick_type=series_tick_type)
        pdf_trade = load_zipped_data(folder, start=start, end=end, series=Series.trade, series_tick_type=series_tick_type)
        assert (pdf_quote.index != pdf_trade.index).sum() == 0
        pdf_trade = pdf_trade.rename({'ts': 'ts_trade', 'bfill': 'bfill_trade'}, axis='columns')
        pdf_qt = pdf_quote.merge(pdf_trade,
                                 # [['ts_trade', 'open', 'high', 'low', 'close', 'volume_buy', 'count_buy', 'volume_sell', 'count_sell', 'grossValue', 'foreignNotional']],
                                 how='left', left_index=True, right_index=True, suffixes=(None, None))
        assert len(pdf_qt) == len(pdf_quote) == len(pdf_trade)

        pdf_qt = pad_volume_quote(pdf_qt)
        pdf_quote = pdf_qt[pdf_quote.columns]
        pdf_trade = pdf_qt[pdf_trade.columns]
        pdf_trade = pdf_trade.rename({'ts_trade': 'ts', 'bfill_trade': 'bfill'}, axis='columns')
        # here filliing logic. fill the missing volume ticks. 1 to first value in df. then all in between. then reset trades.
        # resampling method. it's backward padding of ohlc. for counts ?
        # then insert delta time in microseconds
        df = pdf_trade if series in [series.trade] else pdf_quote
    else:
        # todo: think that'll fail now after refactoring because QC specific reading is messed up. not differentiating bid n ask anymore
        # todo: refactor to just reading both in at same time and return quotes, then can split up into bid ask
        df = load_zipped_data(folder, start=start, end=end, series=series, series_tick_type=series_tick_type)
        df = resample_qc_data(df, series, series_tick_type)
        # df = Qcu.inflate_low_value_assets(df, asset.lower())
        if exchange in [Exchanges.fxcm]:
            df = delete_no_trading_days(df)
        df = fill_ohlcv_nan(df)
    return _transform_index(df, series_tick_type)


def delete_no_trading_days(df):
    ts_remove = []
    for ix_start, ix_end in find_ranges(1000):  # 1 days 1440 min
        ts_remove.append((df.index[ix_start], df.index[ix_end]))
    for ts_start, ts_end in ts_remove:
        df = df.drop(df.loc[ts_start:ts_end].index)
    return df


def find_ranges(lst, n=2):
    """Return ranges for `n` or more repeated values."""
    groups = ((k, tuple(g)) for k, g in groupby(enumerate(lst), lambda x: x[-1]))
    repeated = (idx_g for k, idx_g in groups if len(idx_g) >= n)
    return ((sub[0][0], sub[-1][0]) for sub in repeated)


def fill_ohlcv_nan(df):
    for prefix in ['', 'bid_', 'ask_']:
        if prefix + 'close' in df.columns:
            df[prefix + 'close'] = df[prefix + 'close'].fillna(method='pad')
            for col in ['open', 'high', 'low']:
                df[prefix + col] = df[prefix + col].mask(pd.isna, df[prefix + 'close'])
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(value=0)
    for c in ['ask_size', 'bid_size']:
        if c in df.columns:
            df[c] = df[c].fillna(method='pad')
    return df


def digitize(pdf, cols, n_bins=20.1, def_type=np.float64, right=True) -> (pd.DataFrame, dict):
    bin_edges = {}
    for c in cols:
        max_ = np.max(pdf[c])
        min_ = np.min(pdf[c])
        bins = np.arange(min_, max_, (max_ - min_) / n_bins)
        try:
            pdf[c] = np.digitize(
                pdf[c].astype(def_type),
                bins=bins,
                right=right
            )
        except ValueError as e:
            raise ValueError('{}: - column: {} min: {}, max: {}'.format(e, c, min_, max_))
        bin_edges[c] = list(bins)
    return pdf, bin_edges
