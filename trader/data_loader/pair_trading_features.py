import datetime
import click
import importlib
import numpy as np
import os
import pandas as pd
from talib import abstract
from statsmodels.tsa.stattools import coint
import plotly.offline as py
import plotly.graph_objs as go

from trader.data_loader.utils_features import get_ohlcv_mid_price
from trader.plot.plot_func import create_fig
from common.paths import Paths
from common.modules.dotdict import Dotdict
from common.utils.decorators import time_it
from common.utils.util_func import date_sec_range, rolling_window, to_list
from common.modules.logger import logger


class PairTradingFeatures:
    """
    - provide features that have predictive power for both correlated and cointegrated assets
    - test for cointegration and correlation
    https://medium.com/auquan/pairs-trading-data-science-7dbedafcfe5a
    """
    def __init__(s, params):
        s.params = params
        s.sym_ohlc = {}

    # def run(s, ts, ref_sym, home_params):
    #     s.load_sym(s.params.asset, ts, home_param)

        # @staticmethod
        # def calc_interaction_f_nameseaction_feats_v2(foreign_sym=ref_sym.upper(), hs.p_basearsymaother: [str, lists])
        # s.load_sym(ref_sym.upper(), ts, home_params)
        # interact_feats = s.calc_interaction_feats_v2(foreign_sym=ref_sym.upper(), home_sym=s.params.asset)

    def load_sym(s, sym, ts, home_params=None):
        if home_params is None:
            params = Dotdict(dict(data_start=ts[0] - datetime.timedelta(days=1), data_end=ts[-1] + datetime.timedelta(days=1), asset=sym, exchange=s.params.exchange))
        else:
            params = Dotdict(dict(data_start=ts[0] - datetime.timedelta(days=1), data_end=ts[-1] + datetime.timedelta(days=1), asset=sym, exchange=home_params.exchange, resample_sec=home_params.resample_sec, resample_period=home_params.resample_period))
        s.sym_ohlc[sym.lower()] = get_ohlcv_mid_price(params)

    def add_sym_trends(s, sym):
        df = np.array(s.sym_ohlc[sym.lower()].values)
        inds = {}
        tp = 5
        input_params = dict(timeperiod=tp)
        inds[f'mom_{sym}_{tp}'] = getattr(abstract, 'MOM')(
                    {'open': df[:, 0], 'high': df[:, 1], 'low': df[:, 2], 'close': df[:, 3], 'volume': df[:, 4]}, **input_params)
        # input_params = dict(timeperiod1=5, timeperiod2=30, timeperiod3=60)
        # inds[f'ultosc_{sym}_5_30_60'] = getattr(abstract, 'ULTOSC')(
        #             {'open': df[:,0], 'high': df[:,1], 'low': df[:,2], 'close': df[:,3], 'volume': df[:,4]}, **input_params)
        # input_params = dict(timeperiod=14, fastk_period=5, fastd_period=3)
        # inds[f'stochrsi_{sym}_14_5_3'] = getattr(abstract, 'STOCHRSI')(
        #     {'open': df[:, 0], 'high': df[:, 1], 'low': df[:, 2], 'close': df[:, 3], 'volume': df[:, 4]}, **input_params)
        min_len = min([
            len(v) - next((i for i, x in enumerate(v) if np.isnan(x) == False), None)
            for v in inds.values()])
        return pd.DataFrame(
            np.vstack([v[-min_len:] for v in inds.values()]).transpose(),
            columns=inds.keys(), index=s.sym_ohlc[sym.lower()].index[-min_len:]
        )

    @staticmethod
    def get_sym_interaction_feat_names(params) -> list:
        features = []
        min_period = 14
        time_increases = params.pair_feats_n
        time_increase_power = params.pair_tick_increase_power
        max_delta_time = [int(min_period + i ** time_increase_power) for i in range(time_increases)][-1]
        for sym in to_list(params.pair_sym_other):
            features += [f'pair-{params.asset}-{sym}-price_ratio-{delta_time}' for delta_time in [int(min_period + i ** time_increase_power) for i in range(time_increases)]]
        logger.info(f'Pair Trading Features: {len(features)}')
        return features

    def calc_interaction_feats_v2(s, foreign_sym, home_sym):
        """
        calc change of price ratio for several time intervals.
        does not deliver price ration to models because it correlates with the price, drift over days
        etc, therefore is not normalized and produces models than dont apply to other timeframes..
        :param foreign_sym:
        :param home_sym:
        :return:
        """
        price_ratio = np.divide(s.sym_ohlc[foreign_sym]['close'], s.sym_ohlc[home_sym]['close'])
        out_arr = {}
        time_increases = 10
        time_increase_power = 2.5
        max_delta_time = [int(1 + i ** time_increase_power) for i in range(time_increases)][-1]
        for delta_time in [int(1 + i ** time_increase_power) for i in range(time_increases)]:
            out_arr[f'price_ratio_d_{delta_time}_{foreign_sym}'] = np.subtract(price_ratio[delta_time:].values, price_ratio[:-delta_time].values)
        shortest_arr_len = min([len(arr) for arr in out_arr.values()])
        out_arr = pd.DataFrame(
            np.vstack([
                arr[-shortest_arr_len:] for arr in out_arr.values()
            ]).transpose(),
            columns=out_arr.keys(), index=s.sym_ohlc[foreign_sym].index[-shortest_arr_len:]
        )
        return out_arr

    def get_pair_trade_features(s, sym, ts, home_sym, home_ohlcv, home_params=None):
        """with respect to home_sym, get a bunch of features for sym/home_sym price ratio
        different time periods for different forecasting periods. i should match with the labels
        and another set of i should match short term forecasting: optional
        for a list of time periods i get:
            rolling_mean_i
            rolling_mean_fast - rolling_mean_slow   # gives signal!
            direction & speed of price ratio's divergence or convergence
            how's it a signal. imagine btc gainer faster than eth? must also send trend info about sym:
            - MOM/price for varying i. stochRSI or ULTOSC of xbt? why send oscillators?
            since both are highly correlated, move in tandem hence better accuracy perhaps. pick i
            from chart matching my overlaying my pv.
            consider zscoring it. what's the deal
        """
        s.sym_ohlc[home_sym] = home_ohlcv
        s.load_sym(sym, ts, home_params)
        # at beginning and end of trading periods, that
        #
        # @staticmethod
        # def calc_interacame_basesesymfother: [str, listrst/last non-null signal is not at the same second across CCYs, hence leadto
        # differnt filled timestamp. reducing those now to one commong ts denominator
        # interact_feats_old = s.calc_interaction
        #
        # @staticmethod
        # def calc_interaction_f_nameseats(_feats(foreign_sym=sym, home_sym=homomme__basesysymwother: [str, listm)
        # s.sym_ohlc[home_sym], s.sym_ohlc[sym] = reduce_to_intersect_ts(s.sym_ohlc[home_sym], s.sym_ohlc[sym])
        # interact_feats = s.calc_interaction_feats_v2(foreign_sym=sym.lower(), home_sym=home_sym.lower())
        # interact_feats = s.cut_off_nan(interact_feats)
        # trend_feats = s.add_sym_trends(sym)
        # interact_feats, trend_feats = reduce_to_intersect_ts(interact_feats, trend_feats)
        # out_arr = pd.concat([interact_feats, trend_feats], axis=1)
        # return out_arr

    def cut_off_nan(s, pdf: pd.DataFrame):
        ix_keep_start = np.argmin(pdf.isna().values, axis=0).max()
        ix_keep_end = np.argmin(pdf.sort_values(by='ts', ascending=False).isna().values, axis=0).max()
        if ix_keep_end == 0:
            return pdf.iloc[ix_keep_start:]
        else:
            return pdf.iloc[ix_keep_start:-ix_keep_end]

    def test_coint(s, foreign_sym, home_sym):
        score, pvalue, _ = coint(s.sym_ohlc[foreign_sym]['close'], s.sym_ohlc[home_sym]['close'])
        print(pvalue)

    def test_correlation(s, foreign_sym, home_sym):
        score_corr = np.corrcoef(s.sym_ohlc[foreign_sym]['close'], s.sym_ohlc[home_sym]['close'])
        print(score_corr)

    def plot_price_ratios(s, foreign_sym, home_sym, win_len=[1, 5, 60, 300, 600]):
        traces = {(1, 1): []}
        for i in win_len:
            y = s.mavg(i, foreign_sym, home_sym)
            traces[(1, 1)].append(
                go.Scatter(name=i,
                           x=s.sym_ohlc[foreign_sym].index[-len(y):],
                           y=y)
            )
        fig = create_fig(traces)
        py.plot(fig,
                filename=os.path.join(Paths.projectDir, 'test.html'),
                auto_open=True)

    def plot_zscore(s):
        pass

    def mavg(s, win_len, foreign_sym, home_sym):
        s.price_ratio = np.divide(s.sym_ohlc[foreign_sym]['close'], s.sym_ohlc[home_sym]['close'])
        return rolling_window(s.price_ratio, window=win_len).mean(axis=1)


@time_it
def main(ctx):
    params = ctx.obj.params
    ts = list(date_sec_range(params.data_start, params.data_end))
    inst = PairTradingFeatures(params)
    inst.run(ts, ref_sym='gbpusd', home_params=params)


@click.command()
@click.pass_context
def main(ctx):
    ctx.obj = Dotdict(dict(
        fn_params='forex',
        fn_settings='settings'
    ))
    params = importlib.import_module('{}.{}'.format(Paths.path_config_supervised, ctx.obj.fn_params)).Params()
    params.data_start = datetime.datetime(2018, 3, 2, 0, 0, 0)
    params.data_end = datetime.datetime(2019, 3, 3, 23, 59, 59)
    params.asset = ['BTCUSD', 'ETHUSD', 'XRPXBT', 'LTCXBT', 'XBTUSD', 'XRPUSD', 'EURUSD'][-1]
    params.exchange = 'FXCM'
    params.resample_sec = 60
    params.resample_period = '60S'
    ctx.obj.params = params
    ctx.forward(main)


if __name__ == '__main__':
    main()
