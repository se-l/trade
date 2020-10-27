import os
import pickle
import re

import datetime
from talib import abstract
from scipy.signal import find_peaks
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
from trader.backtest.order import Order
from common.modules import dotdict
from common.modules import order_type
from common.modules import DataStore
from common import Paths
from trader.raw_data_fetcher.load_features import LoadFeatures
from common.utils.util_func import date_sec_range, df_to_npa, reduce_to_intersect_ts, insert_nda_col, to_list
from common.modules import timing, direction, side, the
from connector import InfluxClientWrapper as Influx
from common.utils.util_func import make_struct_nda, join_struct_arrays
from trader.backtest.vectorized_backtest_base import VectorizedBacktestBase
from trader.backtest.stops_finder import StopsFinder


class Backtest(VectorizedBacktestBase, StopsFinder):

    def __init__(s, params):
        super().__init__()
        s.lud = None
        s.params = params
        s.data = DataStore()
        s.df_full = None
        s.strat_entry = {}
        s.arr = None
        s.ix_entry_preds = []
        s.ex_rl_model = params.ex_rl_model
        s.rl_model_ids = params.rl_model_ids
        s.rl_models = []
        s.orders: [Order] = []
        s.future_orders = None
        s.ex = params.ex
        s.ts_start = params.ts_start
        s.ts_end = params.ts_end
        s.influx = Influx()
        s.assume_late_limit_fill = True  # QC alignment when True
        s.assume_late_limit_fill_entry = True  # QC alignment when True
        s.use_simply_limit_price_at_fill = False  # QC alignment when False
        s.asset = params.asset
        s.ini_dic = {}
        s.sym_dic = {}
        s.regr = {}
        s.chunk = ['ho']
        s.fn_p_exec = 'p_exec_{}.json'.format('-'.join([str(i) for i in s.chunk]))
        s.preds_cols = ['ts', 'long', 'short']
        s.preds_exit_cols = ['ts', 'long_exit', 'short_exit']
        s.talib_cols = ['EMA_real_540', 'MOM_real_23', 'MOM_real_360']
        # s.trailing_stop_b = s.p_opt['trailing_stop_a'] / s.p_opt['profit_target']

    def setup(s):
        # if not True:
        #     vbm.exitModelFromEntry = ExitModelFromGivenEntry()
        #     vbm.exitModelFromEntry.load_models('model_class_lgb')
        #     from trader.entry_model.load_features import LoadFeatures
        #     req_feats = ['ULTOSC_real_tp15250_tp210500_tp321000', 'WILLR_real_8442', 'WILLR_real_4914', 'ULTOSC_real_tp18190_tp216380_tp332760', 'CMO_real_798', 'APO_real_fp9504_sp20592', 'AROON_aroondown_3738', 'WILLR_real_1092', 'APO_real_fp10512_sp22776', 'AROON_aroondown_8148', 'MOM_real_5820', 'WILLR_real_9618', 'ADOSC_real_fp2502_sp8340', 'AROON_aroonup_17556', 'AROON_aroondown_2856', 'CMO_real_4032', 'AROON_aroondown_17556', 'PPO_real_fp10512_sp22776', 'WILLR_real_7560', 'WILLR_real_1974', 'PPO_real_fp12780_sp27690', 'MOM_real_10230', 'AROON_aroonup_13146', 'APO_real_fp14796_sp32058', 'AROON_aroonup_8736', 'APO_real_fp7488_sp16224', 'AROON_aroonup_17262', 'ULTOSC_real_tp17602_tp215204_tp330408', 'PPO_real_fp15300_sp33150', 'PPO_real_fp13788_sp29874', 'AROON_aroonup_9030', 'APO_real_fp5220_sp11310', 'CMO_real_9618', 'AROON_aroondown_7266', 'WILLR_real_13146', 'PPO_real_fp1440_sp3120', 'PPO_real_fp7992_sp17316', 'APO_real_fp3456_sp7488', 'AROON_aroonup_9324', 'AROON_aroondown_9618', 'ULTOSC_real_tp18043_tp216086_tp332172', 'PPO_real_fp2952_sp6396', 'WILLR_real_2268', 'WILLR_real_798', 'APO_real_fp432_sp936', 'APO_real_fp15300_sp33150', 'PPO_real_fp12024_sp26052', 'AROON_aroondown_7560', 'APO_real_fp9000_sp19500', 'PPO_real_fp5220_sp11310', 'WILLR_real_8736', 'ULTOSC_real_tp18778_tp217556_tp335112', 'AROON_aroondown_14616', 'WILLR_real_17850', 'MOM_real_6870', 'WILLR_real_2856', 'WILLR_real_7266', 'PPO_real_fp11772_sp25506', 'CMO_real_14322', 'PPO_real_fp1692_sp3666', 'WILLR_real_3150', 'AROON_aroonup_16968', 'AROON_aroonup_10206', 'ULTOSC_real_tp14074_tp28148_tp316296', 'WILLR_real_1386', 'AROON_aroonup_15204', 'PPO_real_fp13536_sp29328', 'WILLR_real_14910', 'MOM_real_8760', 'PPO_real_fp14040_sp30420', 'APO_real_fp13284_sp28782', 'AROON_aroondown_15498', 'AROON_aroonup_11676', 'WILLR_real_3444', 'PPO_real_fp14292_sp30966', 'WILLR_real_504', 'AROON_aroonup_2856', 'PPO_real_fp13032_sp28236', 'APO_real_fp11016_sp23868', 'APO_real_fp7740_sp16770', 'AROON_aroonup_12852', 'ADOSC_real_fp3447_sp11490', 'ULTOSC_real_tp16426_tp212852_tp325704', 'APO_real_fp14040_sp30420', 'WILLR_real_210', 'CMO_real_16674', 'AROON_aroondown_6090', 'PPO_real_fp11268_sp24414', 'AROON_aroonup_12264', 'WILLR_real_12852', 'WILLR_real_5502', 'ULTOSC_real_tp18925_tp217850_tp335700', 'PPO_real_fp12528_sp27144', 'PPO_real_fp1944_sp4212', 'CMO_real_6972', 'WILLR_real_16968', 'APO_real_fp14544_sp31512', 'WILLR_real_2562', 'AROON_aroondown_16086', 'AROON_aroonup_13440', 'MOM_real_9600', 'PPO_real_fp10260_sp22230', 'WILLR_real_6090', 'AROON_aroonup_5208', 'MOM_real_7500', 'AROON_aroondown_6972']
        #     params_LF = copy.copy(params)
        #     params_LF.data_start = ts_start - datetime.timedelta(days=3)
        #     params_LF.data_end = ts_end
        #     print('Load talib features...')
        #     vbm.dfFull, _ = LoadFeatures(params_LF, dotdict(dict(ex='ex2019-04-01_10-10-10-ethusd-exitforecast')), req_feats=req_feats, use_midPrice=True).load()
        #     vbm.dfFull.index = _['ts'].values
        #     vbm.dfFull.index.name = 'ts'
        # vbm.tickForecast = TickForeCast(s.params)
        # vbm.qt = vbm.tickForecast.load_qt_db(s.params)
        # vbm.tickForecast_raw_preds = vbm.tickForecast.load_raw_preds_db(ts_start=s.ts_start, ts_end=s.ts_end)
        # vbm.load_ohlc()
        s.ohlc_mid = s.ohlc = LoadFeatures.get_ohlcv_mid_price(s.params, index='ts')
        s.ix_close = s.ohlc_mid.columns.get_loc('close')
        s.ix_high = s.ohlc_mid.columns.get_loc('high')
        s.ix_open = s.ohlc_mid.columns.get_loc('open')
        s.ix_low = s.ohlc_mid.columns.get_loc('low')
        s.ohlc_ask = LoadFeatures.get_ohlc(
            start=s.params.data_start,
            end=s.params.data_end,
            asset=s.params.asset,
            exchange=s.params.exchange,
            series='ask',
            index='ts',
            res=s.params.resample_period
        )
        s.ohlc_bid = LoadFeatures.get_ohlc(
            start=s.params.data_start,
            end=s.params.data_end,
            asset=s.params.asset,
            exchange=s.params.exchange,
            series='ask',
            index='ts',
            res=s.params.resample_period
        )
        s.extract_inds_sql()
        match_ts = np.intersect1d(pd.to_datetime(s.ts), s.ohlc.index, assume_unique=True)
        ohlc_ts_filter = np.isin(pd.to_datetime(s.ohlc.index), match_ts)

        for ohlc_object in ['ohlc', 'ohlc_bid', 'ohlc_ask']:
            s.__setattr__(ohlc_object, s.__getattribute__(ohlc_object).loc[ohlc_ts_filter])
            s.__getattribute__(ohlc_object)['ts'] = s.__getattribute__(ohlc_object).index
            s.__setattr__(ohlc_object, s.__getattribute__(ohlc_object).reset_index(drop=True))
        s.cross_val_index = s.ohlc.index

    def set_strategy_lib(s, strategy_lib):
        s.strategy_lib = strategy_lib

    def invalidating_future_orders(s, new_orders: list):
        if new_orders.__len__() == 0 or s.future_orders.__len__() == 0:
            return
        # check if this entry invalidates any exit orders
        rm = []
        for o in new_orders:
            for i in range(len(s.future_orders)):
                if o.ix_signal < s.future_orders[i].ix_signal and o.direction == s.future_orders[i].direction:
                    # there are only 2 directions, hence equality works
                    # a long/short entry preceded a short/long exit -> hence 1 exit will not occur
                    rm.append(i)
        rm.reverse()
        for i in rm:
            del s.future_orders[i]

    def get_talib_inds(s, ohlc, ind):
        keys = ind.split('_real_')
        f = keys[0]
        # input_params = getattr(abstract, f)._Function__opt_inputs.keys()
        # out_names = getattr(abstract, f).output_names
        # params = getattr(abstract, f).get_parameters()
        # matype = getattr(abstract, f).get_parameters()['matype']
        input_params = dict(timeperiod=int(keys[1]))
        out_val = getattr(abstract, f)({
            'open': ohlc.iloc[:, s.ix_open],
            'high': ohlc.iloc[:, s.ix_high],
            'low': ohlc.iloc[:, s.ix_low],
            'close': ohlc.iloc[:, s.ix_close]},
            **input_params)
        np.nan_to_num(out_val, 0)
        # make relative
        return out_val
        # arr.append(out_val)
        # return np.vstack(arr).transpose()

    def calc_bband(s, ohlc, strategy):
        f = 'BBANDS'
        input_params = getattr(abstract, f)._Function__opt_inputs.keys()
        # out_names = getattr(abstract, f).output_names
        # params = getattr(abstract, f).get_parameters()
        if strategy.bband_matype != 0:
            matype = strategy.bband_matype
        else:
            matype = getattr(abstract, f).get_parameters()['matype']
        input_params = dict(timeperiod=strategy.bband_tp,
                            nbdevup=strategy.bband_nbdevup,
                            nbdevdn=strategy.bband_nbdevdn,
                            matype=matype)
        upperband, middleband, lowerband = getattr(abstract, f)({'open': ohlc.iloc[:, s.ix_open],
                                                                 'high': ohlc.iloc[:, s.ix_high],
                                                                 'low': ohlc.iloc[:, s.ix_low],
                                                                 'close': ohlc.iloc[:, s.ix_close]},
                                                                **input_params)
        ix_first_non_nan = sum(np.isnan(middleband))
        upperband[:ix_first_non_nan] = upperband[ix_first_non_nan]
        middleband[:ix_first_non_nan] = middleband[ix_first_non_nan]
        lowerband[:ix_first_non_nan] = lowerband[ix_first_non_nan]
        return upperband, middleband, lowerband

    @staticmethod
    def preds_smooth(preds, strategy):
        f = strategy.preds_smooth_f
        input_params = getattr(abstract, f)._Function__opt_inputs.keys()
        # out_names = getattr(abstract, f).output_names
        # params = getattr(abstract, f).get_parameters()
        input_params = dict(timeperiod=strategy.preds_smooth_tp)
        middleband = getattr(abstract, f)({'close': preds.astype(np.float)}, **input_params)
        ix_first_non_nan = sum(np.isnan(middleband))
        middleband[:ix_first_non_nan] = middleband[ix_first_non_nan]
        return middleband

    def process_proposed_orders(s, new_orders: list) -> list:
        """strategies can submit orders in every time index
        all entries are accepted (for now / 2 entries on same asset is soso..)
        exits, reversals can be superseded if a strategies entry invests in the same direction as another exit
        difference of this compared to QC are the exits (they look ahead in time)

        since exit orders are look-ahead orders, they shall be future until the entry_ix loop is past their
        signal, then confirm / append.

        exit orders become invalidated when an entry order for the same asset (long vs short algo) precedes the exit
        in time
        """
        if new_orders.__len__() == 0:
            return []
        for o in new_orders:
            if o.timing == timing.entry:
                s.orders.append(o)
            elif o.timing == timing.exit:
                s.future_orders.append(o)
        return []

    def process_proposed_orders_invalidate_future(s, new_orders: list) -> list:
        if s.future_orders.__len__() > 0:
            # check if this entry invalidates any exit orders
            rm = []
            for o in new_orders:
                for i in range(len(s.future_orders)):
                    if o.ix_signal < s.future_orders[i].ix_signal or \
                            o.ix_signal < s.future_orders[i].fill.ix_fill:  # and o.direction == s.future_orders[i].direction:
                        # there are only 2 directions, hence equality works
                        # a long/short entry preceded a short/long exit -> hence 1 exit will not occur
                        rm.append(i)
            rm = list(set(rm))
            rm.reverse()
            for i in rm:
                del s.future_orders[i]

        for o in new_orders:
            s.future_orders.append(o)
        return []

    def confirm_future_orders(s, ix_curr):
        if s.future_orders.__len__() == 0:
            return
        # check if any exits can be confirmed
        rm = []
        for i in range(len(s.future_orders)):
            if s.future_orders[i].fill.ix_fill < ix_curr:
                s.orders.append(s.future_orders[i])
                rm.append(i)
        rm.reverse()
        for i in rm:
            del s.future_orders[i]

    def filter_peak_valley_ratio(s, stop_ix, strategy):
        for ix in stop_ix:
            if strategy.direction == direction.long and \
                    s.arr[ix, s.lud.fit_savgol_long] / s.arr[0, s.lud.fit_savgol_long] < strategy.min_peak_valley_ratio:
                return ix
            elif strategy.direction == direction.short and \
                    s.arr[ix, s.lud.fit_savgol_short] / s.arr[0, s.lud.fit_savgol_short] < strategy.min_peak_valley_ratio:
                return ix
            else:
                continue
        return 0

    def order_timed_out(s, order, strategy):
        if order.fill.ix_fill - order.ix_signal > strategy.time_entry_cancelation:
            return True
        else:
            return False

    def order_vetoed(s, ix_entry, strategy):
        if strategy.direction == direction.short:
            if s.regr_ens_veto[strategy.id][ix_entry] > strategy.regr_veto_stop:  # the close preds are the same for each asset independent of long or short
                return True
        elif strategy.direction == direction.long:
            if s.regr_ens_veto[strategy.id][ix_entry] < strategy.regr_veto_stop:  # the close preds are the same for each asset independent of long or short
                # s.stats['ts_entry_regr_vetoed'].append(s.sym_dic['idx_ts'].index[ix_entry])
                return True
        else:
            return False

    @staticmethod
    def is_signal(signal_direction, portf_side):
        if portf_side == side.hold:
            return True
        else:
            # when portfolio already long, enter another long
            # had problem that stops triggered a portf_side of long.
            return signal_direction != portf_side

    @staticmethod
    def tv_ho_to_str(tv_ho):
        if tv_ho == the.ho:
            return 'ho'
        elif tv_ho == the.tv:
            return 'tv'
        elif tv_ho == the.extra:
            return 'extra'
        else:
            raise ('Unkown data split. Expecting train, holdout, extra')

    def extract_inds_sql(s):
        ohlc_mid = LoadFeatures.get_ohlcv_mid_price(s.params, index='ts')
        talibs = np.vstack([s.get_talib_inds(ohlc_mid, ind=ind) for ind in s.talib_cols]).transpose()
        # preds = s.load_preds_from_db(tbl='model_preds')
        # s.data['regr'] = s.load_entry_predictions_from_db([6, 7, 8, 9, 10, 11])
        # features = s.load_entry_predictions_from_db([0, 1, 2, 3, 6, 7, 8, 9, 10, 11])
        # features_post_ls = [Features.y_post_valley, Features.y_post_peak]
        # features_pre_ls = [Features.y_pre_valley, Features.y_pre_peak]
        preds = s.load_entry_predictions_from_db()
        new_col = []
        for c in preds.columns:
            if re.search('dwin-\d', c):
                new_col.append('p_long')
            elif re.search('dwin--\d', c):
                new_col.append('p_short')
        preds.columns = new_col
        preds, s.ohlc_mid, talibs = reduce_to_intersect_ts(preds, s.ohlc_mid, pd.DataFrame(talibs, index=s.ohlc_mid.index))
        talibs = talibs.values
        # pre_post reversed assignment. in lgb fl train targeting the pre preds...
        # preds = features[:, [0, 2, 4]]
        # preds_pre = features[:, [1, 3]]
        # correct assignment # +1 due to time stamp
        # preds = features.values  #features[:, [0] + [el + 1 for el in features_post_ls]]
        # preds_pre = features[:, [el + 1 for el in features_pre_ls]]
        # s.data['regr'] = features[:, list(range(5, features.shape[1]))]
        if len(talibs) != len(preds):
            print('Len of curves and preds is not equal...Calc may take a while...')
            print('Missing ts in preds: {}'.format(np.setdiff1d(talibs[:, 0], preds.index, assume_unique=True)))
            # [preds[i,0] for i in range(len(preds[:, 0])) if i> 1 and preds[i, 0] != (preds[i-1,0] + datetime.timedelta(seconds=1))]
            print('Missing ts in curves: {}'.format(np.setdiff1d(preds[:, 0], preds.index, assume_unique=True)))
        assert len(talibs) == len(preds), 'curves and preds dont have the same length. Cannot merge safely'
        # transform to zscore
        # window = 3600
        # for i in range(1, 3):
        #     mean = np.mean(rolling_window(preds[:, 1], window), axis=1)
        #     std = np.std(np.array(rolling_window(preds[:, 1], window)), axis=1, dtype=np.float64)
        #     preds[window - 1:, i] = np.divide(np.subtract(preds[window-1:, i], mean), std)
        #     preds[:window-1, i] = 0
        s.ts = preds.index
        s.data['ohlc_mid'] = ohlc_mid.loc[s.ts]

        # tick_forecast = TickForeCast(s.params)
        # tick_forecast.load_raw_preds_db(ts_start=s.ts[0], ts_end=s.ts[-1])

        # ticks = make_struct_nda(, tick_forecast.raw_preds.columns)
        print('Concatenating loaded data into curves...')
        # s.data['inds'] = np.concatenate([inds, preds[:, 1:], tick_forecast.raw_preds.values], axis=1)
        # s.data['inds'] = np.concatenate([inds, preds[:, 1:], np.zeros(shape=(len(inds), 3))], axis=1)
        # s.data['inds'] = make_struct_nda(s.data['inds'], s.inds_cols + ['long', 'short'] + list(tick_forecast.raw_preds.columns))
        s.data['curve'] = join_struct_arrays([
            make_struct_nda(talibs, s.talib_cols),
            df_to_npa(preds)
        ])
        assert len(s.data['curve']) == len(s.data['ohlc_mid']), 'ohld mid curve length not matching'
        # s.data['curve'] = join_struct_arrays([s.data['curve'], ticks])
        # s.preds_short_mean = s.data['curve']['short']
        # s.preds_long_mean = s.data['curve']['long']

    @staticmethod
    def slice_data_by_ts(data, ts_start, ts_end):
        if type(data) == pd.DataFrame:
            return data.loc[ts_start:ts_end, :]
        elif type(data) == np.ndarray:
            return data[ts_start:ts_end, :]
        else:
            raise('Type not known')

    @staticmethod
    def add_timeperiod(ix_5, time_delta):
        return [ix_5 + i for i in range(0, time_delta)]

    def add_ix_entry_tp(s, ix_entry_preds_mom10, time_delta):
        ix_entry_preds_precursor = []
        for ix_5 in ix_entry_preds_mom10:
            ix_entry_preds_precursor += s.add_timeperiod(ix_5, time_delta)
        return np.sort(list(set(ix_entry_preds_precursor)))

    def get_entry_ix(s, strategies):
        if 'EMA_540_300d_rel' not in s.data['curve'].dtype.names:
            print('Initial indicator calculations...')
            ema_540_300d = np.zeros_like(s.data['curve']['EMA_real_540'])
            i = 300
            ema_540_300d[i:] = s.data['curve']['EMA_real_540'][i:] - s.data['curve']['EMA_real_540'][:-i]
            # preds_net_d1 = np.zeros_like(s.data['curve']['net'])
            # preds_net_d1[1:] = np.subtract(s.data['curve']['net'][1:], s.data['curve']['net'][:-1])
            preds_short_d1 = np.zeros_like(s.data['curve']['p_short'])
            preds_short_d1[1:] = np.subtract(s.data['curve']['p_short'][1:], s.data['curve']['p_short'][:-1])
            preds_long_d1 = np.zeros_like(s.data['curve']['p_long'])
            preds_long_d1[1:] = np.subtract(s.data['curve']['p_long'][1:], s.data['curve']['p_long'][:-1])
            preds_short_d30 = np.zeros_like(s.data['curve']['p_short'])
            preds_short_d30[30:] = np.subtract(s.data['curve']['p_short'][30:], s.data['curve']['p_short'][:-30])
            preds_long_d30 = np.zeros_like(s.data['curve']['p_long'])
            preds_long_d30[30:] = np.subtract(s.data['curve']['p_long'][30:], s.data['curve']['p_long'][:-30])

            s.data['curve'] = join_struct_arrays([
                s.data['curve'],
                make_struct_nda(np.divide(ema_540_300d, s.data['ohlc_mid']['close']), cols=['EMA_540_300d_rel'], def_type=np.float64),
                make_struct_nda(np.divide(s.data['curve']['MOM_real_23'], s.data['ohlc_mid']['close']), cols=['MOM_real_23_rel'], def_type=np.float64),
                make_struct_nda(np.divide(s.data['curve']['MOM_real_360'], s.data['ohlc_mid']['close']), cols=['MOM_real_360_rel'], def_type=np.float64),
                make_struct_nda(preds_short_d1, cols=['p_short_d1'], def_type=np.float64),
                make_struct_nda(preds_long_d1, cols=['p_long_d1'], def_type=np.float64),
                make_struct_nda(preds_short_d30, cols=['p_short_d30'], def_type=np.float64),
                make_struct_nda(preds_long_d30, cols=['p_long_d30'], def_type=np.float64),
                # make_struct_nda(preds_net_d1, cols=['preds_net_d1'], def_type=np.float64),

                # make_struct_nda(sav_poly_long_peak, cols=['sav_poly_long_peak'], def_type=np.float64),
                # make_struct_nda(sav_poly_long_valley, cols=['sav_poly_long_valley'], def_type=np.float64),
                # make_struct_nda(sav_poly_short_peak, cols=['sav_poly_short_peak'], def_type=np.float64),
                # make_struct_nda(sav_poly_short_valley, cols=['sav_poly_short_valley'], def_type=np.float64),
            ])
            try:
                for i in range(s.data['regr'].shape[1]):
                    s.data['regr'][:, i] = np.add(
                        s.ohlc['close'],
                        np.multiply(s.data['regr'][:, i], s.ohlc['close'])
                    )
            except KeyError:
                pass

        print('selecting entry points...')
        for strategy in strategies:
            if strategy.direction == direction.short:
                # above_middle_bband = np.where(s.data['curve']['bband_20_2_2_middle'] <= s.ohlc['close'])[0]
                # above_middle_bband = np.where(s.data['curve']['bband_20_2_2_middle'] <= s.ohlc['close'])[0]
                # cross_bband = np.where(s.data['curve']['bband_upper_s'] <= s.ohlc['high'] + strategy.bband_entry_delta)[0]
                # bullish_range = np.where(s.data['curve']['ema_180_1d'] >= strategy.bullish_range)[0]
                extr_bull_bear_stretch = np.where(s.data['curve']['EMA_540_300d_rel'] >= strategy.bull_bear_stretch)[0]
                # volatility_thresh = np.where(np.subtract(s.data['curve']['bband_20_2_2_upper'], s.data['curve']['bband_20_2_2_lower']) <= strategy.min_bband_volat)[0]
                # bullish_range = np.where(s.data['curve']['ema_180'] <= s.ohlc['close'])[0]
                # mom_extreme_mom_60 = np.where(s.data['curve']['mom_60'] <= strategy.mom_extreme_mom_60)[0]
                # min_mom_3h = np.where(s.data['curve']['mom_10800'] >= strategy.min_mom_300)[0]
                ix_rebounds = np.where(s.data['curve']['MOM_real_360_rel'] <= strategy.rebound_mom)[0]
                # min_bband_volat = np.where(np.subtract(s.data['curve']['bband_upper_s'], s.data['curve']['bband_lower_s']) < strategy.min_bband_volat)[0]

                # ix_entry_preds_mom300 = np.where(s.data['curve']['mom_300'] >= strategy.min_mom_300)[0]
                # ix_entry_preds_mom10 = np.where(s.data['curve']['mom_10'] <= strategy.min_mom_10)[0]
                # ix_entry_preds_mom10 = s.add_ix_entry_tp(ix_entry_preds_mom10, strategy.mom_short_long_tp)
                # ix_entry_preds_mom60 = np.where(s.data['curve']['mom_60'] <= strategy.min_mom_60)[0]
                # ix_short_exit_preds = np.where(s.data['curve']['short_exit'] >= strategy.veto_p_exit)[0]
                # ix_preds_entry_smooth_short_entry_dx = np.where(
                #     (s.data['curve']['short'] > strategy.preds_sl_thresh_dx) &
                #     (s.data['curve']['preds_smooth_short_dx'] < 0)
                # )[0]
                ix_preds_entry_short_entry = np.where(s.data['curve']['p_short'] > strategy.preds_sl_thresh)[0]
                # ix_preds_entry_short_d1 = np.where(s.data['curve']['preds_short_d1'] > 0)[0]
                # ix_preds_entry_short_entry = np.where(s.data['preds_pre']['p_short_pre'] > strategy.preds_sl_thresh)[0]
                ix_entry_preds = ix_preds_entry_short_entry

                # ix_entry_preds = np.where(s.data['curve']['net'] <= strategy.preds_net_thresh)[0]
                # find_peak_entries = s.filter_low_value_opp(find_peak_entries, strategy)
                # ix_entry_preds = np.union1d(ix_entry_preds, np.where(s.data['curve']['sav_poly_short_peak'] == 1)[0])
                # ix_entry_preds = np.union1d(ix_entry_preds, np.where(s.data['curve']['sav_poly_long_valley'] == 1)[0])
                # ix_entry_preds = np.union1d(ix_entry_preds, ix_preds_entry_smooth_short_entry_dx)
                # ix_entry_preds = np.intersect1d(ix_preds_entry_short_entry, ix_preds_entry_short_d1)
                # ix_entry_preds = np.union1d(ix_entry_preds, cross_bband)
                # if not strategy.assume_simulated_future:
                #     ix_entry_preds_d1 = np.where(preds_net_d1 <= 0)[0]
                #     ix_entry_preds = np.setdiff1d(ix_entry_preds, ix_entry_preds_d1)
                # ix_entry_preds = np.setdiff1d(ix_entry_preds, min_bband_volat)

                # ix_entry_preds = np.intersect1d(ix_entry_preds_mom10, ix_entry_preds_mom60)
                # ix_entry_preds = np.intersect1d(ix_entry_preds, ix_entry_preds_mom300)
                # ix_entry_preds = np.setdiff1d(ix_entry_preds, volatility_thresh)
                # ix_entry_preds = np.setdiff1d(ix_entry_preds, ix_rebounds)
                # ix_entry_preds = np.intersect1d(ix_entry_preds, min_preds_net)
                # ix_entry_preds = np.setdiff1d(ix_entry_preds, bullish_range)
                # ix_entry_preds = np.union1d(ix_entry_preds, mom_extreme_mom_60)
                ix_entry_preds = np.setdiff1d(ix_entry_preds, extr_bull_bear_stretch)
                # ix_entry_preds = np.setdiff1d(ix_entry_preds, ix_short_exit_preds)
                # ix_entry_preds = np.intersect1d(ix_entry_preds, min_mom_3h)
                # ix_entry_preds = np.union1d(ix_entry_preds, bearish_range_ema)
                ix_entry_preds = np.unique(ix_entry_preds)
                s.strat_entry[strategy.id] = ix_entry_preds
                print('Short entries: {}'.format(len(ix_entry_preds)))
            elif strategy.direction == direction.long:
                # ix_entry_preds_mom300 = np.where(s.data['curve']['mom_300'] <= strategy.min_mom_300)[0]
                # ix_entry_preds_mom10 = np.where(s.data['curve']['mom_10'] <= strategy.min_mom_10)[0]
                # ix_entry_preds_mom10 = s.add_ix_entry_tp(ix_entry_preds_mom10, strategy.mom_short_long_tp)
                # cross_bband = np.where(s.data['curve']['bband_lower_l'] >= s.ohlc['low'] + strategy.bband_entry_delta)[0]
                # ix_entry_preds_mom60 = np.where(s.data['curve']['mom_60'] >= strategy.min_mom_60)[0]
                ix_rebounds = np.where(s.data['curve']['MOM_real_360_rel'] >= strategy.rebound_mom)[0]
                # min_bband_volat = np.where(np.subtract(s.data['curve']['bband_upper_l'], s.data['curve']['bband_lower_l']) < strategy.min_bband_volat)[0]

                # bullish_range = np.where(s.data['curve']['ema_180'] <= s.ohlc['close'])[0]
                extr_bull_bear_stretch = np.where(s.data['curve']['EMA_540_300d_rel'] <= strategy.bull_bear_stretch)[0]
                # ix_long_exit_preds = np.where(s.data['curve']['long_exit'] >= strategy.veto_p_exit)[0]
                # ix_preds_entry_smooth_long_entry_dx = np.where(
                #     (s.data['curve']['long'] > strategy.preds_sl_thresh_dx) &
                #     (s.data['curve']['preds_smooth_long_dx'] < 0)
                # )[0]

                ix_preds_entry_long_entry = np.where(s.data['curve']['p_long'] > strategy.preds_sl_thresh)[0]
                # ix_preds_entry_long_entry = np.where(s.data['curve']['long'] > strategy.preds_sl_thresh)[0]
                # ix_preds_entry_long_d1 = np.where(s.data['curve']['preds_long_d1'] > 0)[0]
                ix_entry_preds = ix_preds_entry_long_entry
                # ix_entry_preds = np.intersect1d(ix_preds_entry_long_entry, ix_preds_entry_long_d1)
                # mom_extreme_mom_60 = np.where(s.data['curve']['mom_60'] >= strategy.mom_extreme_mom_60)[0]
                # below_middle_bband = np.where(s.data['curve']['bband_20_2_2_middle'] >= s.ohlc['close'])[0]
                # volatility_thresh = np.where(np.subtract(s.data['curve']['bband_20_2_2_upper'], s.data['curve']['bband_20_2_2_lower']) <= strategy.min_bband_volat)[0]
                # ix_entry_preds = np.where(s.data['curve']['net'] >= strategy.preds_net_thresh)[0]
                # find_peak_entries = s.filter_low_value_opp(find_peak_entries, strategy)
                # ix_entry_preds = np.union1d(ix_entry_preds, np.where(s.data['curve']['sav_poly_long_peak'] == 1)[0])
                # ix_entry_preds = np.union1d(ix_entry_preds, np.where(s.data['curve']['sav_poly_short_valley'] == 1)[0])
                # ix_entry_preds = np.union1d(ix_entry_preds, ix_preds_entry_smooth_long_entry_dx)
                # ix_entry_preds = np.union1d(ix_entry_preds, ix_preds_entry_long_entry)
                # ix_entry_preds = np.union1d(ix_entry_preds, cross_bband)
                # if not strategy.assume_simulated_future:
                #     ix_entry_preds_d1 = np.where(preds_net_d1 >= 0)[0]
                #     ix_entry_preds = np.setdiff1d(ix_entry_preds, ix_entry_preds_d1)
                # ix_entry_preds = np.setdiff1d(ix_entry_preds, min_bband_volat)

                # ix_entry_preds = np.intersect1d(ix_entry_preds_mom10, ix_entry_preds_mom60)
                # ix_entry_preds = np.intersect1d(ix_entry_preds, below_middle_bband)
                ix_entry_preds = np.setdiff1d(ix_entry_preds, ix_rebounds)
                # ix_entry_preds = np.intersect1d(ix_entry_preds, min_preds_net)
                # ix_entry_preds = np.intersect1d(ix_entry_preds, bullish_range)
                ix_entry_preds = np.setdiff1d(ix_entry_preds, extr_bull_bear_stretch)
                # ix_entry_preds = np.setdiff1d(ix_entry_preds, ix_long_exit_preds)
                # ix_entry_preds = np.union1d(ix_entry_preds, bullish_range_ema)
                ix_entry_preds = np.unique(ix_entry_preds)
                s.strat_entry[strategy.id] = ix_entry_preds
                print('Long entries: {}'.format(len(ix_entry_preds)))

        # merging entries from each strategy into a ix, (strat_tupel) list
        all_ix = []
        for l in [arr.tolist() for arr in s.strat_entry.values()]:
            all_ix += l
        all_ix = list(set(all_ix))
        all_ix.sort()
        # merge all entry arrays and sort by entry_ix. if multiple strategies overlap on ix, they get merged into the list
        for ix in all_ix:
            strat_ids = []
            for id, strat_entry_l in s.strat_entry.items():
                if ix in strat_entry_l:
                    strat_ids.append(id)
            s.ix_entry_preds.append((ix, strat_ids))
        # needs to become list of tupels (ix, strategy id)

    def set_trendspotter_pv_prices(s, trendspotter_pv_ts, series, pv):
        print('Loading trendspotter values...')
        trendspotter_pv_ts[series] = s.load_trendspotter_pv_from_db(series)
        if len(trendspotter_pv_ts[series]) == 0:
            return trendspotter_pv_ts
        high_p = trendspotter_pv_ts[series][trendspotter_pv_ts[series]['pv'] == pv]
        trendspotter_pv_ts[f'{series}_{pv}'] = pd.DataFrame(index=s.ohlc['ts'], columns=['price'])
        trendspotter_pv_ts[f'{series}_{pv}'].loc[high_p.index, 'price'] = high_p['price'].values
        trendspotter_pv_ts[f'{series}_{pv}']['price'].interpolate(method='pad', inplace=True)
        trendspotter_pv_ts[f'{series}_{pv}']['price'].fillna(
            99999 if pv == 'p' else 0,
            inplace=True)
        return trendspotter_pv_ts

    def filter_low_value_opp(s, find_peak_entries, strategy):
        filtered_entries = []
        find_peak_entries = sorted(find_peak_entries)
        peak_valley_dic = {}
        if strategy.direction == direction.long:
            valleys_pos, props = find_peaks(-1 * s.data['curve']['fit_savgol_long'] + 2 * max(s.data['curve']['fit_savgol_long']),
                                        height=0.1, prominence=0.06, distance=300)

        elif strategy.direction == direction.short:
            valleys_pos, props = find_peaks(-1 * s.data['curve']['fit_savgol_short'] + 2 * max(s.data['curve']['fit_savgol_short']),
                                        height=0.1, prominence=0.06, distance=300)
        for i in range(len(find_peak_entries)):
            peak_valley_dic[find_peak_entries[i]] = {}
            peak_valley_dic[find_peak_entries[i]]['val_peak'] = s.ohlc.iloc[find_peak_entries[i], s.ix_high]
            try:
                pot_valley = valleys_pos[np.where((valleys_pos < find_peak_entries[i+1]) & (valleys_pos > find_peak_entries[i]))]
            except IndexError:
                pot_valley = []
            if len(pot_valley) > 0:
                if strategy.direction == direction.long:
                    val_valley = s.ohlc.iloc[pot_valley, s.ix_high]
                elif strategy.direction == direction.short:
                    val_valley = s.ohlc.iloc[pot_valley, s.ix_low]
                peak_valley_dic[find_peak_entries[i]]['ix_valley'] = pot_valley[np.argmin(val_valley.values)]
                peak_valley_dic[find_peak_entries[i]]['val_valley'] = np.min(val_valley.values)
            else:
                peak_valley_dic[find_peak_entries[i]]['ix_valley'] = None
                peak_valley_dic[find_peak_entries[i]]['val_valley'] = None

        # FILTER ENTRIES
        for i in range(len(find_peak_entries)):
            if i < 3:
                filtered_entries.append(find_peak_entries[i])
                continue
            avg_opp = []
            for k in range(1, 4):
                if peak_valley_dic[find_peak_entries[i-k]]['ix_valley'] is not None:
                    avg_opp.append(peak_valley_dic[find_peak_entries[i-k]]['val_peak'] - peak_valley_dic[find_peak_entries[i-k]]['val_valley'])
            if len(avg_opp) < 2 or np.mean(avg_opp) > strategy.min_opp_thresh:
                filtered_entries.append(find_peak_entries[i])
        return np.array(filtered_entries)

    def get_entry_ix_mom60only(s, strategies):
        strat_entry = {}
        for strategy in strategies:
            if strategy.direction == direction.short:
                ix_entry_preds = np.where((s.data['curve']['mom_60'] <= strategy.min_mom_60)
                                          & (s.data['curve']['mom_60'] <= strategy.min_mom_60))[0]
            elif strategy.direction == direction.long:
                ix_entry_preds = np.where((s.data['curve']['mom_60'] >= strategy.min_mom_60)
                                          & (s.data['curve']['mom_60'] >= strategy.min_mom_60))[0]
            strat_entry[strategy.id] = ix_entry_preds
        all_ix = []
        for l in [arr.tolist() for arr in strat_entry.values()]:
            all_ix += l
        all_ix = list(set(all_ix))
        all_ix.sort()
        # merge all entry arrays and sort by entry_ix. if multiple strategies overlap on ix, they get merged into the list
        s.ix_entry_preds = []
        for ix in all_ix:
            strat_ids = []
            for id, strat_entry_l in strat_entry.items():
                if ix in strat_entry_l:
                    strat_ids.append(id)
            s.ix_entry_preds.append((ix, strat_ids))
        # needs to become list of tupels (ix, strategy id)

    def init_arr_lud(s, col: str, ix_start, ix_end):
        if not s.lud:
            s.lud = dotdict()
            s.lud[col] = 0
            s.arr = s.ohlc.iloc[ix_start:ix_end, s.ix_close].values.reshape(-1, 1)
        else:
            s.arr = np.empty((len(s.ohlc.iloc[ix_start:ix_end, s.ix_close]), len(s.lud.keys())))
            s.arr[:, s.lud[col]] = s.ohlc.iloc[ix_start:ix_end, s.ix_close].values

    def add_curves(s, curve_names, ix_start, ix_end):
        s.create_arr_ix(curve_names)
        for curve in curve_names:
            s.arr[:, s.lud[curve]] = s.data['curve'][curve][ix_start:ix_end]

    def create_arr_ix(s, cols):
        for c in to_list(cols):
            if c not in s.lud.keys():
                s.lud[c] = s.arr.shape[-1]
                s.arr = insert_nda_col(s.arr)

    def add_p(s, order, strategy):
        ix_start = order.fill.ix_fill
        s.create_arr_ix(['p_short', 'p_long'])
        for curve in ['p_short', 'p_long']:
            s.arr[:, s.lud[curve]] = s.data['curve'][curve][ix_start:ix_start + strategy.max_trade_length]

    def add_regr(s, order, strategy):
        ix_start = order.fill.ix_fill
        cols = [s.lud.regression_0, s.lud.regression_1, s.lud.regression_2, s.lud.regression_3, s.lud.regression_4, s.lud.regression_5]
        s.arr[:, cols] = s.data['regr'][ix_start:ix_start + strategy.max_trade_length, :]

    def add_p_exit(s, order, strategy):
        ix_start = order.fill.ix_fill
        s.create_arr_ix(['long_exit', 'short_exit'])
        # s.arr[:, s.lud.p_long] = s.preds_long[ix_start:ix_start + strategy.max_trade_length]
        s.arr[:, s.lud.long_exit] = s.data['curve']['long_exit'][ix_start:ix_start + strategy.max_trade_length]
        s.arr[:, s.lud.short_exit] = s.data['curve']['short_exit'][ix_start:ix_start + strategy.max_trade_length]
        # s.arr[:, s.lud.p_short] = s.preds_short[ix_start:ix_start + strategy.max_trade_length]

    def add_delta_profit(s, entry_price, strategy):
        # not in line with VS which uses midPrice to calculate delta profit. preferred
        s.create_arr_ix(['delta_profit', 'delta_profit_rel'])
        if True:  # using tradeBar
            s.arr[:, s.lud.delta_profit] = (s.arr[:, s.lud.close] - entry_price) * (-1 if strategy.direction == direction.short else 1)
            s.arr[:, s.lud.delta_profit_rel] = s.arr[:, s.lud.delta_profit] / entry_price
        else:
            s.arr[:, s.lud.delta_profit] = (s.arr[:, s.lud.mid_close] - entry_price) * (-1 if strategy.direction == direction.short else 1)
            s.arr[:, s.lud.delta_profit_rel] = s.arr[:, s.lud.delta_profit] / entry_price

    def add_rolling_max_profit(s, entry_price):
        s.create_arr_ix(['roll_max_profit', 'roll_max_profit_rel'])
        m = [0]
        for i in range(1, len(s.arr)):
            if s.arr[i, s.lud.delta_profit] > m[i - 1]:
                m.append(s.arr[i, s.lud.delta_profit])
            else:
                m.append(m[i - 1])
        s.arr[:, s.lud.roll_max_profit] = m
        s.arr[:, s.lud.roll_max_profit_rel] = np.divide(s.arr[:, s.lud.roll_max_profit], entry_price)  #s.arr[:, s.lud.close])

    def add_trailing_profit(s, entry_price):
        s.create_arr_ix(['trailing_profit', 'trailing_profit_rel'])
        s.arr[:, s.lud.trailing_profit] = s.arr[:, s.lud.roll_max_profit] - s.arr[:, s.lud.delta_profit]
        s.arr[:, s.lud.trailing_profit_rel] = s.arr[:, s.lud.trailing_profit] / entry_price

    def trail_profit_stop_price(s, order, strategy):
        s.create_arr_ix(['trail_profit_stop_price'])
        if order.direction == direction.long:
            s.arr[:, s.lud.trail_profit_stop_price] = order.fill.avg_price + s.arr[:, s.lud.roll_max_profit] - np.multiply(s.arr[:, s.lud.roll_max_profit] + order.fill.avg_price, strategy.trail_profit_stop)
        elif order.direction == direction.short:
            s.arr[:, s.lud.trail_profit_stop_price] = order.fill.avg_price - s.arr[:, s.lud.roll_max_profit] + np.multiply(order.fill.avg_price - s.arr[:, s.lud.roll_max_profit], strategy.trail_profit_stop)

    def add_trailing_stop_loss(s, entry_price, strategy):
        s.create_arr_ix(['stop_loss'])
        s.arr[:, s.lud.stop_loss] = entry_price * strategy.max_trailing_stop_a - \
                                  strategy.trailing_stop_b * s.arr[:, s.lud.roll_max_profit]
        s.arr[:, s.lud.stop_loss] = np.where(s.arr[:, s.lud.stop_loss] < strategy.min_stop_loss * entry_price,
                                           strategy.min_stop_loss * entry_price, s.arr[:, s.lud.stop_loss])

    def load_rl_exit_model(s):
        with open(os.path.join(Paths.model_rl, s.ex_rl_model, f'bins_{s.params.asset.lower()}.obj'), 'rb') as f:
            s.rl_bin_dict = pickle.load(f)
        for i in s.rl_model_ids:
            with open(os.path.join(Paths.model_rl, s.ex_rl_model, i), 'rb') as f:
                s.rl_models.append(pickle.load(f))

    def add_rl_exit(s, order):
        s.create_arr_ix(['rl_action_0', 'rl_action_1'])
        # assemble state
        # dynamic_state_feats = ['roll_max_profit', 'trailing_profit_rel', 'delta_profit_rel']
        # merge with df_full, preload into vb, load short term forecast preds and all order book stuff
        # end_ts = order.ts_fill + datetime.timedelta(seconds=len(s.arr))
        # ts_relevant = list(date_sec_range(order.ts_fill, end_ts))
        # convert regr back into original. better is just using the orignal I guess. refactor later
        # ['elapsed', 'p_long', 'p_long_pre', 'p_short', 'p_short_pre', 'profit_rel', 'roll_max_profit_rel', 'trailing_max_profit_rel', 'EMA_real_540', 'MOM_real_23', 'MOM_real_360', 'action']

        schema = {'elapsed': 0,
                  'p_long': 1,
                  'p_long_pre': 2,
                  'p_short': 3,
                  'p_short_pre': 4,
                  'profit_rel': 5,
                  'roll_max_profit_rel': 6,
                  'trailing_max_profit_rel': 7,
                  'EMA_real_540': 8,
                  'MOM_real_23': 9,
                  'MOM_real_360': 10
                  }
        # this had to be leaded from disk. very unreliable. any state_space change in RL script can change
        # this unpredictably
        state_space = {k: v for k, v in schema.items()}

        states = np.concatenate([
                np.array([int((pd.Timestamp(ts) - pd.Timestamp(s.arr[0, s.lud.ts])).total_seconds()) for ts in s.arr[:, s.lud.ts]]).reshape(-1, 1),
                s.arr[:, s.lud.p_long].reshape(-1, 1),
                np.zeros((s.arr.shape[0])).reshape(-1, 1),
                s.arr[:, s.lud.p_short].reshape(-1, 1),
                np.zeros((s.arr.shape[0])).reshape(-1, 1),
                s.arr[:, s.lud.delta_profit_rel].reshape(-1, 1),
                s.arr[:, s.lud.roll_max_profit_rel].reshape(-1, 1),
                s.arr[:, s.lud.trailing_profit_rel].reshape(-1, 1),
                s.arr[:, s.lud.EMA_540_300d_rel].reshape(-1, 1),
                s.arr[:, s.lud.MOM_real_23_rel].reshape(-1, 1),
                s.arr[:, s.lud.MOM_real_360_rel].reshape(-1, 1)
            ], axis=1
        )
        # normalize state
        for key in s.rl_bin_dict.keys():
            states[:, state_space[key]] = s.rl_bin_dict[key].transform(states[:, state_space[key]].reshape(-1, 1)).reshape(1, -1)[0]

        states_0 = np.concatenate([states, np.zeros(shape=(len(states), 1))], axis=1)
        states_1 = np.concatenate([states, np.ones(shape=(len(states), 1))], axis=1)
        # get expected rewards
        act_0 = []
        act_1 = []
        for m in s.rl_models:
            if type(m) == xgb.Booster:
                act_0.append(m.predict(xgb.DMatrix(states_0)))
                act_1.append(m.predict(xgb.DMatrix(states_1)))
            elif type(m) == lgb.Booster:
                act_0.append(m.predict(states_0))
                act_1.append(m.predict(states_1))
        s.arr[:, s.lud.rl_action_0] = np.average(act_0, axis=0)
        s.arr[:, s.lud.rl_action_1] = np.average(act_1, axis=0)

        pdf = pd.DataFrame(states, columns=schema, index=pd.to_datetime(s.arr[:, s.lud.ts]))
        pdf['rl_hold'] = s.arr[:, s.lud.rl_action_0]
        pdf['rl_exit'] = s.arr[:, s.lud.rl_action_1]
        # s.db_insert_preds(pdf, 'eurusd', measurement='unbinned_research_entry')
        s.influx.write_pdf(pdf,
                           measurement='research_entry_unbinned',
                           tags=dict(
                               asset='eurusd',
                               ex='2020-01-05_5',
                           ),
                           field_columns=pdf.columns,
                           # tag_columns=[]
                           )
        return

    def add_p_exit_given_entry(s, order):
        s.create_arr_ix(['p_exit_given_entry'])
        live_cols = ['roll_max_profit', 'trailing_profit_rel', 'delta_profit_rel']
        # merge with df_full, preload into vb, load short term forecast preds and all order book stuff
        end_ts = order.ts_fill + datetime.timedelta(seconds=len(s.arr))
        ts_relevant = list(date_sec_range(order.ts_fill, end_ts))
        df = pd.DataFrame(s.arr[:, [s.lud.roll_max_profit, s.lud.trailing_profit_rel, s.lud.delta_profit_rel]], columns=live_cols,
                          index=ts_relevant)
        df.index.name = 'ts'
        df['min_elapsed'] = np.divide(list(range(len(s.arr))), 60).astype(int)
        # aux_data = pd.concat([
        #     s.df_full.loc[order.ts_fill:end_ts, :],
        #     s.qt.loc[order.ts_fill:end_ts, :],
        #     s.tick_forecast_raw_preds.loc[order.ts_fill:end_ts, :]
        # ], axis=1)
        # df = df.merge(aux_data, how='left', on='ts')
        df = df.merge(s.df_full.loc[order.ts_fill:end_ts, :], how='left', on='ts')
        s.arr[:, s.lud.p_exit_given_entry] = s.exit_model_from_entry.full_predict_bt(df, direction=order.direction)

        # print('No Exit found. Using last ix of array')
        # return s.ohlc.index[-1], s.ohlc.iloc[-1, s.ix_close]

    def store_backtest(s):
        pdf = pd.DataFrame(None, columns=['price', 'direction', 'order_type', 'signal_source', 'fill'])
        for o in s.orders:
            if o.order_type == order_type.limit:
                pdf = pdf.append(pd.Series(
                    [o.price_limit, o.direction, o.fill.order_type, o.signal_source, o.quantity, False],
                    index=['price', 'direction', 'order_type', 'signal_source', 'quantity', 'fill'],
                    name=o.ts_signal
                ))
            pdf = pdf.append(pd.Series(
                [o.fill.avg_price, o.direction, o.fill.order_type, o.signal_source, o.fill.quantity, True],
                index=['price', 'direction', 'order_type', 'signal_source', 'quantity', 'fill'],
                name=o.fill.ts_fill
            ))
        # Logger.info('Saving backtest in influx  ex and backtest time...')
        s.influx.write_pdf(pdf,
                           measurement='backtest',
                           tags=dict(
                               asset=s.params.asset.lower(),
                               ex=s.ex,
                               backtest_time=s.params.backtest_time
                           ),
                           field_columns=['price', 'quantity'],
                           tag_columns=['fill', 'direction', 'order_type', 'signal_source']
                           )

    def store_input_curves(s):
        pdf = pd.DataFrame(s.data['curve'], columns=s.data['curve'].dtype.names, index=s.ohlc_mid.index)
        s.influx.write_pdf(pdf,
                           measurement='backtest_curves',
                           tags=dict(
                               asset=s.params.asset.lower(),
                               ex=s.ex,
                               backtest_time=s.params.backtest_time
                           ),
                           field_columns=pdf.columns
                           )

    def store_arr(s, pdf, fields):
        s.influx.write_pdf(pdf,
                           measurement='backtest_curves',
                           tags=dict(
                               asset=s.params.asset.lower(),
                               ex=s.ex,
                               backtest_time=s.params.backtest_time
                           ),
                           field_columns=fields
                           )
