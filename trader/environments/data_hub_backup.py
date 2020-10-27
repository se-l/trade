# import operator
# import re
# import copy
# import datetime
# import numpy as np
# import pandas as pd
#
# from talib import abstract
# from typing import Union
#
# from trader.common.globals import OHLC
# from trader.backtest.order import Order
# from trader.backtest.signal_type import SignalType
# from trader.common.modules.Dotdict import Dotdict
# from trader.common.modules.Series import Series
# from trader.common.modules.data_store import DataStore
# from trader.common.refdata.named_tuples import nda_schema
# from trader.common.utils.util_func import date_sec_range, df_to_npa, insert_nda_col, to_list, get_intersect_ts
# from trader.connector.influxdb.influxdb_wrapper import InfluxClientWrapper as Influx
# from trader.common.utils.util_func import make_struct_nda, join_struct_arrays
# from trader.backtest.stops_finder import StopsFinder
# from trader.backtest.fill import Fill
# from trader.common.refdata.tick_size import tick_size
# from trader.common.utils.util_func import todec, reduce_to_intersect_ts
# from trader.common.modules.enums import Assets, Direction, Timing, OrderType, Side, THE, SignalSource, Exchanges
# from trader.common.modules.Logger import Logger
# from trader.data_loader.utils_features import get_ohlc, get_ohlcv_mid_price
# from trader.train.reinforced.rl_agent_v2 import RLAgent
# from trader.common.refdata.curves_reference import CurvesReference as Cr
#
#
# class DataHub(StopsFinder):
#     """
#     Dataframe for decision making relevant data during RL Training & Backtesting.
#     """
#     def __init__(s, params):
#         s.handler_rl: RLAgent = None
#         s.ts = None
#         s.ohlc_mid = None
#         s.data = DataStore()
#         s.lud = None
#         s.params = params
#         s.df_full = None
#         s.strat_entry = {}
#         s.arr = None
#         s.ix_entry_preds = []
#         s.future_orders = None
#         s.ts_start = params.ts_start
#         s.ts_end = params.ts_end
#         s.assume_late_limit_fill = True  # QC alignment when True
#         s.assume_late_limit_fill_entry = True  # QC alignment when True
#         s.use_simply_limit_price_at_fill = False  # QC alignment when False
#         s.asset = params.asset
#         s.ini_dic = {}
#         s.sym_dic = {}
#         s.regr = {}
#         s.chunk = ['ho']
#         s.fn_p_exec = 'p_exec_{}.json'.format('-'.join([str(i) for i in s.chunk]))
#         s.preds_cols = ['ts', 'long', 'short']
#         s.preds_exit_cols = ['ts', 'long_exit', 'short_exit']
#         s.talib_cols = ['EMA_real_540', 'MOM_real_23', 'MOM_real_360']
#         # s.trailing_stop_b = s.p_opt['trailing_stop_a'] / s.p_opt['profit_target']
#
#     def setup(s):
#         # vbm.tickForecast = TickForeCast(s.params)
#         # vbm.qt = vbm.tickForecast.load_qt_db(s.params)
#         s.load_ohlc()
#         s.extract_inds_sql()
#         ohlc_ts_filter = get_intersect_ts(s.ohlc_mid, pd.to_datetime(s.ts))
#         # match_ts = np.intersect1d(pd.to_datetime(s.ts), s.ohlc.index, assume_unique=True)
#         # ohlc_ts_filter = np.isin(pd.to_datetime(s.ohlc.index), match_ts)
#
#         for ohlc_object in ['ohlc_mid', 'ohlc_bid', 'ohlc_ask']:
#             s.__setattr__(ohlc_object, s.__getattribute__(ohlc_object).loc[ohlc_ts_filter])
#             s.__getattribute__(ohlc_object)['ts'] = s.__getattribute__(ohlc_object).index
#             s.__setattr__(ohlc_object, s.__getattribute__(ohlc_object).reset_index(drop=True))
#
#     def _exchange_has_trade_data(s):
#         return True if s.params.exchange in [Exchanges.bitmex] else False
#
#     def reset(s):
#         s.__init__(s.params)
#         s.setup()
#
#     def state_size(s):
#         return s.arr.shape[-1]
#
#     def set_p_opt(s, p_opt):
#         s.p_opt = Dotdict(p_opt)
#         s.strategy_lib.update_p_opt(p_opt)
#         s.strategy_lib.set_trailing_stop_b()
#
#     def get_exit_signals(s, order: Order, strategy) -> Union[Dotdict, None]:
#         n_ix_remaining = len(s.ohlc_mid) - order.fill.ix_fill
#         ix_end = order.fill.ix_fill + min(strategy.max_trade_length, n_ix_remaining)
#         s.set_ix_re_entry(order, strategy, ix_end)
#         s.set_order_curves(order, strategy, ix_end)
#         return s._get_ix_exit_signal(order, strategy)
#
#     def get_rl_entries(s, order_exit: Order, strategy) -> Union[Dotdict, None]:
#         n_ix_remaining = len(s.ohlc_mid) - order_exit.fill.ix_fill
#         ix_end = order_exit.fill.ix_fill + min(strategy.max_trade_length, n_ix_remaining)
#         s.ix_reentry_set = []
#         s.set_order_curves(order_exit, strategy, ix_end)
#         return s._get_ix_exit_signal(order_exit, strategy)
#
#     def _get_ix_exit_signal(s, order, strategy):
#         ix_signals = SignalType.init()
#         ix_ema_veto = None  # todo: re-introduce this
#         # todo - type of spot check need to be config driven. A large set of methods. So also optimizable
#         # ix_signals.ix_other_peak_entry_stop = s.find_other_peak_entry_stop(order, strategy, veto_ix=ix_ema_veto)
#         if strategy.use_rl_exit:
#             ix_signals.ix_rl_exit = s.find_exit_rl_stop(order, strategy, veto_ix=ix_ema_veto)
#         ix_signals.ix_elapsed = len(s.data['ohlc_mid']) - 1
#         return ix_signals
#
#     def set_order_curves(s, order, strategy, ix_end):
#         s.init_arr_lud(col='close', ix_start=order.fill.ix_fill, ix_end=ix_end)
#         for c in ['ohlc_mid']:
#             s.create_arr_ix(c)
#             s.arr[:, s.lud[c]] = s.data['ohlc_mid'].iloc[order.fill.ix_fill:ix_end, s.ix_close]
#         s.create_arr_ix('ts')
#         s.arr[:, s.lud['ts']] = s.data['ohlc_mid'].iloc[order.fill.ix_fill:ix_end].index
#
#         s.add_curves(['EMA_540_300d_rel', 'MOM_real_23_rel', 'MOM_real_360_rel', 'p_long', 'p_short'], ix_start=order.fill.ix_fill, ix_end=ix_end)
#
#         # s.add_p(order, strategy)
#         # try:
#         #     s.add_regr(order, strategy)
#         # except KeyError:
#         #     pass
#         # s.add_p_exit(order, strategy)
#         # s.add_delta_profit(order.fill.avg_price, strategy)
#         # s.add_rolling_max_profit(order.fill.avg_price)
#         # s.add_trailing_profit(order.fill.avg_price)
#         # s.trail_profit_stop_price(order, strategy)
#         s.add_delta_profit(s.arr[0, s.lud.close], strategy)
#         s.add_rolling_max_profit(s.arr[0, s.lud.close])
#         s.add_trailing_profit(s.arr[0, s.lud.close])
#         s.trail_profit_stop_price(order, strategy)
#         # s.add_half_bband_middle_lower()
#         # s.add_half_bband_middle_upper()
#         s.add_trailing_stop_loss(order.fill.avg_price, strategy)
#         if strategy.use_rl_exit:
#             s.add_rl_exit()
#
#     def set_ix_re_entry(s, order, strategy, ix_end):
#         try:
#             s.ix_reentry_set = list(set(
#                 np.intersect1d(s.strat_entry[strategy.id][order.fill.ix_fill < s.strat_entry[strategy.id]],
#                                s.strat_entry[strategy.id][s.strat_entry[strategy.id] < ix_end])
#             ))
#         except KeyError:
#             s.ix_reentry_set = []
#
#     def would_reenter_same_sec(s, min_ix):
#         """
#         only for the entry orders as still finding exit order.
#         checks if strategy going opposite direction to exit order exists for the exit time.
#         """
#         return min_ix in s.ix_reentry_set
#
#     def get_talib_inds(s, ohlc, ind):
#         keys = ind.split('_real_')
#         f = keys[0]
#         # input_params = getattr(abstract, f)._Function__opt_inputs.keys()
#         # out_names = getattr(abstract, f).output_names
#         # params = getattr(abstract, f).get_parameters()
#         # matype = getattr(abstract, f).get_parameters()['matype']
#         input_params = dict(timeperiod=int(keys[1]))
#         out_val = getattr(abstract, f)({
#             'open': ohlc.iloc[:, s.ix_open],
#             'high': ohlc.iloc[:, s.ix_high],
#             'low': ohlc.iloc[:, s.ix_low],
#             'close': ohlc.iloc[:, s.ix_close]},
#             **input_params)
#         np.nan_to_num(out_val, 0)
#         # make relative
#         return out_val
#         # arr.append(out_val)
#         # return np.vstack(arr).transpose()
#
#     @staticmethod
#     def preds_smooth(preds, strategy):
#         f = strategy.preds_smooth_f
#         input_params = getattr(abstract, f)._Function__opt_inputs.keys()
#         # out_names = getattr(abstract, f).output_names
#         # params = getattr(abstract, f).get_parameters()
#         input_params = dict(timeperiod=strategy.preds_smooth_tp)
#         middleband = getattr(abstract, f)({'close': preds.astype(np.float)}, **input_params)
#         ix_first_non_nan = sum(np.isnan(middleband))
#         middleband[:ix_first_non_nan] = middleband[ix_first_non_nan]
#         return middleband
#
#     def order_timed_out(s, order, strategy):
#         if order.fill.ix_fill - order.ix_signal > strategy.time_entry_cancelation:
#             return True
#         else:
#             return False
#
#     def order_vetoed(s, ix_entry, strategy):
#         if strategy.direction == Direction.short:
#             if s.regr_ens_veto[strategy.id][ix_entry] > strategy.regr_veto_stop:  # the close preds are the same for each asset independent of long or short
#                 return True
#         elif strategy.direction == Direction.long:
#             if s.regr_ens_veto[strategy.id][ix_entry] < strategy.regr_veto_stop:  # the close preds are the same for each asset independent of long or short
#                 # s.stats['ts_entry_regr_vetoed'].append(s.sym_dic['idx_ts'].index[ix_entry])
#                 return True
#         else:
#             return False
#
#     @staticmethod
#     def is_signal(signal_direction, portf_side):
#         if portf_side == Side.hold:
#             return True
#         else:
#             # when portfolio already long, enter another long
#             # had problem that stops triggered a portf_side of long.
#             return signal_direction != portf_side
#
#     @staticmethod
#     def tv_ho_to_str(tv_ho):
#         if tv_ho == THE.ho:
#             return 'ho'
#         elif tv_ho == THE.tv:
#             return 'tv'
#         elif tv_ho == THE.extra:
#             return 'extra'
#         else:
#             raise ('Unkown data split. Expecting train, holdout, extra')
#
#     def extract_inds_sql(s):
#         ohlc_mid = get_ohlcv_mid_price(s.params)
#         talibs = np.vstack([s.get_talib_inds(ohlc_mid, ind=ind) for ind in s.talib_cols]).transpose()
#         # preds = s.load_preds_from_db(tbl='model_preds')
#         # s.data['regr'] = s.load_entry_predictions_from_db([6, 7, 8, 9, 10, 11])
#         # features = s.load_entry_predictions_from_db([0, 1, 2, 3, 6, 7, 8, 9, 10, 11])
#         # features_post_ls = [Features.y_post_valley, Features.y_post_peak]
#         # features_pre_ls = [Features.y_pre_valley, Features.y_pre_peak]
#         preds = s.load_entry_predictions_from_db()
#         new_col = []
#         for c in preds.columns:
#             if re.search('dwin-\d', c):
#                 new_col.append('p_long')
#             elif re.search('dwin--\d', c):
#                 new_col.append('p_short')
#         preds.columns = new_col
#         preds, s.ohlc_mid, talibs = reduce_to_intersect_ts(preds, s.ohlc_mid, pd.DataFrame(talibs, index=s.ohlc_mid.index))
#         talibs = talibs.values
#         # pre_post reversed assignment. in lgb fl train targeting the pre preds...
#         # preds = features[:, [0, 2, 4]]
#         # preds_pre = features[:, [1, 3]]
#         # correct assignment # +1 due to time stamp
#         # preds = features.values  #features[:, [0] + [el + 1 for el in features_post_ls]]
#         # preds_pre = features[:, [el + 1 for el in features_pre_ls]]
#         # s.data['regr'] = features[:, list(range(5, features.shape[1]))]
#         if len(talibs) != len(preds):
#             Logger.info('Len of curves and preds is not equal...Calc may take a while...')
#             Logger.info('Missing ts in preds: {}'.format(np.setdiff1d(talibs[:, 0], preds.index, assume_unique=True)))
#             # [preds[i,0] for i in range(len(preds[:, 0])) if i> 1 and preds[i, 0] != (preds[i-1,0] + datetime.timedelta(seconds=1))]
#             Logger.info('Missing ts in curves: {}'.format(np.setdiff1d(preds[:, 0], preds.index, assume_unique=True)))
#         assert len(talibs) == len(preds), 'curves and preds dont have the same length. Cannot merge safely'
#         # transform to zscore
#         # window = 3600
#         # for i in range(1, 3):
#         #     mean = np.mean(rolling_window(preds[:, 1], window), axis=1)
#         #     std = np.std(np.array(rolling_window(preds[:, 1], window)), axis=1, dtype=np.float64)
#         #     preds[window - 1:, i] = np.divide(np.subtract(preds[window-1:, i], mean), std)
#         #     preds[:window-1, i] = 0
#         s.ts = preds.index
#         s.data['ohlc_mid'] = ohlc_mid.loc[s.ts]
#
#         # tick_forecast = TickForeCast(s.params)
#         # tick_forecast.load_raw_preds_db(ts_start=s.ts[0], ts_end=s.ts[-1])
#
#         # ticks = make_struct_nda(, tick_forecast.raw_preds.columns)
#         Logger.info('Concatenating loaded data into curves...')
#         # s.data['inds'] = np.concatenate([inds, preds[:, 1:], tick_forecast.raw_preds.values], axis=1)
#         # s.data['inds'] = np.concatenate([inds, preds[:, 1:], np.zeros(shape=(len(inds), 3))], axis=1)
#         # s.data['inds'] = make_struct_nda(s.data['inds'], s.inds_cols + ['long', 'short'] + list(tick_forecast.raw_preds.columns))
#         s.data['curve'] = join_struct_arrays([
#             make_struct_nda(talibs, s.talib_cols),
#             df_to_npa(preds)
#         ])
#         assert len(s.data['curve']) == len(s.data['ohlc_mid']), 'ohld mid curve length not matching'
#         # s.data['curve'] = join_struct_arrays([s.data['curve'], ticks])
#         # s.preds_short_mean = s.data['curve']['short']
#         # s.preds_long_mean = s.data['curve']['long']
#
#     @staticmethod
#     def slice_data_by_ts(data, ts_start, ts_end):
#         if type(data) == pd.DataFrame:
#             return data.loc[ts_start:ts_end, :]
#         elif type(data) == np.ndarray:
#             return data[ts_start:ts_end, :]
#         else:
#             raise('Type not known')
#
#     @staticmethod
#     def add_timeperiod(ix_5, time_delta):
#         return [ix_5 + i for i in range(0, time_delta)]
#
#     def add_ix_entry_tp(s, ix_entry_preds_mom10, time_delta):
#         ix_entry_preds_precursor = []
#         for ix_5 in ix_entry_preds_mom10:
#             ix_entry_preds_precursor += s.add_timeperiod(ix_5, time_delta)
#         return np.sort(list(set(ix_entry_preds_precursor)))
#
#     def get_entry_ix(s, strategies):
#         if 'EMA_540_300d_rel' not in s.data['curve'].dtype.names:
#             Logger.info('Initial indicator calculations...')
#             ema_540_300d = np.zeros_like(s.data['curve']['EMA_real_540'])
#             i = 300
#             ema_540_300d[i:] = s.data['curve']['EMA_real_540'][i:] - s.data['curve']['EMA_real_540'][:-i]
#             # preds_net_d1 = np.zeros_like(s.data['curve']['net'])
#             # preds_net_d1[1:] = np.subtract(s.data['curve']['net'][1:], s.data['curve']['net'][:-1])
#             preds_short_d1 = np.zeros_like(s.data['curve']['p_short'])
#             preds_short_d1[1:] = np.subtract(s.data['curve']['p_short'][1:], s.data['curve']['p_short'][:-1])
#             preds_long_d1 = np.zeros_like(s.data['curve']['p_long'])
#             preds_long_d1[1:] = np.subtract(s.data['curve']['p_long'][1:], s.data['curve']['p_long'][:-1])
#             preds_short_d30 = np.zeros_like(s.data['curve']['p_short'])
#             preds_short_d30[30:] = np.subtract(s.data['curve']['p_short'][30:], s.data['curve']['p_short'][:-30])
#             preds_long_d30 = np.zeros_like(s.data['curve']['p_long'])
#             preds_long_d30[30:] = np.subtract(s.data['curve']['p_long'][30:], s.data['curve']['p_long'][:-30])
#
#             s.data['curve'] = join_struct_arrays([
#                 s.data['curve'],
#                 make_struct_nda(np.divide(ema_540_300d, s.data['ohlc_mid']['close']), cols=['EMA_540_300d_rel'], def_type=np.float64),
#                 make_struct_nda(np.divide(s.data['curve']['MOM_real_23'], s.data['ohlc_mid']['close']), cols=['MOM_real_23_rel'], def_type=np.float64),
#                 make_struct_nda(np.divide(s.data['curve']['MOM_real_360'], s.data['ohlc_mid']['close']), cols=['MOM_real_360_rel'], def_type=np.float64),
#                 make_struct_nda(preds_short_d1, cols=['p_short_d1'], def_type=np.float64),
#                 make_struct_nda(preds_long_d1, cols=['p_long_d1'], def_type=np.float64),
#                 make_struct_nda(preds_short_d30, cols=['p_short_d30'], def_type=np.float64),
#                 make_struct_nda(preds_long_d30, cols=['p_long_d30'], def_type=np.float64),
#                 # make_struct_nda(preds_net_d1, cols=['preds_net_d1'], def_type=np.float64),
#
#                 # make_struct_nda(sav_poly_long_peak, cols=['sav_poly_long_peak'], def_type=np.float64),
#                 # make_struct_nda(sav_poly_long_valley, cols=['sav_poly_long_valley'], def_type=np.float64),
#                 # make_struct_nda(sav_poly_short_peak, cols=['sav_poly_short_peak'], def_type=np.float64),
#                 # make_struct_nda(sav_poly_short_valley, cols=['sav_poly_short_valley'], def_type=np.float64),
#             ])
#             try:
#                 for i in range(s.data['regr'].shape[1]):
#                     s.data['regr'][:, i] = np.add(
#                         s.ohlc_mid['close'],
#                         np.multiply(s.data['regr'][:, i], s.ohlc_mid['close'])
#                     )
#             except KeyError:
#                 pass
#
#         Logger.info('selecting entry points...')
#         for strategy in strategies:
#             if strategy.direction == Direction.short:
#                 # above_middle_bband = np.where(s.data['curve']['bband_20_2_2_middle'] <= s.ohlc_mid['close'])[0]
#                 # above_middle_bband = np.where(s.data['curve']['bband_20_2_2_middle'] <= s.ohlc_mid['close'])[0]
#                 # cross_bband = np.where(s.data['curve']['bband_upper_s'] <= s.ohlc_mid['high'] + strategy.bband_entry_delta)[0]
#                 # bullish_range = np.where(s.data['curve']['ema_180_1d'] >= strategy.bullish_range)[0]
#                 extr_bull_bear_stretch = np.where(s.data['curve']['EMA_540_300d_rel'] >= strategy.bull_bear_stretch)[0]
#                 # volatility_thresh = np.where(np.subtract(s.data['curve']['bband_20_2_2_upper'], s.data['curve']['bband_20_2_2_lower']) <= strategy.min_bband_volat)[0]
#                 # bullish_range = np.where(s.data['curve']['ema_180'] <= s.ohlc_mid['close'])[0]
#                 # mom_extreme_mom_60 = np.where(s.data['curve']['mom_60'] <= strategy.mom_extreme_mom_60)[0]
#                 # min_mom_3h = np.where(s.data['curve']['mom_10800'] >= strategy.min_mom_300)[0]
#                 ix_rebounds = np.where(s.data['curve']['MOM_real_360_rel'] <= strategy.rebound_mom)[0]
#                 # min_bband_volat = np.where(np.subtract(s.data['curve']['bband_upper_s'], s.data['curve']['bband_lower_s']) < strategy.min_bband_volat)[0]
#
#                 # ix_entry_preds_mom300 = np.where(s.data['curve']['mom_300'] >= strategy.min_mom_300)[0]
#                 # ix_entry_preds_mom10 = np.where(s.data['curve']['mom_10'] <= strategy.min_mom_10)[0]
#                 # ix_entry_preds_mom10 = s.add_ix_entry_tp(ix_entry_preds_mom10, strategy.mom_short_long_tp)
#                 # ix_entry_preds_mom60 = np.where(s.data['curve']['mom_60'] <= strategy.min_mom_60)[0]
#                 # ix_short_exit_preds = np.where(s.data['curve']['short_exit'] >= strategy.veto_p_exit)[0]
#                 # ix_preds_entry_smooth_short_entry_dx = np.where(
#                 #     (s.data['curve']['short'] > strategy.preds_sl_thresh_dx) &
#                 #     (s.data['curve']['preds_smooth_short_dx'] < 0)
#                 # )[0]
#                 ix_preds_entry_short_entry = np.where(s.data['curve']['p_short'] > strategy.preds_sl_thresh)[0]
#                 # ix_preds_entry_short_d1 = np.where(s.data['curve']['preds_short_d1'] > 0)[0]
#                 # ix_preds_entry_short_entry = np.where(s.data['preds_pre']['p_short_pre'] > strategy.preds_sl_thresh)[0]
#                 ix_entry_preds = ix_preds_entry_short_entry
#
#                 # ix_entry_preds = np.where(s.data['curve']['net'] <= strategy.preds_net_thresh)[0]
#                 # find_peak_entries = s.filter_low_value_opp(find_peak_entries, strategy)
#                 # ix_entry_preds = np.union1d(ix_entry_preds, np.where(s.data['curve']['sav_poly_short_peak'] == 1)[0])
#                 # ix_entry_preds = np.union1d(ix_entry_preds, np.where(s.data['curve']['sav_poly_long_valley'] == 1)[0])
#                 # ix_entry_preds = np.union1d(ix_entry_preds, ix_preds_entry_smooth_short_entry_dx)
#                 # ix_entry_preds = np.intersect1d(ix_preds_entry_short_entry, ix_preds_entry_short_d1)
#                 # ix_entry_preds = np.union1d(ix_entry_preds, cross_bband)
#                 # if not strategy.assume_simulated_future:
#                 #     ix_entry_preds_d1 = np.where(preds_net_d1 <= 0)[0]
#                 #     ix_entry_preds = np.setdiff1d(ix_entry_preds, ix_entry_preds_d1)
#                 # ix_entry_preds = np.setdiff1d(ix_entry_preds, min_bband_volat)
#
#                 # ix_entry_preds = np.intersect1d(ix_entry_preds_mom10, ix_entry_preds_mom60)
#                 # ix_entry_preds = np.intersect1d(ix_entry_preds, ix_entry_preds_mom300)
#                 # ix_entry_preds = np.setdiff1d(ix_entry_preds, volatility_thresh)
#                 # ix_entry_preds = np.setdiff1d(ix_entry_preds, ix_rebounds)
#                 # ix_entry_preds = np.intersect1d(ix_entry_preds, min_preds_net)
#                 # ix_entry_preds = np.setdiff1d(ix_entry_preds, bullish_range)
#                 # ix_entry_preds = np.union1d(ix_entry_preds, mom_extreme_mom_60)
#                 ix_entry_preds = np.setdiff1d(ix_entry_preds, extr_bull_bear_stretch)
#                 # ix_entry_preds = np.setdiff1d(ix_entry_preds, ix_short_exit_preds)
#                 # ix_entry_preds = np.intersect1d(ix_entry_preds, min_mom_3h)
#                 # ix_entry_preds = np.union1d(ix_entry_preds, bearish_range_ema)
#                 ix_entry_preds = np.unique(ix_entry_preds)
#                 s.strat_entry[strategy.id] = ix_entry_preds
#                 Logger.info('Short entries: {}'.format(len(ix_entry_preds)))
#             elif strategy.direction == Direction.long:
#                 # ix_entry_preds_mom300 = np.where(s.data['curve']['mom_300'] <= strategy.min_mom_300)[0]
#                 # ix_entry_preds_mom10 = np.where(s.data['curve']['mom_10'] <= strategy.min_mom_10)[0]
#                 # ix_entry_preds_mom10 = s.add_ix_entry_tp(ix_entry_preds_mom10, strategy.mom_short_long_tp)
#                 # cross_bband = np.where(s.data['curve']['bband_lower_l'] >= s.ohlc_mid['low'] + strategy.bband_entry_delta)[0]
#                 # ix_entry_preds_mom60 = np.where(s.data['curve']['mom_60'] >= strategy.min_mom_60)[0]
#                 ix_rebounds = np.where(s.data['curve']['MOM_real_360_rel'] >= strategy.rebound_mom)[0]
#                 # min_bband_volat = np.where(np.subtract(s.data['curve']['bband_upper_l'], s.data['curve']['bband_lower_l']) < strategy.min_bband_volat)[0]
#
#                 # bullish_range = np.where(s.data['curve']['ema_180'] <= s.ohlc_mid['close'])[0]
#                 extr_bull_bear_stretch = np.where(s.data['curve']['EMA_540_300d_rel'] <= strategy.bull_bear_stretch)[0]
#                 # ix_long_exit_preds = np.where(s.data['curve']['long_exit'] >= strategy.veto_p_exit)[0]
#                 # ix_preds_entry_smooth_long_entry_dx = np.where(
#                 #     (s.data['curve']['long'] > strategy.preds_sl_thresh_dx) &
#                 #     (s.data['curve']['preds_smooth_long_dx'] < 0)
#                 # )[0]
#
#                 ix_preds_entry_long_entry = np.where(s.data['curve']['p_long'] > strategy.preds_sl_thresh)[0]
#                 # ix_preds_entry_long_entry = np.where(s.data['curve']['long'] > strategy.preds_sl_thresh)[0]
#                 # ix_preds_entry_long_d1 = np.where(s.data['curve']['preds_long_d1'] > 0)[0]
#                 ix_entry_preds = ix_preds_entry_long_entry
#                 # ix_entry_preds = np.intersect1d(ix_preds_entry_long_entry, ix_preds_entry_long_d1)
#                 # mom_extreme_mom_60 = np.where(s.data['curve']['mom_60'] >= strategy.mom_extreme_mom_60)[0]
#                 # below_middle_bband = np.where(s.data['curve']['bband_20_2_2_middle'] >= s.ohlc_mid['close'])[0]
#                 # volatility_thresh = np.where(np.subtract(s.data['curve']['bband_20_2_2_upper'], s.data['curve']['bband_20_2_2_lower']) <= strategy.min_bband_volat)[0]
#                 # ix_entry_preds = np.where(s.data['curve']['net'] >= strategy.preds_net_thresh)[0]
#                 # find_peak_entries = s.filter_low_value_opp(find_peak_entries, strategy)
#                 # ix_entry_preds = np.union1d(ix_entry_preds, np.where(s.data['curve']['sav_poly_long_peak'] == 1)[0])
#                 # ix_entry_preds = np.union1d(ix_entry_preds, np.where(s.data['curve']['sav_poly_short_valley'] == 1)[0])
#                 # ix_entry_preds = np.union1d(ix_entry_preds, ix_preds_entry_smooth_long_entry_dx)
#                 # ix_entry_preds = np.union1d(ix_entry_preds, ix_preds_entry_long_entry)
#                 # ix_entry_preds = np.union1d(ix_entry_preds, cross_bband)
#                 # if not strategy.assume_simulated_future:
#                 #     ix_entry_preds_d1 = np.where(preds_net_d1 >= 0)[0]
#                 #     ix_entry_preds = np.setdiff1d(ix_entry_preds, ix_entry_preds_d1)
#                 # ix_entry_preds = np.setdiff1d(ix_entry_preds, min_bband_volat)
#
#                 # ix_entry_preds = np.intersect1d(ix_entry_preds_mom10, ix_entry_preds_mom60)
#                 # ix_entry_preds = np.intersect1d(ix_entry_preds, below_middle_bband)
#                 ix_entry_preds = np.setdiff1d(ix_entry_preds, ix_rebounds)
#                 # ix_entry_preds = np.intersect1d(ix_entry_preds, min_preds_net)
#                 # ix_entry_preds = np.intersect1d(ix_entry_preds, bullish_range)
#                 ix_entry_preds = np.setdiff1d(ix_entry_preds, extr_bull_bear_stretch)
#                 # ix_entry_preds = np.setdiff1d(ix_entry_preds, ix_long_exit_preds)
#                 # ix_entry_preds = np.union1d(ix_entry_preds, bullish_range_ema)
#                 ix_entry_preds = np.unique(ix_entry_preds)
#                 s.strat_entry[strategy.id] = ix_entry_preds
#                 Logger.info('Long entries: {}'.format(len(ix_entry_preds)))
#
#         # merging entries from each strategy into a ix, (strat_tupel) list
#         all_ix = []
#         for l in [arr.tolist() for arr in s.strat_entry.values()]:
#             all_ix += l
#         all_ix = list(set(all_ix))
#         all_ix.sort()
#         # merge all entry arrays and sort by entry_ix. if multiple strategies overlap on ix, they get merged into the list
#         for ix in all_ix:
#             strat_ids = []
#             for id, strat_entry_l in s.strat_entry.items():
#                 if ix in strat_entry_l:
#                     strat_ids.append(id)
#             s.ix_entry_preds.append((ix, strat_ids))
#         # needs to become list of tupels (ix, strategy id)
#
#     def get_delta_close(s, t_start: datetime, t_end: datetime, tdelta_sec: int) -> np.ndarray:
#         tdelta = datetime.timedelta(seconds=tdelta_sec)
#         # with (tdelta := datetime.timedelta(seconds=tdelta_sec)):
#         return s.ohlc_mid.loc[t_start:t_end, 'close'].values - s.ohlc_mid.loc[t_start-tdelta:t_end-tdelta, 'close'].values
#
#     def get_entry_ix_mom60only(s, strategies):
#         strat_entry = {}
#         for strategy in strategies:
#             if strategy.direction == Direction.short:
#                 ix_entry_preds = np.where((s.data['curve']['mom_60'] <= strategy.min_mom_60)
#                                           & (s.data['curve']['mom_60'] <= strategy.min_mom_60))[0]
#             elif strategy.direction == Direction.long:
#                 ix_entry_preds = np.where((s.data['curve']['mom_60'] >= strategy.min_mom_60)
#                                           & (s.data['curve']['mom_60'] >= strategy.min_mom_60))[0]
#             strat_entry[strategy.id] = ix_entry_preds
#         all_ix = []
#         for l in [arr.tolist() for arr in strat_entry.values()]:
#             all_ix += l
#         all_ix = list(set(all_ix))
#         all_ix.sort()
#         # merge all entry arrays and sort by entry_ix. if multiple strategies overlap on ix, they get merged into the list
#         s.ix_entry_preds = []
#         for ix in all_ix:
#             strat_ids = []
#             for id, strat_entry_l in strat_entry.items():
#                 if ix in strat_entry_l:
#                     strat_ids.append(id)
#             s.ix_entry_preds.append((ix, strat_ids))
#         # needs to become list of tupels (ix, strategy id)
#
#     def init_arr_lud(s, col: str, ix_start, ix_end):
#         if not s.lud:
#             s.lud = Dotdict()
#             s.lud[col] = 0
#             s.arr = s.ohlc_mid.iloc[ix_start:ix_end, s.ix_close].values.reshape(-1, 1)
#         else:
#             s.arr = np.empty((len(s.ohlc_mid.iloc[ix_start:ix_end, s.ix_close]), len(s.lud.keys())))
#             s.arr[:, s.lud[col]] = s.ohlc_mid.iloc[ix_start:ix_end, s.ix_close].values
#
#     def add_curves(s, curve_names, ix_start, ix_end):
#         s.create_arr_ix(curve_names)
#         for curve in curve_names:
#             s.arr[:, s.lud[curve]] = s.data['curve'][curve][ix_start:ix_end]
#
#     def create_arr_ix(s, cols):
#         for c in to_list(cols):
#             if c not in s.lud.keys():
#                 s.lud[c] = s.arr.shape[-1]
#                 s.arr = insert_nda_col(s.arr)
#
#     def add_p(s, order, strategy):
#         ix_start = order.fill.ix_fill
#         s.create_arr_ix([Cr.p_short, Cr.p_long])
#         for curve in [Cr.p_short, Cr.p_long]:
#             s.arr[:, s.lud[curve]] = s.data['curve'][curve][ix_start:ix_start + strategy.max_trade_length]
#
#     def add_regr(s, order, strategy):
#         ix_start = order.fill.ix_fill
#         cols = [s.lud.regression_0, s.lud.regression_1, s.lud.regression_2, s.lud.regression_3, s.lud.regression_4, s.lud.regression_5]
#         s.arr[:, cols] = s.data['regr'][ix_start:ix_start + strategy.max_trade_length, :]
#
#     def add_p_exit(s, order, strategy):
#         ix_start = order.fill.ix_fill
#         s.create_arr_ix([Cr.long_exit, Cr.short_exit])
#         # s.arr[:, s.lud.p_long] = s.preds_long[ix_start:ix_start + strategy.max_trade_length]
#         s.arr[:, s.lud.long_exit] = s.data['curve'][Cr.long_exit][ix_start:ix_start + strategy.max_trade_length]
#         s.arr[:, s.lud.short_exit] = s.data['curve'][Cr.short_exit][ix_start:ix_start + strategy.max_trade_length]
#         # s.arr[:, s.lud.p_short] = s.preds_short[ix_start:ix_start + strategy.max_trade_length]
#
#     def add_delta_profit(s, entry_price, strategy):
#         # not in line with VS which uses midPrice to calculate delta profit. preferred
#         s.create_arr_ix([Cr.delta_profit, Cr.delta_profit_rel])
#         if True:  # using tradeBar
#             s.arr[:, s.lud.delta_profit] = (s.arr[:, s.lud.close] - entry_price) * (-1 if strategy.direction == Direction.short else 1)
#             s.arr[:, s.lud.delta_profit_rel] = s.arr[:, s.lud.delta_profit] / entry_price
#         else:
#             s.arr[:, s.lud.delta_profit] = (s.arr[:, s.lud.mid_close] - entry_price) * (-1 if strategy.direction == Direction.short else 1)
#             s.arr[:, s.lud.delta_profit_rel] = s.arr[:, s.lud.delta_profit] / entry_price
#
#     def add_rolling_max_profit(s, entry_price):
#         s.create_arr_ix([Cr.roll_max_profit, Cr.roll_max_profit_rel])
#         m = [0]
#         for i in range(1, len(s.arr)):
#             if s.arr[i, s.lud.delta_profit] > m[i - 1]:
#                 m.append(s.arr[i, s.lud.delta_profit])
#             else:
#                 m.append(m[i - 1])
#         s.arr[:, s.lud.roll_max_profit] = m
#         s.arr[:, s.lud.roll_max_profit_rel] = np.divide(s.arr[:, s.lud.roll_max_profit], entry_price)  #s.arr[:, s.lud.close])
#
#     def add_trailing_profit(s, entry_price):
#         s.create_arr_ix([Cr.trailing_profit, Cr.trailing_profit_rel])
#         s.arr[:, s.lud.trailing_profit] = s.arr[:, s.lud.roll_max_profit] - s.arr[:, s.lud.delta_profit]
#         s.arr[:, s.lud.trailing_profit_rel] = s.arr[:, s.lud.trailing_profit] / entry_price
#
#     def trail_profit_stop_price(s, order, strategy):
#         s.create_arr_ix([Cr.trail_profit_stop_price])
#         if order.direction == Direction.long:
#             s.arr[:, s.lud.trail_profit_stop_price] = order.fill.avg_price + s.arr[:, s.lud.roll_max_profit] - np.multiply(s.arr[:, s.lud.roll_max_profit] + order.fill.avg_price, strategy.trail_profit_stop)
#         elif order.direction == Direction.short:
#             s.arr[:, s.lud.trail_profit_stop_price] = order.fill.avg_price - s.arr[:, s.lud.roll_max_profit] + np.multiply(order.fill.avg_price - s.arr[:, s.lud.roll_max_profit], strategy.trail_profit_stop)
#
#     def add_trailing_stop_loss(s, entry_price, strategy):
#         s.create_arr_ix([Cr.stop_loss])
#         s.arr[:, s.lud.stop_loss] = entry_price * strategy.max_trailing_stop_a - \
#                                     strategy.trailing_stop_b * s.arr[:, s.lud.roll_max_profit]
#         s.arr[:, s.lud.stop_loss] = np.where(s.arr[:, s.lud.stop_loss] < strategy.min_stop_loss * entry_price,
#                                            strategy.min_stop_loss * entry_price, s.arr[:, s.lud.stop_loss])
#
#     def get_state_attr(s, curve):
#         if curve == Cr.ts:
#             return np.array([int((pd.Timestamp(ts) - pd.Timestamp(s.arr[0, s.lud[Cr.ts]])).total_seconds()) for ts in s.arr[:, s.lud[Cr.ts]]]).reshape(-1, 1)
#         else:
#             try:
#                 return s.arr[:, s.lud[curve]].reshape(-1, 1)
#             except KeyError as e:
#                 Logger.error(f'Curve {curve} not found in arr: {e}. Insert zeroes instead')
#                 return np.zeros((s.arr.shape[0])).reshape(-1, 1)
#
#     def add_rl_exit(s):
#         s.create_arr_ix([Cr.rl_action_hold, Cr.rl_action_exit])
#         states_schema = nda_schema(
#             np.concatenate([
#                 s.get_state_attr(rl_state_col.name) for rl_state_col in s.handler_rl.schema
#             ], axis=1),
#             s.handler_rl.schema
#         )
#         s.arr[:, s.lud[Cr.rl_action_hold]], s.arr[:, s.lud[Cr.rl_action_exit]] = s.handler_rl.get_rl_exit(states_schema)
#         # s.store_rl_values(states_schema)
#
#     def store_rl_values(s, states: nda_schema):
#         pdf = pd.DataFrame(states.nda, columns=states.schema, index=pd.to_datetime(s.arr[:, s.lud[Cr.ts]]))
#         pdf[Cr.rl_action_hold] = s.arr[:, s.lud[Cr.rl_action_hold]]
#         pdf[Cr.rl_action_exit] = s.arr[:, s.lud[Cr.rl_action_exit]]
#         # s.db_insert_preds(pdf, 'eurusd', measurement='unbinned_research_entry')
#         Influx().write_pdf(pdf,
#                            measurement='research_entry_unbinned',
#                            tags=dict(
#                                asset='eurusd',
#                                ex='2020-01-05_5',
#                            ),
#                            field_columns=pdf.columns,
#                            # tag_columns=[]
#                            )
#
#         # def store_rl_data(s):
#         #     field_names = ['rl_action_exit', 'rl_action_hold']
#         #     s.store_arr(
#         #         pd.DataFrame(s.arr[:ix_signals.ix_rl_exit - order.fill.ix_fill, [s.lud[Cr.rl_action_exit], s.lud.[Cr.rl_action_hold]]],
#         #                      columns=field_names,
#         #                      index=s.ts[order.fill.ix_fill:ix_signals.ix_rl_exit]
#         #                      ),
#         #         fields=field_names
#         #     )
#
#     def add_p_exit_given_entry(s, order):
#         s.create_arr_ix(['p_exit_given_entry'])
#         live_cols = ['roll_max_profit', 'trailing_profit_rel', 'delta_profit_rel']
#         # merge with df_full, preload into vb, load short term forecast preds and all order book stuff
#         end_ts = order.ts_fill + datetime.timedelta(seconds=len(s.arr))
#         ts_relevant = list(date_sec_range(order.ts_fill, end_ts))
#         df = pd.DataFrame(s.arr[:, [s.lud.roll_max_profit, s.lud.trailing_profit_rel, s.lud.delta_profit_rel]], columns=live_cols,
#                           index=ts_relevant)
#         df.index.name = 'ts'
#         df['min_elapsed'] = np.divide(list(range(len(s.arr))), 60).astype(int)
#         # aux_data = pd.concat([
#         #     s.df_full.loc[order.ts_fill:end_ts, :],
#         #     s.qt.loc[order.ts_fill:end_ts, :],
#         #     s.tick_forecast_raw_preds.loc[order.ts_fill:end_ts, :]
#         # ], axis=1)
#         # df = df.merge(aux_data, how='left', on='ts')
#         df = df.merge(s.df_full.loc[order.ts_fill:end_ts, :], how='left', on='ts')
#         s.arr[:, s.lud.p_exit_given_entry] = s.exit_model_from_entry.full_predict_bt(df, direction=order.direction)
#
#     def store_arr(s, pdf, fields):
#         Influx().write_pdf(pdf,
#                            measurement='backtest_curves',
#                            tags=dict(
#                                asset=s.params.asset.lower(),
#                                ex=s.params.ex,
#                                backtest_time=s.params.backtest_time
#                            ),
#                            field_columns=fields
#                            )
#
#     def ix2ts(s, ix):
#         return s.ts[ix]
#
#     def to_ix(s, ts: datetime.datetime):
#         return s.ts.get_loc(ts)
#
#     def arr_to_ix(s, arr_ts):
#         return np.where(np.isin(pd.to_datetime(s.ts), arr_ts.index, assume_unique=True))[0]
#
#     def to_ts(s, ix: int = None):
#         return s.ts[ix] if ix else s.ts[s.ix]
#
#     def _price_moved_away_too_much(s, starting_limit_price, exit_limit, strategy, order_exit) -> bool:
#         return abs(starting_limit_price - exit_limit) >= todec(strategy.mo_n_ticks * tick_size[order_exit.asset])
#
#     def load_entry_predictions_from_db(s) -> pd.DataFrame:
#         return Influx().load_p(s.params.asset, [], ex=s.params.ex_entry, from_ts=s.ts_start, to_ts=s.ts_end, load_from_training_set=s.params.load_from_training_set)
#
#     def get_curve(s, curve, start=None, end=None) -> np.ndarray:
#         """every curve should register itself. identified for frame location and column"""
#         if not all([isinstance(p, int) for p in [start, end]]):
#             start = s.to_ix(start)
#             end = s.to_ix(end)
#         return s.data[curve].iloc[start:end]
#
#     def load_ohlc(s):
#         Logger.info("Load ohlc for {} to {}".format(s.params.data_start, s.params.data_end))
#         s.ohlc_mid, s.ohlc_ask, s.ohlc_bid = get_ohlcv_mid_price(s.params, return_ask_bid=True)
#         if s._exchange_has_trade_data():
#             s.ohlc_trade = get_ohlc(
#                 end=s.params.data_end,
#                 series=Series.trade,
#                 index='ts',
#                 **s.params
#             )
#             s.ohlc_mid, s.ohlc_ask, s.ohlc_bid, s.ohlc_trade = reduce_to_intersect_ts(s.ohlc_mid, s.ohlc_ask, s.ohlc_bid, s.ohlc_trade)
#         else:
#             s.ohlc_trade = s.ohlc_mid
#         s.ix_open, s.ix_high, s.ix_low, s.ix_close = (s.ohlc_mid.columns.get_loc(c) for c in OHLC)
#         assert len(s.ohlc_trade) == len(s.ohlc_mid), 'Trade and QuoteBar Array length not identical'
#
#     def is_order_canceled_before_fill(s, order: Order, strategy) -> Union[int, None]:
#         """
#         cancel order if not filled before preds see it as opp
#         check first instance where preds are too low. if that inex before fill. cancel and set forward accordingly
#         first approach. cancel when opportunity is gone. but with >1 entry method, not aware of which opp it is
#         new approach. cancel when order price and bba is more than tick away. currently not having spike catching order
#         """
#         if order.order_type == OrderType.market:
#             return None
#         else:
#             df_quote_near, ix_hl_near = s._get_near_quote_ref(order.direction)
#             op_greater_lower = operator.ge if order.direction == Direction.long else operator.le
#             op_delta_limit = operator.add if order.direction == Direction.long else operator.sub
#             delta_ix_cancel = np.argmax(
#                 op_greater_lower(todec(df_quote_near.iloc[order.ix_signal:order.ix_signal + strategy.max_trade_length, ix_hl_near]), todec(op_delta_limit(order.price_limit, 2*0.05)))
#             )
#             if delta_ix_cancel == 0 and delta_ix_cancel + order.ix_signal < order.fill.ix_fill:
#                 return None
#             else:
#                 return delta_ix_cancel + order.ix_signal
