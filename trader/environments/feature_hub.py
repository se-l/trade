import copy
import importlib
import itertools
import pickle
import os
import click
import datetime
import numpy as np
import pandas as pd

from collections import defaultdict
from functools import partial, reduce, lru_cache
from typing import Union
from trader.backtest.order import Order
from trader.backtest.strategy_info import Strategy
from common.globals import OHLC, OHLCV
from common.modules.ctx import Ctx
from common.modules.dotdict import Dotdict
from common.modules.series import Series
from common.modules.data_store import DataStore
from common.paths import Paths
from common.refdata.named_tuples import nda_schema
from common.utils.normalize import Normalize
from common.utils.pandas_frame_plus import PandasFramePlus
from common.utils.util_func import to_list, resolve_col_name, get_model_features, rolling_window, standard_params_setup, downside_deviation, SeriesTickType
from connector.influxdb.influxdb_wrapper import InfluxClientWrapper as Influx
from common.utils.util_func import reduce_to_intersect_ts
from common.modules.enums import Direction, Exchanges
from common.modules.logger import logger
from trader.data_loader.config.talib_function_defaults import talib_selected
from trader.data_loader.load_features import load_features
from trader.data_loader.utils_features import get_ohlc, get_ohlcv_mid_price
from trader.train.reinforced.rl_agent_v2 import RLAgent
from common.refdata.curves_reference import CurvesReference as Cr

curve2f_map = {
    "return": 'norm2return',
    "bin": 'norm2kmeans_bin',
    "zscore": 'norm2zscore',
    "p_y_valley": ('p_model', {'model_name': 'p_y_valley'}),
    "p_y_peak": ('p_model', {'model_name': 'p_y_peak'}),
    # 'rl_risk_reward_ls': 'predict_rl_model',
    # 'rl_risk_reward_neutral': 'predict_rl_model',
    'rl_risk_reward_ls': ('p_model', {'model_name': 'rl_risk_reward_ls'}),
    'rl_risk_reward_neutral': ('p_model', {'model_name': 'rl_risk_reward_neutral'}),
    'regr_reward_ls_weighted': ('regr_reward_weighted', {'model_name': 'ls'}),
    'regr_reward_neutral_weighted': ('regr_reward_weighted', {'model_name': 'neutral'}),
    'rl_risk_reward_ls_actual': 'calc_risk_reward',
    'rl_risk_reward_neutral_actual': 'calc_risk_reward',
    'downside_deviation': 'downside_deviation',
    'volume_usd_10000': 'x_ref_selector',
    'second': 'x_ref_selector',
    'slope_n': 'slope_n'
}


# issues mid.close and bid.close replace each other in frame
# not merging difference time axis together
# - how to merge feats coming from different xasis. label change...
# - time based / event based. merge both and forward pad
# 	all feats derived from different x-axis need a prefix, talibs, state preds, regressions etc.


class FeatureHub:
    """Dataframe for decision making relevant data during RL Training & Backtesting."""

    def __init__(s, params, dependencies=None, datastore=None, data=None, index=None, columns=None, dtype=None, copy_=False):
        s.params = params
        s.data = datastore or DataStore()
        s.pdp = s.data['root'] = PandasFramePlus(feature_hub=s, data=data, index=index, columns=columns, dtype=dtype, copy=copy_)
        s.dependencies = dependencies or defaultdict(dict)
        s.ix_open, s.ix_high, s.ix_low, s.ix_close = (None,) * 4
        s.prefix = None

    def col_location(s, col: str) -> (bool, bool):
        category, col_name = resolve_col_name(col)
        if category:
            if s.data.get(category, {}).__contains__(col_name):
                return category, col_name
            else:
                return False, False
        else:
            found_in = []
            for category, df in s.data.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                if col in df.columns:
                    found_in.append(category)
            if len(found_in) == 1:
                return found_in[0], col
            elif not found_in:
                return False, False
            elif len(found_in) > 1:
                raise ValueError(f'Found {col} in multiple frames {",".join(found_in)}. Specify...')
        return False, False

    def _generate_curves(s, curves: list):
        # 2 processing axis. 1 curve by curve. second - within each curve. | pipe
        # curve_func_dct = defaultdict(list)
        for curve in curves:
            logger.info(f'PDP generates {curve}')
            # category, col_name = resolve_col_name(curve)
            # call necessary f to build it with ts lala constraints, might need to buffer / add buffer inputs around index
            cat, col = s.col_location(curve)
            if s._col_in_root(cat, col) or s._cp_col2root(cat, col):
                continue
            s.chain(col=curve, operations=[s.instructions2callable(instructions) for instructions in s.curve2instructions(curve)])
            # curve_func_dct[s._find_curve_func_group(col_name, category)].append(col_name)
            # for func_group, col_names in curve_func_dct.items():
            #     s._execute_curve_generators(col_names, func_group)
            #     for cat, col in [s.col_location(curve) for curve in curves]:
            cat, col = s.col_location(curve)
            if s._col_in_root(cat, col) or s._cp_col2root(cat, col):
                continue
            elif curve.startswith(s.params.asset_pair):
                continue
            else:
                raise NotImplementedError(f'Failed to generate {cat} {col}')

    def _col_in_root(s, category, column):
        return True if (f'{category}.{column}' in s.data['root'].columns) or \
                       (category and category in [None, 'root'] and column in s.data['root'].columns) else None

    def _cp_col2root(s, category, column):
        if category and column and category not in [None, 'root']:
            s.pdp[f'{category}.{column}'] = s.data[category][column]
            return True

    def block2f_kw(s, curve, out_kwargs) -> tuple:
        category, col_name = resolve_col_name(curve)
        if out_kwargs.get('in_curve') == s.params.asset_pair:
            cat = category
            col = col_name
        else:
            cat, col = s.col_location(col_name)
        if (s._col_in_root(cat, col) or s._cp_col2root(cat, col)) and out_kwargs.get('in_curve') != getattr(s.params, 'asset_pair', ''):
            return '_get_curve', out_kwargs
        elif curve.startswith('pair'):
            return '_get_pair_trade_feat', out_kwargs
        elif any((curve.startswith(f'{ab}_size_update_') for ab in ('bid', 'ask'))):
            return '_get_order_book_feat', out_kwargs
        elif any((curve.startswith(f'trade_{metric}_') for metric in ('volume', 'count'))):
            return '_get_trade_metric_feat', out_kwargs
        elif any((name in curve.upper() for name in talib_selected + ['EMA_'])):
            return '_gen_talibs', out_kwargs
        elif s.params.asset_pair and (curve.startswith(s.params.asset_pair) or out_kwargs.get('out_curve', '').startswith(s.params.asset_pair)):
            return 'load_ohlc_other_sym', out_kwargs
        elif any((snippet in name for name in [curve.lower()] for snippet in ('mid.', 'ask.', 'bid.', 'trade.'))):
            return 'load_ohlc', out_kwargs
        else:
            f_out = curve2f_map.get(curve, curve)
            return (f_out, out_kwargs) if isinstance(f_out, str) else (f_out[0], {**f_out[1], **out_kwargs})

    def curve2instructions(s, curve: str) -> list:
        curve_bldg_blocks = curve.split('|')
        curve_bldg_blocks = s.shorten_curve_instructions(curve_bldg_blocks)
        return [s.block2f_kw(block, {'out_curve': '|'.join(curve_bldg_blocks[:i + 1]),
                                     'in_curve': '|'.join(curve_bldg_blocks[:i])}) for i, block in enumerate(curve_bldg_blocks)]

    def shorten_curve_instructions(s, curve_bldg_blocks):
        # start from first actually present one
        i_present = 0
        for i, block in enumerate(curve_bldg_blocks[::-1]):
            if '|'.join(curve_bldg_blocks[:-i]) in s.pdp.columns:
                i_present = len(curve_bldg_blocks) - i
                break
        if i_present > 1:
            return ['|'.join(curve_bldg_blocks[:i_present])] + curve_bldg_blocks[i_present:]
        else:
            return curve_bldg_blocks

    def instructions2callable(s, instructions: Union[str, tuple]) -> callable:
        if isinstance(instructions, str):
            return s.__getattribute__(instructions)
        elif isinstance(instructions, tuple):
            return partial(s.__getattribute__(instructions[0]), **instructions[1])
        else:
            raise TypeError('Cannot resolve the executable instructions. Pass str(FunctionName) or tuple(FunctionName, InputDict)')

    def chain(s, col, operations: list) -> np.ndarray:
        s.pdp[col] = reduce(lambda res_prev, f_next: f_next(res_prev), operations[1:], operations[0]())
        return s.pdp[col].values

    def get_curves(s, curves: Union[list, str]) -> Union[pd.DataFrame, np.ndarray]:
        curves = to_list(curves)
        to_generate = []
        for curve in curves:
            cat, col = s.col_location(curve)
            if s._col_in_root(cat, col) or s._cp_col2root(cat, col):
                continue
            else:
                to_generate.append(curve)
        s._generate_curves(to_generate)
        # cols = [resolve_col_name(c)[-1] for c in curves]
        return s.pdp[curves] if len(curves) > 1 else s.pdp[curves[0]].values

    def _get_curve(s, curves: list = None, **kwargs):
        curves = curves or to_list(kwargs.get('out_curve'))
        return s.pdp[curves].values

    def _gen_talibs(s, curves: list = None, **kwargs):
        curves = curves or to_list(kwargs.get('out_curve'))
        params = copy.copy(s.params)
        # workaround to remove the x_ref_selector at start
        params.req_feats = [c.split('|')[-1] for c in to_list(curves)]
        params.quantize_inds = False
        df_talib_feats, df_ohlc = load_features(params, use_mid_price=True, df_ohlc=s.pdp[['mid.' + c for c in OHLCV]].rename({k: k.replace('mid.', '') for k in s.pdp.columns}, axis='columns')[OHLCV])
        return df_talib_feats[params.req_feats].values
        # s.pdp[curves] = df_talib_feats[params.req_feats]
        # s.pdp[curves].fillna(0, inplace=True)
        # return s.pdp[curves].values

    @staticmethod
    def downside_deviation(s, nda: np.ndarray, **kwargs):
        return downside_deviation(nda).values

    def slope_n(s, nda, **kwargs) -> np.ndarray:
        n = kwargs.get('slope_n') or 5
        return np.subtract(s.pdp[kwargs.get('in_curve')].values, s.pdp[kwargs.get('in_curve')].shift(n).fillna(method='bfill').values) / n

    def load_ohlc(s, **kwargs):
        logger.info("Load ohlc for {} to {}".format(s.params.data_start, s.params.data_end))
        s.data['mid'], s.data['ask'], s.data['bid'] = get_ohlcv_mid_price(s.params, return_ask_bid=True)
        if s._exchange_has_trade_data():
            s.data['trade'] = get_ohlc(
                start=s.params.data_start,
                end=s.params.data_end,
                series=Series.trade,
                **{name: s.params.__getattribute__(name) for name in ['exchange', 'series_tick_type', 'asset']}
            )
            s.data['mid'], s.data['ask'], s.data['bid'], s.data['trade'] = reduce_to_intersect_ts(s.data['mid'], s.data['ask'], s.data['bid'], s.data['trade'])
        else:
            s.data['trade'] = s.data['mid']
        s.ix_open, s.ix_high, s.ix_low, s.ix_close = (s.data['mid'].columns.get_loc(c) for c in OHLC)
        assert len(s.data['trade']) == len(s.data['mid']), 'Trade and QuoteBar Array length not identical'
        category, col_name = resolve_col_name(kwargs.get('out_curve'))
        return s.data[category][col_name]

    def load_ohlc_other_sym(s, *args, **kwargs):
        other_params = copy.deepcopy(s.params)
        other_params.asset = s.params.asset_pair
        other_params.series_tick_type = SeriesTickType('ts', 1, 'second')
        other_params.ts_start = other_params.data_start = other_params.data_start - datetime.timedelta(days=1)
        other_params.ts_end = other_params.data_end = other_params.data_end + datetime.timedelta(days=1)
        s.feat_hub_pair = FeatureHub(other_params) if not hasattr(s, 'feat_hub_pair') else s.feat_hub_pair
        if kwargs.get('out_curve') == s.params.asset_pair:
            s.pdp[kwargs.get('out_curve')] = np.nan
        else:
            # need to align shape and ts index here
            req_col = '|'.join(kwargs.get('out_curve').split('|')[1:])
            s.feat_hub_pair.pdp[req_col]
            req_ix_other = s.pdp['mid.ts'].map(s.map_ts_base2ix_other())
            s.pdp[kwargs.get('out_curve')] = s.feat_hub_pair.pdp.loc[req_ix_other, req_col].values
        return s.pdp[kwargs.get('out_curve')].values

    @lru_cache()
    def map_ts_base2ix_other(s, *args, **kwargs):
        """In order to join other ccy into base, need to align ts index
        to be cached
        """
        ts_base = s.pdp[['mid.ts']]
        ts_base = ts_base[~ts_base['mid.ts'].duplicated(keep='last')]
        ts_base['ix_base'] = ts_base.index
        ts_base['base'] = True
        ts_other = s.feat_hub_pair.pdp[['mid.ts']]
        ts_other = ts_other[~ts_other['mid.ts'].duplicated(keep='last')]
        ts_other['ix_other'] = ts_other.index
        ts_other['other'] = True
        all_ts = pd.concat((ts_base.set_index('mid.ts', drop=True), ts_other.set_index('mid.ts', drop=True)), axis=0).sort_index()
        c_other = ['ix_other', 'other']
        all_ts = all_ts.iloc[all_ts.index.get_loc(ts_base['mid.ts'].iloc[0]) - 1:
                             all_ts.index.get_loc(ts_base['mid.ts'].iloc[-1]) + 1]
        eq_ts = all_ts[(all_ts['base'] & all_ts['other'])]
        map_ts_base2ix_other = eq_ts['ix_other'].to_dict() if not eq_ts.empty else {}
        all_ts[c_other] = all_ts[c_other].shift(1)
        all_ts.drop(all_ts.index[0], inplace=True)
        all_ts['ix_other'] = all_ts['ix_other'].fillna(method='ffill').astype(int)
        all_ts = all_ts[all_ts['base'].fillna(False)]
        map_ts_base2ix_other.update(all_ts['ix_other'].to_dict())
        assert len(s.pdp['mid.ts'].unique()) == len(map_ts_base2ix_other.keys())
        return map_ts_base2ix_other

    def _exchange_has_trade_data(s):
        return True if s.params.exchange in [Exchanges.bitmex] else False

    def export_mini_arr(self, ix):
        raise NotImplemented

    def norm2kmeans_bin(s, nda: np.ndarray, normalize=None, **kwargs) -> np.ndarray:
        normalize = normalize or s.dependencies.get('normalize') or Normalize(ex=s.params.ex, range_fn='load_feature')
        if isinstance(normalize, dict):
            for k, obj in normalize.items():
                obj.store = obj.load_normalize_store('KBins') if obj.store_exists('KBins') else {}
                if kwargs.get('in_curve') in obj.load_normalize_store('KBins').keys():
                    normalize = obj
                    break
        first_nan = np.nanargmin(np.isnan(nda))
        # if first_nan > 0:
        #     logger.warning(f'Column {kwargs.get("in_curve")} contains {first_nan} null values. Consider trimming dframe.')
        result = normalize.kmeans_bin_nda(nda[first_nan:], kwargs.get("in_curve"))
        return np.concatenate((np.full(first_nan, np.nan), result.flatten()))

    def norm2return(s, nda: np.ndarray, **kwargs) -> np.ndarray:
        entry_price: float = kwargs.get('entry_price') or s.dependencies.get('entry_price')
        return nda / entry_price

    def x_ref_selector(s, *args, **kwargs):
        """until a hub, contains both x axis in one, this is not needed"""
        return None

    @staticmethod
    def norm2zscore(nda: np.ndarray, **kwargs) -> np.ndarray:
        window = kwargs.get('window', 3600)
        mean = np.mean(rolling_window(nda, window), axis=1)
        std = np.std(np.array(rolling_window(nda, window)), axis=1, dtype=np.float64)
        nda[window - 1:] = np.divide(np.subtract(nda[window - 1:], mean), std)
        nda[:window - 1] = 0
        return nda

    def elapsed(s, **kwargs):
        ts_entry: datetime.datetime = kwargs.get('ts_entry') or s.dependencies.get('ts_entry')
        s.pdp[Cr.elapsed] = (pd.to_datetime(s.pdp['mid.ts']) - ts_entry).dt.total_seconds()
        return s.pdp[Cr.elapsed].values

    def p_model(s, *args, **kwargs) -> np.ndarray:
        """
        (1) pre-computed. load from Influx
        (2) Reference model in dependencies and compute with inputs from pdp
        """
        # return s.load_p_model(kwargs.get('model_name')) or s._predict_p_model(kwargs.get('model_name'))
        return s._predict_p_model(kwargs.get('model_name'))

    def load_p_model(s, model_name, source='influx'):
        if source == 'influx' and False:  # model_name_present():
            return Influx().load_p(s.params.asset, model_name, ex=s.params.ex_entry, from_ts=s.params.ts_start, to_ts=s.params.ts_end, load_from_training_set=s.params.load_from_training_set)

    def _predict_p_model(s, model_name: str) -> np.ndarray:
        model_dct = s.dependencies.get('models', {}).get(model_name)
        load_model_from_influx = s.dependencies.get('load_model_from_influx', False)
        if load_model_from_influx:
            m2m_influx = s.dependencies.get('m2m_influx', {})
            res_ts = Influx().load_p(asset=s.params.asset, model=m2m_influx.get(model_name, model_name), ex=s.params.ex_entry, from_ts=s.pdp['ts'].iloc[0], to_ts=s.pdp['ts'].iloc[-1])
            res = res_ts.reset_index()['p'].values
            diff = len(s.pdp) - len(res)
            return np.concatenate([res, res[-diff:]])
        else:
            feature_names = list(set.union(*[set(get_model_features(m)) for m in model_dct.values()]))
            # with Pool(processes=min((multiprocessing.cpu_count() // 2)-1, len(model_dct.keys()))) as p:
            #     return np.mean(p.map(partial(EstimatorLgb.predict, target=s.pdp[feature_names]), model_dct.values()), axis=0)
            # res = np.mean([EstimatorLgb.predict(model=m, target=s.pdp[feature_names]) for m in model_dct.values()], axis=0)
            # with open(os.path.join(Paths.path_buffer, 'prediction_995'), 'wb') as f:
            #     pickle.dump(res, f)
            # with open(os.path.join(Paths.path_buffer, 'prediction_995_ts'), 'wb') as f:
            #     pickle.dump(s.pdp['ts'], f)
            with open(os.path.join(Paths.path_buffer, 'prediction_995'), 'rb') as f:
                return pickle.load(f)
            # res = res / s.dependencies.get('regr_reward_norm', {}).get(model_name, 1)

    def predict_rl_model(s, **kwargs) -> np.ndarray:
        handler_rl: RLAgent = kwargs.get('handler_rl') or s.dependencies.get('handler_rl')
        s.pdp[Cr.rl_risk_reward_neutral], s.pdp[Cr.rl_risk_reward_ls] = handler_rl.estimate_risk_reward(s.pdp, direction=Direction.long)
        return s.pdp[kwargs.get('out_curve')].values

    @staticmethod
    def decay_vec(discount_decay, min_threshold):
        if discount_decay >= 1:
            raise ValueError('Discount decay needs to be smaller than 1.')
        decay_vec = []
        x = 1
        while x > min_threshold:
            decay_vec.append(x)
            x *= discount_decay
        return decay_vec

    def calc_risk_reward(s, **kwargs) -> np.ndarray:
        """Reward of decision at t0 is the average pnl of all future ti discounted by decay_vec[-i]. i_max is reward horizon."""
        direction: Direction = kwargs.get('direction') or s.dependencies.get('direction') or Direction.long
        entry_price = s.pdp['bid.close' if direction == direction.long else 'ask.close']
        exit_price = s.pdp['ask.close' if direction == direction.long else 'bid.close']
        # *(-1 if direction == Direction.short else 1)
        decay_vec = s.decay_vec(s.params.discount_decay, s.params.discount_decay_min_threshold)
        l_decay_vec = len(decay_vec)
        res = [np.matmul(decay_vec[:len((vec := (entry_price[i:i + l_decay_vec] - exit_price[i]) / entry_price[i]))], vec.values.transpose()) for i in range(len(entry_price))]
        s.pdp[Cr.rl_risk_reward_ls_actual] = np.multiply(res, 10 ** 6) / l_decay_vec
        s.pdp[Cr.rl_risk_reward_neutral_actual] = s.pdp[Cr.rl_risk_reward_ls_actual] * -1
        return s.pdp[kwargs.get('out_curve')].values

        # if s.params.downside_risk_multiplier > 0:
        #     downside_risk = 1 + downside_deviation_rolling(pnl_return)
        #     # should be rolling downside risk. dont punish early ticks for late losses.
        #     tick_reward = np.divide(delta_pnl_return, downside_risk * s.params.downside_risk_multiplier)
        # else:
        #     tick_reward = delta_pnl_return
        # risk adjustment on tick level
        # at training time can perform risk adjustment on portfolio level
        # decay_vec = list(decay_vec)
        # mat_decay = np.array([decay_vec] + [([0] * i + decay_vec[:-i]) for i in range(1, len_delta_pnl)])
        # res = np.matmul(tick_reward, mat_decay.transpose())
        s.pdp[kwargs.get('out_curve')] = 10 ** 6 * res / ix_reward_summing_cutoff
        return s.pdp[kwargs.get('out_curve')].values

    def rl_action(s, **kwargs) -> np.ndarray:
        handler_rl: RLAgent = kwargs.get('handler_rl') or s.dependencies.get('handler_rl')
        return handler_rl.rl_action(s.pdp)

    def regr_reward_weighted(s, **kwargs) -> np.ndarray:
        if 'neutral' in kwargs.get('out_curve'):
            s.pdp[kwargs.get('out_curve')] = s.pdp[kwargs.get('out_curve').replace('neutral', 'ls')] * -1
        else:
            weights = kwargs.get('regr_reward_weights') or s.dependencies.get('regr_reward_weights')
            for model_nm in s.dependencies.get('models', {}).keys():
                s.pdp[model_nm] = s._predict_p_model(model_nm)
            s.pdp[kwargs.get('out_curve')] = np.average(s.pdp[list(weights.keys())], weights=list(weights.values()), axis=1)
        return s.pdp[kwargs.get('out_curve')].values

    def profit(s, **kwargs) -> np.ndarray:
        entry_price: float = kwargs.get('entry_price') or s.dependencies.get('entry_price')
        direction: Direction = kwargs.get('direction') or s.dependencies.get('direction')
        c = 'bid.close' if direction == direction.long else 'ask.close'
        s.pdp[Cr.profit] = (s.pdp[c] - entry_price) * (-1 if direction == direction.short else 1)
        return s.pdp[Cr.profit].values

    def rolling_max_profit(s, **kwargs):
        try:
            ix_profit = s.pdp.columns.get_loc(Cr.profit.name)
        except KeyError:
            s.pdp[Cr.profit]
            ix_profit = s.pdp.columns.get_loc(Cr.profit.name)
        m = [s.pdp[Cr.profit].iloc[0]]
        for i in range(1, len(s.pdp)):
            if s.pdp.iloc[i, ix_profit] > m[i - 1]:
                m.append(s.pdp.iloc[i, ix_profit])
            else:
                m.append(m[i - 1])
        s.pdp[Cr.rolling_max_profit] = m
        return s.pdp[Cr.rolling_max_profit].values

    def trailing_profit(s, **kwargs):
        s.pdp[Cr.trailing_profit] = s.pdp[Cr.rolling_max_profit] - s.pdp[Cr.profit]
        return s.pdp[Cr.trailing_profit].values

    def trail_profit_stop_price(s, **kwargs):
        return s.pdp['mid.close'].values
        order: Order = kwargs.get('order') or s.dependencies.get('order')
        strategy: Strategy = kwargs.get('strategy') or s.dependencies.get('strategy')
        if order.direction == direction.long:
            s.pdp[Cr.trail_profit_stop_price] = order.fill.avg_price + s.pdp[Cr.rolling_max_profit] - np.multiply(s.pdp[Cr.rolling_max_profit] + order.fill.avg_price, strategy.trail_profit_stop)
        elif order.direction == direction.short:
            s.pdp[Cr.trail_profit_stop_price] = order.fill.avg_price - s.pdp[Cr.rolling_max_profit] + np.multiply(order.fill.avg_price - s.pdp[Cr.rolling_max_profit], strategy.trail_profit_stop)
        return s.pdp[Cr.trail_profit_stop_price].values

    def _get_pair_trade_feat(s, **kwargs):
        """
        f'pair-{params.asset}-{sym}-price_ratio-{delta_time}
        pair-ethusd-xbtusd-price_ratio-123
        """
        kwargs.update(s.pair_feat_name2kwargs(kwargs.get('out_curve')))
        sym_base = kwargs.get('sym_base')
        sym_other = kwargs.get('sym_other')
        d_tick = kwargs.get('d_tick')
        col_intermediate = f'mid.close|div_{sym_other}_mid.close'
        if col_intermediate not in s.pdp.columns:
            s.pdp[col_intermediate] = np.divide(s.pdp['mid.close'], s.pdp[f'{sym_other}|mid.close'])
        # s.pdp[kwargs.get('out_curve')] = s.pdp[col_intermediate].subtract(s.pdp[col_intermediate].shift(d_tick)).fillna(0)
        return s.pdp[col_intermediate].subtract(s.pdp[col_intermediate].shift(d_tick)).fillna(0).values
        # return s.pdp[kwargs.get('out_curve')].values

    @staticmethod
    def pair_feat_name2kwargs(name):
        # pair-{params.asset}-{sym}-price_ratio-{delta_time}
        return {
            'sym_base': name[5:][:6],
            'sym_other': name[12:][:6],
            'd_tick': int(name.split('|')[0][31:]),
        }

    def _get_trade_metric_feat(s, **kwargs):
        kwargs.update(s.trade_metric_feat_name2kwargs(kwargs.get('out_curve')))
        col_name = f'trade.{kwargs.get("metric")}_{kwargs.get("direction")}'
        nda = np.full(len(s.pdp), np.nan)
        nda[kwargs.get('d_tick')-1:] = np.sum(rolling_window(s.pdp[col_name], kwargs.get('d_tick')), axis=1)
        return nda
        # s.pdp[kwargs.get('out_curve')] = np.nan
        # s.pdp[kwargs.get('out_curve')].iloc[kwargs.get('d_tick')-1:] = np.sum(rolling_window(s.pdp[col_name], kwargs.get('d_tick')), axis=1)
        # return s.pdp[kwargs.get('out_curve')].values

    @staticmethod
    def trade_metric_feat_name2kwargs(name):
        """trade_count_buy_sum_{delta_time}"""
        return {
            'metric': name.split('_')[1],
            'direction': name.split('_')[2],
            'd_tick': int(name.split('_')[-1]),
        }

    def weekday(s, **kwargs):
        return s.pdp['mid.ts'].dt.weekday

    def _get_order_book_feat(s, **kwargs):
        kwargs.update(s.order_book_feat_name2kwargs(kwargs.get('out_curve')))
        ba_col_name = f'{kwargs.get("bid_ask")}.size_{kwargs.get("direction")}' + ('_count' if kwargs.get('count') else '')
        nda = np.full(len(s.pdp), np.nan)
        nda[kwargs.get('d_tick') - 1:] = np.sum(rolling_window(s.pdp[ba_col_name], kwargs.get('d_tick')), axis=1)
        return nda
        # s.pdp[kwargs.get('out_curve')] = np.nan
        # s.pdp[kwargs.get('out_curve')].iloc[kwargs.get('d_tick')-1:] = np.sum(rolling_window(s.pdp[ba_col_name], kwargs.get('d_tick')), axis=1)
        # return s.pdp[kwargs.get('out_curve')].values

    @staticmethod
    def order_book_feat_name2kwargs(name):
        """{ab}_size_update_remove_cnt_sum_{delta_time}
            ask_size_update_remove_cnt_sum_123
        """
        return {
            'bid_ask': name.split('_')[0],
            'direction': {'remove': 'removed', 'add': 'added'}.get(name.split('_')[3]),
            'metric': 'count' if name.split('_')[4] == 'cnt' else 'size',
            'd_tick': int(name.split('_')[-1]),
        }

    def trailing_stop_loss(s, entry_price=None, max_trailing_stop_a=None, trailing_stop_b=None, **kwargs):
        entry_price: float = entry_price or s.dependencies.get('entry_price')
        max_trailing_stop_a: float = max_trailing_stop_a or s.dependencies.get('max_trailing_stop_a')
        trailing_stop_b: float = trailing_stop_b or s.dependencies.get('trailing_stop_b')
        s.pdp[Cr.trailing_stop_loss] = entry_price * max_trailing_stop_a - trailing_stop_b * s.pdp[Cr.rolling_max_profit]
        # s.pdp[Cr.stop_loss] = np.where(s.pdp[Cr.stop_loss] < min_stop_loss * entry_price, min_stop_loss * entry_price, s.pdp[Cr.stop_loss])
        return s.pdp[Cr.trailing_stop_loss].values

    def store_rl_values(s, states: nda_schema):
        pdf = pd.DataFrame(states.nda, columns=states.schema, index=pd.to_datetime(s.pdp[Cr[Cr.ts]]))
        pdf[Cr.rl_action_hold] = s.pdp[Cr[Cr.rl_action_hold]]
        pdf[Cr.rl_action_exit] = s.pdp[Cr[Cr.rl_action_exit]]
        # s.db_insert_preds(pdf, 'eurusd', measurement='unbinned_research_entry')
        Influx().write_pdf(pdf,
                           measurement='research_entry_unbinned',
                           tags=dict(
                               asset='eurusd',
                               ex=s.params.ex,
                           ),
                           field_columns=pdf.columns,
                           # tag_columns=[]
                           )

    def copy_extract(s, start: Union[int, datetime.datetime], end: Union[int, datetime.datetime] = None):
        start = s.to_ix(start) if isinstance(start, datetime.datetime) else start
        end = s.to_ix(end) if isinstance(end, datetime.datetime) else end
        return FeatureHub(params=s.params, dependencies=s.dependencies,
                          datastore=DataStore({name: df.loc[start:end].reset_index(drop=True) for name, df in s.data.items() if isinstance(df, pd.DataFrame) if name != 'root'}),
                          data=s.data['root'].loc[start:end].reset_index(drop=True)
                          )

    def store_arr(s, pdf, fields):
        Influx().write_pdf(pdf,
                           measurement='backtest_curves',
                           tags=dict(
                               asset=s.params.asset.lower(),
                               ex=s.params.ex,
                               backtest_time=s.params.backtest_time
                           ),
                           field_columns=fields
                           )

    def ts(s):
        s.pdp[Cr.ts] = pd.to_datetime(s.data.get('mid')['ts'])
        return s.pdp[Cr.ts].values
        # pd.to_datetime((
        #         s.data.get('mid') or s.data.get('trade') or s.data.get('ask') or s.data.get('bid')
        # )['ts'])

    def ix2ts(s, ix):
        return s.pdp['mid.ts'][ix]

    def to_ix(s, ts: datetime.datetime):
        # may receive datetimes that are not in index, pick closest in future
        try:
            eq_range = s.pdp.index[s.pdp[Cr.ts] == ts]
            return eq_range[-1] if len(eq_range) > 0 else s.pdp.index[s.pdp[Cr.ts] >= ts][0]
        except IndexError:
            logger.warning(f'Index {ts} not out of range. Returning last: {s.pdp[Cr.ts].iloc[-1]}')
            return s.pdp.index[-1]

    def arr_to_ix(s, arr_ts):
        return np.where(np.isin(pd.to_datetime(s.pdp[Cr.ts]), arr_ts.index, assume_unique=True))[0]

    def to_ts(s, ix: int = None):
        return s.pdp[Cr.ts][ix]

    def assign_event_series_prefix(s, prefix=None):
        s.prefix = prefix or s.params.series_tick_type.folder
        for key, frame in s.data.items():
            if isinstance(frame, str):
                continue
            s.data[key].columns = [s.prefix + '|' + c for c in s.data[key].columns]

    def merge_cols(s, pdf):
        for target_col in ['ts'] + [''.join(tup) for tup in itertools.product(['mid.', 'ask.', 'bid.', 'trade.'], OHLC)]:
            match_cols = [c for c in pdf.columns if c.endswith(target_col)]
            if not match_cols:
                continue
            elif len(match_cols) == 1:
                pdf = pdf.rename({match_cols[0]: target_col}, axis='columns')
            else:
                pdf[target_col] = pdf[match_cols[0]]
                for c in match_cols[1:]:
                    pdf[target_col].fillna(pdf[c], inplace=True)
                pdf.drop(match_cols, inplace=True, axis=1)
        return pdf

    def merge_event_series(s, fhub):
        for hub in [fhub, s]:
            if not any((c.endswith('ts') for c in hub.pdp.columns)):
                hub.pdp['mid.ts']
        # if not s.prefix:
        #     s.assign_event_series_prefix()
        # if not fhub.prefix:
        #     fhub.assign_event_series_prefix()
        for df_name, df in fhub.data.items():
            if df_name in s.data.keys() and not isinstance(df, str):  # and df_name == 'root'
                s.data[df_name] = pd.concat((s.data[df_name], df))
                s.data[df_name] = s.merge_cols(s.data[df_name])
                s.data[df_name] = s.data[df_name].sort_values('ts')
                s.data[df_name] = s.data[df_name][~s.data[df_name]['ts'].duplicated(keep='last')]
                s.data[df_name] = s.data[df_name].bfill().ffill().reset_index(drop=True)
        s.pdp = s.data['root'] = PandasFramePlus(feature_hub=s, data=s.data['root'])

    def rm_duplicated_ts(s):
        """For RL training purposed, there is no point in trying to keep multiple event per micro sec, just last event."""
        for df_name, df in s.data.items():
            if df_name in s.data.keys() and not isinstance(df, str):
                s.data[df_name] = s.data[df_name][~s.data[df_name]['ts'].duplicated(keep='last')]
        s.pdp = s.data['root'] = PandasFramePlus(feature_hub=s, data=s.data['root'])

    def trim_frame(s):
        max_ = s.pdp.isna().max()
        logger.info(f'Dropping {max_} leading rows.')
        s.pdp.drop(s.pdp.index[:max_], inplace=True)
        # issue here - need to also drop other dfs in s.datastore

    def curtail_ts(s, start, end):
        for df_name, df in s.data.items():
            if not isinstance(df, pd.DataFrame):
                continue
            s.data[df_name] = df.loc[(df['ts'] >= start) & (df['ts'] <= end)].reset_index(drop=True)
        s.pdp = s.data['root'] = PandasFramePlus(feature_hub=s, data=s.data['root'])


@click.command('test_fhub')
@click.pass_context
def test_fhub(ctx: Ctx):
    params = importlib.import_module('{}.{}'.format(Paths.path_config_reinforced, ctx.obj.fn_params)).Params()
    standard_params_setup(params, Paths.backtests)
    fh = FeatureHub(params)
    # fh.pdp['mid.close']
    # fh.pdp['mid.volume']
    # fh.pdp[['MOM_real_360', "MOM_real_23"]]  # need to provide quantizer object in
    # fh.pdp['mid.ts']

    # fh.pdp['profit|return']
    # fh.pdp["MOM_real_360|bin"]
    # fh.pdp["MOM_real_23|return|bin"]
    fh.dependencies['entry_price'] = 100
    fh.dependencies['max_trailing_stop_a'] = 100
    fh.dependencies['trailing_stop_b'] = 10
    fh.dependencies['direction'] = Direction.long
    import pickle
    with open(os.path.join(r'C:\repos\trade2\model\supervised\ex2020-05-24_10-51-43-ethusd', 'model_classification_lgb_rd-n_ts-1590288912.309'), "rb") as f:
        obj_ = pickle.load(f)
    fh.dependencies['models'] = {'p_y_peak': obj_, 'p_y_valley': obj_}
    fh.dependencies['ts_entry'] = datetime.datetime(2019, 8, 25, 1, 1, 1)
    fh.dependencies['handler_rl'] = RLAgent(params, fh).setup()  # mem_size=len(fh.pdp['mid.close']) * params.mem_x_instances_factor)
    fh.get_curves('rl_risk_reward_ls')
    fh.pdp[['trail_profit_stop_price|return', 'profit|return', 'rolling_max_profit|return', 'MOM_real_360|return|bin', 'p_y_peak', 'elapsed', 'EMA_real_540|return|bin', 'trailing_profit|return',
            'p_y_valley', 'MOM_real_23|return|bin', 'trailing_stop_loss|return']]
    # fh.pdp['rl_risk_reward_neutral']
    # fh.pdp['rl_action']
    # copy little array


@click.command('test_axis_merge')
@click.pass_context
def test_axis_merge(ctx: Ctx):
    params = importlib.import_module('{}.{}'.format(Paths.path_config_supervised, ctx.obj.fn_params)).Params()
    standard_params_setup(params, Paths.backtests)
    # params.series_tick_type = SeriesTickType('ts', params.resample_sec, 'second')
    # fh_ts = FeatureHub(params)
    params_vol = copy.copy(params)
    params_vol.series_tick_type = SeriesTickType('volume_usd', 10000, 'volume_usd_10000')
    fh_vol = FeatureHub(params_vol)
    fh_vol.pdp['mid.close']
    fh_vol.pdp['mid.ts']
    delta_time = 123
    fh_vol.pdp['xbtusd|mid.close']
    fh_vol.pdp['pair-ethusd-xbtusd-price_ratio-123|bin']
    fh_vol.pdp['MOM_real_8730|bin']
    # for ab in ['ask', 'bid']:
    #     fh_vol.pdp[f'{ab}_size_update_add_sum_{delta_time}']
    #     fh_vol.pdp[f'{ab}_size_update_remove_sum_{delta_time}']
    #     fh_vol.pdp[f'{ab}_size_update_add_cnt_sum_{delta_time}']
    #     fh_vol.pdp[f'{ab}_size_update_remove_cnt_sum_{delta_time}']
    # fh_vol.pdp['weekday']
    #
    # fh_vol.pdp[f'trade_volume_buy_sum_{delta_time}']
    # fh_vol.pdp[f'trade_volume_sell_sum_{delta_time}']
    # fh_vol.pdp[f'trade_count_buy_sum_{delta_time}']
    # fh_vol.pdp[f'trade_count_sell_sum_{delta_time}']

    # fh_ts.assign_event_series_prefix('ts')
    # fh_vol.assign_event_series_prefix('volume_usd_10000')
    # fh_ts.merge_event_series(fh_vol)
    # fh_ts.pdp['mid.close']
    a = 1


@click.command()
@click.pass_context
def main(ctx):
    ctx.obj = Dotdict(dict(
        fn_params='ethusd_train'
    ))
    # ctx.forward(test_fhub)
    ctx.forward(test_axis_merge)


if __name__ == '__main__':
    main()
