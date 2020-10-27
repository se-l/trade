from functools import reduce

import operator
import numpy as np
import pandas as pd

from hyperopt import pyll, STATUS_OK
from typing import Union
from collections import deque

from trader.backtest.order import Order
from trader.backtest.stops_finder import StopsFinder
from trader.backtest.strategy_info import StrategyLib, Strategy
from common.utils.util_func import todec
from connector import store_bt_pnl, store_rewards
from trader.environments.brokerage import Brokerage
from trader.backtest.fast_forward import FastForward
from trader.backtest.signal_type import SignalType
from trader.backtest.system import System
from common.modules import dotdict
from trader.backtest.config.optimize_params import OptParams
from common.modules import direction, order_type, side, signal_source
from trader.environments.feature_hub import FeatureHub
from common.modules.logger import logger
from common.refdata import CurvesReference as Cr


class AgentBacktest(StopsFinder):
    """Handles
    - Sending orders to broker
    - Decision when to order
    - Providing logic when to cancel, exit
    - Drive Training & Backtesting
    """

    def __init__(s, params, brokerage, system, feature_hub, handler_rl=None):
        s.strategy_lib, s.p_opt, s.fhub, s.fast_forward = (None,) * 4
        s.params = params
        s.system: System = system
        s.brokerage: Brokerage = brokerage
        s.feature_hub: FeatureHub = feature_hub
        s.handler_rl = handler_rl
        s.portfolio_side = side.hold
        s.new_orders, s.ix_entry_preds = ([],) * 2
        s.strat_entry = {}
        s.trade_len = deque(maxlen=30)

    def get_best_entry_min_delta_rl_risk_reward(s, indices, pnl):
        delta_rewards = (s.feature_hub.pdp[Cr.rl_risk_reward_ls] - s.feature_hub.pdp[Cr.rl_risk_reward_neutral]).iloc[indices]
        pdf = pd.DataFrame(zip(pnl, delta_rewards), columns=['pnl', 'dr']).sort_values('dr', ascending=False)
        pdf['cum_pnl'] = pdf['pnl'].cumsum()
        return 0 if pdf.empty else pdf.loc[np.argmax(pdf['cum_pnl']), 'dr'] / 2

    def get_best_exit_min_delta_rl_risk_reward(s, indices, pnl):
        # if actual profit's higher going forward shouldn have exited.
        pass

    def find_next_entry(s, direction=direction.long):
        # every time this generator is exhausted we'd expect a new RL model, hence need to recalc the preds
        strategy = s.strategy_lib.get_strategy(0)
        s.drop_estimated_rewards(s.feature_hub.pdp)
        s.feature_hub.pdp = s.set_order_dependent_feats2tick0(s.feature_hub.pdp, direction)
        rl_risk_reward_ls = s.feature_hub.pdp[Cr.regr_reward_ls_weighted]
        rl_risk_reward_neutral = s.feature_hub.pdp[Cr.regr_reward_neutral_weighted]
        # for thresh in np.arange(0.002, 0.01, 0.001):
        #     entries_class = s.feature_hub.pdp['volume_usd_10000|p_y_valley'] >= thresh  # s.strategy_lib.get_strategy(0).
        #     # entries_class = reduce(lambda x, y: np.logical_and(x, y), [
        #     #     s.feature_hub.pdp["second|MOM_real_23"] >= 0,
        #     #     s.feature_hub.pdp['ts'] >= s.params.ts_start,
        #     #     s.feature_hub.pdp['ts'] <= s.params.ts_end
        #     # ])
        #     if entries_class.sum() < len(s.feature_hub.pdp) // 10:
        #         logger.info(f'Entry thresh: {thresh}')
        #         logger.info(f'{round(100 * entries_class.sum() / len(entries_class), 2)}% of episode tagged for entry CLASS.')
        #         break
        entries = reduce(lambda x, y: np.logical_and(x, y), [
            rl_risk_reward_ls > 0,
            rl_risk_reward_ls > rl_risk_reward_neutral,
            (rl_risk_reward_ls - rl_risk_reward_neutral) > strategy.entry_min_delta_rl_risk_reward,
            # s.feature_hub.pdp['rl_risk_reward_ls|slope_n'] < s.params.entry_max_slope_rl_risk_reward,
            # s.feature_hub.pdp['regr_reward_ls_weighted|slope_n'] < strategy.entry_max_slope_rl_risk_reward,
            s.feature_hub.pdp['ts'] >= s.params.ts_start,
            s.feature_hub.pdp['ts'] <= s.params.ts_end
        ])
        logger.info(f'{round(100 * entries.sum() / len(entries), 2)}% of episode tagged for entry.')
        s.drop_order_dependent_feats(s.feature_hub.pdp)
        for ix in entries[entries].index:  # np.where(entries)[0]:
            yield ix, [0]  # 0 as we default strategy for now. just 1 - long or short.

    def train_via_backtest(s):
        strategy_lib = StrategyLib([
            Strategy(0, s.params.asset, direction.long),
            # Strategy(1, s.params.asset, Direction.short),
        ])
        s.strategy_lib, p_opts = s.setup_opt_params(strategy_lib, s.params)
        s.set_p_opt(p_opts)

        for j in range(s.params.num_training_iterations):
            s.reset_episode()
            s.backtest_single_ensemble()
            logger.info(f'Episode: {j} | {round((float(j) / float(s.params.num_training_iterations)) * 100.0, 2)} %')
            logger.info(s.brokerage.episode_summary())
            # logger.info(s.handler_rl.train_summary())
            if True:  # s.backtest_agent.handler_rl.exploration_rate < s.params.exploration_rate_to_min_thresh:  # and s.num_training_iterations % 10 == 0:
                s.brokerage.store_backtest(j)
                store_bt_pnl(s.brokerage.orders, j, s.params)
                store_rewards(s.feature_hub.pdp,
                              [Cr.regr_reward_ls_weighted, Cr.regr_reward_neutral_weighted],
                              j, s.params, overwrite=True)
                # store_rewards(s.handler_rl.trade_rewards2episode_rewards(),
                #               [Cr.rl_risk_reward_neutral_actual, Cr.rl_risk_reward_ls_actual], #, Cr.rl_risk_reward_neutral, Cr.rl_risk_reward_ls],
                #               j, s.params, overwrite=True)

    def backtest_single_ensemble(s):
        n_exits = 0
        for ix_entry, strategy_ids in s.find_next_entry():
            s.new_orders = s.brokerage.process_proposed_orders_invalidate_future(s.new_orders)
            if ix_entry is None:
                continue
            s.brokerage.confirm_future_orders(ix_entry)
            s.fast_forward.update_w_orders(s.brokerage.future_orders, ix_entry)
            for strategy_id in strategy_ids:
                if ix_entry <= s.fast_forward.ff[strategy_id]:
                    continue
                strategy = s.strategy_lib.get_strategy(strategy_id)
                # FOR MODEL ENTRY ON SIGNAL
                s.portfolio_side = s.brokerage.get_portfolio_side()
                if not s.is_signal(strategy.direction, s.portfolio_side):
                    continue
                else:
                    # get rl_predictions to test whether order gets an immediate exit...
                    # if s.arr[0, s.lud.rl_action_exit] > s.arr[0, s.lud.rl_action_hold]:
                    #     return None
                    direction = direction.long if strategy.direction == direction.long else direction.short
                    order = s.brokerage.create_order_ticket(ix_entry, direction, signal_source.model_p, order_type.market, strategy)
                    order.ts_signal = s.feature_hub.ix2ts(order.ix_signal)
                order.fill = s.brokerage.get_order_fill(order, strategy, move_to_nearest_bid_ask=False)
                order.ix_cancel = s.is_order_canceled_before_fill(order, strategy)
                if order.ix_cancel:
                    s.fast_forward.update_ff_timeout(strategy.id, order.ix_cancel)
                    # if s.order_timed_out(order, strategy):
                    #     s.fast_forward.update_ff_timeout(strategy.id, ix_entry + strategy.time_entry_cancelation)
                    continue
                s.set_feathub_mini(order, strategy)
                ix_exit_signals = s.get_exit_signals(order, strategy)
                order_exit = s._get_order_exit(ix_exit_signals, strategy)
                # rl_risk_reward_neutral[order.ix_signal:order_exit.ix_signal]
                # rl_risk_reward_ls[order.ix_signal:order_exit.ix_signal]
                # if s.feature_hub.pdp[Cr.rl_risk_reward_ls].iloc[order.fill.ix_fill] - s.feature_hub.pdp[Cr.rl_risk_reward_neutral].iloc[order.fill.ix_fill] < 0:
                #     a = 2
                s.buffer_trade_window(order_exit.fill.ts_fill - order.fill.ts_fill)
                # if isinstance(order_exit, int):  # todo: refactor. hella mis-named
                #     s.fast_forward.update_ff_timeout(strategy.id, order_exit)
                # else:
                s.new_orders += [order, order_exit]
                # s.handler_rl.store_trade_reward(s.arr, order, order_exit)
                if s.params.max_evals < 2:
                    s.brokerage.log_profit(order, order_exit)
                n_exits += 1
                if n_exits > 800:
                    s.brokerage.confirm_future_orders(len(s.feature_hub.data['mid']))
                    return
                # s.brokerage.log_tick_pnl(order, order_exit)

        # last ix entry passed. 1 order may still be in the future orders queue
        s.brokerage.confirm_future_orders(len(s.feature_hub.data['mid']))
        # s.params.entry_min_delta_rl_risk_reward = s.get_best_entry_min_delta_rl_risk_reward(
        #     [s.brokerage.orders[i].ix_signal for i in range(len(s.brokerage.orders)) if i % 2 == 0],
        #     s.brokerage.profits
        # )
        # logger.info(f'entry_min_delta_rl_risk_reward: {s.params.entry_min_delta_rl_risk_reward}')

    def reset(s):
        s.portfolio_side = side.hold
        s.new_orders = []
        s.fast_forward = FastForward(s.strategy_lib)

    def reset_episode(s):
        """resetting all objects, simulating new start without reloading data unnecessarily"""
        s.reset()
        s.brokerage.reset()

    def set_feathub_mini(s, order: Order, strategy, buffer_trade_window=None):
        s.fhub = s.feature_hub.copy_extract(order.fill.ts_fill, order.fill.ts_fill + (buffer_trade_window or s.buffer_trade_window()))
        s.fhub.dependencies['ts_entry'] = order.fill.ts_fill
        s.fhub.dependencies['entry_price'] = order.fill.avg_price
        s.fhub.dependencies['direction'] = strategy.direction
        s.arr = s.fhub.pdp
        s.drop_order_dependent_feats(s.arr)
        s.drop_estimated_rewards(s.arr)

    @staticmethod
    def drop_estimated_rewards(pdp):
        pdp.drop([c for c in pdp.columns if c in [Cr.rl_risk_reward_neutral, Cr.rl_risk_reward_ls, 'rl_risk_reward_ls|slope_n']], axis=1, inplace=True)

    @staticmethod
    def drop_order_dependent_feats(pdp):
        c_drop = [Cr.elapsed, 'profit', 'rolling_max_profit', 'rolling_max_profit|return',
                  'trailing_profit', 'trailing_profit|return', 'profit|return',
                  'trailing_stop_loss', 'trailing_stop_loss|return',
                  'trail_profit_stop_price|return']
        c_drop += [c + '|return' for c in ['second|MOM_real_360', 'second|MOM_real_23', 'second|EMA_real_540']]
        pdp.drop([c for c in pdp.columns if c in c_drop], axis=1, inplace=True)

    @staticmethod
    def set_order_dependent_feats2tick0(pdp, direction):
        pdp[Cr.elapsed] = 0
        c_exit, c_entry = ('bid.close', 'ask.close') if direction == direction.long else ('ask.close', 'bid.close')
        pdp[Cr.profit] = (pdp[c_exit] - pdp[c_entry]) * (-1 if direction == direction.short else 1)
        pdp[Cr.rolling_max_profit] = pdp[Cr.profit]
        pdp['rolling_max_profit|return'] = pdp[Cr.rolling_max_profit] / pdp[c_entry]
        for c in ['second|MOM_real_360', 'second|MOM_real_23', 'second|EMA_real_540']:
            pdp[c + '|return'] = pdp[c] / pdp[c_entry]
        return pdp

    def _get_order_exit(s, ix_signals, strategy):
        min_ix, exit_signal_source = SignalType.determine_exit_signal_source(ix_signals)
        # if exit_signal_source == SignalSource.ix_elapsed:
        #     return min_ix
        order_type = order_type.market if exit_signal_source in s.params.exit_signal_market_order else order_type.limit
        direction = direction.long if strategy.direction == direction.short else direction.short
        order_ticket_exit = s.brokerage.create_order_ticket(min_ix, direction, exit_signal_source, order_type, strategy)
        order_ticket_exit.ts_signal = s.feature_hub.ix2ts(min_ix)
        order_ticket_exit.fill = s.brokerage.get_order_fill(order_ticket_exit, strategy, move_to_nearest_bid_ask=False)
        return order_ticket_exit

    def set_p_opt(s, p_opt):
        s.p_opt = p_opt
        s.strategy_lib.update_p_opt(p_opt)
        # s.strategy_lib.set_trailing_stop_b()

    def get_score(s):
        s.brokerage.assign_order_quantities()
        trades = s.brokerage.calc_portfolio_value()
        # assert sum(trades) == s.brokerage.cash, 'sum of trades not equal to cash. check...'
        s.brokerage.calc_win_loss_ratio()
        logger.info('Win Loss ratio: {}  - n-win_loss-trades: {}'.format(s.brokerage.win_loss_ratio, len(s.brokerage.win_loss_trades)))
        if s.brokerage.params.max_evals == 1:
            logger.info('Trades: {}'.format(trades))
        logger.info('n orders: {}, profit: {}'.format(len(s.brokerage.orders), s.brokerage.cash))  # np.sum(s.brokerage.stats['profit_after_fees']))
        logger.info('Max. Drawdown: {}'.format(s.brokerage.calc_max_drawdown(trades)))
        logger.info('LO / MO fill loss: {}'.format(sum([(o.fill.avg_price - o.price_limit) for o in s.brokerage.orders])))

        s.system.store_p_exec(s.p_opt, {'profit': s.brokerage.cash})
        return {
            'loss': -1 * sum(trades),  # s.brokerage.cash,  # np.sum(s.brokerage.stats['profit_after_fees']),
            # 'loss': -1 * len([t for t in np.divide(s.brokerage.stats['profit_after_fees'], s.brokerage.stats['price_limit_entry']) if t > 10*bp]),  # maximise trades. ensure some profit made 10 bp
            'status': STATUS_OK,
            'stats': {},  # s.brokerage.stats
            'orders': s.brokerage.orders,
            # 'vb': vb
        }

    def setup_opt_params(s, strategy_lib, params, opt_params_str=None):
        for i, strategy in enumerate(strategy_lib.lib):
            strategy_lib.lib[i].p_opt = OptParams(strategy_lib.lib[i].direction)
        strategy_lib.set_p_opt_map()
        strategy_lib.override_p_opt_labels()
        if params.max_evals == 1 and params.use_exec_opt_param and s.system.load_p_exec() and opt_params_str is None:
            p_opt = s.system.load_p_exec()
        elif opt_params_str is not None:
            p_opt = s.system.load_p_exec()
        elif params.max_evals > 1 and params.replace_exec_param_where_scan and s.system.load_p_exec():
            p_opt = s.system.load_p_exec()
            p_opt_scan = strategy_lib.merge_p_opts(params.max_evals)
            for key, val in p_opt_scan.items():
                if type(val) == pyll.base.Apply:
                    p_opt[key] = val
        else:
            p_opt = strategy_lib.merge_p_opts(params.max_evals)
        return strategy_lib, dotdict(p_opt)

    def get_exit_signals(s, order: Order, strategy, recursion=None) -> Union[dotdict, None]:
        n_ix_remaining = len(s.feature_hub.data['mid']) - order.fill.ix_fill
        ix_end = order.fill.ix_fill + n_ix_remaining
        s.set_ix_re_entry(order, strategy, ix_end)
        ix_exit_signals = s._get_ix_exit_signal(order, strategy)
        min_ix, exit_signal_source = SignalType.determine_exit_signal_source(ix_exit_signals)
        if exit_signal_source == signal_source.ix_elapsed and min_ix < ix_end - 1 and (recursion or 0) < 5:
            # arr was too small request bigger and call recursive
            recursion = 0 if recursion is None else recursion + 1
            if recursion == 4:
                logger.info('Recursion at 4. Not extending trade window any further.')
            s.set_feathub_mini(order, strategy, buffer_trade_window=(s.arr.iloc[-1]['ts'] - s.arr.iloc[0]['ts']) * 2)
            return s.get_exit_signals(order, strategy, recursion=recursion)
        return ix_exit_signals

    def _get_ix_exit_signal(s, order, strategy):
        ix_signals = SignalType.init()
        ix_ema_veto = None  # todo: re-introduce this
        # todo - type of spot check need to be config driven. A large set of methods. So also optimizable
        # ix_signals.ix_other_peak_entry_stop = s.find_other_peak_entry_stop(order, strategy, veto_ix=ix_ema_veto)
        if strategy.use_rl_exit:
            ix_signals.ix_rl_exit = s.find_exit_rl_stop(order, strategy, veto_ix=ix_ema_veto)
        if strategy.use_regr_reward_exit:
            ix_signals.ix_regr_reward_exit = s.find_exit_regr_reward_stop(order, strategy, veto_ix=ix_ema_veto)
        ix_signals.ix_elapsed = s.find_exit_elapsed(order, strategy)
        return ix_signals

    def would_reenter_same_sec(s, min_ix):
        """
        only for the entry orders as still finding exit order.
        checks if strategy going opposite direction to exit order exists for the exit time.
        """
        return min_ix in s.ix_reentry_set

    def set_ix_re_entry(s, order, strategy, ix_end):
        try:
            s.ix_reentry_set = list(set(
                np.intersect1d(s.strat_entry[strategy.id][order.fill.ix_fill < s.strat_entry[strategy.id]],
                               s.strat_entry[strategy.id][s.strat_entry[strategy.id] < ix_end])
            ))
        except KeyError:
            s.ix_reentry_set = []

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

    def is_order_canceled_before_fill(s, order: Order, strategy) -> Union[int, None]:
        """
        cancel order if not filled before preds see it as opp
        check first instance where preds are too low. if that inex before fill. cancel and set forward accordingly
        first approach. cancel when opportunity is gone. but with >1 entry method, not aware of which opp it is
        new approach. cancel when order price and bba is more than tick away. currently not having spike catching order
        """
        if order.order_type == order_type.market:
            return None
        else:
            df_quote_near, ix_hl_near = s._get_near_quote_ref(order.direction)
            op_greater_lower = operator.ge if order.direction == direction.long else operator.le
            op_delta_limit = operator.add if order.direction == direction.long else operator.sub
            delta_ix_cancel = np.argmax(
                op_greater_lower(todec(df_quote_near.iloc[order.ix_signal:order.ix_signal + strategy.max_trade_length, ix_hl_near]), todec(op_delta_limit(order.price_limit, 2 * 0.05)))
            )
            if delta_ix_cancel == 0 and delta_ix_cancel + order.ix_signal < order.fill.ix_fill:
                return None
            else:
                return delta_ix_cancel + order.ix_signal

    @staticmethod
    def is_signal(signal_direction, portf_side):
        if portf_side == side.hold:
            return True
        else:
            # when portfolio already long, enter another long
            # had problem that stops triggered a portf_side of long.
            return signal_direction != portf_side

    def buffer_trade_window(s, duration=None):
        if duration:
            s.trade_len.append(duration)
        else:
            durations = list(s.trade_len)
            if len(durations) < 5:
                return s.params.max_trade_window
            else:
                tdelta = np.percentile(durations, 90) * 1.5
                # logger.debug(f'Buffer + : {tdelta}')
                return tdelta
