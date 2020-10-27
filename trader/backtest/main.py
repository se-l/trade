import importlib

import click
import datetime
import numpy as np
import pandas as pd
import os
from functools import partial

from trader.backtest.signal_type import SignalType
from common.modules import ctx
from common.utils.util_func import create_dir, set_ex
from hyperopt import STATUS_OK, Trials, pyll
from common.modules.logger import Logger
from trader.train.supervised.estimators.estimator_base import EstimatorBase
from trader.backtest.vectorized_backtest import Backtest
from trader.backtest.config.optimize_params import OptParams
from trader.backtest.order import Order
from trader.backtest.strategy_info import Strategy, StrategyLib
from trader.backtest.fast_forward import FastForward
from common.modules import direction, side,  signal_source, timing, order_type
from common import Paths
from common.modules import dotdict


def backtest_multiple_ensembles(strategy_lib, params):
    hp_results, vb = optimize_single_ensemble(
        strategy_lib=strategy_lib,
        params=params.to_dict()
    )
    merged_param = hp_results[0][0][0]
    if params.max_evals == 1:
        vb.store_backtest()
    if params.store_input_curves:
        vb.store_input_curves()
    Logger.info('Best merged params: {}'.format(merged_param))
    return merged_param


def optimize_single_ensemble(strategy_lib, params=None):
    params = dotdict(params)
    vbm = Backtest(params=params)
    vbm.setup()
    strategy_lib, vbm.p_opt = setup_opt_params(strategy_lib, vbm, params)

    vbm.set_strategy_lib(strategy_lib)
    hyperopt_trials, best_params = hp_single_ensemble(vbm, params.max_evals)

    kf_result = [(hyperopt_trials.best_trial['result']['orders'], best_params)]
    if params.max_evals > 1:
        return kf_result, None
    else:
        return kf_result, vbm


def setup_opt_params(strategy_lib, vbm, params, opt_params_str=None):
    strategy_lib.lib[0].p_opt = OptParams(strategy_lib.lib[0].direction)
    strategy_lib.lib[1].p_opt = OptParams(strategy_lib.lib[1].direction)
    strategy_lib.set_p_opt_map()
    strategy_lib.override_p_opt_labels()
    if params.max_evals == 1 and params.use_exec_opt_param and vbm.load_p_exec() and opt_params_str is None:
        p_opt = vbm.load_p_exec()
    elif opt_params_str is not None:
        p_opt = vbm.load_p_exec()
    elif params.max_evals > 1 and params.replace_exec_param_where_scan and vbm.load_p_exec():
        p_opt = vbm.load_p_exec()
        p_opt_scan = strategy_lib.merge_p_opts(params.max_evals)
        for key, val in p_opt_scan.items():
            if type(val) == pyll.base.Apply:
                p_opt[key] = val
    else:
        p_opt = strategy_lib.merge_p_opts(params.max_evals)
    return strategy_lib, dotdict(p_opt)


def hp_single_ensemble(vb, max_evals):
    partial_func = partial(backtest_single_ensemble, vb=vb)
    hyperopt_trials = Trials()
    best_params = EstimatorBase.optimize(
        space=vb.p_opt,
        score=partial_func,
        trials=hyperopt_trials,
        max_evals=max_evals
    )
    return hyperopt_trials, best_params


def backtest_single_ensemble(p_opt, vb):
    vb.p_opt = dotdict(p_opt)
    strategy_lib = vb.strategy_lib
    strategy_lib.update_p_opt(p_opt)
    strategy_lib.set_trailing_stop_b()
    total_profit = 0

    # go through each signal
    for strategy in strategy_lib.get_all_strategies():
        vb.assume_late_limit_fill = strategy.assume_late_limit_fill
        vb.assume_late_limit_fill_entry = strategy.assume_late_limit_fill_entry
        if strategy.use_rl_exit:
            vb.load_rl_exit_model()
    # try:
    #     _ = vb.trendSpotter
    # except AttributeError:
    #     vb.trendSpotter = TrendSpotter(vb.ohlc, list(strategy_lib.lib.values())[1], no_resampling=True)
    #     vb.trendSpotter.ohlc_ts = vb.ohlc['ts']
    vb.get_entry_ix(strategy_lib.get_all_strategies())
    vb.orders = []
    vb.future_orders = []
    vb.holding = 0
    vb.cash = 0
    vb.trades = []
    vb.future_orders = []
    status_start_date = datetime.datetime(2010, 1, 1)
    new_orders = []
    fast_forward = FastForward(strategy_lib)
    portfolio_side = side.hold  # neither long nor short, all enter
    # the entries are tupel of (ts/ix and a list of objects indicating the strategy (asset, Direction, potentially more here)
    for ix_entry, strategy_ids in vb.ix_entry_preds:
        new_orders = vb.process_proposed_orders_invalidate_future(new_orders)
        vb.confirm_future_orders(ix_entry)
        fast_forward.update_w_orders(vb.future_orders, ix_entry)
        for strategy_id in strategy_ids:
            if ix_entry <= fast_forward.ff[strategy_id]:
                continue
            strategy = strategy_lib.get_strategy(strategy_id)
            # FOR MODEL ENTRY ON SIGNAL
            # entry constraint: given current portfolio status. entry feasible of model
            if len(vb.orders) > 0:
                if vb.orders[-1].signal_source != signal_source.model_p:
                    portfolio_side = side.hold
                else:  # last order must have been a stop since model_p is the type for entries
                    portfolio_side = vb.orders[-1].direction
            if not vb.is_signal(strategy.direction, portfolio_side):
                continue
            else:
                order_type = order_type.market
                order = Order(**dict(
                    timing=timing.entry,
                    ix_signal=ix_entry,
                    ts_signal=vb.ts[ix_entry],
                    # price_limit=vb.ohlc.iloc[ix_entry, vb.ohlc.columns.get_loc('close')] + sim_tick,
                    # quantity=999,
                    direction=strategy.direction,
                    strategy_id=strategy.id,
                    signal_source=signal_source.model_p,
                    asset=vb.asset,
                    order_type=order_type,
                    exchange=vb.params.exchange)
                              )
            # MODEL FILL
            # STATUS
            if order.ts_signal.date() > status_start_date.date():
                status_start_date = order.ts_signal
                print(f'Backtesting {order.ts_signal.date()}')
            order.fill = vb.get_order_fill(order, strategy, delta_limit=strategy.delta_limit, move_to_bba=False)
            # because we assume this order has been a limit order initiated in anticiapation of p_entry,
            # we assume a limit order fee instead of market fee
            # if strategy.assume_simulated_future:
            #     order.fill.avg_price -= sim_tick
            #     order.price_limit -= sim_tick
            #     order.order_type = OrderType.limit
            #     order.set_fee()
            # cancel order if not filled before preds see it as opp
            # check first instance where preds are too low. if that inex before fill. cancel and set forward accordingly
            if True or order.order_type == order_type.limit:
                ix_cancel = vb.check_p_cancel_b4_fill(order, strategy)
                if ix_cancel != 0 and ix_cancel + order.ix_signal < order.fill.ix_fill:
                    fast_forward.update_ff_timeout(strategy_id, ix_cancel + order.ix_signal)
                    continue
            order.fill.ts_signal = pd.to_datetime(order.fill.ts_signal)
            # if vb.order_timed_out(order, strategy):
            #     fast_forward.update_ff_timeout(strategy.id, ix_entry + strategy.time_entry_cancelation)
            #     continue
            # create the subarray to work with for this order
            order.ts_signal = vb.ts[order.ix_signal]
            order.ts_fill = vb.ts[order.fill.ix_fill]
            n_ix_remaining = len(vb.ohlc) - order.fill.ix_fill
            ix_end = order.fill.ix_fill + min(strategy.max_trade_length, n_ix_remaining)

            vb.init_arr_lud(col='close', ix_start=order.fill.ix_fill, ix_end=ix_end)
            for c in ['ohlc_mid']:
                vb.create_arr_ix(c)
                vb.arr[:, vb.lud[c]] = vb.data['ohlc_mid'].iloc[order.fill.ix_fill:ix_end, vb.ix_close]
            vb.create_arr_ix('ts')
            vb.arr[:, vb.lud['ts']] = vb.data['ohlc_mid'].iloc[order.fill.ix_fill:ix_end].index

            vb.add_curves(['EMA_540_300d_rel', 'MOM_real_23_rel', 'MOM_real_360_rel', 'p_long', 'p_short'], ix_start=order.fill.ix_fill, ix_end=ix_end)
            # vb.add_p(order, strategy)
            # try:
            #     vb.add_regr(order, strategy)
            # except KeyError:
            #     pass
            # vb.add_p_exit(order, strategy)
            # vb.add_delta_profit(order.fill.avg_price, strategy)
            # vb.add_rolling_max_profit(order.fill.avg_price)
            # vb.add_trailing_profit(order.fill.avg_price)
            # vb.trail_profit_stop_price(order, strategy)
            vb.add_delta_profit(vb.arr[0, vb.lud.close], strategy)
            vb.add_rolling_max_profit(vb.arr[0, vb.lud.close])
            vb.add_trailing_profit(vb.arr[0, vb.lud.close])
            vb.trail_profit_stop_price(order, strategy)
            # vb.add_half_bband_middle_lower()
            # vb.add_half_bband_middle_upper()
            vb.add_trailing_stop_loss(order.fill.avg_price, strategy)
            if strategy.use_rl_exit:
                # vb.add_p_exit_given_entry(order)
                vb.add_rl_exit(order)
            # dont even enter when it would exit right afterwards
            if vb.arr[0, vb.lud.rl_action_1] > vb.arr[0, vb.lud.rl_action_0]:
                continue
            try:
                vb.ix_reentry_set = list(set(
                    np.intersect1d(vb.strat_entry[strategy_id][order.fill.ix_fill < vb.strat_entry[strategy_id]],
                                   vb.strat_entry[strategy_id][vb.strat_entry[strategy_id] < ix_end])
                ))
            except KeyError:
                vb.ix_reentry_set = []

            # if order.direction == Direction.short:
            #     a=1

            # MODEL EXIT
            # ix_limit = vb.find_moving_limit_trigger(order, strategy)
            # market order stops
            ix_signals = SignalType.init()
            ix_ema_veto = None  # vb.veto_stop_ema(strategy)
            # ix_signals.ix_trail_stop = vb.find_regr_trail_stop(order, veto_ix=ix_ema_veto)
            # ix_signals.ix_trail_profit_stop = vb.find_trail_profit_stop(order, strategy, veto_ix=ix_ema_veto)

            # ix_signals.ix_spike_mom_stop = vb.find_spike_mom_stop(order, strategy, veto_ix=ix_ema_veto)
            # ix_signals.ix_bband_loss_stop = vb.find_bband_loss_stop(order, strategy, veto_ix=ix_ema_veto)
            # ix_signals.ix_opp_exceeded_stop = vb.find_opp_exceeded_stop(order, strategy, veto_ix=ix_ema_veto)
            # ix_signals.ix_cheat_valley_stop = vb.find_cheat_valley_stop(order, strategy, veto_ix=ix_ema_veto)
            # ix_signals.ix_valley_stop = vb.find_valley_stop(order, strategy, veto_ix=ix_ema_veto)
            ix_signals.ix_other_peak_entry_stop = vb.find_other_peak_entry_stop(order, strategy, veto_ix=ix_ema_veto)
            # add an exit if momentum goes against AFTER trade has been entered. dont wait for trail_profit...
            # limit order stops
            # ix_signals.ix_bband_take_profit_stop = vb.find_bband_take_profit_stop(order, strategy, veto_ix=ix_ema_veto)
            # ix_signals.ix_losing_peaks_stop = vb.find_losing_twin_peaks_stop(order, strategy, ix_signals.ix_trail_stop, veto_ix=ix_ema_veto)
            # ix_signals.ix_preds_net_stop = vb.find_preds_net_stop(order, strategy, veto_ix=ix_ema_veto)
            # ix_signals.ix_exit_preds_stop = vb.find_exit_preds_stop(order, strategy, veto_ix=ix_ema_veto)
            # ix_signals.ix_entry_preds_stop = vb.find_preds_entry_stop(order, strategy, veto_ix=ix_ema_veto)
            # ix_signals.ix_entry_preds_dx_stop = vb.find_preds_entry_dx_stop(order, strategy, veto_ix=ix_ema_veto)
            # ix_signals.ix_take_abs_profit_stop = vb.find_take_abs_profit_stop(order, strategy, veto_ix=ix_ema_veto)
            # ix_signals.ix_cheat_preds_stop = vb.find_cheat_preds_stop(order, strategy, veto_ix=ix_ema_veto)
            # ix_signals.ix_between_bbands_stop = vb.find_between_bbands_stop(order, strategy)
            # ix_signals.ix_exit_given_entry = vb.find_exit_given_entry_model_stop(order, strategy, veto_ix=ix_ema_veto)
            if strategy.use_rl_exit:
                ix_signals.ix_rl_exit = vb.find_exit_rl_stop(order, strategy, veto_ix=ix_ema_veto)
                field_names = ['rl_action_1', 'rl_action_0']
                vb.store_arr(
                    pd.DataFrame(vb.arr[:ix_signals.ix_rl_exit - order.fill.ix_fill, [vb.lud.rl_action_1, vb.lud.rl_action_0]],
                                 columns=field_names,
                                 index=vb.ts[order.fill.ix_fill:ix_signals.ix_rl_exit]
                                 ),
                    fields=field_names
                )
            # ix_signals.ix_tree_regression_stop = vb.find_tree_regression_stop(order, strategy, veto_ix=ix_ema_veto)
            # ix_signals.ix_dont_go_minus_again_stop = vb.find_dont_go_minus_again_stop(order, strategy, veto_ix=ix_ema_veto)
            # ix_signals.ix_elapsed = len(vb.arr) + order.fill.ix_fill - 1
            ix_signals.ix_elapsed = len(vb.data['ohlc_mid'])-1
            min_ix = min([ix for ix in list(ix_signals.values()) if ix > 0])
            exit_ix_signal = min_ix
            exit_signal_source = SignalType.determine_exit_signal_source(min_ix, ix_signals)
            # order_exit.ix_fill, order_exit.price_limit = vb.add_handle_exit(order_exit, strategy)
            if exit_signal_source in [signal_source.between_bbands_stop,
                                      signal_source.losing_peaks_stop,
                                      signal_source.trail_stop,
                                      signal_source.ix_preds_net_stop,
                                      signal_source.ix_exit_preds_stop,
                                      signal_source.ix_entry_preds_stop,
                                      signal_source.ix_entry_preds_dx_stop,
                                      signal_source.ix_bband_take_profit_stop,
                                      signal_source.ix_bband_loss_stop,
                                      signal_source.elapsed,
                                      signal_source.ix_take_abs_profit_stop,
                                      signal_source.ix_valley_stop,
                                      signal_source.ix_other_peak_entry_stop,
                                      signal_source.ix_dont_go_minus_again_stop,
                                      signal_source.ix_tree_regression_stop,
                                      signal_source.ix_rl_exit
                                      # SignalSource.ix_cheat_preds_stop
                                      ]:
                order_type = order_type.market
            else:
                order_type = order_type.market  # OrderType.market if abs(vb.data['inds']['mom_5'][exit_ix_signal]) >= strategy.limit_market_mom_thresh else OrderType.limit
            # order_type = OrderType.market
            order_exit = create_order_exit(vb, exit_ix_signal, strategy, exit_signal_source, order_type)
            # double check that ts_fill of a limit order exit is not AFTER the ts_fill of a market order exit
            # ix_first_stop_market_orders = min([ix for ix in [
            #     # ix_signals.ix_trail_stop,
            #     ix_signals.ix_trail_profit_stop,
            #     ix_signals.ix_bband_loss_stop] if ix > 0])
            # if order_exit.fill.ix_fill > ix_first_stop_market_orders:
            #     for stop in [#'ix_between_bbands_stop',
            #                  'ix_losing_peaks_stop',
            #                  # 'ix_bband_take_profit_stop',
            #                     'ix_opp_exceeded_stop',
            #                   'ix_cheat_preds_stop',
            #                  'ix_preds_net_stop',
            #                  'ix_entry_preds_stop',
            #                 'ix_entry_preds_dx_stop',
            #                  'ix_exit_preds_stop']:
            #         ix_signals[stop] = 0
            #     order_exit = create_order_exit(vb, ix_first_stop_market_orders, strategy,
            #                                    determine_exit_signal_source(ix_first_stop_market_orders, ix_signals),
            #                                    OrderType.market)
            new_orders += [order, order_exit]
            profit = (order_exit.fill.avg_price - order.fill.avg_price) * (-1 if order.direction == direction.short else 1)
            total_profit += profit
            print(f'ENTER {order.direction.name} | {order.fill.ts_fill} | {order.fill.avg_price}\nEXIT {signal_source[exit_signal_source]} | '
                  f'{order_exit.fill.ts_fill} | {order_exit.fill.avg_price}\n\tPROFIT: {round(profit, 4)} | TOTAL: {round(total_profit, 4)}')
    # last ix entry passed. 1 order may still be in the future orders queue
    vb.confirm_future_orders(len(vb.ohlc))

    vb.assign_order_quantities()
    trades = vb.calc_portfolio_value()
    # assert sum(trades) == vb.cash, 'sum of trades not equal to cash. check...'
    vb.calc_win_loss_ratio()
    print('Win Loss ratio: {}  - n-win_loss-trades: {}'.format(vb.win_loss_ratio, len(vb.win_loss_trades)))
    if vb.params.max_evals == 1:
        print('Trades: {}'.format(trades))
    print('n orders: {}, profit: {}'.format(len(vb.orders), vb.cash))  # np.sum(vb.stats['profit_after_fees']))
    print('Max. Drawdown: {}'.format(vb.calc_max_drawdown(trades)))
    print('LO / MO fill loss: {}'.format(sum([(o.fill.avg_price - o.price_limit) for o in vb.orders])))

    vb.store_p_exec(p_opt, {'profit': vb.cash})
    return {
        'loss': -1 * sum(trades),  # vb.cash,  # np.sum(vb.stats['profit_after_fees']),
        # 'loss': -1 * len([t for t in np.divide(vb.stats['profit_after_fees'], vb.stats['price_limit_entry']) if t > 10*bp]),  # maximise trades. ensure some profit made 10 bp
        'status': STATUS_OK,
        'stats': {},  # vb.stats
        'orders': vb.orders,
        # 'vb': vb
    }


def create_order_exit(vb, exit_ix_signal, strategy, exit_signal_source, order_type):
    order_exit = Order(**dict(
        timing=timing.exit,
        ix_signal=exit_ix_signal,
        ts_signal=vb.ts[exit_ix_signal],
        # price_limit=vb.ohlc.iloc[exit_ix_signal, vb.ohlc.columns.get_loc('close')],
        # quantity=1,
        direction=direction.long if strategy.direction == direction.short else direction.short,
        strategy_id=strategy.id,
        signal_source=exit_signal_source,
        asset=vb.asset,
        order_type=order_type,
        exchange=vb.params.exchange
    )
                       )
    order_exit.fill = vb.get_order_fill(order_exit, strategy, move_to_nearest_bid_ask=False)
    return order_exit


@click.command('backtest')
@click.pass_context
def backtest(ctx: ctx):
    params = importlib.import_module('{}.{}'.format(Paths.path_config_backtest, ctx.obj.fn_params)).Params()
    strategy_lib = StrategyLib([
        Strategy(0, params.asset.upper(), direction.short),
        Strategy(1, params.asset.upper(), direction.long),
    ])
    params.data_start = params.ts_start - datetime.timedelta(days=1)
    params.data_end = params.ts_end + datetime.timedelta(days=1)
    params.ex = set_ex(params.ex, params.asset)
    params.ex_path = os.path.join(Paths.backtests, params.ex)
    create_dir(params.ex_path)
    Logger.init_log(os.path.join(Paths.backtests, params.ex, 'log_{}'.format(datetime.date.today())))
    Logger.debug('Params: {}'.format(params))
    Logger.debug('DataPeriod: {} - {}'.format(params.data_start, params.data_end))

    best_param = backtest_multiple_ensembles(
        strategy_lib=strategy_lib,
        params=params
    )


@click.command()
@click.pass_context
def main(ctx):
    ctx.obj = dotdict(dict(
        fn_params='eurusd'
    ))
    ctx.forward(backtest)


if __name__ == '__main__':
    main()
