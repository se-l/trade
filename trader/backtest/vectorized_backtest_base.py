import datetime
import json
import os
import numpy as np
import copy
import pandas as pd
from trader.backtest.fill import Fill
from common.globals import OHLC
from common.modules import series
from common.modules import DataStore
from common import Paths
from common.refdata import tick_size
from trader.data_loader.utils_features import get_ohlc
from common.utils.util_func import default_to_py_type, todec, reduce_to_intersect_ts, SeriesTickType
from common.modules import direction, timing, order_type


class VectorizedBacktestBase:

    def __init__(s):
        s.ts_start = None
        s.ex = None
        s.params = None
        s.fn_p_exec = None
        s.orders = []
        s.assume_late_limit_fill_entry = False
        s.assume_late_limit_fill = False
        s.ts = None
        s.data = DataStore()

    def get_order_fill(s, order, strategy, delta_limit=0, move_to_bba=True):
        if order.order_type == order_type.market:
            avg_price = s.get_market_fill_price(order.ix_signal, order.direction)
            order.price_limit = avg_price
            return Fill(**dict(ix_fill=order.ix_signal, avg_price=avg_price, ts_fill=order.ts_signal,
                               quantity=order.quantity, direction=order.direction, asset=order.asset, order_type=order_type.market))
        else:
            # find out when tickforecaster overwrites with market order
            if order.direction == direction.long:
                if s.params.use_tick_forecast:
                    ix_tickforecast_mo = np.argmax(s.data['curve']['p_1tick_down'][order.ix_signal:] > strategy.p_tick_against)
                    tickforecast_found_none = ix_tickforecast_mo == 0 and s.data['curve']['p_1tick_down'][0] <= strategy.p_tick_against
                    ix_tickforecast_fill = order.ix_signal + ix_tickforecast_mo
                else:
                    tickforecast_found_none = True
                    ix_tickforecast_fill = order.ix_signal
            elif order.direction == direction.short:
                if s.params.use_tick_forecast:
                    ix_tickforecast_mo = np.argmax(s.data['curve']['p_1tick_up'][order.ix_signal:] > strategy.p_tick_against)
                    tickforecast_found_none = ix_tickforecast_mo == 0 and s.data['curve']['p_1tick_up'][0] <= strategy.p_tick_against
                    ix_tickforecast_fill = order.ix_signal + ix_tickforecast_mo
                else:
                    tickforecast_found_none = True
                    ix_tickforecast_fill = order.ix_signal
            else:
                raise ("Order direction unknown")
            if move_to_bba:
                ix, exit_limit = s.add_handle_exit(order, strategy)
                if not tickforecast_found_none and ix_tickforecast_fill < ix:
                    order.order_type = order_type.market
                    avg_price = s.get_market_fill_price(ix_tickforecast_fill, order.direction)
                    return Fill(**dict(ix_fill=ix_tickforecast_fill, avg_price=avg_price, ts_fill=s.to_ts(ix_tickforecast_fill),
                                       quantity=order.quantity, direction=order.direction, asset=order.asset, order_type=order_type.market))
                else:
                    return Fill(**dict(ix_fill=ix, avg_price=exit_limit, ts_fill=s.to_ts(ix),
                                       quantity=order.quantity, direction=order.direction, asset=order.asset))
            else:
                # when limit order would actually get hit
                price_limit_entry = s.get_limit_entry_price(order.ix_signal, strategy)
                order.price_limit = price_limit_entry
                # Below routine also might return a fill price slightly better than limit entry price
                ix_limit_entry, price_limit_entry = s.get_limit_fill_ix(order.ix_signal, price_limit_entry, strategy)
                if not tickforecast_found_none and ix_tickforecast_fill < ix_limit_entry:
                    order.order_type = order_type.market
                    avg_price = s.get_market_fill_price(ix_tickforecast_fill, order.direction)
                    ix_tickforecast_fill = order.ix_signal + ix_tickforecast_mo
                    return Fill(**dict(ix_fill=ix_tickforecast_fill, avg_price=avg_price, ts_fill=s.to_ts(ix_tickforecast_fill),
                                       quantity=order.quantity, direction=order.direction, asset=order.asset, order_type=order_type.market))
                else:
                    return Fill(**dict(ix_fill=ix_limit_entry, avg_price=price_limit_entry, ts_fill=s.to_ts(ix_limit_entry),
                                       quantity=order.quantity, direction=order.direction, asset=order.asset))

    def to_ix(s, ts: datetime.datetime):
        return s.ts.get_loc(ts)

    def arr_to_ix(s, arr_ts):
        return np.where(np.isin(pd.to_datetime(s.ts), arr_ts.index, assume_unique=True))[0]

    def to_ts(s, ix: int):
        return s.ts[ix]

    def get_limit_fill_ix_price(s, ix_signal, strategy):
        # when limit order would actually get hit
        price_limit_entry = s.get_limit_entry_price(ix_signal, strategy)
        # Below routine also might return a fill price slightly better than limit entry price
        ix_limit_entry, price_limit_entry = s.get_limit_fill_ix(ix_signal, price_limit_entry, strategy)
        return ix_limit_entry, price_limit_entry

    def get_limit_entry_price(s, ix_signal, strategy):
        if strategy.direction == direction.long:
            return s.ohlc_bid.iloc[ix_signal, s.ix_close] - s.ohlc_bid.iloc[
                ix_signal, s.ix_close] * strategy.delta_limit
        elif strategy.direction == direction.short:
            return s.ohlc_ask.iloc[ix_signal, s.ix_close] + s.ohlc_ask.iloc[
                ix_signal, s.ix_close] * strategy.delta_limit
        else:
            raise ('Direction of strategy required for limit fill')

    @staticmethod
    def limit_fill_not_found(arg_max_series, ix_limit, ix_entry):
        return arg_max_series.iloc[0] == False and ix_limit == 1 + ix_entry

    def get_limit_fill_ix(s, ix_entry, price_limit_entry, strategy):
        if strategy.direction == direction.long:
            # +1 as we can only have an order execute in the next second. Not <= which assumes we're last in order book queue
            if s.assume_late_limit_fill_entry:
                ask_lt_limit = todec(s.ohlc_ask.iloc[1 + ix_entry:, s.ix_low]) < todec(price_limit_entry)
                tradebar_lt_limit = todec(s.ohlc.iloc[1 + ix_entry:, s.ix_low]) < todec(price_limit_entry)
                ix_limit_fill = min(np.argmax(ask_lt_limit),
                                    np.argmax(tradebar_lt_limit)
                                    )
                if s.limit_fill_not_found(ask_lt_limit, ix_limit_fill, ix_entry) \
                        and s.limit_fill_not_found(tradebar_lt_limit, ix_limit_fill, ix_entry):
                    return s.ohlc.index[-1], price_limit_entry
            else:
                ask_lt_limit = todec(s.ohlc.iloc[1 + ix_entry:, s.ix_low]) <= todec(price_limit_entry)
                ix_limit_fill = np.argmax(ask_lt_limit)
                if s.limit_fill_not_found(ask_lt_limit, ix_limit_fill, ix_entry):
                    return s.ohlc.index[-1], price_limit_entry
            return ix_limit_fill, min(s.ohlc_ask.iloc[ix_limit_fill, s.ix_high], price_limit_entry)

        elif strategy.direction == direction.short:
            if s.assume_late_limit_fill_entry:
                bid_gt_limit = todec(s.ohlc_bid.iloc[1 + ix_entry:, s.ix_high]) > todec(price_limit_entry)
                tradebar_gt_limit = todec(s.ohlc.iloc[1 + ix_entry:, s.ix_high]) > todec(price_limit_entry)
                ix_limit_fill = min(
                    np.argmax(bid_gt_limit),
                    np.argmax(tradebar_gt_limit)
                )
                if s.limit_fill_not_found(bid_gt_limit, ix_limit_fill, ix_entry) \
                        and s.limit_fill_not_found(tradebar_gt_limit, ix_limit_fill, ix_entry):
                    return s.ohlc.index[-1], price_limit_entry
            else:
                bid_gt_limit = todec(s.ohlc.iloc[1 + ix_entry:, s.ix_high]) >= todec(price_limit_entry)
                ix_limit_fill = np.argmax(bid_gt_limit)
                if s.limit_fill_not_found(bid_gt_limit, ix_limit_fill, ix_entry):
                    return s.ohlc.index[-1], price_limit_entry
            return ix_limit_fill, max(s.ohlc_bid.iloc[ix_limit_fill, s.ix_low], price_limit_entry)
        else:
            raise ('Direction of strategy required for limit fill')

    def add_handle_exit(s, order_exit, strategy):
        ix_exit_trail_stop = order_exit.ix_signal
        # ix_left_trade = ix_exit_trail_stop - ix_limit_entry
        if ix_exit_trail_stop >= len(s.ohlc):
            return len(s.ohlc) - 1, s.ohlc.iloc[min(ix_exit_trail_stop, len(s.ohlc) - 1), s.ix_close] - \
                   strategy.delta_limit_exit * s.ohlc.iloc[min(ix_exit_trail_stop, len(s.ohlc) - 1), s.ix_close]
        else:
            if order_exit.direction == direction.long:  # of the backtest, not the trade.
                # not best bid if theres a bid-ask gap greater than 1 tick
                # exit_limit = s.ohlc_bid.iloc[ix_exit_trail_stop, s.ix_close] \
                # - s.ohlc_bid.iloc[ix_exit_trail_stop, s.ix_close] * strategy.delta_limit_exit
                exit_limit = todec(s.ohlc_ask.iloc[ix_exit_trail_stop, s.ix_close] - tick_size[s.asset])

            elif order_exit.direction == direction.short:
                # exit_limit = s.ohlc_ask.iloc[ix_exit_trail_stop, s.ix_close] \
                # + s.ohlc_ask.iloc[ix_exit_trail_stop, s.ix_close] * strategy.delta_limit_exit
                exit_limit = todec(s.ohlc_bid.iloc[ix_exit_trail_stop, s.ix_close] + tick_size[s.asset])
            else:
                raise ('Missing direction')
        starting_limit_price = copy.copy(exit_limit)
        order_exit.price_limit = float(starting_limit_price)

        # for ix in range(ix_exit_trail_stop + 1, min(ix_exit_trail_stop + s.p_opt.max_trade_length, s.ohlc.index[-1])):
        # this exit handling goes beyond size of vb.arr open end
        if order_exit.direction == direction.long:
            for ix in range(ix_exit_trail_stop + 1, s.ohlc.index[-1]):
                # if market drops while short and hits limit order just placed before current close
                # below is synced with VS. But may not be correct. quote low instead of close is better.
                if s.assume_late_limit_fill and (
                        todec(s.ohlc_ask.iloc[ix, s.ix_low]) < exit_limit
                        or todec(s.ohlc.iloc[ix, s.ix_low]) < exit_limit
                ):
                    return ix, min(s.ohlc_ask.iloc[ix, s.ix_high], float(exit_limit))
                elif not s.assume_late_limit_fill and (
                        todec(s.ohlc.iloc[ix, s.ix_low]) <= exit_limit
                ):
                    return ix, min(s.ohlc_ask.iloc[ix, s.ix_high], float(exit_limit))
                # if market went up and price difference between limit order and current close is so large that gap needs closing
                elif todec(s.ohlc_bid.iloc[
                               ix, s.ix_close]) - exit_limit > 0:  # asset_price_incr: #s.p_opt.delta_limit_exit_update * s.ohlc_ask.iloc[ix, s.ix_close]:  # and s.ens_p_entry[ix] < s.p_opt.min_entry_p_short:
                    # ix + 1 gets closer to how VS works. first cancel, then re-create order is actually canceled. assuming it takes 1 sec/iter
                    exit_limit = todec(s.ohlc_bid.iloc[ix, s.ix_close])  # - s.ohlc_bid.iloc[ix, s.ix_close] * s.p_opt.delta_limit_exit
                    # turn into market order. deriving tick
                    ## 0.95 accounts for lacking double precision accuracy
                elif todec(abs(starting_limit_price - exit_limit)) >= todec(strategy.mo_n_ticks * tick_size[order_exit.asset]):
                    avg_price = s.get_market_fill_price(ix, order_exit.direction)
                    order_exit.order_type = order_type.market
                    return ix, avg_price
                else:
                    continue
            print('No Exit fill found')
            return s.ohlc.index[-1], min(s.ohlc_ask.iloc[s.ohlc.index[-1], s.ix_high], float(exit_limit))
        elif order_exit.direction == direction.short:
            for ix in range(ix_exit_trail_stop + 1, s.ohlc.index[-1]):
                # if market drops while short and hits limit order just placed before current close
                if s.assume_late_limit_fill and (
                        todec(s.ohlc_bid.iloc[ix, s.ix_high]) > exit_limit
                        or todec(s.ohlc.iloc[ix, s.ix_high]) > exit_limit
                ):
                    return ix, max(s.ohlc_bid.iloc[ix, s.ix_low], float(exit_limit))
                if not s.assume_late_limit_fill and \
                        (
                                todec(s.ohlc.iloc[ix, s.ix_high]) >= exit_limit
                        ):
                    return ix, max(s.ohlc_bid.iloc[ix, s.ix_low], float(exit_limit))
                # if market went up and price difference between limit order and current close is so large that gap needs closing
                elif exit_limit - todec(s.ohlc_ask.iloc[
                                            ix, s.ix_close]) > 0:  # asset_price_incr:  #s.p_opt.delta_limit_exit_update * s.ohlc_bid.iloc[ix, s.ix_close]:  # and s.ens_p_entry[ix] < s.p_opt.min_entry_p_short:
                    # ix + 1 gets closer to how VS works. first cancel, then re-create order is actually canceled. assuming it takes 1 sec/iter
                    exit_limit = todec(s.ohlc_ask.iloc[ix, s.ix_close])  # - s.ohlc_ask.iloc[ix, s.ix_close] * s.p_opt.delta_limit_exit
                elif abs(starting_limit_price - exit_limit) >= todec(strategy.mo_n_ticks * tick_size[order_exit.asset]):
                    avg_price = s.get_market_fill_price(ix, order_exit.direction)
                    order_exit.order_type = order_type.market
                    return ix, avg_price
                else:
                    continue
            print('No Exit fill found')
            return s.ohlc.index[-1], max(s.ohlc_bid.iloc[s.ohlc.index[-1], s.ix_low], float(exit_limit))
        else:
            raise ('Missing order direction')

    def get_market_fill_price(s, ix_signal, order_direction):
        if order_direction == direction.long:
            # in volatile seconds there can be a huge gap in bid/ask_close and traded price in the data
            # conservatively assume worst of the worst bid/ask price or trading price for that second)
            # return s.ohlc.iloc[ix_signal, s.ix_close]
            # qc applies below for limit, for above, s.ix_close for market order
            return max(s.ohlc_ask.iloc[ix_signal, s.ix_high], s.ohlc.iloc[ix_signal, s.ix_high])
        elif order_direction == direction.short:
            # return s.ohlc.iloc[ix_signal, s.ix_close]
            return min(s.ohlc_bid.iloc[ix_signal, s.ix_low], s.ohlc.iloc[ix_signal, s.ix_low])
        else:
            raise ('Direction of strategy required for limit fill')

    @staticmethod
    def calc_max_drawdown(trades):
        if len(trades) == 0:
            return 0
        roll_max_cash = [trades[0]]
        drawdown = [0]
        for i in range(1, len(trades)):
            c = sum(trades[:i + 1])
            if c > roll_max_cash[-1]:
                roll_max_cash.append(c)
            else:
                roll_max_cash.append(roll_max_cash[-1])
            drawdown.append(roll_max_cash[-1] - c)
        return max(drawdown)

    def calc_win_loss_ratio(s):
        if len(s.win_loss_trades) > 0:
            s.win_loss_ratio = len([t for t in s.win_loss_trades if t > 0]) / len(s.win_loss_trades)
        else:
            s.win_loss_ratio = None

    def assign_order_quantities(s):
        # simple for now, just 1 asset. so either long or short
        # exits go to 0 portfolio. entries fully swing
        s.holding = 0
        for o in s.orders:
            if o.direction == direction.long:
                if s.holding >= 1:
                    o.quantity = 0
                    continue
                elif o.timing == timing.entry and s.holding < 1:
                    o.quantity = 1 - s.holding
                    s.holding = s.holding + o.quantity
                elif o.timing == timing.exit:
                    o.quantity = -1 * s.holding
                    s.holding = 0
                else:
                    print('Unaccounted scenario')

            elif o.direction == direction.short:
                if s.holding <= -1:
                    o.quantity = 0
                    continue
                elif o.timing == timing.entry and s.holding > -1:
                    o.quantity = -1 - s.holding
                    s.holding = s.holding + o.quantity
                elif o.timing == timing.exit:
                    o.quantity = -1 * s.holding
                    s.holding = 0
                else:
                    print('Unaccounted scenario')
            for o in s.orders:
                o.fill.quantity = o.quantity

    @staticmethod
    def get_convex_capital(n_loosing_trades):
        max_cap = 1
        if n_loosing_trades == 0:
            return max_cap
        else:
            cap_reduction_factor = 2
            # want to take last 2 trades into account
            # this eq allows for 3 caps: [1, 0.5, 0.25]
            capital = max_cap - max_cap / (min(n_loosing_trades, 2) * cap_reduction_factor)
            return capital

    @staticmethod
    def get_n_loosing_trades(profit_trade):
        bool_win = [profit < 0 for profit in profit_trade[-min(2, len(profit_trade)):]]
        if bool_win[-1] is True:
            # if last trade was a winner, go max cap immediately
            return 0
        else:
            return sum(bool_win)

    def assign_order_quantities_convex(s):
        """
        risk more capital when it's winning. reduce capital when loosing.
        assumption: winning and loosing trades come in streaks as prevailing market conditions match algo params
        or dont...
        trade completions criteria: entry -> exit  or entry -> entry   no: exit->entry
        params:
        cap = f_cap_adjustment(
         profit last n trades,
         max cap, min cap,
        """
        n_loosing_trades = 0
        profit_trade = []
        s.holding = 0
        for i in range(len(s.orders)):
            o = s.orders[i]
            if len(profit_trade) > 0:
                n_loosing_trades = s.get_n_loosing_trades(profit_trade)
            if o.direction == direction.long:
                # already long - dont long more. scenario shouldnt exit anyway
                if s.holding >= 1:
                    o.quantity = 0
                    continue
                # currently short and performing a full turnaround
                elif o.timing == timing.entry and s.holding < 1:
                    if i > 0:
                        profit_trade.append(s.orders[i - 1].fill.avg_price - o.fill.avg_price)
                        o.quantity = s.get_convex_capital(n_loosing_trades) - s.holding
                    else:
                        o.quantity = 1
                    s.holding = s.holding + o.quantity
                elif o.timing == timing.exit:
                    profit_trade.append(s.orders[i - 1].fill.avg_price - o.fill.avg_price)
                    o.quantity = -1 * s.holding
                    s.holding = 0
                else:
                    print('Unaccounted scenario')

            elif o.direction == direction.short:
                if s.holding <= -1:
                    o.quantity = 0
                    continue
                elif o.timing == timing.entry and s.holding > -1:
                    if i > 0:
                        profit_trade.append(o.fill.avg_price - s.orders[i - 1].fill.avg_price)
                        o.quantity = -s.get_convex_capital(n_loosing_trades) - s.holding
                    else:
                        o.quantity = -1
                    s.holding = s.holding + o.quantity
                elif o.timing == timing.exit:
                    profit_trade.append(o.fill.avg_price - s.orders[i - 1].fill.avg_price)
                    o.quantity = -1 * s.holding
                    s.holding = 0
                else:
                    print('Unaccounted scenario')
        for o in s.orders:
            o.fill.quantity = o.quantity

    def calc_portfolio_value(s):
        port_prev = [0]
        trades = []
        holding = 0
        ts = [0]
        s.cash = 0
        s.win_loss_trades = []
        i = 0
        for o in s.orders:
            if o.direction == direction.long:
                s.cash -= o.fill.avg_price * o.quantity + o.fee
                holding += o.quantity
            elif o.direction == direction.short:
                s.cash += o.fill.avg_price * -o.quantity - o.fee
                holding += o.quantity
            # print((holding * o.price_limit + s.cash))
            trades.append((holding * o.fill.avg_price + s.cash) - port_prev[-1])
            port_prev.append((holding * o.fill.avg_price + s.cash))
            ts.append(o.fill.ts_fill.strftime('%b-%d %H:%M:%S'))
            s.update_win_loss_trade(o, trades, i)
            i += 1
        # end of algo
        if s.params.max_evals == 1:
            print('Cash History: {}'.format(list(zip(ts, port_prev))))
        print('Q fees: {}'.format(sum([o.fee for o in s.orders])))
        s.cash += s.holding * s.ohlc.iloc[-1, s.ix_close]
        return trades

    def update_win_loss_trade(s, o, trades, i):
        if o.timing == timing.exit:
            s.win_loss_trades.append(trades[-1])
        elif i > 0 and o.timing == timing.entry and s.orders[i - 1].timing == timing.entry:
            s.win_loss_trades.append(trades[-1])

    def calc_trade_opportunity(s, order_entry, strategy):
        """
        we define the earning opportunity in mean reversion fashion. Assuming the price will, at most,
        return to a minimum(short) or maximum(long) over a predefined past time window.
        Then may place a stop loss at half the opportunity
        """
        ### -1 because at the time of OnOrderEvent() the sec bar for that seconds has not been emitted yet and
        ## the close_hist price is from endTime or previous second
        ix_fill = order_entry.fill.ix_fill - 1
        if strategy.direction == direction.long:
            max_ = np.max(s.ohlc.iloc[ix_fill - strategy.opp_time_window:ix_fill, s.ix_close])
            opportunity = max_ - s.ohlc.iloc[ix_fill, s.ix_close]
        elif strategy.direction == direction.short:
            min_ = np.min(s.ohlc.iloc[ix_fill - strategy.opp_time_window:ix_fill, s.ix_close])
            opportunity = s.ohlc.iloc[ix_fill, s.ix_close] - min_
        if np.isnan(opportunity):
            return s.ohlc.iloc[ix_fill, s.ix_close]
        else:
            return opportunity

    def load_entry_predictions_from_db(s) -> pd.DataFrame:
        return s.influx.load_p(s.params.asset, [], ex=s.params.ex_entry, from_ts=s.ts_start, to_ts=s.ts_end, load_from_training_set=s.params.load_from_training_set)

    def load_ohlc(s):
        print("Load ohlc for {} to {}".format(s.params.data_start, s.params.data_end))
        s.ohlc = get_ohlc(start=s.params.data_start, end=s.params.data_end, asset=s.params.asset,
                          exchange=s.params.exchange, series=series.trade, series_tick_type=SeriesTickType('ts', s.params.resample_period, 'second'))
        pdf_qoute = get_ohlc(start=s.params.data_start, end=s.params.data_end, asset=s.params.asset,
                             exchange=s.params.exchange, series=series.quote, series_tick_type=SeriesTickType('ts', s.params.resample_period, 'second'))
        s.ohlc, pdf_qoute = reduce_to_intersect_ts(s.ohlc, pdf_qoute)
        s.ohlc_ask = pdf_qoute[['ask_' + c for c in OHLC + ['size']]].rename({c: c.replace('ask_', '') for c in pdf_qoute.columns}, axis='columns')
        s.ohlc_bid = pdf_qoute[['bid_' + c for c in OHLC + ['size']]].rename({c: c.replace('ask_', '') for c in pdf_qoute.columns}, axis='columns')
        assert len(s.ohlc) == len(pdf_qoute), 'Trade and QuoteBar Array lenght not identical'

        s.ix_close = s.ohlc.columns.get_loc('close')
        s.ix_high = s.ohlc.columns.get_loc('high')
        s.ix_open = s.ohlc.columns.get_loc('open')
        s.ix_low = s.ohlc.columns.get_loc('low')

    def store_p_exec(s, p, stats):
        try:
            with open(os.path.join(Paths.backtests, s.ex, s.fn_p_exec), 'r') as f:
                p_exec = json.load(f)
            if p_exec['stats']['profit'] < stats['profit']:
                print('Opt Params with higher profit. Overwriting {}...'.format(s.fn_p_exec))
                s.overwrite_p_exec(p, stats)
            else:
                return
        except FileNotFoundError:
            print('Creating new {} in {}...'.format(s.fn_p_exec, s.ex))
            s.overwrite_p_exec(p, stats)
            return

    def overwrite_p_exec(s, p, stats):
        for k, v in p.items():
            p[k] = default_to_py_type(v)
        p_exec = {'stats': {'profit': stats['profit'], 'asset': s.params.asset}, 'p_exec': p}
        with open(os.path.join(os.path.join(Paths.backtests, s.ex), s.fn_p_exec), 'wt') as f:
            json.dump(p_exec, f)

    def load_p_exec(s):
        try:
            with open(os.path.join(os.path.join(Paths.backtests, s.ex), s.fn_p_exec), 'r') as f:
                'Loaded opt params file {} from: {}'.format(s.fn_p_exec, s.ex)
                p_exec = json.load(f)
            return p_exec['p_exec']
        except FileNotFoundError:
            print('{} not found: Using default values...'.format(s.fn_p_exec))
            return False

    def would_reenter_same_sec(s, min_ix):
        """
        only for the entry orders as still finding exit order.
        checks if strategy going opposite direction to exit order exists for the exit time.
        """
        return min_ix in s.ix_reentry_set
        # for id, strategy in s.strategyLib.lib.items():
        #     if strategy.direction == order.direction:
        #         return (min_ix, [id]) in s.ix_entry_preds

    def check_p_cancel_b4_fill(s, order, strategy):
        """first approach. cancel when opportunity is gone. but with >1 entry method, not aware of which opp it is
        new approach. cancel when order price and bba is more than tick away. currently not having spike catching order
        """
        if order.direction == direction.short:
            # ix_cancel = np.argmax(s.data['curve']['net'][order.ix_signal:order.ix_signal+strategy.max_trade_length] > strategy.preds_net_thresh + strategy.d_net_cancel_entry)
            ### beware: pd.df carries index. np. array  doesnt
            ix_cancel_price_move = np.argmax(
                todec(s.ohlc_ask.iloc[order.ix_signal:order.ix_signal + strategy.max_trade_length, 3]) <= todec(order.price_limit - 2 * 0.05)) - order.ix_signal
        elif order.direction == direction.long:
            # ix_cancel = np.argmax(s.data['curve']['net'][order.ix_signal:order.ix_signal+strategy.max_trade_length] < strategy.preds_net_thresh - strategy.d_net_cancel_entry)
            ix_cancel_price_move = np.argmax(
                todec(s.ohlc_bid.iloc[order.ix_signal:order.ix_signal + strategy.max_trade_length, 3]) >= todec(order.price_limit + 2 * 0.05)) - order.ix_signal
        else:
            return 0
        ix_cancel = ix_cancel_price_move
        return ix_cancel
