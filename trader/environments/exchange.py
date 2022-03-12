import copy
import operator
import datetime
import numpy as np
import pandas as pd

from common.refdata import tick_size
from trader.backtest.fill import Fill
from common.utils.util_func import todec, pdf_col_ix
from common.modules import direction, order_type, exchange
from common.modules.logger import logger


class Exchange:
    """Responsible for
    - Returning FILL information given an order
    """
    def __init__(s, params, feature_hub):
        s.params = params
        s.feature_hub = feature_hub
        s.pdp = s.feature_hub.data['root']
        s.ts = None
        s.lud = None
        s.arr = None

        s.ohlc_mid, s.ohlc_ask, s.ohlc_bid, s.ohlc_trade = (None, ) * 4

    def setup(s):
        s.feature_hub.get_curves('mid.close')
        s.ohlc_mid = s.feature_hub.data['mid']
        s.ohlc_ask = s.feature_hub.data['ask']
        s.ohlc_bid = s.feature_hub.data['bid']
        s.ohlc_trade = s.feature_hub.data['trade']
        return s

    def _exchange_has_trade_data(s):
        return True if s.params.exchange in [exchange.bitmex] else False

    def reset(s):
        s.__init__(s.params, s.feature_hub)
        s.setup()

    def ix2ts(s, ix):
        return s.ts[ix]

    def get_order_fill(s, order, strategy, move_to_nearest_bid_ask=True):
        if order.order_type == order_type.market:
            order.price_limit = s.get_market_fill_price(order.ix_signal, order.direction)
            return Fill(**dict(ix_fill=order.ix_signal, avg_price=order.price_limit, ts_fill=order.ts_signal, ts_signal=order.ts_signal,
                               quantity=order.quantity, direction=order.direction, asset=order.asset, order_type=order.order_type))
        else:
            if move_to_nearest_bid_ask:
                ix_limit_fill, price_limit_fill = s.add_handle_exit(order, strategy)
            else:
                order.price_limit = s.get_limit_entry_price(order, strategy)
                ix_limit_fill, price_limit_fill = s.get_limit_fill_ix(order)
            return Fill(**dict(ix_fill=ix_limit_fill, avg_price=price_limit_fill, ts_fill=s.to_ts(ix_limit_fill), ts_signal=order.ts_signal,
                               quantity=order.quantity, direction=order.direction, asset=order.asset, order_type=order.order_type))

            # find out when tickforecaster overwrites with market order
            # if s.params.use_tick_forecast:
            #     ix_tickforecast_mo = np.argmax(s.data['curve']['p_1tick_down'][order.ix_signal:] > strategy.p_tick_against)
            #     tickforecast_found_none = ix_tickforecast_mo == 0 and s.data['curve']['p_1tick_down'][0] <= strategy.p_tick_against
            #     ix_tickforecast_fill = order.ix_signal + ix_tickforecast_mo
            # else:
            #     tickforecast_found_none = True
            #     ix_tickforecast_fill = order.ix_signal

            # if not tickforecast_found_none and ix_tickforecast_fill < ix:
            #     order.order_type = OrderType.market
            #     avg_price = s.get_market_fill_price(ix_tickforecast_fill, order.direction)
            #     return Fill(**dict(ix_fill=ix_tickforecast_fill, avg_price=avg_price, ts_fill=s.to_ts(ix_tickforecast_fill),
            #                        quantity=order.quantity, direction=order.direction, asset=order.asset, order_type=order.order_type))

            # if not tickforecast_found_none and ix_tickforecast_fill < ix_limit_entry:
            #     order.order_type = OrderType.market
            #     avg_price = s.get_market_fill_price(ix_tickforecast_fill, order.direction)
            #     ix_tickforecast_fill = order.ix_signal + ix_tickforecast_mo
            #     return Fill(**dict(ix_fill=ix_tickforecast_fill, avg_price=avg_price, ts_fill=s.to_ts(ix_tickforecast_fill),
            #                        quantity=order.quantity, direction=order.direction, asset=order.asset, order_type=order.order_type))

    def add_handle_exit(s, order_exit, strategy):
        order_direction = order_exit.direction
        df_quote_near, ix_hl_near, df_quote_far, ix_hl_far = s._get_near_far_quote_reference(order_direction)
        ix_close = pdf_col_ix(df_quote_near, 'close')
        op_delta_limit_price = operator.sub if order_exit.direction == direction.long else operator.add

        ix_exit_trail_stop = order_exit.ix_signals
        # ix_left_trade = ix_exit_trail_stop - ix_limit_entry
        if ix_exit_trail_stop >= len(s.ohlc_mid):
            ix_close = pdf_col_ix(s.ohlc_mid, 'close')
            return len(s.ohlc_mid) - 1, s.ohlc_mid.iloc[min(ix_exit_trail_stop, len(s.ohlc_mid) - 1), ix_close] - \
                   strategy.delta_limit_exit * s.ohlc_mid.iloc[min(ix_exit_trail_stop, len(s.ohlc_mid) - 1), ix_close]
        else:
            # not best bid if theres a bid-ask gap greater than 1 tick
            # exit_limit = s.ohlc_bid.iloc[ix_exit_trail_stop, s.ix_close] \
            # - s.ohlc_bid.iloc[ix_exit_trail_stop, s.ix_close] * strategy.delta_limit_exit
            exit_limit = op_delta_limit_price(todec(df_quote_near.iloc[ix_exit_trail_stop, ix_close]), tick_size[s.params.asset])
        # below is for adaptive limit order adjustment to get a better price (Bitmex)
        starting_limit_price = copy.copy(exit_limit)
        order_exit.price_limit = float(starting_limit_price)

        # for ix in range(ix_exit_trail_stop + 1, min(ix_exit_trail_stop + s.p_opt.max_trade_length, s.ohlc_mid.index[-1])):
        # this exit handling goes beyond size of vb.arr open end
        op = s._get_late_limit_fill_operator()

        for ix in range(ix_exit_trail_stop + 1, s.ohlc_mid.index[-1]):
            # if market moves towards limit & hits limit order just placed before current close
            # below is synced with VS. But may not be correct. quote low instead of close is better.
            if op(todec(df_quote_near.iloc[ix, ix_hl_near]), exit_limit) or op(todec(s.ohlc_trade.iloc[ix, ix_hl_near]), exit_limit):
                return ix, min(df_quote_near.iloc[ix, ix_hl_far], float(exit_limit))
            # if market moved aways from market and price difference between limit order and current close is so large that gap needs closing
            elif todec(df_quote_far.iloc[ix, ix_close]) - exit_limit > 0:
                # ix + 1 gets closer to how VS works. first cancel, then re-create order is actually canceled. assuming it takes 1 sec/iter
                exit_limit = todec(df_quote_far.iloc[ix, ix_close])
                # turn into market order. deriving tick
            elif s._price_moved_away_too_much(starting_limit_price, exit_limit, strategy, order_exit):
                order_exit.order_type = order_type.market
                return ix, s.get_market_fill_price(ix, order_exit.direction)
            else:
                continue
            logger.info('No Exit fill found')
            return s.ohlc_trade.index[-1], min(df_quote_near.iloc[s.ohlc_trade.index[-1], ix_hl_far], float(exit_limit))

    @staticmethod
    def _price_moved_away_too_much(starting_limit_price, exit_limit, strategy, order_exit) -> bool:
        return abs(starting_limit_price - exit_limit) >= todec(strategy.mo_n_ticks * tick_size[order_exit.asset])

    def to_ix(s, ts: datetime.datetime):
        return s.ts.get_loc(ts)

    def to_ts(s, ix: int = None):
        return s.ts[ix]

    # def get_limit_fill_ix_price(s, ix_signals, strategy):
    #     # when limit order would actually get hit
    #     order.price_limit = s.get_limit_entry_price(ix_signals, strategy)
    #     # Below routine also might return a fill price slightly better than limit entry price
    #     ix_limit_entry, price_limit_entry = s.get_limit_fill_ix(order)
    #     return ix_limit_entry, price_limit_entry

    def get_limit_entry_price(s, order, strategy):
        df_quote_far, ix_hl_far = s._get_far_quote_ref(order.direction)
        op = operator.sub if order.direction == direction.long else operator.add
        ix_close = df_quote_far.columns.get_loc('close')
        return op(df_quote_far.iloc[order.ix_signal, ix_close], df_quote_far.iloc[order.ix_signal, ix_close] * strategy.delta_limit)

    @staticmethod
    def limit_fill_not_found(arg_max_series, ix_limit, ix_entry):
        return arg_max_series.iloc[0] is False and ix_limit == 1 + ix_entry

    def get_limit_fill_ix(s, order):
        price_limit_entry = order.price_limit
        ix_entry = order.ix_signal
        df_quote_near, ix_hl_near, df_quote_far, ix_hl_far = s._get_near_far_quote_reference(order.direction)
        op = s._get_late_limit_fill_operator()
        op_max_min = min if order.direction == direction.long else max

        # +1 as we can only have an order execute in the next second. Not <= which assumes we're last in order book queue
        quote_near_limit = op(todec(df_quote_near.iloc[1 + ix_entry:, ix_hl_near]), todec(price_limit_entry))
        tradebar_lt_limit = op(todec(s.ohlc_trade.iloc[1 + ix_entry:, ix_hl_near]), todec(price_limit_entry))
        ix_limit_fill = min(np.argmax(quote_near_limit), np.argmax(tradebar_lt_limit))
        if s.limit_fill_not_found(quote_near_limit, ix_limit_fill, ix_entry) and s.limit_fill_not_found(tradebar_lt_limit, ix_limit_fill, ix_entry):
            return s.ohlc_trade.index[-1], price_limit_entry
        ix_high = pdf_col_ix(s.ohlc_ask, 'high')
        return ix_limit_fill, op_max_min(s.ohlc_ask.iloc[ix_limit_fill, ix_high], price_limit_entry)

    def _get_late_limit_fill_operator(s):
        return operator.lt if s.params.assume_late_limit_fill else operator.le

    def _get_near_far_quote_reference(s, order_direction) -> (pd.DataFrame, int, pd.DataFrame, int):
        """
        :param order_direction:
        :return: df_quote_near, ix_hl_near, df_quote_far, ix_hl_far
        """
        return (
            *s._get_near_quote_ref(order_direction),
            *s._get_far_quote_ref(order_direction)
        )

    def _get_near_quote_ref(s, order_direction):
        pdf = s.ohlc_ask if order_direction == direction.long else s.ohlc_bid
        return pdf, pdf_col_ix(pdf, 'low' if order_direction == direction.long else 'high')

    def _get_far_quote_ref(s, order_direction):
        pdf = s.ohlc_bid if order_direction == direction.long else s.ohlc_ask
        return pdf, pdf_col_ix(pdf, 'high' if order_direction == direction.long else 'low')

    def get_market_fill_price(s, ix_signals, order_direction):
        """
        in volatile seconds there can be a huge gap in bid/ask_close and traded price in the data
        conservatively assume worst of the worst bid/ask price or trading price for that second)
        return s.ohlc.iloc[ix_signals, s.ix_close]
        qc applies below for limit, for above, s.ix_close for market order
        """
        df_quote_near, ix_hl_near, df_quote_far, ix_hl_far = s._get_near_far_quote_reference(order_direction)
        side = 'low' if order_direction == direction.long else 'high'
        op_max_min = max if order_direction == direction.long else min
        return op_max_min(df_quote_near.iloc[ix_signals, ix_hl_near], s.ohlc_trade.iloc[ix_signals, pdf_col_ix(s.ohlc_trade, side)])
