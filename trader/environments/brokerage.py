import pandas as pd

from functools import reduce
from trader.backtest.order import Order
from common.modules import direction, timing, order_type, side, signal_source
from common.modules.logger import logger
from common.utils import property_plus
from connector import InfluxClientWrapper as Influx


class Brokerage:
    """
    Handle
    - Portfolio statistics
    - Order Tracking
    - Send Orders to Exchange
    - Agent's requests (new orders, cancelations, state queries)
    """

    def __init__(s, params, exchange, feature_hub):
        s.params = params
        s.exchange = exchange
        s.feature_hub = feature_hub
        s.win_loss_trades = []
        s.orders = []
        s.future_orders = []
        s.holding = 0
        s.cash = 0
        s.trades = []
        s.future_orders = []
        s.episode_tick_pnl = []
        s.win_loss_ratio = None

    def reset(s):
        s.win_loss_trades = []
        s.orders = []
        s.future_orders = []
        s.holding = 0
        s.cash = 0
        s.trades = []
        s.future_orders = []
        s.episode_tick_pnl = []

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

    def get_order_fill(s, order, strategy, move_to_nearest_bid_ask=True):
        return s.exchange.get_order_fill(order, strategy, move_to_nearest_bid_ask)

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

    def log_profit(s, order, order_exit):
        fee = order.fill.fee + order_exit.fill.fee
        profit = (order_exit.fill.avg_price - order.fill.avg_price) * (-1 if order.direction == direction.short else 1) - fee
        logger.info(f'ENTER {order.direction.name} | {order.fill.ts_fill} | {order.fill.avg_price}\nEXIT {signal_source[order_exit.signal_source]} | '
                    f'{order_exit.fill.ts_fill} | {order_exit.fill.avg_price}\n\tPROFIT: {round(profit, 4)} | FEE: {round(fee, 4)} | TOTAL: {round(s.total_profit + profit, 4)}'
                    f' | DURATION: {order_exit.fill.ts_fill - order.fill.ts_fill}')

    def log_tick_pnl(s, order, order_exit):
        mid_prices = s.feature_hub.pdp['mid.close'].loc[s.feature_hub.to_ix(order.fill.ts_fill):s.feature_hub.to_ix(order_exit.fill.ts_fill)]
        s.episode_tick_pnl.append(mid_prices - order.fill.avg_price)

    def get_episode_tick_pnl(s):
        pdf = pd.DataFrame(pd.concat(s.episode_tick_pnl))
        pdf.columns = ['pnl']
        return pdf

    def calc_portfolio_value(s):
        port_prev = [0]
        trades = []
        holding = 0
        ts = [0]
        s.cash = 0
        i = 0
        for o in s.orders:
            if o.direction == direction.long:
                s.cash -= o.fill.avg_price * o.quantity + o.fee
                holding += o.quantity
            elif o.direction == direction.short:
                s.cash += o.fill.avg_price * -o.quantity - o.fee
                holding += o.quantity
            # logger.info((holding * o.price_limit + s.cash))
            trades.append((holding * o.fill.avg_price + s.cash) - port_prev[-1])
            port_prev.append((holding * o.fill.avg_price + s.cash))
            ts.append(o.fill.ts_fill.strftime('%b-%d %H:%M:%S'))
            s.update_win_loss_trade(o, trades, i)
            i += 1
        # end of algo
        if s.params.max_evals == 1:
            logger.info('Cash History: {}'.format(list(zip(ts, port_prev))))
        logger.info('Q fees: {}'.format(sum([o.fee for o in s.orders])))
        s.cash += s.holding * s.feature_hub.data['mid'].iloc[-1, s.feature_hub.ix_close]
        return trades

    def update_win_loss_trade(s, o, trades, i):
        if o.timing == timing.exit:
            s.win_loss_trades.append(trades[-1])
        elif i > 0 and o.timing == timing.entry and s.orders[i - 1].timing == timing.entry:
            s.win_loss_trades.append(trades[-1])

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
                    logger.info('Unaccounted scenario')

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
                    logger.info('Unaccounted scenario')
        for o in s.orders:
            o.fill.quantity = o.quantity

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

    def get_portfolio_side(s):
        if len(s.orders) > 0:
            if s.orders[-1].signal_source != signal_source.model_p:
                return side.hold
            else:  # last order must have been a stop since model_p is the type for entries
                return s.orders[-1].direction
        else:
            return side.hold

    def create_order_ticket(s, ix_signal, direction, signal_source, order_type, strategy):
        return Order(**dict(
            ix_signal=ix_signal,
            # ts_signal=s.exchange.ts[exit_ix_signals],
            # price_limit=s.ohlc_mid.iloc[exit_ix_signals, s.ohlc_mid.columns.get_loc('close')],
            # quantity=1,
            direction=direction,
            strategy_id=strategy.id,
            signal_source=signal_source,
            asset=s.params.asset,
            order_type=order_type,
            exchange=s.params.exchange
        )
                     )

    def sortino_ratio(s):
        """for every realized trade, get sortino ratio"""
        return [o.fill.avg_price for o in s.orders]

    @property_plus()
    def order_durations(s):
        return [order.fill.ts_fill - s.orders[i - 1].fill.ts_fill for i, order in enumerate(s.orders) if i % 2 != 0]

    @property
    def profits(s):
        return [(order.fill.avg_price - s.orders[i - 1].fill.avg_price) * (-1 if order.direction == direction.long else 1) - order.fill.fee for i, order in enumerate(s.orders) if i % 2 != 0]

    @property
    def fees(s):
        return sum([order.fill.fee for order in s.orders])

    @property_plus()
    def total_profit(s):
        return sum(s.profits)

    def store_backtest(s, bt_i=0, overwrite=True):
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
        # logger.info('Saving backtest in influx  ex and backtest time...')
        Influx().write_pdf(
            pdf,
            measurement='backtest',
            tags=dict(
                asset=str(s.params.asset),
                ex=s.params.ex,
                backtest_time=s.params.backtest_time,
                bt_i=str(bt_i)
            ),
            field_columns=['price', 'quantity'],
            tag_columns=['fill', 'direction', 'order_type', 'signal_source'],
            overwrite=overwrite
        )
        # Influx().write_pdf(
        #     pdf=s.get_episode_tick_pnl(),
        #     measurement='backtest',
        #     field_columns=['pnl'],
        #     tags=dict(
        #         asset=s.params.asset.name,
        #         ex=s.params.ex,
        #         backtest_time=s.params.backtest_time
        #     ),
        #     overwrite=False
        # )

    def store_episode_pnl(s, overwrite=False):
        Influx().write_pdf(
            pdf=s.get_episode_tick_pnl(),
            measurement='backtest',
            field_columns=['pnl'],
            tags=dict(
                asset=s.params.asset.name,
                ex=s.params.ex,
                backtest_time=s.params.backtest_time
            ),
            overwrite=overwrite
        )

    def episode_summary(s):
        if order_durations := s.order_durations:
            cum_order_duration = reduce(lambda x, y: x + y, order_durations)
            avg_order_duration = cum_order_duration / len(order_durations)/2
        else:
            cum_order_duration = avg_order_duration = 0
        return '\n'.join([
            f'Total PnL: {s.total_profit}',
            f'Total Fee: {s.fees}',
            f'Cum Trade duration: {cum_order_duration}',
            f'Avg Trade duration: {avg_order_duration}',
            f'# Trades: {len(s.orders) / 2}'
        ])
