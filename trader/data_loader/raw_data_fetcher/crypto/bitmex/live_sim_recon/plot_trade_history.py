from plot.plot_func import plotly_backtest
from trade_mysql.mysql_conn import Db
import pandas as pd
from qc.load_features import LoadFeatures
import datetime
from qc.py_backtest.Order_v2 import Order
from qc.common.enums import *
from qc.py_backtest.Fill import Fill
from utils.utilFunc import dotdict


class PlotTradeHistory:
    def __init__(s):
        s.db = Db()
        s.data_start = datetime.datetime(2019, 5, 25)
        s.data_end = datetime.datetime(2019, 6, 3)
        s.asset = 'ETHUSD'
        s.exchange = 'bitmex'

    def run(s):
        s.db.close()
        trade_history = s.get_trade_history()

        plotly_backtest(
            ohlc=LoadFeatures.get_ohlc(start=s.data_start, end=s.data_end, asset=s.asset, exchange=s.exchange, index=int),
            orders=s.convert_bitmex_exec_history_to_orders(trade_history),
            ex='live_review'
        )

    def get_trade_history(s):
        sql = '''select * from trade.bitmex_execution_history where timestamp > '2019-05-01 01:01:01' order by timestamp ;'''
        trade_history = s.db.fetchall(sql)
        trade_history_cols = [el[0] for el in s.db.fetchall('''DESCRIBE trade.bitmex_execution_history;''')]
        trade_history = pd.DataFrame(trade_history, columns=trade_history_cols)
        return trade_history

    def convert_bitmex_exec_history_to_orders(s, trade_history):
        orders = []
        # trade_history.index = trade_history['timestamp']
        # trade_history.sort_values(by='timestamp')
        for i in range(len(trade_history)):
            item = trade_history.iloc[i]
            if item['text'] == "Funding":
                continue
            if item['symbol'] == s.asset:
                # remove short entries.
                if (item['side'] == "Sell" and item['execInst'] == 'ParticipateDoNotInitiate') or \
                    (item['side'] == "Buy" and item['execInst'] != 'ParticipateDoNotInitiate'):
                    # make an exception for the total reversal
                    if i > 0 and (item['execInst'] == 'ParticipateDoNotInitiate' and trade_history.iloc[i-1]['execInst'] == 'ParticipateDoNotInitiate'):
                        item['cumQty'] = trade_history.iloc[i-1]['cumQty']
                        item['execInst'] = 'FakeExit'
                    else:
                        continue
                if i > 0 and (item['side'] == "Buy" and item['execInst'] == 'ParticipateDoNotInitiate' and trade_history.iloc[i - 1]['execInst'] == 'ParticipateDoNotInitiate'):
                    item['cumQty'] = trade_history.iloc[i - 1]['cumQty']

                if item['side'] == "Buy":
                    direction = Direction.long
                elif item['side'] == "Sell":
                    direction = Direction.short
                else:
                    raise
                o = dotdict(dict(ts_signal=item['timestamp'], price_limit=item['avgPx'],
                                 quantity=item['cumQty'] if direction==Direction.long else -item['cumQty'],
                                 signal_source=SignalSource.model_p if item['execInst'] == 'ParticipateDoNotInitiate' else SignalSource.ix_rl_exit,
                                 timing=Timing.entry if item['execInst'] == 'ParticipateDoNotInitiate' else Timing.exit,
                                 fill=dotdict(dict(ts_fill=item['timestamp'], price_limit=item['avgPx'], avg_price=item['avgPx'])),
                                 fee=item['commission']*item['cumQty']*item['avgPx'],
                                 direction=direction
                                 ))
                orders.append(o)
        return orders


if __name__ == '__main__':
    plotTradeHistory = PlotTradeHistory()
    plotTradeHistory.run()
