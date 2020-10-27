import json
import os

import pandas as pd

from common.utils.util_func import default_to_py_type
from common import Paths
from common.modules import order_type
from connector import InfluxClientWrapper as Influx


class System:
    def __init__(s, params):
        s.params = params
        s.chunk = ['ho']
        s.fn_p_exec = 'p_exec_{}.json'.format('-'.join([str(i) for i in s.chunk]))
        s.influx = Influx()

    def store_p_exec(s, p, stats):
        try:
            with open(os.path.join(Paths.backtests, s.params.ex, s.fn_p_exec), 'r') as f:
                p_exec = json.load(f)
            if p_exec['stats']['profit'] < stats['profit']:
                print('Opt Params with higher profit. Overwriting {}...'.format(s.fn_p_exec))
                s.overwrite_p_exec(p, stats)
            else:
                return
        except FileNotFoundError:
            print('Creating new {} in {}...'.format(s.fn_p_exec, s.params.ex))
            s.overwrite_p_exec(p, stats)
            return

    def overwrite_p_exec(s, p, stats):
        for k, v in p.items():
            p[k] = default_to_py_type(v)
        p_exec = {'stats': {'profit': stats['profit'], 'asset': s.params.asset}, 'p_exec': p}
        with open(os.path.join(os.path.join(Paths.backtests, s.params.ex), s.fn_p_exec), 'wt') as f:
            json.dump(p_exec, f)

    def load_p_exec(s):
        try:
            with open(os.path.join(os.path.join(Paths.backtests, s.params.ex), s.fn_p_exec), 'r') as f:
                'Loaded opt params file {} from: {}'.format(s.fn_p_exec, s.params.ex)
                p_exec = json.load(f)
            return p_exec['p_exec']
        except FileNotFoundError:
            print('{} not found: Using default values...'.format(s.fn_p_exec))
            return False

    def store_backtest(s, orders):
        """todo: move into a more high level / admin module"""
        pdf = pd.DataFrame(None, columns=['price', 'direction', 'order_type', 'signal_source', 'fill'])
        for o in orders:
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
                               ex=s.params.ex,
                               backtest_time=s.params.backtest_time
                           ),
                           field_columns=['price', 'quantity'],
                           tag_columns=['fill', 'direction', 'order_type', 'signal_source']
                           )

    def store_input_curves(s, data, ohlc_mid):
        pdf = pd.DataFrame(data['curve'], columns=data['curve'].dtype.names, index=ohlc_mid.index)
        s.influx.write_pdf(pdf,
                           measurement='backtest_curves',
                           tags=dict(
                               asset=s.params.asset.lower(),
                               ex=s.params.ex,
                               backtest_time=s.params.backtest_time
                           ),
                           field_columns=pdf.columns
                           )
