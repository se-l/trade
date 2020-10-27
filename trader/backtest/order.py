import os
import math

from common.modules.order_type import OrderType
from trader.backtest.fee import Fee
from common.paths import Paths
from trader.backtest.fill import Fill


class Order(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(s, **kwargs):
        # super().__init__()
        s.path_buffer = os.path.join(Paths.path_buffer)
        s.asset = None
        # s.timing = None
        s.order_type = OrderType.limit
        s.ix_signal = None
        s.ts_signal = None
        s.ix_cancel = None
        s.ts_cancel = None
        s.ix_order_place = None
        s.ts_order_place = None
        s.price_order = None
        s.direction = None
        s.timing = None
        s.exchange = None

        s.fill: Fill
        # s.fee = s.fill.fee
        # s.ix_fill = s.fill.ix_fill
        # s.ts_fill = s.fill.ts_fill

        s.price_limit = None
        s.signal_source = None
        s.quantity = None

        for k, v in kwargs.items():
            s[k] = v

    def set_fee(s):
        try:
            if s.fill.avg_price != 0:
                s.fee = s.fill.avg_price * math.fabs(s.quantity) * Fee.fee(s.exchange, s.order_type, s.asset)
            else:
                s.fee = s.price_limit * math.fabs(s.quantity) * Fee.fee(s.exchange, s.order_type, s.asset)
        except AttributeError:
            s.fee = s.price_limit * math.fabs(s.quantity) * Fee.fee(s.order_type, s.asset)

    def to_dict(s):
        return {
            'asset': s.asset,
            'exchange': s.exchange,
            'order_type': s.order_type,
            'ts_signal': s.ts_signal,
            'ts_cancel': s.ts_cancel,
            'ts_order_place': s.ts_order_place,
            'price_order': s.price_order,
            'direction': s.direction,
            'ts_fill': s.fill.ts_fill,
            'price_limit': s.price_limit,
            'signal_source': s.signal_source,
            'quantity': s.fill.quantity,
        }

    def __getstate__(s):
        # Deliberately do not return self.value or self.last_change.
        # We want to have a "blank slate" when we unpickle.
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
        return s

    def __setstate__(s, state):
        # Make self.history = state and last_change and value undefined
        __getattr__ = object.__getattribute__
        __setattr__ = object.__setattr__
        __delattr__ = object.__delattr__
