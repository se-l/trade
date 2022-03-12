from trader.backtest.fee import Fee
import math


# @aliased
from common.modules import exchange


class Fill(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(s, ts_fill, avg_price, direction, quantity=0, **kwargs):
        s.asset = None
        s.ix_fill = None
        s.ts_fill = ts_fill
        s.direction = direction
        s.quantity = quantity
        s.signal_source = None
        s.signal_fill_slippage = None
        s.avg_price = avg_price
        s.order_type = None
        s.exchange = exchange.bitmex

        for k, v in kwargs.items():
            s[k] = v

    @property
    def fee(s):
        try:
            return s.avg_price * math.fabs(s.quantity or 1) * Fee.fee(s.exchange, s.order_type, s.asset)
        except (AttributeError, TypeError):
            return 0

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
