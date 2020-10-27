from common.modules import timing


class FastForward(object):
    """fast forward is set by entry limit timeouts and future order
        future order can be removed mid-simulation, hence need to update fast forward accordingly"""
    no_ff = 0
    timeout = 1
    future_order = 2

    def __init__(s, strategyLib):
        s.ff = {i: 0 for i in strategyLib.lib.keys()}
        s.ff_type = {i: None for i in strategyLib.lib.keys()}

    def update_w_orders(s, future_orders: list, ix_entry):
        if future_orders.__len__() == 0:
            for i, val in s.ff_type.items():
                if val == s.future_order:
                    s.ff[i] = ix_entry - 1
                    s.ff_type[i] = s.no_ff
        else:
            # only valid for entry order, not exits
            for o in future_orders:
                if o.fill.ix_fill > ix_entry and o.fill.ix_fill > s.ff[o.strategy_id] and o.timing == timing.entry:
                    s.ff[o.strategy_id] = o.fill.ix_fill
                    s.ff_type[o.strategy_id] = s.future_order

    def update_ff_timeout(s, strategy_id, ix):
        s.ff[strategy_id] = ix
        s.ff_type[strategy_id] = s.timeout
        # invalidate other strategies forward
        if strategy_id == 1:
            s.ff[0] = 0
        elif strategy_id == 0:
            s.ff[1] = 0
