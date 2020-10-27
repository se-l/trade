from common import Paths
from common.utils import aliased


@aliased
class Trade(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    ix_entry = None
    ts_entry = None
    ix_limit_entry = None
    ts_limit_entry = None
    price_limit_entry = None

    ix_exit_fill = None
    ts_exit_fill = None
    price_exit_limit = None
    price_exit_fill = None
    trade_profit_mean = None
    trade_profit_bp = None
    trade_len = None
    entry_to_limit = None
    fees = 0
    profit_after_fees = None
    ix_exit_signal = None
    ts_exit_signal = None

    def __init__(s, Order0, Order1):
        s.path_buffer = Paths.path_buffer

    def ix_to_ts(s, ix):
        pass
