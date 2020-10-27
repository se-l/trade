from common.modules import signal_source
from common.modules import dotdict
from enum import Enum


class SignalType:
    # def __init__(s):
    @classmethod
    def init(s) -> dotdict:
        return dotdict({enum: 0 for enum in signal_source._member_map_.values()})

    @staticmethod
    def determine_exit_signal_source(ix_signals: dotdict) -> (int, Enum):
        min_ix = min([ix for ix in list(ix_signals.values()) if ix > 0])
        for ix_stop_cat, ix in ix_signals.items():
            if ix == min_ix:
                return min_ix, ix_stop_cat
        raise ('Need to come up with an exit. Timeout, end of backtest at least')

    # def reset(s):
    #     s.ix_signal = Dotdict({enum: 0 for enum in SignalSource._member_map_.values()})
