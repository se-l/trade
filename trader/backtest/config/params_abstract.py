import types
import datetime
from common.modules import exchange
from common.modules import signal_source


class ParamsBase:
    exchange = exchange.fxcm
    data_start = None
    data_end = None
    asset = None
    ex = None
    resample_sec = 60
    reduced_data_len = 30000
    load_from_training_set = False
    backtest_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    max_trade_window = datetime.timedelta(hours=2)

    assume_late_limit_fill = True  # QC alignment when True
    assume_late_limit_fill_entry = True  # QC alignment when True
    use_simply_limit_price_at_fill = False
    asset_pair = None

    def to_dict(self):
        return {k: self.__getattribute__(k) for k in self.__dir__() if k[:2] != '__' and not isinstance(self.__getattribute__(k), types.BuiltinFunctionType)}

    exit_signal_market_order = [signal_source.between_bbands_stop,
                                signal_source.losing_peaks_stop,
                                signal_source.trail_stop,
                                signal_source.ix_preds_net_stop,
                                signal_source.ix_exit_preds_stop,
                                signal_source.ix_entry_preds_stop,
                                signal_source.ix_entry_preds_dx_stop,
                                signal_source.ix_bband_take_profit_stop,
                                signal_source.ix_bband_loss_stop,
                                signal_source.ix_elapsed,
                                signal_source.ix_take_abs_profit_stop,
                                signal_source.ix_valley_stop,
                                signal_source.ix_other_peak_entry_stop,
                                signal_source.ix_dont_go_minus_again_stop,
                                signal_source.ix_tree_regression_stop,
                                signal_source.ix_rl_exit,
                                signal_source.ix_regr_reward_exit
                                # SignalSource.ix_cheat_preds_stop
                                ]

