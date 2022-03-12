from .enum_utils import EnumStr
from enum import Enum


class SignalSource(EnumStr, Enum):
    model_p = 'model_p'
    regr_exit = 'regr_exit'
    trail_stop = 'trail_stop'
    veto_entry = 'veto_entry'
    ix_elapsed = 'ix_elapsed'
    net_p = 'net_p'
    nn_p = 'nn_p'
    trail_profit = 'trail_profit'
    between_bbands_stop = 'between_bbands_stop'
    losing_peaks_stop = 'losing_peaks_stop'
    ix_preds_net_stop = 'ix_preds_net_stop'
    ix_exit_preds_stop = 'ix_exit_preds_stop'
    ix_entry_preds_stop = 'ix_entry_preds_stop'
    ix_spike_mom_stop = 'ix_spike_mom_stop'
    ix_bband_take_profit_stop = 'ix_bband_take_profit_stop'
    ix_bband_loss_stop = 'ix_bband_loss_stop'
    ix_opp_exceeded_stop = 'ix_opp_exceeded_stop'
    ix_take_abs_profit_stop = 'ix_take_abs_profit_stop'
    ix_cheat_preds_stop = 'ix_cheat_preds_stop'
    ix_entry_preds_dx_stop = 'ix_entry_preds_dx_stop'
    ix_cheat_valley_stop = 'ix_cheat_valley_stop'
    ix_valley_stop = 'ix_valley_stop'
    ix_other_peak_entry_stop = 'ix_other_peak_entry_stop'
    ix_trend_stop = 'ix_trend_stop'
    ix_exit_given_entry = 'ix_exit_given_entry'
    ix_dont_go_minus_again_stop = 'ix_dont_go_minus_again_stop'
    ix_tree_regression_stop = 'ix_tree_regression_stop'
    ix_rl_exit = 'ix_rl_exit'
    ix_regr_reward_exit = 'ix_regr_reward_exit'