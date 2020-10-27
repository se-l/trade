import numpy as np
from common.modules import direction
from common.utils import PandasFramePlus
from common.utils.util_func import rolling_window, todec
from scipy.signal import find_peaks
from common.refdata import CurvesReference as Cr


class StopsFinder:
    arr: PandasFramePlus = None

    def would_reenter_same_sec(self, *args):
        return False

    def find_class_p_exit(s, order, strategy):
        if order.direction == direction.short:
            stop = s.arr[Cr.p_net] > strategy.exit_net_p
        elif order.direction == direction.long:
            stop = s.arr[Cr.p_net] < strategy.exit_net_p
        # switch on regression stop after x seconds
        ix_stop = np.argmax(stop)
        ix_stop = ix_stop + order.fill.ix_fill if ix_stop != 0 else len(
            s.arr) + order.fill.ix_fill
        return ix_stop

    def find_losing_twin_peaks_stop(s, order, strategy, earlier_stop=None, veto_ix=None):
        lookback_window = 60
        if len(s.arr) < lookback_window:
            return s.ignore_no_stop_found(order, len(s.arr))
        # looks wrong  - review rolling window is used correctly.
        min_avg_close = np.mean(rolling_window(s.arr[Cr.close], lookback_window), axis=1)[::60]
        # min_avg_close2 = []
        # for i in range(60, len(s.arr) + 1, 60):
        #     min_avg_close2.append(np.mean(s.arr[i - lookback_window: i, s.lud.close]))
        peaks = []
        position = []
        ix_stop = 0
        for ix in range(1, len(min_avg_close)):
            # if earlier_stop is not None and ix * lookback_window > earlier_stop:
            #     return earlier_stop * 2  # just something afterwards
            tri = list(min_avg_close[ix - 1:ix + 2])
            if len(tri) < 3:
                break
            if strategy.direction == direction.long:
                if tri.index(max(tri)) == 1:
                    peaks.append(max(tri))
                    position.append(ix + 2)
                if len(peaks) >= 2 \
                        and peaks[-1] < peaks[-2] \
                        and s.arr[position[-1] * lookback_window, Cr.trailing_profit_rel] > strategy.twin_peak_trailing_profit:
                    ix_stop = position[-1] * lookback_window
                    if veto_ix is not None and ix_stop in veto_ix:
                        continue
                    else:
                        break
            elif strategy.direction == direction.short:
                if tri.index(min(tri)) == 1:
                    peaks.append(min(tri))
                    position.append(ix + 2)
                if len(peaks) >= 2 \
                        and peaks[-1] > peaks[-2] \
                        and s.arr[position[-1] * lookback_window, Cr.trailing_profit_rel] > strategy.twin_peak_trailing_profit:
                    ix_stop = position[-1] * lookback_window
                    if veto_ix is not None and ix_stop in veto_ix:
                        continue
                    else:
                        break
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def ignore_no_stop_found(s, order, ix_stop):
        ix_stop = ix_stop + order.fill.ix_fill if ix_stop != 0 else (len(s.arr) - 1) + order.fill.ix_fill
        return ix_stop

    def veto_stop_ema(s, strategy):
        if strategy.direction == direction.long:
            return np.where((s.arr[Cr.ema_9_300d_rel] > strategy.veto_stop_ema_d300) == True)[0]
        elif strategy.direction == direction.short:
            return np.where((s.arr[Cr.ema_9_300d_rel] < strategy.veto_stop_ema_d300) == True)[0]
        else:
            return None

    def find_preds_entry_stop(s, order, strategy, veto_ix=None):
        if strategy.direction == direction.long:
            stop_ix = s.arr[Cr.p_entry_long] <= strategy.p_entry_sl_exit
        elif strategy.direction == direction.short:
            stop_ix = s.arr[Cr.p_entry_short] <= strategy.p_entry_sl_exit
        stop_ix[0] = False
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_preds_entry_dx_stop(s, order, strategy, veto_ix=None):
        if strategy.direction == direction.long:
            stop_ix = np.multiply(s.arr[Cr.p_entry_long] <= strategy.p_entry_sl_dx_exit,
                                  s.arr[Cr.p_entry_smooth_long_dx] >= 0)
        elif strategy.direction == direction.short:
            stop_ix = np.multiply(s.arr[Cr.p_entry_short] <= strategy.p_entry_sl_dx_exit,
                                  s.arr[Cr.p_entry_smooth_short_dx] <= 0)
        stop_ix[0] = False
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_preds_net_stop(s, order, strategy, veto_ix=None):
        if strategy.direction == direction.long:
            stop_ix = np.multiply(s.arr[Cr.p_net] < strategy.preds_net_exit,
                                  s.arr[Cr.p_net_d1] <= 0)
        elif strategy.direction == direction.short:
            stop_ix = np.multiply(s.arr[Cr.p_net] > strategy.preds_net_exit,
                                  s.arr[Cr.p_net_d1] >= 0)
        stop_ix[0] = False
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_exit_preds_stop(s, order, strategy, veto_ix=None):
        if strategy.direction == direction.long:
            stop_ix = s.arr[Cr.p_long_exit] >= strategy.p_exit
        elif strategy.direction == direction.short:
            stop_ix = s.arr[Cr.p_short_exit] >= strategy.p_exit
        stop_ix[0] = False
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_regr_trail_stop(s, order, veto_ix=None):
        stop_ix = s.arr[Cr.trailing_profit] > s.arr[Cr.stop_loss]
        stop_ix[0] = False
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        # protect after immediate entry after exit due to np.argmax returning 0 if all False.
        # coz series[0] cannot be True as no profit has been made, hence no trailing
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_spike_mom_stop(s, order, strategy, veto_ix=None):
        if strategy.direction == direction.long:
            stop_ix = s.arr[Cr.mom_23_rel] < strategy.mom_spike_stop
        elif strategy.direction == direction.short:
            stop_ix = s.arr[Cr.mom_23_rel] > strategy.mom_spike_stop
        stop_ix[0] = False  # a trade shouldnt be entered in which we are immediately in loss. That's because of the min/max fill model, assuming worst case scenario
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_take_abs_profit_stop(s, order, strategy, veto_ix=None):
        if strategy.direction == direction.long:
            stop_ix = s.arr[Cr.close] >= order.fill.avg_price + strategy.take_abs_profit
        elif strategy.direction == direction.short:
            stop_ix = s.arr[Cr.close] <= order.fill.avg_price - strategy.take_abs_profit
        stop_ix[0] = False  # a trade shouldnt be entered in which we are immediately in loss. That's because of the min/max fill model, assuming worst case scenario
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_opp_exceeded_stop(s, order, strategy, veto_ix=None):
        opp = todec(s.calc_trade_opportunity(order, strategy))
        if strategy.direction == direction.long:
            stop_ix = s.arr[Cr.close] < todec(order.fill.avg_price) - opp * todec(strategy.opp_stop_scaling_factor)
        elif strategy.direction == direction.short:
            stop_ix = s.arr[Cr.close] > todec(order.fill.avg_price) + opp * todec(strategy.opp_stop_scaling_factor)
        stop_ix[0] = False  # a trade shouldnt be entered in which we are immediately in loss. That's because of the min/max fill model, assuming worst case scenario
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_cheat_valley_stop(s, order, strategy, veto_ix=None):
        if strategy.direction == direction.long:
            stop_ix, props = find_peaks(-1 * s.arr[Cr.p_entry_long] + 2 * max(s.arr[Cr.p_entry_long]), height=0.05, prominence=0.06, distance=300)
        elif strategy.direction == direction.short:
            stop_ix, props = find_peaks(-1 * s.arr[Cr.p_entry_short] + 2 * max(s.arr[Cr.p_entry_short]), height=0.05, prominence=0.06, distance=300)
        if veto_ix is not None:
            stop_ix = np.setdiff1d(stop_ix, veto_ix)
        if len(stop_ix) > 0:
            ix_stop = stop_ix[0]
        else:
            ix_stop = 0
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_valley_stop(s, order, strategy, veto_ix=None):
        if strategy.direction == direction.long:
            stop_ix = s.arr[Cr.sav_poly_long_valley] == 1
        elif strategy.direction == direction.short:
            stop_ix = s.arr[Cr.sav_poly_short_valley] == 1
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_other_peak_entry_stop(s, order, strategy, veto_ix=None):
        if strategy.direction == direction.long:
            stop_ix = s.arr[Cr.p_short] >= strategy.p_other_side_entry_sl_exit
        elif strategy.direction == direction.short:
            stop_ix = s.arr[Cr.p_long] >= strategy.p_other_side_entry_sl_exit
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_exit_given_entry_model_stop(s, order, strategy, veto_ix=None):
        stop_ix = s.arr[Cr.p_exit_given_entry] >= strategy.p_exit_given_entry
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_dont_go_minus_again_stop(s, order, strategy, veto_ix=None):
        stop_ix = np.multiply(np.multiply(
            s.arr[Cr.roll_max_profit] > 0,
            s.arr[Cr.trailing_profit_rel] > strategy.dont_go_minus_again),
            s.arr[Cr.delta_profit] <= 0
        )
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_other_peak_entry_stop_savgol(s, order, strategy, veto_ix=None):
        if strategy.direction == direction.long:
            stop_ix = s.arr[Cr.sav_poly_short_peak] == 1
        elif strategy.direction == direction.short:
            stop_ix = s.arr[Cr.sav_poly_long_peak] == 1
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_trail_profit_stop(s, order, strategy, veto_ix=None):
        stop_ix = s.arr[Cr.trailing_profit_rel] > strategy.trail_profit_stop
        stop_ix[0] = False  # a trade shouldnt be entered in which we are immediately in loss. That's because of the min/max fill model, assuming worst case scenario
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_exit_regr_reward_stop(s, order, strategy, veto_ix=None):
        stop_ix = s.arr[Cr.regr_reward_ls_weighted] < strategy.regr_reward_exit
        stop_ix.iloc[0] = False  # a trade shouldnt be entered in which we are immediately in loss. That's because of the min/max fill model, assuming worst case scenario
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_exit_elapsed(s, order, strategy, veto_ix=None):
        ix_stop = len(s.arr) - 1  # len(s.feature_hub.data['mid']) - 1
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_exit_rl_stop(s, order, strategy, veto_ix=None):
        stop_ix = s.arr[Cr.rl_action]
        stop_ix.iloc[0] = False  # a trade shouldnt be entered in which we are immediately in loss. That's because of the min/max fill model, assuming worst case scenario
        if veto_ix is not None:
            stop_ix[veto_ix] = False
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_tree_regression_stop(s, order, strategy, veto_ix=None):
        stop_ix = {}
        ix_stops = []
        for col in [Cr.regression_0, Cr.regression_1, Cr.regression_2,
                    Cr.regression_3, Cr.regression_4,
                    # Cr.regression_5
                    ]:
            if order.direction == direction.long:
                stop_ix[col] = s.arr[:, col] < s.arr[Cr.trail_profit_stop_price]
            elif order.direction == direction.short:
                stop_ix[col] = s.arr[:, col] > s.arr[Cr.trail_profit_stop_price]
            stop_ix[col][0] = False  # a trade shouldnt be entered in which we are immediately in loss. That's because of the min/max fill model, assuming worst case scenario
            if veto_ix is not None:
                stop_ix[col][veto_ix] = False
            ix_stops.append(np.argmax(stop_ix[col]))
        ix_stop = min(ix_stops)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_stop

    def find_regr_delta_stop(s, order, strategy):
        # regr stop and avoid immediate exit after entry
        # Exit Signal: min_regr_delta_stop
        if order.direction == direction.short:
            stop_ix = s.arr[Cr.regr_delta] > strategy.min_regr_delta_stop
        elif order.direction == direction.long:
            stop_ix = s.arr[Cr.regr_delta] < strategy.min_regr_delta_stop
        # switch on regression stop after x seconds
        stop_ix[:strategy.regr_switch_after_sec] = False
        assert stop_ix[0] == False, 'Error if time regr switch is on'
        ix_stop = np.argmax(stop_ix)
        while ix_stop > 0 and s.would_reenter_same_sec(ix_stop + order.fill.ix_fill):
            stop_ix[ix_stop] = False
            ix_stop = np.argmax(stop_ix)
        ix_regr_delta_stop = s.ignore_no_stop_found(order, ix_stop)
        return ix_regr_delta_stop

    def order_timed_out(s, order, strategy):
        if order.fill.ix_fill - order.ix_signal > strategy.time_entry_cancelation:
            return True
        else:
            return False

    def find_cheat_preds_stop(s, order, strategy, veto_ix=None):
        """on a min scale, look ahead and fin min/max in preds long n short.
        if min/max </> thresh (say below above mean of preds series), then
        want to have LO at price of that stop. the stop_ix is where bba is 1 tick away, so usual handling.
        problem: still wouldnt be filled by coz need 2 ticks. switch on on-touch fill model for this.
        """
        try:
            if order.direction == direction.short:
                ix_short_cross_mean_above_down = np.argmax(
                    s.arr[Cr.p_entry_short] < s.arr[Cr.preds_short_mean] - strategy.delta_preds_mean)
                ix_short_cross_mean_above_up = ix_short_cross_mean_above_down + strategy.preds_tp + np.argmax(
                    s.arr[ix_short_cross_mean_above_down + strategy.preds_tp:, Cr.p_entry_short] > s.arr[Cr.preds_short_mean] - strategy.delta_preds_mean)
                min_short = ix_short_cross_mean_above_down + np.argmin(
                    s.arr[ix_short_cross_mean_above_down:ix_short_cross_mean_above_up, Cr.p_entry_short])
                ix_stop = min_short
            elif order.direction == direction.long:
                ix_long_cross_mean_above_down = np.argmax(
                    s.arr[Cr.p_entry_long] < s.arr[Cr.preds_long_mean] - strategy.delta_preds_mean)
                ix_long_cross_mean_above_up = ix_long_cross_mean_above_down + strategy.preds_tp + np.argmax(
                    s.arr[ix_long_cross_mean_above_down + strategy.preds_tp:, Cr.p_entry_long] > s.arr[Cr.preds_long_mean] - strategy.delta_preds_mean)
                min_long = ix_long_cross_mean_above_down + np.argmin(
                    s.arr[ix_long_cross_mean_above_down:ix_long_cross_mean_above_up, Cr.p_entry_long])
                ix_stop = min_long

            ix_stop = s.ignore_no_stop_found(order, ix_stop)
            return ix_stop
        except ValueError:
            print('Had Valueerror')
            return 0
