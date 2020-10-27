import numpy as np
from hyperopt import hp
from common.globals import bp
from common.modules import direction
from trader.backtest.config.optimize_params_base import OptParamsBase, HpTup


class OptParams(OptParamsBase):
    def __init__(s, direction):
        if direction == direction.short:
            s.p_opt = [
                HpTup('assume_late_limit_fill', True, hp.choice, [True, False], True),
                HpTup('assume_late_limit_fill_entry', True, hp.choice, [True, False], True),
                HpTup('assume_simulated_future', False, hp.choice, [True, False], True),
                HpTup('delta_limit', 0 * bp, hp.choice, np.arange(3, 6, 2) * bp, True),
                HpTup('profit_target', 590 * bp, hp.choice, np.arange(440, 801, 30) * bp, True),
                HpTup('trailing_stop_a', 400 * bp, hp.choice, np.arange(340, 621, 30) * bp, True),
                HpTup('max_trailing_stop_a', 260 * bp, hp.choice, np.arange(190, 300, 30) * bp, True),
                HpTup('trail_profit_stop', 95 * bp, hp.choice, np.arange(35, 85, -5) * bp, True),
                HpTup('preds_net_exit', 0.12, hp.choice, np.arange(0.15, 0.35, 0.02), True),
                HpTup('p_entry_sl_exit', 0, hp.choice, np.arange(0.15, 0.35, 0.02), True),
                HpTup('twin_peak_trailing_profit', 155 * bp, hp.choice, np.arange(-30, 10, 5) * bp, True),
                # range needs to lower half of range
                HpTup('veto_stop_ema_d300', 10000*-10 * bp, hp.choice, np.arange(-10, -45, -5) * bp, True),
                HpTup('veto_p_exit', 10.85, hp.choice, np.arange(0.7, 1.01, 0.05), True),
                HpTup('p_exit', 11.0, hp.choice, np.arange(0.7, 1.01, 0.03), True),
                HpTup('mom_spike_stop', 11.0, hp.choice, np.arange(0.7, 1.01, 0.03), True),

                HpTup('min_stop_loss', 0 * bp, hp.choice, np.arange(0, 40, 5) * bp, True),
                HpTup('delta_limit_exit', 0 * bp, hp.choice, np.arange(.2, .8, .2), True),
                HpTup('delta_limit_exit_update', 0 * bp, hp.quniform, np.arange(.6, 1.1, .2), True),
                HpTup('max_trade_length', 80000, hp.choice, np.arange(20000, 35001, 5000), True),

                # range needs to be upper half of range
                HpTup('bull_bear_stretch', 910 * bp, hp.choice, np.arange(13, 22, 1) * bp, True),
                # HpTup('min_bband_volat', 0, hp.choice, np.arange(0, 45, 5), True),
                # mst exclude values leading to 0 entries
                HpTup('preds_net_thresh', -0.2, hp.choice, np.arange(-0.18, -0.25, -0.01), True),  # single best 0.17
                HpTup('d_net_cancel_entry', 0.02, hp.choice, np.arange(0, 0.04, 0.01), True),
                HpTup('bband_tp', 30, hp.choice, np.arange(15, 60, 15), True),
                HpTup('bband_nbdevup', 2, hp.choice, np.arange(1, 4, 1), True),
                HpTup('bband_nbdevdn', 2, hp.choice, np.arange(1, 4, 1), True),
                HpTup('bband_matype', 0, hp.choice, np.arange(1, 4, 1), True),
                HpTup('rebound_mom', -90.02, hp.choice, np.arange(1, 4, 1), True),

                HpTup('height', 0.012, hp.choice, np.arange(0.03, 0.02, 0.03), True),
                HpTup('prominence', 0.006, hp.choice, np.arange(0.03, 0.02, 0.03), True),
                HpTup('distance', 20, hp.choice, np.arange(5, 25, 5), True),
                HpTup('finder_lookback', 120, hp.choice, np.arange(1, 4, 1), True),
                HpTup('slope_long_thresh', 0.006, hp.choice, np.arange(0, 0.025, 0.001), True),
                HpTup('slope_short_thresh', -0.02, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('min_pv_cnt', 2, hp.choice, np.arange(2, 8, 1), True),
                HpTup('stop_sideline', False, hp.choice, [True, False], True),
                HpTup('preds_sl_thresh', 10.07, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('p_tick_against', 1, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('use_rl_exit', True, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('p_other_side_entry_sl_exit', 0.2, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('rl_exit_sep_factor', 1, hp.choice,  [1], True),
                HpTup('mo_n_ticks', 5, hp.choice, [5], True),
            ]
        elif direction == direction.long:
            s.p_opt = [
                HpTup('assume_late_limit_fill', False, hp.choice, [True, False], True),
                HpTup('assume_late_limit_fill_entry', False, hp.choice, [True, False], True),
                HpTup('assume_simulated_future', False, hp.choice, [True, False], True),
                HpTup('delta_limit', 0 * bp, hp.choice, np.arange(3, 6, 2) * bp, True),
                HpTup('profit_target', 620 * bp, hp.choice, np.arange(300, 801, 30) * bp, True),
                HpTup('trailing_stop_a', 180 * bp, hp.choice, np.arange(150, 350, 30) * bp, True),
                HpTup('max_trailing_stop_a', 220 * bp, hp.choice, np.arange(100, 300, 20) * bp, True),
                HpTup('trail_profit_stop', 95 * bp, hp.choice, np.arange(35, 76, 5) * bp, True),
                HpTup('preds_net_exit', -10.22, hp.choice, np.arange(-0.12, -0.25, 0.02), True),  # -0.22 single best
                HpTup('p_entry_sl_exit', 0.12, hp.choice, np.arange(0.1, 0.2, 0.02), False),
                HpTup('twin_peak_trailing_profit', 55 * bp, hp.choice, np.arange(-45, 11, 10) * bp, True),
                HpTup('veto_stop_ema_d300', 1000*23 * bp, hp.choice, np.arange(5, 30, 2) * bp, True),  # range needs to be upper half of range
                HpTup('veto_p_exit', 11, hp.choice, np.arange(0.7, 1.01, 0.05), True),
                HpTup('p_exit', 11, hp.choice, np.arange(0.7, 1.01, 0.05), True),
                HpTup('mom_spike_stop', 11.0, hp.choice, np.arange(0.7, 1.01, 0.03), True),

                HpTup('min_stop_loss', 0 * bp, hp.choice, np.arange(0, 40, 5) * bp, True),
                HpTup('delta_limit_exit', 0 * bp, hp.choice, np.arange(.2, .8, .2), True),
                HpTup('delta_limit_exit_update', 0 * bp, hp.quniform, np.arange(.6, 1.1, .2), True),
                HpTup('max_trade_length', 80000, hp.choice, np.arange(20000, 35001, 5000), True),

                HpTup('bull_bear_stretch', -911 * bp, hp.choice, np.arange(-14, -5, 1) * bp, True),  # range needs to be lower half of range
                # HpTup('min_bband_volat', 30, hp.choice, np.arange(0, 45, 5), True),
                HpTup('preds_net_thresh', 10.17, hp.choice, np.arange(0.18, 0.40, 0.02), True),  # mst exclude values leading to 0 entries
                HpTup('d_net_cancel_entry', 0.03, hp.choice, np.arange(0, 0.04, 0.01), True),

                HpTup('rebound_mom', 90.02, hp.choice, np.arange(1, 4, 1), True),

                HpTup('height', 0.01, hp.choice, np.arange(0.003, 0.02, 0.003), True),
                HpTup('prominence', 0.007, hp.choice, np.arange(0.003, 0.02, 0.003), True),
                HpTup('distance', 10, hp.choice, np.arange(5, 25, 5), True),
                HpTup('finder_lookback', 120, hp.choice, np.arange(120, 4, 1), True),
                HpTup('slope_long_thresh', 0.002, hp.choice, np.arange(0, 0.025, 0.001), True),
                HpTup('slope_short_thresh', -0.05, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('min_pv_cnt', 3, hp.choice, np.arange(2, 8, 1), True),
                HpTup('stop_sideline', True, hp.choice, [True, False], True),
                HpTup('preds_sl_thresh', 0.11, hp.choice, np.arange(0.11, 0.16, 0.02), False),
                HpTup('p_tick_against', 1, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('use_rl_exit', False, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('use_regr_reward_exit', True, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('regr_reward_exit', 0, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                # HpTup('entry_min_delta_rl_risk_reward', 0, hp.choice, np.arange(0, 300, 50), False),
                HpTup('p_other_side_entry_sl_exit', 0.1, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('rl_exit_sep_factor', 1.1, hp.choice,  [0.8, 1, 1.2], False),
                HpTup('mo_n_ticks', 5, hp.choice, [5], True),
            ]
