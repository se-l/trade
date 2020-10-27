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
                HpTup('p_entry_sl_exit', 0, hp.choice, np.arange(0.1, 0.2, 0.02), True),
                HpTup('twin_peak_trailing_profit', 155 * bp, hp.choice, np.arange(-30, 10, 5) * bp, True),
                # range needs to lower half of range
                HpTup('veto_stop_ema_d300', 10000*-10 * bp, hp.choice, np.arange(-10, -45, -5) * bp, True),
                HpTup('veto_p_exit', 10.85, hp.choice, np.arange(0.7, 1.01, 0.05), True),
                HpTup('p_exit', 11.0, hp.choice, np.arange(0.7, 1.01, 0.03), True),
                HpTup('mom_spike_stop', 11.0, hp.choice, np.arange(0.7, 1.01, 0.03), True),

                HpTup('min_stop_loss', 0 * bp, hp.choice, np.arange(0, 40, 5) * bp, True),
                HpTup('delta_limit_exit', 0 * bp, hp.choice, np.arange(.2, .8, .2), True),
                HpTup('delta_limit_exit_update', 0 * bp, hp.quniform, np.arange(.6, 1.1, .2), True),
                HpTup('max_trade_length', 800, hp.choice, np.arange(20000, 35001, 5000), True),

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
                HpTup('preds_sl_thresh', 10.1, hp.choice, np.arange(0.1, 0.15, 0.02), False),
                HpTup('p_tick_against', 1, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('use_rl_exit', True, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('p_other_side_entry_sl_exit', 0.1, hp.choice, np.arange(0.08, 0.13, 0.02), False),
                HpTup('rl_exit_sep_factor', 1, hp.choice,  [0.8, 1, 1.2], True),
                HpTup('rl_exit_delta_diff', 0.00000005, hp.choice, [-0.00005, 0.0001, 0.00005], False),
                HpTup('mo_n_ticks', 5, hp.choice, [5], True),
            ]
        elif direction == direction.long:
            s.p_opt = [
                HpTup('assume_late_limit_fill', False, hp.choice, [True, False], True),
                HpTup('assume_late_limit_fill_entry', False, hp.choice, [True, False], True),
                HpTup('use_rl_exit', False, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('use_regr_reward_exit', True, hp.choice, np.arange(-0.025, 0.01, 0.005), True),
                HpTup('mo_n_ticks', 5, hp.choice, [5], True),
                HpTup('entry_min_delta_rl_risk_reward', 4, hp.choice, np.arange(4, 8.01, 0.2), True),
                HpTup('entry_max_slope_rl_risk_reward', 1000, hp.choice, np.arange(-0.15, 0.151, 0.02), True),
                HpTup('regr_reward_exit', -1, hp.choice, np.arange(-1.8, 1.01, 0.2), True),
                # HpTup('entry_min_delta_rl_risk_reward', 2.65, hp.choice, np.arange(-0.1, 5.01, 0.05), False),
                # HpTup('entry_max_slope_rl_risk_reward', -0.15, hp.choice, np.arange(-0.15, 0.151, 0.02), False),
                # HpTup('regr_reward_exit', 0.8, hp.choice, np.arange(-4, 3.01, 0.05), False),
                HpTup('rl_risk_reward_ls_9999', 1, hp.choice, np.arange(1, 101, 33), True),
                HpTup('rl_risk_reward_ls_999', 1, hp.choice, np.arange(1, 101, 33), True),
                HpTup('rl_risk_reward_ls_995', 1, hp.choice, np.arange(1, 101, 33), True),
                HpTup('rl_risk_reward_ls_99', 1, hp.choice, np.arange(1, 101, 33), True),
            ]
