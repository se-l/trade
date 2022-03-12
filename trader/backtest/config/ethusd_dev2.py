from common.modules import direction
from common.modules import exchange
from common.utils.util_func import SeriesTickType
from trader.backtest.config.params_abstract import ParamsBase
from common.modules import assets
import datetime


class Params(ParamsBase):
    exchange = exchange.bitmex
    train = True
    data_start = ts_start = datetime.datetime(2019, 8, 2)
    data_end = ts_end = datetime.datetime(2019, 8, 4, 23, 59, 59)
    load_model_from_influx = True
    # data_start = ts_start = datetime.datetime(2019, 8, 5)
    # data_end = ts_end = datetime.datetime(2019, 8, 6, 23, 59, 59)
    # load_model_from_influx = False
    asset = assets.ethusd
    series_tick_type = SeriesTickType('volume_usd', 10000, 'volume_usd_10000')
    ex = None  # 'ex2019-12-30_23-52-54-eurusd'
    ex_entry = 'ex2020-09-27_20-44-16-ethusd'
    max_evals = 1
    use_exec_opt_param = True
    replace_exec_param_where_scan = False
    load_from_training_set = False
    dependency_models = {
        'volume_usd_10000':
            {
                # 'p_y_peak': 'ex2020-08-01_04-33-55-ethusd/model_classification_lgb_rd-n_ts-1596228213.25',
                # 'p_y_valley': 'ex2020-08-01_04-33-55-ethusd/model_classification_lgb_rd-n_ts-1596228214.375',
                # 'rl_risk_reward_ls_500_999': 'ex2020-09-27_20-44-16-ethusd/model_regression_lgb_rd-n_horizon-500_decay-0.999_ts-1600600349.809',
                # 'rl_risk_reward_ls_1000_999': 'ex2020-09-27_20-44-16-ethusd/model_regression_lgb_rd-n_horizon-1000_decay-0.999_ts-1600600349.831',
                # 'rl_risk_reward_ls_2000_999': 'ex2020-09-27_20-44-16-ethusd/model_regression_lgb_rd-n_horizon-2000_decay-0.999_ts-1600600349.741',
                # 'rl_risk_reward_ls_4000_999': 'ex2020-09-27_20-44-16-ethusd/model_regression_lgb_rd-n_horizon-4000_decay-0.999_ts-1600600349.842',
                'rl_risk_reward_ls_9999': 'ex2020-09-27_20-44-16-ethusd/model_regression_lgb_rd-n_ts-1601210817.201',
                'rl_risk_reward_ls_999': 'ex2020-09-27_20-44-16-ethusd/model_regression_lgb_rd-n_ts-1601210818.25',
                'rl_risk_reward_ls_995': 'ex2020-09-27_20-44-16-ethusd/model_regression_lgb_rd-n_ts-1601210831.656',
                'rl_risk_reward_ls_99': 'ex2020-09-27_20-44-16-ethusd/model_regression_lgb_rd-n_ts-1601210971.337',
            },
        'second': {}
    }
    dependency_normalize = {
        'volume_usd_10000': {
            'load_feature': 'ex2020-09-27_20-44-16-ethusd'
        },
        # 'second': {
        #     'load_feature': 'ex2020-09-07_01-40-45-ethusd'
        # }
    }
    load_exit_bins = False
    num_test_iterations = 5
    num_training_iterations = 5
    pnl_model_store_interval = 2  # test all models on validation set
    store_int = [[]]
    # Env
    model_direction = direction.long

    # Entry Exit params
    # entry_min_delta_rl_risk_reward = 0  # By strategy params
    # entry_max_slope_rl_risk_reward = 0  # By strategy params
    # exit_min_delta_rl_risk_reward = 0  # By strategy params
    # reward_horizon = 2500  # By model params
    # discount_decay = 0.999  # By model params
    # discount_decay_min_threshold = 0.01  # By model params
    use_tick_forecast = False

    # validation
    store_input_curves = False
