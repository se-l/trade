from trader.backtest.config.params_abstract import ParamsBase
from common.modules import assets
import datetime


class Params(ParamsBase):
    data_start = ts_start = datetime.datetime(2018, 6, 21)
    data_end = ts_end = datetime.datetime(2018, 7, 25, 23, 59, 59)
    # data_start = ts_start = datetime.datetime(2019, 1, 1)
    # data_end = ts_end = datetime.datetime(2019, 6, 10, 23, 59, 59)
    asset = assets.eurusd
    ex = None  # 'ex2019-12-30_23-52-54-eurusd'
    ex_entry = 'ex2020-01-07_08-42-44-eurusd'
    max_evals = 1
    use_exec_opt_param = True
    replace_exec_param_where_scan = False
    load_from_training_set = False
    # LONG RL MODELS
    # ex_rl_model = 'ex2020-01-08_22-57-59-eurusd'  # long
    # rl_model_ids = ['model_rl_long_0.44_2020-01-09_01-41-12']  # -0.009
    # rl_model_ids = ['model_rl_long_0.63_2020-01-09_04-32-29']  # .003    model_rl_long_0.77_2020-01-09_06-12-09', 'model_rl_long_0.66_2020-01-09_00-51-50']
    # rl_model_ids = ['model_rl_long_0.97_2020-01-09_05-47-35']  # 0.003
    # rl_model_ids = ['model_rl_long_0.44_2020-01-09_01-41-12', 'model_rl_long_0.63_2020-01-09_04-32-29']
    # SHORT RL MODELS
    ex_rl_model = 'ex2020-01-15_14-08-02-eurusd'  # short
    # rl_model_ids = ['model_rl_short_0.53_2020-01-16_07-32-34']  # -0.0393
    rl_model_ids = ['model_rl_short_0.44_2020-01-16_06-26-32']  #
    # rl_model_ids = ['model_rl_short_0.26_2020-01-16_06-53-12']  # -0.0277
    # rl_model_ids = ['model_rl_short_0.48_2020-01-16_05-11-33']  #
    # rl_model_ids = ['model_rl_short_0.48_2020-01-16_05-11-33', 'model_rl_short_0.44_2020-01-16_06-26-32']  # 0.0514
    resample_sec: int = 60
    resample_period: str = '{}S'.format(resample_sec)
    use_tick_forecast = False

    # validation
    store_input_curves = False
