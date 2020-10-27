import pickle

import importlib

import click
import os
from functools import partial

from trader.backtest.agent_backtest import AgentBacktest
from common.globals import OHLCV
from common.utils.normalize import Normalize
from trader.environments.brokerage import Brokerage
from trader.backtest.system import System
from common.modules import ctx
from common.utils.util_func import standard_params_setup
from hyperopt import Trials, space_eval, fmin, atpe, STATUS_OK
from trader.environments.exchange import Exchange
from trader.environments.feature_hub import FeatureHub
from trader.backtest.strategy_info import Strategy, StrategyLib
from common.modules import direction
from common import Paths
from common.modules import dotdict
from common.modules.logger import logger


class Director:
    def __init__(s, params, strategy_lib):
        s.params = params
        s.num_training_iterations = params.num_training_iterations
        s.strategy_lib = strategy_lib
        s.feature_hub = FeatureHub(params)
        # ideally this whole block is also config driven. for each ts, need to have the normalizers and models ready
        # params_ts = copy.copy(params)
        # params_ts.series_tick_type = SeriesTickType('ts', params.resample_sec, 'second')
        hubs = [
            s.feature_hub,
            # FeatureHub(params_ts)
        ]
        hubs[0].prefix = 'volume_usd_10000'
        # hubs[0].prefix = 'second'
        # hubs[-1].pdp[["second|MOM_real_23", "second|MOM_real_360", "second|EMA_real_540"]]
        for hub in hubs:
            hub.dependencies['load_model_from_influx'] = s.params.load_model_from_influx
            hub.dependencies['m2m_influx'] = {
                # 'rl_risk_reward_ls_9999': 'model_regression_lgb_rd-n_ts-1601230252.107',
                # 'rl_risk_reward_ls_999': 'model_regression_lgb_rd-n_ts-1601230332.939',
                'rl_risk_reward_ls_995': 'model_regression_lgb_rd-n_ts-1603245630.012',
                # 'rl_risk_reward_ls_99': 'model_regression_lgb_rd-n_ts-1601234729.784',
            }
            hub.dependencies['regr_reward_weights'] = {
                # 'rl_risk_reward_ls_9999': strategy.rl_risk_reward_ls_9999,
                # 'rl_risk_reward_ls_999': strategy.rl_risk_reward_ls_999,
                'rl_risk_reward_ls_995': 1,
                # 'rl_risk_reward_ls_99': strategy.rl_risk_reward_ls_99,
            }
            for key, ex_model_fn in s.params.dependency_models[hub.prefix].items():
                if 'influx' in ex_model_fn:
                    continue
                with open(os.path.join(Paths.trade_model, ex_model_fn), "rb") as f:
                    hub.dependencies['models'][key] = pickle.load(f)
            for key, ex in s.params.dependency_normalize[hub.prefix].items():
                hub.dependencies['normalize'][key] = Normalize(ex=ex, range_fn=key)
            if hub.dependencies['models']:
                hub.pdp[list(hub.dependencies['models'][list(hub.dependencies['models'].keys())[0]].values())[0].feature_name()]
            standard_params_setup(params, Paths.backtests)
        # s.feature_hub.merge_event_series(hubs[-1])
        s.feature_hub.pdp[['mid.' + c for c in OHLCV + ['ts']] + ['ask.close', 'bid.close']]
        s.feature_hub.pdp['ts'] = s.feature_hub.pdp['mid.ts']
        s.feature_hub.curtail_ts(s.params.ts_start, s.params.ts_end)

        s.system = System(s.params)
        s.exchange = Exchange(s.params, s.feature_hub).setup()
        s.brokerage = Brokerage(s.params, s.exchange, s.feature_hub)
        s.backtest_agent = AgentBacktest(s.params, s.brokerage, s.system, s.feature_hub)
        s.backtest_agent.strategy_lib, p_opts = s.backtest_agent.setup_opt_params(strategy_lib, params)

        s.sortino_ratios = None
        s.training_rewards = []
        s.schema = []

    def backtest(s):
        logger.info('Optimizing...')
        s.backtest_agent.train_via_backtest()


def hyperopt_backtest_params(s, strategy_lib):
    s.strategy_lib, p_opts = s.backtest_agent.setup_opt_params(strategy_lib, s.params)
    hyperopt_trials = Trials()
    best_params = space_eval(
        p_opts, fmin(
            partial(dosth, s=s),
            space=p_opts,
            algo=atpe.suggest,
            trials=hyperopt_trials,
            verbose=True,
            max_evals=s.params.max_evals
        ))
    logger.info(best_params)
    return hyperopt_trials, best_params


def dosth(p_opt, s: Director):
    if s.params.max_evals > 1:
        logger.info(p_opt)
    s.brokerage.reset()
    s.backtest_agent.reset_episode()
    s.backtest_agent.set_p_opt(p_opt)
    strategy = s.strategy_lib.get_strategy(0)
    # s.feature_hub.dependencies['regr_reward_norm'] = {
    #     # 'rl_risk_reward_ls_9999': 893.3139800617058,
    #     # 'rl_risk_reward_ls_999': 1040.2635276676667,
    #     # 'rl_risk_reward_ls_995': 2060.3458369375603,
    #     # 'rl_risk_reward_ls_99': 8266.274055442218
    # }
    s.feature_hub.dependencies['regr_reward_weights'] = {
        # 'rl_risk_reward_ls_9999': strategy.rl_risk_reward_ls_9999,
        # 'rl_risk_reward_ls_999': strategy.rl_risk_reward_ls_999,
        'rl_risk_reward_ls_995': strategy.rl_risk_reward_ls_995,
        # 'rl_risk_reward_ls_99': strategy.rl_risk_reward_ls_99,
    }
    s.backtest_agent.backtest_single_ensemble()
    logger.info(s.brokerage.episode_summary())
    return {
        'loss': -1 * s.brokerage.total_profit,
        'status': STATUS_OK,
    }
    # if True:
    # s.brokerage.store_backtest(j)
    # store_bt_pnl(s.brokerage.orders, j, s.params)
    # store_rewards(s.feature_hub.pdp,
    #               [Cr.regr_reward_ls_weighted, Cr.regr_reward_neutral_weighted],
    #               j, s.params, overwrite=True)


def optimize(p_opt, agent):
    agent.set_p_opt(p_opt)
    agent.backtest_single_ensemble()
    total_profit = agent.get_score()
    return total_profit


def init_ix_stops():
    return dotdict({k: 0 for k in [
        'ix_trail_stop', 'ix_elapsed', 'ix_trail_profit_stop',
        'ix_between_bbands_stop', 'ix_losing_peaks_stop', 'ix_preds_net_stop',
        'ix_exit_preds_stop', 'ix_entry_preds_stop', 'ix_spike_mom_stop',
        'ix_bband_take_profit_stop', 'ix_bband_loss_stop', 'ix_opp_exceeded_stop',
        'ix_take_abs_profit_stop', 'ix_cheat_preds_stop', 'ix_entry_preds_dx_stop',
        'ix_cheat_valley_stop', 'ix_other_peak_entry_stop', 'ix_exit_given_entry',
        'ix_dont_go_minus_again_stop', 'ix_rl_exit'
    ]})


@click.command('backtest')
@click.pass_context
def backtest(ctx: ctx):
    params = importlib.import_module('{}.{}'.format(Paths.path_config_backtest, ctx.obj.fn_params)).Params()
    strategy_lib = StrategyLib([
        # Strategy(0, params.asset.upper(), Direction.short),
        Strategy(0, params.asset.upper(), direction.long),
    ])
    standard_params_setup(params, Paths.backtests)
    director = Director(params, strategy_lib)
    # hyperopt_backtest_params(director, strategy_lib)
    director.backtest()


@click.command()
@click.pass_context
def main(ctx):
    ctx.obj = dotdict(dict(
        fn_params='ethusd'
    ))
    ctx.forward(backtest)


if __name__ == '__main__':
    main()
