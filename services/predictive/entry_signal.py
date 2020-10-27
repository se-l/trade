#from clr import AddReference
#AddReference("System")
#//from System import *
#from QuantConnect import *
import pickle
from functools import partial
import json
import copy
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import datetime
import statistics
from sklearn.preprocessing import KBinsDiscretizer

def get_feature_names(d):
    if type(d) == pd.DataFrame:
        return d.columns
    elif type(d) == np.ndarray:
        return list(d.dtype.names)
    else:
        return ['f{}'.format(i) for i in np.arange(0, len(d))]

def get_trade_ini_key():
    return 'ex_path_rel'

def default_to_py_type(o):
    if isinstance(o, np.int64):
        return int(o)
    elif isinstance(o, np.int32):
        return int(o)
    else:
        return o

projectDir = 'http://sebastian-lueneburg.com/model'
def get_paths():
    projectDir = 'http://sebastian-lueneburg.com/model'
    trade_ini_fn = 'http://sebastian-lueneburg.com/model/trade_ini.json'
    return trade_ini_fn


class GetPredictionsBase:
    def __init__(s, model_id):
        s.model_id = model_id
        s.req_symbols = []
        s.ini_dic = {}
        s.dicin = {}
        s.args = {}
        s.ex_path_rel = {}
        s.live_mode = False
        s.normalizer_talib = {}
        s.normalizer_pairsTraing = {}
        s.normalizer_qt = {}
        s.influx = None

    def load_normalizers(s, sym):
        try:
            ex = s.ini_dic[sym]['ex']
        except KeyError:
            return
        if ex == "":  # means the ini file has no model for this ex, hence cannot normalize anything
            return
        s.normalizer_talib[sym] = Normalize(save_range=False, load_range=True, load_kbins=True, ex=ex, range_fn='talibs')
        for sym2 in ['gbpusd', 'usdjpy']:
            s.normalizer_pairsTraing[sym2] = Normalize(save_range=False, load_range=True, load_kbins=True, ex=ex, range_fn=f'pairsTrading_{sym2}')
        # s.normalizer_qt[sym] = Normalize(save_range=False, load_range=True, load_kbins=True, ex=ex, range_fn='quotetrade')

    def load_ini_json(s, trade_ini):
        trade_ini_fn = get_paths()
        trade_ini = json.loads(s.files['trade_ini.json'])
        # with open(trade_ini_fn) as f:
        #     trade_ini = json.load(f)
        trade_ini_key = get_trade_ini_key()

        for sym, ex_path_rel in trade_ini[trade_ini_key].items():
            if sym in s.req_symbols:
                try:
                    s.ex_path_rel[sym] = trade_ini[trade_ini_key][sym][s.model_id]
                    if s.ex_path_rel[sym] in ["", None]:
                        continue
                    # s.ini_dic[sym] = {'ex': os.path.join(model_repo_path, s.ex_path_rel[sym])}
                    s.ini_dic[sym] = {'ex': s.ex_path_rel[sym]}
                except KeyError:
                    continue

        for sym, sym_dic in s.ini_dic.items():
            if sym in s.req_symbols:
                # with open(os.path.join(sym_dic['ex'], 'ini.json')) as f:
                #     ini = json.load(f)
                try:
                    ini = json.loads(s.files['ini.json'])
                except Exception as e:
                    print(sym_dic['ex'] + '/ini.json')
                    raise e
                for k, v in ini.items():
                    s.ini_dic[sym][k] = v

    def set_args(s, **kwargs):
        s.args = dict(zip(kwargs['keys'], kwargs['values']))
        for k in ['time', 'live', 'symbol']:
            s.args[k] = kwargs[k]

    def get_binned_indicators(s, sym, features_names, args=None):
        args = args if args is not None else s.args
        inp = copy.copy(args)
        inp = s.normalizer_talib[sym].normalize_kbin_kwargs(inp.values(), inp.keys())
        # inp = s.normalizer_qt[sym].normalize_kbin_kwargs(inp.values(), inp.keys())
        inp = s.normalizer_pairsTraing[sym].normalize_kbin_kwargs(inp.values(), inp.keys())
        return np.array([inp[col] for col in features_names], ndmin=2)

    def get_binned_indicators_wo_regr(s, sym, feature_names, regr_feats, args=None):
        args = args if args is not None else s.args
        feats = list(args.keys())
        inp = copy.copy(args)
        inp = s.normalizer_talib[sym].normalize_kbin_kwargs(list(inp.values()), feats)
        # inp = s.normalizer_qt[sym].normalize_kbin_kwargs(inp, feats)
        for sym2 in ['gbpusd', 'usdjpy']:
            inp = s.normalizer_pairsTraing[sym2].normalize_kbin_kwargs(inp, feats)
        inp = dict(zip(feats, inp))

        pdf = pd.DataFrame(inp, index=[pd.to_datetime(s.args['time'])])
        s.db_insert_class_preds(pdf, s.args['symbol'].lower(), measurement='vs_entry')

        return np.array([inp[col] for col in feature_names if col not in regr_feats], ndmin=2)

    def set_live_mode(s, live_mode):
        s.live_mode = live_mode

    @staticmethod
    def get_algo_feat_str(m):
        if type(m) == xgb.core.Booster:
            algo = 'xgb'
            m_feat_str = m.feature_names.__str__()
        elif type(m) == lgb.basic.Booster:
            algo = 'lgb'
            m_feat_str = m.feature_name().__str__()
        else:
            raise ('Algo not understood')
        return algo, m_feat_str

    def db_insert_class_preds(s, pdf, sym, measurement):
        s.influx.write_pdf(pdf,
                           measurement=measurement,
                           tags=dict(
                               asset=sym,
                               ex='2020-01-05_4',
                           ),
                           field_columns=pdf.columns,
                           # tag_columns=[]
                           )


class EntrySignal(GetPredictionsBase):
    def __init__(s):
        super().__init__(model_id="model_entry_post")
        s.models = {}
        s.avg_class_preds_dic = {}
        s.avg_regr_preds_dic = {}
        s.checked_class_model_features_present = {}
        s.checked_regr_model_features_present = {}
        # s.load_ini_json()
        # s.fill_ini_dic()
        s.preds_cols = ['ts', 'long', 'short']

        s.schema_mask = {}

        s.schema_pairs_df = ['price_ratio_d_1_gbpusd_newv',
                             'price_ratio_d_2_gbpusd_newv',
                             'price_ratio_d_6_gbpusd_newv',
                             'price_ratio_d_16_gbpusd_newv',
                             'price_ratio_d_33_gbpusd_newv',
                             'price_ratio_d_56_gbpusd_newv',
                             'price_ratio_d_89_gbpusd_newv',
                             'price_ratio_d_130_gbpusd_newv',
                             'price_ratio_d_182_gbpusd_newv',
                             'price_ratio_d_244_gbpusd_newv',
                             'mom_gbpusd_5_newv',
                             # 'ultosc_gbpusd_5_30_60_newv'
                             ] + \
                            ['price_ratio_d_1_usdjpy_newv',
                             'price_ratio_d_2_usdjpy_newv',
                             'price_ratio_d_6_usdjpy_newv',
                             'price_ratio_d_16_usdjpy_newv',
                             'price_ratio_d_33_usdjpy_newv',
                             'price_ratio_d_56_usdjpy_newv',
                             'price_ratio_d_89_usdjpy_newv',
                             'price_ratio_d_130_usdjpy_newv',
                             'price_ratio_d_182_usdjpy_newv',
                             'price_ratio_d_244_usdjpy_newv',
                             'mom_usdjpy_5_newv',
                             # 'ultosc_usdjpy_5_30_60_newv'
                             ]

        s.schema_out = ['bid_size_open', 'bid_size_high', 'bid_size_low', 'bid_size_close',
                        'bid_size_mean', 'bool_bid_size_update', 'size_bid_add_sum',
                        'size_bid_remove_sum', 'ask_size_open', 'ask_size_high', 'ask_size_low',
                        'ask_size_close', 'ask_size_mean', 'bool_ask_size_update',
                        'size_ask_add_sum', 'size_ask_remove_sum', 'side_buy', 'side_sell',
                        'trade_size_total', 'trade_size_mean', 'trade_size_buy',
                        'trade_size_sell',
                        'spread_ticks_close', 'spread_ticks_max',
                        'mid_price_delta_2', 'size_ask_add_cumsum_2',
                        'size_ask_remove_cumsum_2', 'bool_ask_size_update_cumsum_2',
                        'size_bid_add_cumsum_2', 'size_bid_remove_cumsum_2',
                        'bool_bid_size_update_cumsum_2', 'mid_price_delta_6',
                        'size_ask_add_cumsum_6', 'size_ask_remove_cumsum_6',
                        'bool_ask_size_update_cumsum_6', 'size_bid_add_cumsum_6',
                        'size_bid_remove_cumsum_6', 'bool_bid_size_update_cumsum_6',
                        'mid_price_delta_16', 'size_ask_add_cumsum_16',
                        'size_ask_remove_cumsum_16', 'bool_ask_size_update_cumsum_16',
                        'size_bid_add_cumsum_16', 'size_bid_remove_cumsum_16',
                        'bool_bid_size_update_cumsum_16', 'mid_price_delta_33',
                        'size_ask_add_cumsum_33', 'size_ask_remove_cumsum_33',
                        'bool_ask_size_update_cumsum_33', 'size_bid_add_cumsum_33', 'size_bid_remove_cumsum_33',
                        'bool_bid_size_update_cumsum_33',
                        'mid_price_delta_56', 'size_ask_add_cumsum_56',
                        'size_ask_remove_cumsum_56', 'bool_ask_size_update_cumsum_56',
                        'size_bid_add_cumsum_56', 'size_bid_remove_cumsum_56',
                        'bool_bid_size_update_cumsum_56', 'mid_price_delta_89',
                        'size_ask_add_cumsum_89', 'size_ask_remove_cumsum_89',
                        'bool_ask_size_update_cumsum_89', 'size_bid_add_cumsum_89',
                        'size_bid_remove_cumsum_89', 'bool_bid_size_update_cumsum_89',
                        'mid_price_delta_130', 'size_ask_add_cumsum_130',
                        'size_ask_remove_cumsum_130', 'bool_ask_size_update_cumsum_130',
                        'size_bid_add_cumsum_130', 'size_bid_remove_cumsum_130',
                        'bool_bid_size_update_cumsum_130', 'trade_cnt_total', 'trade_cnt_net',
                        'trade_size_net', 'trade_size_buy_cumsum_2', 'trade_side_buy_cumsum_2',
                        'trade_size_sell_cumsum_2', 'trade_side_sell_cumsum_2',
                        'trade_size_buy_cumsum_6', 'trade_side_buy_cumsum_6',
                        'trade_size_sell_cumsum_6', 'trade_side_sell_cumsum_6',
                        'trade_size_buy_cumsum_16', 'trade_side_buy_cumsum_16',
                        'trade_size_sell_cumsum_16', 'trade_side_sell_cumsum_16',
                        'trade_size_buy_cumsum_33', 'trade_side_buy_cumsum_33',
                        'trade_size_sell_cumsum_33', 'trade_side_sell_cumsum_33',
                        'trade_size_buy_cumsum_56', 'trade_side_buy_cumsum_56',
                        'trade_size_sell_cumsum_56', 'trade_side_sell_cumsum_56',
                        'trade_size_buy_cumsum_89', 'trade_side_buy_cumsum_89',
                        'trade_size_sell_cumsum_89', 'trade_side_sell_cumsum_89',
                        'trade_size_buy_cumsum_130', 'trade_side_buy_cumsum_130',
                        'trade_size_sell_cumsum_130', 'trade_side_sell_cumsum_130']
        s.schema_out = [el + '_newv' for el in s.schema_out]
        s.newv_feats = s.schema_out + s.schema_pairs_df

        s.regr_schema_out = [
            'regr_lgb_close_rd-n_rp-1_tp-30_dwin--1',
            'regr_lgb_close_rd-n_rp-1_tp-60_dwin--1',
            'regr_lgb_close_rd-n_rp-1_tp-120_dwin--1',
            'regr_lgb_close_rd-n_rp-1_tp-300_dwin--1',
            'regr_lgb_close_rd-n_rp-1_tp-600_dwin--1',
            'regr_lgb_close_rd-n_rp-1_tp-1800_dwin-1']

    def process_ini_json(s, req_symbols, trade_ini=None):
        print(f'Processing ini.json for: {req_symbols}')
        s.req_symbols = [x.lower() for x in req_symbols]
        s.load_ini_json(trade_ini)
        s.fill_ini_dic()
        for sym in s.req_symbols:
            s.load_normalizers(sym)

    def store_files(s, keys, values):
        s.files = {}
        for i, key in enumerate(keys):
            s.files[key] = values[i]

    def fill_ini_dic(s):
        for sym, sym_dic in s.ini_dic.items():
            ex = sym_dic['ex']
            sym_dic['model_long_entry'] = [fn.replace('model_', '').replace('_ubuntu', '').replace('_seb', '').replace('_ubuntu', '') for fn in sym_dic['model_long_entry_fn']]
            sym_dic['model_short_entry'] = [fn.replace('model_', '').replace('_ubuntu', '').replace('_seb', '').replace('_ubuntu', '') for fn in sym_dic['model_short_entry_fn']]
            sym_dic['m_long'] = {}
            sym_dic['m_short'] = {}
            for f in sym_dic['model_long_entry_fn']:
                # sym_dic['m_long'][f] = pickle.load(open(os.path.join(ex, f), 'rb'))
                try:
                    sym_dic['m_long'][f] = pickle.loads(s.files[f])
                except Exception as e:
                    print(ex + '/' + f)
            for f in sym_dic['model_short_entry_fn']:
                # sym_dic['m_short'][f] = pickle.load(open(os.path.join(ex, f), 'rb'))
                try:
                    sym_dic['m_short'][f] = pickle.loads(s.files[f])
                except Exception as e:
                    print(ex + '/' + f)
            sym_dic['m_regr'] = {}
            feats = []
            for m_name, m_dic in {**sym_dic['m_long'], **sym_dic['m_short']}.items():
                for m_name, m in m_dic.items():
                    s.checked_class_model_features_present[m_name] = False
                    s.schema_mask[m_name] = False
                    if type(m) == xgb.core.Booster:
                        m.set_param('predictor', 'cpu_predictor')
                        feats += m.feature_names
                    elif type(m) == lgb.basic.Booster:
                        # m.params['device_type'] = 'cpu'
                        feats += m.feature_name()
            s.regr_m_name = [f for f in list(set(feats)) if 'regr' in f]
            # for f in s.regr_m_name:
            #     try:
            #         model_fn = 'model_' + f + '_seb'
            #         sym_dic['m_regr'][f] = pickle.load(open(os.path.join(ex, model_fn), 'rb'))
            #         for m_name, m in sym_dic['m_regr'][f].items():
            #             s.checked_regr_model_features_present[m_name] = False
            #             s.schema_mask[m_name] = False
            #             if type(m) == xgb.core.Booster:
            #                 m.set_param('predictor', 'cpu_predictor')
            #     except FileNotFoundError:
            #         print('File not found exception. Better check as it jumped over relevant code')
            #         model_fn = 'model_' + f + '_seb'
            #         sym_dic['m_regr'][f] = pickle.load(open(os.path.join(ex, model_fn), 'rb'))

            sym_dic['algos_long'] = list(set([m for m in ['xgb', 'lgb'] if any([p for p in s.ini_dic[sym]['model_long_entry'] if m in p])]))
            sym_dic['algos_short'] = list(set([m for m in ['xgb', 'lgb'] if any([p for p in s.ini_dic[sym]['model_short_entry'] if m in p])]))
            # sym_dic['dicin'] = json.load(open(os.path.join(ex, 'bins.txt'), 'r'))
            s.ini_dic[sym] = sym_dic
            del sym_dic
    #
    # def load_preds_from_db(s, sym_list, ts_start, ts_end):
    #     db = Db()
    #     for sym in sym_list:
    #         sql = '''select ts, p from trade.model_preds
    #             where asset = {0} and
    #             feature in ({4}) and
    #             ex = {3} and
    #             ts >= '{1}' and
    #             ts <= '{2}' order by ts ;'''.format(
    #             Assets.__getattribute__(Assets, sym.upper()),
    #             ts_start, ts_end,
    #             Ex.__getattribute__(Ex, s.ini_dic[sym.lower()]['ex'].split(r'/')[-1].replace('-', 'D')),
    #             (Features.y_post_valley)
    #         )
    #         arr_long = np.array(db.fetchall(sql))
    #         sql = '''select ts, p from trade.model_preds
    #                         where asset = {0} and
    #                         feature in ({4}) and
    #                         ex = {3} and
    #                         ts >= '{1}' and
    #                         ts <= '{2}' order by ts ;'''.format(
    #             Assets.__getattribute__(Assets, sym.upper()),
    #             ts_start, ts_end,
    #             Ex.__getattribute__(Ex, s.ini_dic[sym.lower()]['ex'].split(r'/')[-1].replace('-', 'D')),
    #             (Features.y_post_peak)
    #         )
    #         arr_short = np.array(db.fetchall(sql))
    #         s.ini_dic[sym.lower()]['idx_ts'] = arr_short[:, 0]
    #         s.ini_dic[sym.lower()]['ens_con'] = np.concatenate([arr_long[:, 1:], arr_short[:, 1:]], axis=1)
    #         s.ini_dic[sym.lower()]['db_preds'] = pd.DataFrame(
    #             s.ini_dic[sym.lower()]['ens_con'],
    #             index=arr_short[:, 0], columns=['long', 'short']
    #         )
    #     db.close()
    #     return
    #
    # def load_regr_preds_from_db(s, sym_list, ts_start, ts_end):
    #     db = Db()
    #     regr_features = (Features.regr_lgb_close_30, Features.regr_lgb_close_60, Features.regr_lgb_close_120,
    #                      Features.regr_lgb_close_300, Features.regr_lgb_close_600, Features.regr_lgb_close_1800)
    #     for sym in sym_list:
    #         sql = '''select ts, feature, avg(p) from trade.model_preds where
    #         feature in {3} and
    #         asset = {0} and
    #         ts >= '{1}' and
    #         ts <= '{2}'
    #         group by ts, feature
    #         order by ts ;'''.format(
    #             Assets.__getattribute__(Assets, sym.upper()),
    #             ts_start, ts_end, regr_features
    #         )
    #         data = np.array(db.fetchall(sql))
    #         ts = np.unique(data[:, 0])
    #         ts.sort()
    #         s.ini_dic[sym.lower()]['regr_preds'] = pd.DataFrame(
    #             np.array([data[np.where(data[:, 1] == ix_feat), 2][0] for ix_feat in regr_features]).transpose(),
    #             index=ts, columns=regr_features
    #         )
    #     db.close()

    def get_xgb_matrix(s, sym, m, m_name, expanded_args=None, regr_feats=None, digitize=True):
        # risk in below approach. if regression features done exactly follow indicator
        # features in model, then data order is messed. avoided by assembling data
        # features exactly as in order from model. so first generate sth else
        if not digitize and regr_feats is not None:
            # merges args data and regression prediction for classification
            return xgb.DMatrix(np.hstack([
                s.get_binned_indicators_wo_regr(sym, m.feature_names, regr_feats, args=expanded_args),
                np.array([s.args[col] for col in m.feature_names if col in s.newv_feats], ndmin=2),
                np.array(
                    [s.avg_regr_preds_dic[col] for col in m.feature_names if col in s.regr_m_name],
                    ndmin=2
                )
            ]), feature_names=m.feature_names)
        else:
            raise("Need to add newv feats is this case matters")
            return xgb.DMatrix(
                s.get_binned_indicators_wo_regr(sym, m.feature_names, s.regr_m_name, args=s.args),
                feature_names=m.feature_names)

    def get_lgb_matrix(s, sym, m, m_name, expanded_args=None, regr_feats=None, digitize=True):
        # risk in below approach. if regression features done exactly follow indicator
        # features in model, then data order is messed. avoided by assembling data
        # features exactly as in order from model. so first generate sth else
        if not digitize and regr_feats is not None:
            if not s.schema_mask[m_name]:
                # first time a matrix is build for this model - save the schema rearrange matric
                s.schema_mask[m_name] = s.get_schema_conversion_matrix(
                    input_schema=[col for col in m.feature_name() if col not in s.regr_m_name and '_newv' not in col] \
                                 + [col for col in m.feature_name() if col in s.newv_feats] \
                                 + [col for col in m.feature_name() if col in regr_feats],
                    target_schema=m.feature_name()
                )
            return np.hstack([
                s.get_binned_indicators_wo_regr(sym, m.feature_name(), regr_feats, args=expanded_args),
                np.array(
                    [s.avg_regr_preds_dic[col] for col in m.feature_name() if col in regr_feats],
                    ndmin=2)
            ])[:, s.schema_mask[m_name]]
        else:
            if not s.schema_mask[m_name]:
                # first time a matrix is build for this model - save the schema rearrange matric
                s.schema_mask[m_name] = s.get_schema_conversion_matrix(
                    input_schema=[col for col in m.feature_name() if col not in s.regr_m_name and '_newv' not in col] \
                                 + [col for col in m.feature_name() if col in s.newv_feats],
                    target_schema=m.feature_name()
                )
            return s.get_binned_indicators_wo_regr(sym, m.feature_name(), s.regr_m_name, args=s.args)[:, s.schema_mask[m_name]]

    def get_schema_conversion_matrix(s, input_schema, target_schema):
        return [input_schema.index(target_schema[i]) for i in range(len(target_schema))]

    def get_class_preds(s, sym, m, m_name, m_avg_name):
        algo, m_feat_str = s.get_algo_feat_str(m)
        if not s.class_matrix_dic[algo].__contains__(m_feat_str):
            s.class_matrix_dic[algo][m_feat_str] = s.__getattribute__('get_classifier_{}_mat'.format(algo))(sym, m, m_name)
        if not s.checked_class_model_features_present[m_name]:
            # during binning indicators a check for indicators is performed, coz KeyError if missing. That excludes
            # all other feats. Model return a results despite missing feats.
            if len(m.feature_name()) != len(s.class_matrix_dic[algo][m_feat_str][0]):
                raise("Missing or too many features for classification predictions")
            s.checked_class_model_features_present[m_name] = True
        s.class_preds_dic[m_avg_name][m_name] = m.predict(s.class_matrix_dic[algo][m_feat_str])

    def get_regr_preds(s, sym, m_regr, m_name, m_avg_name):
        # requested regr features from classification model equal model names
        algo, m_feat_str = s.get_algo_feat_str(m_regr)
        if not s.regr_matrix_dic[algo].__contains__(m_feat_str):
            s.regr_matrix_dic[algo][m_feat_str] = s.__getattribute__('get_{}_matrix'.format(algo))(sym, m_regr, m_name)
        if not s.checked_regr_model_features_present[m_name]:
            # in case of problembs: this probably doesnt work for xgboost data. just lgb
            if len(m_regr.feature_name()) != len(s.regr_matrix_dic[algo][m_feat_str][0]):
                raise("Missing features")
        s.regr_preds_dic[m_avg_name][m_name] = m_regr.predict(s.regr_matrix_dic[algo][m_feat_str])

    def get_classifier_xgb_mat(s, sym, m, m_name):
        # args_feats = [f for f in m.feature_names if 'regr' not in f]
        regr_feats = [f for f in m.feature_names if 'regr' in f]
        return s.get_xgb_matrix(
                sym,
                m,
                m_name,
                {**s.args, **s.regr_preds_dic},
                regr_feats,
                digitize=False
                )

    def get_classifier_lgb_mat(s, sym, m, m_name):
        # args_feats = [f for f in m.feature_name() if 'regr' not in f]
        regr_feats = [f for f in m.feature_name() if 'regr' in f]
        return s.get_lgb_matrix(
                sym,
                m,
                m_name,
                {**s.args, **s.regr_preds_dic},
                regr_feats,
                digitize=False
                )

    def calc_preds(s, sym):
        s.class_matrix_dic = {k: {} for k in s.ini_dic[sym]['algos_long'] + s.ini_dic[sym]['algos_short']}
        if 'xgb' not in s.class_matrix_dic.keys():
            s.class_matrix_dic['xgb'] = {}
        s.regr_matrix_dic = s.class_matrix_dic.copy()
        s.regr_preds_dic = {k: {} for k in s.ini_dic[sym]['m_regr'].keys()}
        for m_avg_name, m_dic in s.ini_dic[sym]['m_regr'].items():
            for m_name, m in m_dic.items():
                s.get_regr_preds(sym, m, m_name, m_avg_name)
        s.avg_regr_preds_dic = {k: np.mean(list(v.values())) for k, v in s.regr_preds_dic.items()}
        # calculate class predictions
        s.class_preds_dic = {k: {} for k in {**s.ini_dic[sym]['m_long'], **s.ini_dic[sym]['m_short']}.keys()}
        for m_avg_name, m_avg_class_dic in {**s.ini_dic[sym]['m_long'], **s.ini_dic[sym]['m_short']}.items():
            for m_name, m in m_avg_class_dic.items():
                s.get_class_preds(sym, m, m_name, m_avg_name)
        s.avg_class_preds_dic = {k: np.mean(list(v.values())) for k, v in s.class_preds_dic.items()}
        return np.nan_to_num([
            np.average([s.avg_class_preds_dic[k] for k in s.ini_dic[sym]['m_long'].keys()]) if len(s.ini_dic[sym]['m_long'].keys()) > 0 else 0,
            np.average([s.avg_class_preds_dic[k] for k in s.ini_dic[sym]['m_short'].keys()]) if len(s.ini_dic[sym]['m_short'].keys()) > 0 else 0
        ])

    def calc_p(s, sym):
        # replaced this with KBin binning
        # s.dicin[sym] = s.ini_dic[sym]['dicin']
        return s.calc_preds(sym)

    def average_regr_batch_models(s):
        """Replaced by dict comprehension"""
        for k, v in s.regr_preds_dic.items():
            s.avg_regr_preds_dic[k] = np.mean(list(v.values()))

    def average_class_batch_models(s):
        """Replaced by dict comprehension"""
        for k, v in s.class_preds_dic.items():
            s.avg_class_preds_dic[k] = np.mean(list(v.values()))

    def get_model_preds(s, **kwargs):
        s.set_args(**kwargs)
        sym = s.args['symbol'].lower()
        if not s.args['live']:
            try:
                return s.get_model_preds_from_db(sym)
            except (IndexError, KeyError):
                return s.get_model_preds_calc(sym, **kwargs)
        else:
            return s.get_model_preds_calc(sym, **kwargs)

    def get_model_regr_preds(s, sym, live, ts=None):
        """returns the latest regr predictions that were calculated as part of the classifiction
        in offline mode, thtat's from db"""
        #         s.regr_preds_dic[m_avg_name][m_name]
        if s.live_mode or live:
            return [statistics.mean([el[0] for el in s.regr_preds_dic[key].values()]) for key in s.regr_schema_out[:-1]]
        else:
            ts = datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
            return list(s.ini_dic[sym.lower()]['regr_preds'].loc[ts])[:-1]

    def get_model_preds_calc(s, sym, **kwargs):
        if kwargs.__contains__('pairs_df'): # and kwargs.__contains__('tickforecast_df'):
            if kwargs['pairs_df'] is not None:
                s.args.update({s.schema_pairs_df[i]: kwargs['pairs_df'][i] for i in range(len(s.schema_pairs_df))})
                pdf = pd.DataFrame(s.args, index=[pd.to_datetime(s.args['time'])])
                pdf = pdf.drop(['time', 'symbol', 'live'], axis=1)
                s.db_insert_class_preds(pdf, s.args['symbol'].lower(), measurement='vs_entry_unbinned')
                return s.calc_p(sym)
            else:
                return [0, 0]
            # if kwargs['tickforecast_df'] is not None:
            #     s.args.update({s.schema_out[i]: kwargs['tickforecast_df'][i] for i in range(len(s.schema_out))})
        else:
            return [0, 0]

    def get_model_preds_from_db(s, sym):
        return s.ini_dic[sym]['db_preds'].loc[datetime.datetime.strptime(s.args['time'], '%Y-%m-%d %H:%M:%S')].values.tolist()
        # idx = np.where(s.ini_dic[sym]['idx_ts'] == np.datetime64(s.args['time']))[0][0]
        # return s.ini_dic[sym]['ens_con'][idx]


class Normalize:
    '''
    Either bin or scale inputs. Load and store values used for normalization
    Each dataset gets their respective instance for normalization
    '''
    ix_min = 0
    ix_max = 1

    def __init__(s, save_range, load_range, **kwargs):
        s.load_range = load_range
        s.save_range = save_range
        if save_range or load_range:
            s.ex = kwargs['ex']
            s.range_fn = kwargs['range_fn']
        s.load_kbins = kwargs['load_kbins']
        if s.load_kbins:
            s.kbins_dict = s.load_normalize_kmeans_bins()

    def ready_norm_min_max(s, arr):
        if not s.load_range:
            s.norm_min_max = {col: (None, None) for col in get_feature_names(arr)}

    def temp_save(s, col, arr_min, arr_max):
        if s.store_normalize_ranges:
            s.norm_min_max[col] = (default_to_py_type(arr_min), default_to_py_type(arr_max))

    def normalize_scale01_ndarr(s, arr):
        s.ready_norm_min_max(arr)
        for col in get_feature_names(arr):  # range(arr.shape[1]):
            arr[col], arr_min, arr_max = s.normalize_scale01_1darr(arr[col], s.norm_min_max[col][s.ix_min], s.norm_min_max[col][s.ix_max])
            s.temp_save(col, arr_min, arr_max)
        if s.save_range:
            s.store_normalize_ranges()
        return arr

    def normalize_kmeans_bin_ndarr(s, arr, n_bins=30):
        if not s.load_kbins:
            if len(arr) < 5000:
                s.kbins_dict = dotdict()
                for col in get_feature_names(arr):
                    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
                    if type(arr) == pd.DataFrame:
                        est.fit(arr[col].values.reshape(-1, 1))
                    else:
                        est.fit(arr[col].reshape(-1, 1))
                    s.kbins_dict[col] = est
            else:
                with ProcessPoolExecutor(max_workers=min(4, len(arr.columns))) as pool:
                    est_name = list(pool.map(partial(s.pp_kbin_transform, n_bins=n_bins), s.arr_num_tup_gen(arr)))
                s.kbins_dict = dotdict({name: est for est, name in est_name})
            s.load_kbins = True
        print('Applying KBins...')
        for col in get_feature_names(arr):
            if col not in s.kbins_dict.keys():
                continue
            if type(arr) == pd.DataFrame:
                arr[col] = s.kbins_dict[col].transform(arr[col].values.reshape(-1, 1)).reshape(1, -1)[0].astype(int)
            else:
                arr[col] = s.kbins_dict[col].transform(arr[col].reshape(-1, 1)).reshape(1, -1)[0].astype(int)
        if s.save_range:
            s.store_normalize_kmeans_bins()
        return arr

    def normalize_kbin_kwargs(s, values: list, feature_names: list):
        """those must have a matching sequence"""
        for i in range(len(feature_names)):
            col = feature_names[i]
            if col not in s.kbins_dict.keys():
                continue
            else:
                values[i] = s.kbins_dict[col].transform([[values[i]]])[0][0].astype(int)
        return values

    @classmethod
    def normalize_scale01_ndarr_static(cls, arr):
        for col in range(arr.shape[1]):
            if type(arr) == np.ndarray:
                arr[:, col], arr_min, arr_max = cls.normalize_scale01_1darr(arr[:, col])
            elif type(arr) == pd.DataFrame:
                arr.iloc[:, col], arr_min, arr_max = cls.normalize_scale01_1darr(arr.iloc[:, col])
            else:
                raise('array type not understood')
        return arr

    @staticmethod
    def normalize_scale01_1darr(arr, arr_min=None, arr_max=None):
        if arr_min is None:
            arr_min = np.min(arr)
        if arr_max is None:
            arr_max = np.max(arr)
        return np.divide(
            np.subtract(arr, arr_min),
            np.subtract(arr_max, arr_min)
        ), arr_min, arr_max

    def load_normalize_kmeans_bins(s):
        # with open(os.path.join(s.ex, '{}.KBins'.format(s.range_fn)), 'rb') as f:
        #     s.kbins_dict = pickle.load(f)
        # try:
        s.kbins_dict = pickle.loads(s.files['{}.KBins'.format(s.range_fn)])
        # except Exception as e:
        #     print('{}.KBins'.format(s.range_fn))
        return s.kbins_dict

    @staticmethod
    def float_to_int_approx(arr, digits=6):
        """
        convert 0-1 scaled inputs to an integer representation to save memory and space
        """
        return np.multiply(arr, 10 ** digits).astype(int)

    def quantize_df(s, dfFull: pd.DataFrame):
        s.norm_min_max = s.load_normalize_range()
        for c in dfFull.columns:
            dfFull[c] = np.digitize(
                dfFull[c],
                bins=s.norm_min_max[c],
                right=True
            )
        return dfFull

    def kmeans_bin_df(s, dfFull: pd.DataFrame, n_bins=20):
        s.kbins_dict = s.load_normalize_kmeans_bins()
        for c in dfFull.columns:
            dfFull[c] = s.kbins_dict[c].transform(dfFull[c].values.reshape(-1, 1)).reshape(1, -1)[0]
        return dfFull

    @staticmethod
    def arr_num_tup_gen(df):
        if type(df) == pd.DataFrame:
            for col in df.columns:
                print(col)
                yield df[col].values, col
        else:
            raise TypeError('Not compatible with object yet. Add handling')

    @staticmethod
    def pp_kbin_transform(arr_name_tup: (np.array, str), n_bins=20):
        arr, name = arr_name_tup
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
        est.fit(arr.reshape(-1, 1))
        return (est, name)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


entrySignal = EntrySignal()
# entrySignal.process_ini_json(['eurusd'])

