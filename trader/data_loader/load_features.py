import os
import json
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from common.globals import OHLCV, OHLC
from common.utils.normalize import Normalize
from common.modules.series import Series
from trader.data_loader.config.config import drop_features
from trader.data_loader.gen_features import EngineerFeatures
from common.utils.util_func import df_to_npa, find_indicator_ready_ix
from common.modules.logger import logger
from common.paths import Paths
from trader.data_loader.utils_features import get_ohlcv_mid_price, get_ohlc, digitize


def load_features(params, use_mid_price, df_ohlc=None):
    if df_ohlc is None and use_mid_price:
        df_ohlc = get_ohlcv_mid_price(params)
    elif df_ohlc is None:
        df_ohlc = get_ohlc(
            start=params.data_start,
            end=params.data_end,
            series=Series.trade,
            **params
        )
    ef = EngineerFeatures(params)
    # SET UP FEATURE LIBRARY
    ef.reg_func_lib(col=OHLC, inputs=None, function='base')
    if not params.req_feats:
        ef.get_potential_talibs(df_ohlc)
    else:
        ef.register_feats(req_feats=params.req_feats)
    # log_len_pot_fn(ef, 'gen_talibs_fast')
    # add feats relying on talib values
    req_feats = [k for k, v in ef.featureLib.items() if v['function'] == 'gen_talibs_fast']
    # test: requested feats in source?
    # logger.info('req_feats not found in source file: {}'.format([c for c in req_feats if c not in df_ohlc.columns]))
    all_feats = list(ef.featureLib.keys())
    # logger.info('Max available feats: {}'.format(len(all_feats)))
    best_feats = all_feats
    model_required_features = all_feats
    df_full = gen_df_full(ef, params, best_feats, model_required_features, df_ohlc)
    try:
        if params.cut_off_talib_warmup:
            # *5 to account for unstable period, su
            last_inv_ix = find_indicator_ready_ix(df_full) * 2
            logger.info('Indicator Warmup: {} sec'.format(last_inv_ix * params.resample_sec))
            df_full = df_full[last_inv_ix:].reset_index(drop=True)
            df_ohlc = df_ohlc[last_inv_ix:].reset_index(drop=True)
    except AttributeError:
        pass

    if params.quantize_inds:
        # df_full = s.quantize_df(df_full)
        normalize = Normalize(True, False, ex=params.ex, range_fn='load_feature')
        df_full = normalize.kmeans_bin_df(df_full)
    return df_full, df_ohlc


def quantize_df(params, df_full):
    try:
        with open(os.path.join(Paths.projectDir, 'model', params.ex, 'bins.txt'), 'r') as f:
            bin_edges = json.load(f)
        for c in df_full.columns:
            df_full[c] = np.digitize(df_full[c], bins=bin_edges[c], right=True)
    except FileNotFoundError:
        logger.info('Creating new bins.txt...')
        df_full, bin_edges = digitize(df_full, df_full.columns, n_bins=20.1)

        with open(os.path.join(Paths.projectDir, 'model', params.ex, 'bins.txt'), 'w') as out:
            json.dump(bin_edges, out)
    return df_full


def gen_df_full(ef, params, best_feats, model_required_features, df_ohlc):
    df_full = get_features_batch(ef, params,
                                 df_ohlc,
                                 best_feats,
                                 i=0,
                                 required=model_required_features
                                 )
    df_full = df_full.drop([c for c in OHLCV + ['ts'] if c in df_full.columns], axis=1)
    # df_ohlc = df_ohlc.loc[df_full.index, :]
    df_full = df_full.drop([k for k in drop_features if k in df_full.columns], axis=1)
    return df_full


def get_features_batch(ef, params, df_talib, all_feats, i, required):
    # req_feats = list(np.unique(all_feats[i * params.featPerBatch: (i + 1) * params.featPerBatch] + required))
    req_feats = all_feats
    # logger.info('Len requested feats in batch: {}'.format(len(req_feats)))

    df_full = df_to_npa(df_talib[OHLCV], def_type=np.float)
    df_full = ef.gen_requested_feats_fast(nd=df_full,
                                          colnames=req_feats,
                                          only_last_row=False,
                                          ohlc_normalized=True)
    df_full = pd.DataFrame(df_full,
                           index=df_talib.index,
                           columns=df_full.dtype.names)
    return df_full


def log_len_pot_fn(ef, fn):
    logger.info('{}: {}'.format(fn, len([k for k, v in ef.featureLib.items() if ef.featureLib[k]['function'] == fn])))


def compress_data_w_non_required_features(pdf: pd.DataFrame, params, df_talib):
    col_before = pdf.columns  # todo: next operation with a struc_npa
    pdf, keep_idx = np.unique(pdf, return_index=True, axis=1)
    rm_idx = [i for i in range(0, len(col_before)) if i not in keep_idx]
    logger.info('Removing {} duplicated columns: {}'.format(len(col_before) - len(keep_idx), col_before[rm_idx]))
    pdf = pd.DataFrame(pdf, index=df_talib.index, columns=col_before[keep_idx])
    # remove feats not requested
    col_not_req = [c for c in pdf.columns if c not in params.req_feats]
    if len(col_not_req) > 0:
        logger.info('Removing {} not-requested columns'.format(len(col_not_req)))
        pdf.drop(col_not_req, axis=1, inplace=True)

    # kick out feats that are only Nan - isnt this what zero variance would do?
    sel = VarianceThreshold(threshold=0.01)
    col_b4 = pdf.columns
    # remove leading nan, otherwise transform fails
    pdf = pd.DataFrame(sel.fit_transform(pdf[params.rmLeadRows:]),
                       columns=col_b4[sel.get_support()],
                       index=pdf.index[params.rmLeadRows:])
    col_rm = [c for c in col_b4 if c not in pdf.columns]
    logger.info('{} Columns removed with Variance < 0.01: {}'.format(len(col_rm), col_rm))

# def reduce_non_required_features(pdf, params, df_talib):
#     # monkeypatch ts into frame. refactor at some point
#     if 'ts' in df_talib.columns:
#         required = required + ['ts']
#         df_full = pd.concat([df_full, df_talib['ts']], axis=1)
#
#     # FEATURE REDUCTION
#     # separate required from what can be reduced
#     if len(required) > 0:
#         req_df = df_full[required]
#         reduce_df = df_full.drop(required, axis=1)
#     else:
#         reduce_df = df_full
#         req_df = None
#     if len(reduce_df.columns) > 0 and s.params.use_xgbClassModel is False:
#         reduce_pdf = compress_data_w_non_required_features
#
#     # add required cols back in
#     if required and req_df:
#         df_full = pd.concat([req_df.iloc[s.params.rmLeadRows:, :],
#                              reduce_df], axis=1)
#         logger.info('Moved: {} back in'.format(len(req_df.columns)))
#     else:
#         df_full = reduce_df
#
#     # PCA - retain 99% of the variance
#     # df = filterBestFeatures(df)
#     return df_full


if __name__ == '__main__':
    import importlib

    params_ = importlib.import_module('{}.{}'.format(Paths.path_config_reinforced, 'ethusd')).Params()
    df_ohlc_test = get_ohlcv_mid_price(params_)
