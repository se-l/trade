import talib
import math
import pandas as pd
import numpy as np
import re
import warnings

from talib import abstract
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
from common.modules.dotdict import Dotdict
from common.utils.util_func import shift5, join_struct_arrays, make_struct_nda, rolling_window, is_int
from trader.data_loader.config.talib_function_defaults import talib_out_2_results, talib_out_3_results, talib_d1, talib_d2, talib_around_x, talib_top_x_ind, talib_above_band, talib_cross_mean, \
    talib_feats_crossed, talib_selected, talib_excluded

warnings.filterwarnings('ignore')  # , r'This code may break in numpy 1.13 because this will return a view instead of a copy -- see release notes for details.')


# list of functions
# print(talib.get_functions())
# dict of functions by group
# print(talib.get_function_groups().keys())
# cats = ['Overlap Studies', 'Momentum Indicators', 'Volume Indicators', 'Cycle Indicators', 'Price Transform', 'Volatility Indicators', 'Pattern Recognition']


class ContinueI(Exception):
    pass


continue_i = ContinueI()


class EngineerFeatures:
    def __init__(self, m_params):
        self.featureLib = Dotdict([])
        self.talib_relev_func = sum(
            [talib.get_function_groups()[x] for x in ['Overlap Studies', 'Momentum Indicators', 'Volume Indicators',
                                                      'Cycle Indicators', 'Math Operators',
                                                      'Pattern Recognition', 'Volatility Indicators',
                                                      'Statistic Functions']], [])
        self.weightedSignals = Dotdict([])
        self.talibAllFunctions = talib.get_functions()
        self.talibOut1 = [x for x in self.talib_relev_func if x not in (talib_out_2_results + talib_out_3_results)]
        self.talibGenSingle = 'gen_talibs_fast'
        self.m_params = m_params
        self.exclude_in_gt_bin = ['LINEARREG_SLOPE', 'LINEARREG_ANGLE', 'STDDEV', 'MININDEX', 'MINMAX', 'MINMAXINDEX',
                                  'MAXINDEX']

    def build_library(self, params):
        df = pd.DataFrame(data=None, columns=['open', 'high', 'low', 'close'])
        self.reg_func_lib(col=df.columns.tolist(), inputs=None, function='base')
        self.get_potential_talibs(df)
        self.get_potential_d1(req_feat=talib_d1)
        self.get_potential_d2(req_feat=talib_d2)
        self.get_potential_top_x_bin(percthreshold=[10, 20, 30], look_back_period=params.lookbackWindow, req_feat=talib_top_x_ind)
        self.get_potential_bottom_x_bin(percthreshold=[10, 20, 30], look_back_period=params.lookbackWindow, req_feat=talib_top_x_ind)
        self.get_potential_around_x_bin(req_feat=talib_around_x, aroundx=[0], thresholdperc=[1, 5],
                                        look_back_period=params.lookbackWindow)
        self.get_potential_above_band_bin(req_feat=talib_above_band)
        self.get_potential_cross_mean_bin(req_feat=talib_cross_mean, look_back_period=params.lookbackWindow)
        self.get_potential_feats_crossed_bin(req_feat=talib_feats_crossed)

    def register_feats(s, req_feats):
        for f in req_feats:
            keys = f.split('_')
            feat_name = keys[0]
            tal_params = {}
            for i in range(1, len(keys)):
                if keys[i] == 'real':
                    continue
                elif 'sp' in keys[i]:
                    tal_params['slowperiod'] = int(keys[i][2:])
                elif 'fp' in keys[i]:
                    tal_params['fastperiod'] = int(keys[i][2:])
                elif 'tp1' in keys[i]:
                    tal_params['timeperiod1'] = int(keys[i][3:])
                elif 'tp2' in keys[i]:
                    tal_params['timeperiod2'] = int(keys[i][3:])
                elif 'tp3' in keys[i]:
                    tal_params['timeperiod3'] = int(keys[i][3:])
                elif is_int(keys[i]):
                    tal_params['timeperiod'] = int(keys[i])
                else:
                    pass

            s.reg_func_lib(f,
                           s.talibGenSingle,
                           s.talib_ordered_dict2list(getattr(abstract, feat_name)._Function__info['input_names'].values()),
                           tal_params=tal_params,
                           rootname=feat_name
                           )

    def get_potential_talibs(self, df, recomb_inp_params=False):
        no_vol = 'volume' not in df.columns

        # for f in self.talibAllFunctions:
        for f in talib_selected:
            # for f in self.talibRelevFunc:
            if no_vol:
                try:
                    for val in getattr(abstract, f)._Function__info['input_names'].values():
                        if type(val) == list:
                            if 'volume' in val:
                                raise continue_i
                        elif val == 'volume':
                            raise continue_i
                except ContinueI:
                    continue

            if f in talib_excluded:
                continue

            input_params = getattr(abstract, f)._Function__opt_inputs.keys()

            if 'timeperiod' in input_params:
                for tp in self.get_timeperiods(f):
                    for k in getattr(abstract, f).output_names:
                        self.reg_func_lib('{}_{}_{}'.format(f, k, tp),
                                          self.talibGenSingle,
                                          self.talib_ordered_dict2list(getattr(abstract, f)._Function__info['input_names'].values()),
                                          tal_params={'timeperiod': tp},
                                          rootname=f)

            if 'fastperiod' in input_params and 'slowperiod' in input_params:
                if recomb_inp_params:
                    fp, sp = self.get_fast_short_period(f)
                    for i in range(len(fp)):
                        for j in range(len(sp)):
                            for k in getattr(abstract, f).output_names:
                                self.reg_func_lib('{}_{}_fp{}_sp{}'.format(f, k, fp[i], sp[j]),
                                                  self.talibGenSingle,
                                                  self.talib_ordered_dict2list(
                                                      getattr(abstract, f)._Function__info['input_names'].values()),
                                                  tal_params={'fastperiod': fp[i],
                                                              'slowperiod': sp[j]},
                                                  rootname=f)
                else:
                    fp, sp = self.get_fast_short_period(f)
                    for i in range(len(fp)):
                        for k in getattr(abstract, f).output_names:
                            self.reg_func_lib('{}_{}_fp{}_sp{}'.format(f, k, fp[i], sp[i]),
                                              self.talibGenSingle,
                                              self.talib_ordered_dict2list(
                                                  getattr(abstract, f)._Function__info['input_names'].values()),
                                              tal_params={'fastperiod': fp[i],
                                                          'slowperiod': sp[i]},
                                              rootname=f)
            if 'timeperiod1' in input_params and 'timeperiod2' in input_params and 'timeperiod3' in input_params:
                if recomb_inp_params:
                    tp1, tp2, tp3 = self.get_timeperiod_123(f)
                    for i in range(len(tp1)):
                        for j in range(len(tp2)):
                            for m in range(len(tp3)):
                                for k in getattr(abstract, f).output_names:
                                    # next talib scan
                                    # self.reg_func_lib('{}_{}_tpa{}_tpb{}_tpc{}'.format(f, k, tp1[i], tp2[j], tp3[m]),
                                    self.reg_func_lib('{}_{}_tp1{}_tp2{}_tp3{}'.format(f, k, tp1[i], tp2[j], tp3[m]),
                                                      self.talibGenSingle,
                                                      self.talib_ordered_dict2list(
                                                          getattr(abstract, f)._Function__info['input_names'].values()),
                                                      tal_params={'timeperiod1': tp1[i],
                                                                  'timeperiod2': tp2[j],
                                                                  'timeperiod3': tp3[m]
                                                                  },
                                                      rootname=f)
                else:
                    tp1, tp2, tp3 = self.get_timeperiod_123(f)
                    for i in range(len(tp1)):
                        for k in getattr(abstract, f).output_names:
                            # next talib scan
                            # self.reg_func_lib('{}_{}_tpa{}_tpb{}_tpc{}'.format(f, k, tp1[i], tp2[i], tp3[i]),
                            self.reg_func_lib('{}_{}_tp1{}_tp2{}_tp3{}'.format(f, k, tp1[i], tp2[i], tp3[i]),
                                              self.talibGenSingle,
                                              self.talib_ordered_dict2list(
                                                  getattr(abstract, f)._Function__info['input_names'].values()),
                                              tal_params={'timeperiod1': tp1[i],
                                                          'timeperiod2': tp2[i],
                                                          'timeperiod3': tp3[i]
                                                          },
                                              rootname=f)
            if 'fastk_period' in input_params and 'fastd_period' in input_params:
                if recomb_inp_params:
                    fp, sp = self.getffkd(f)
                    for i in range(len(fp)):
                        for j in range(len(sp)):
                            for k in getattr(abstract, f).output_names:
                                self.reg_func_lib('{}_{}_fk{}_fd{}'.format(f, k, fp[i], sp[j]),
                                                  self.talibGenSingle,
                                                  self.talib_ordered_dict2list(
                                                      getattr(abstract, f)._Function__info['input_names'].values()),
                                                  tal_params={'fastk_period': fp[i],
                                                              'fastd_period': sp[j]},
                                                  rootname=f)
                else:
                    fp, sp = self.getffkd(f)
                    for i in range(len(fp)):
                        for k in getattr(abstract, f).output_names:
                            self.reg_func_lib('{}_{}_fk{}_fd{}'.format(f, k, fp[i], sp[i]),
                                              self.talibGenSingle,
                                              self.talib_ordered_dict2list(
                                                  getattr(abstract, f)._Function__info['input_names'].values()),
                                              tal_params={'fastk_period': fp[i],
                                                          'fastd_period': sp[i]},
                                              rootname=f)

            else:
                for k in getattr(abstract, f).output_names:
                    self.reg_func_lib('{}_{}'.format(f, k),
                                      self.talibGenSingle,
                                      self.talib_ordered_dict2list(getattr(abstract, f)._Function__info['input_names'].values()),
                                      tal_params=None,
                                      rootname=f)

    def get_timeperiods(self, f):
        if f == 'CCI':
            getattr(abstract, f).parameters = {'timeperiod': 14}
        default_tp = getattr(abstract, f).get_parameters()['timeperiod'] * self.m_params.defMultiplier
        tp = [default_tp]
        for i in range(1, self.m_params.time_period[0] + 1):
            if not self.m_params.extendOnly:
                val = int(math.ceil(default_tp / (1 + i * self.m_params.time_period[1])))
                if val not in tp:
                    tp.append(val)
            val = int(math.ceil(default_tp * (1 + i * self.m_params.time_period[1])))
            if val not in tp:
                tp.append(val)
        return tp

    def get_fast_short_period(self, f):
        default_fast_period = getattr(abstract, f).get_parameters()['fastperiod'] * self.m_params.defMultiplier
        default_short_period = getattr(abstract, f).get_parameters()['slowperiod'] * self.m_params.defMultiplier
        fp = [default_fast_period]
        sp = [default_short_period]
        for i in range(1, self.m_params.fastslow_period[0] + 1):
            if not self.m_params.extendOnly:
                fp1 = int(math.ceil(default_fast_period / (1 + i * self.m_params.fastslow_period[1])))
                sp1 = int(math.ceil(default_short_period / (1 + i * self.m_params.fastslow_period[1])))
                if len(fp) < 3 or (fp1 != fp[-2] or sp1 != sp[-2]):
                    fp.append(fp1)
                    sp.append(sp1)
            fp1 = int(math.ceil(default_fast_period * (1 + i * self.m_params.fastslow_period[1])))
            sp1 = int(math.ceil(default_short_period * (1 + i * self.m_params.fastslow_period[1])))
            if len(fp) < 3 or (fp1 != fp[-2] or sp1 != sp[-2]):
                fp.append(fp1)
                sp.append(sp1)
        return fp, sp

    def getffkd(self, f):
        default_fast_period = getattr(abstract, f).get_parameters()['fastk_period'] * self.m_params.defMultiplier
        default_short_period = getattr(abstract, f).get_parameters()['fastd_period'] * self.m_params.defMultiplier
        fp = [default_fast_period]
        sp = [default_short_period]
        for i in range(1, self.m_params.fastslow_period[0] + 1):
            if not self.m_params.extendOnly:
                fp1 = int(math.ceil(default_fast_period / (1 + i * self.m_params.fastslow_period[1])))
                sp1 = int(math.ceil(default_short_period / (1 + i * self.m_params.fastslow_period[1])))
                if len(fp) < 3 or (fp1 != fp[-2] or sp1 != sp[-2]):
                    fp.append(fp1)
                    sp.append(sp1)
            fp1 = int(math.ceil(default_fast_period * (1 + i * self.m_params.fastslow_period[1])))
            sp1 = int(math.ceil(default_short_period * (1 + i * self.m_params.fastslow_period[1])))
            if len(fp) < 3 or (fp1 != fp[-2] or sp1 != sp[-2]):
                fp.append(fp1)
                sp.append(sp1)
        return fp, sp

    def get_timeperiod_123(self, f):
        def1 = getattr(abstract, f).get_parameters()['timeperiod1'] * self.m_params.defMultiplier
        def2 = getattr(abstract, f).get_parameters()['timeperiod2'] * self.m_params.defMultiplier
        def3 = getattr(abstract, f).get_parameters()['timeperiod3'] * self.m_params.defMultiplier
        tp1 = [def1]
        tp2 = [def2]
        tp3 = [def3]
        for i in range(1, self.m_params.tp_123[0] + 1):
            if not self.m_params.extendOnly:
                t1 = int(math.ceil(def1 / (1 + i * self.m_params.tp_123[1])))
                t2 = int(math.ceil(def2 / (1 + i * self.m_params.tp_123[1])))
                t3 = int(math.ceil(def3 / (1 + i * self.m_params.tp_123[1])))
                if len(tp1) < 3 or (t1 != tp1[-2] or t2 != tp2[-2] or t3 != tp3[-2]):
                    tp1.append(t1)
                    tp2.append(t2)
                    tp3.append(t3)
            t1 = int(math.ceil(def1 * (1 + i * self.m_params.tp_123[1])))
            t2 = int(math.ceil(def2 * (1 + i * self.m_params.tp_123[1])))
            t3 = int(math.ceil(def3 * (1 + i * self.m_params.tp_123[1])))
            if len(tp1) < 3 or (t1 != tp1[-2] or t2 != tp2[-2] or t3 != tp3[-2]):
                tp1.append(t1)
                tp2.append(t2)
                tp3.append(t3)
        # print('{}---{}---{}'.format(tp1, tp2, tp3))
        return tp1, tp2, tp3

    def gen_talibs_fast(self, df, req_feat=None, fast=True):
        self.finalCols = []
        self.initTalibConcat = True
        for f in req_feat:
            if f in self.featureLib.keys():
                # Follow instructions in library
                f_orig = self.featureLib[f].rootname

                try:
                    # Only use the specified parameters, no param sweeping
                    # tp = self.featureLib[f].talParams['timeperiod']
                    # talParams = self.featureLib[f].talParams
                    # print('creating talib {} with: {}'.format(f_orig, self.featureLib[f].talParams))
                    self.get_talib_fast(df, f_orig,
                                        input_params=self.featureLib[f].talParams
                                        )
                except (KeyError, TypeError):
                    # check for default timeperiod params - should run as these are exptectd in library
                    # if 'timeperiod' in getattr(abstract, f)._Function__opt_inputs.keys():
                    #     for tp in self.get_timeperiods(f, params):
                    #         df = self.getTalib(df, f, dict(timeperiod=tp))
                    # else:
                    # print('creating talib: {}'.format(f))
                    print('Missing {} in feature library'.format(f))
                    self.get_talib_fast(df, f_orig)
            else:
                # Generate functions based on talib params
                if 'timeperiod' in getattr(abstract, f)._Function__opt_inputs.keys():
                    for tp in self.get_timeperiods(f):
                        self.get_talib_fast(df, f, dict(timeperiod=tp))
                else:
                    self.get_talib_fast(df, f)
        try:
            if isinstance(self.talibConcat, list):
                ix_return = [self.finalCols.index(c) for c in req_feat]
                return [self.finalCols[i] for i in ix_return], np.vstack(tuple([self.talibConcat[i] for i in ix_return])).transpose()
            else:
                return self.finalCols, self.talibConcat.transpose()
        except Exception as e:
            print(req_feat)
            print(e)

        # return [self.finalCols[i] for i in ix_return], self.talibConcat[ix_return].transpose()

    def get_talib_fast(self, df, f_orig, input_params=None):
        out = []
        if input_params is not None:
            pv = list(input_params.values())
            if 'timeperiod' in input_params:
                for k in getattr(abstract, f_orig).output_names:
                    out.append('{}_{}_{}'.format(f_orig, k, input_params['timeperiod']))

            if 'fastperiod' in input_params and 'slowperiod' in input_params:
                for k in getattr(abstract, f_orig).output_names:
                    out.append('{}_{}_fp{}_sp{}'.format(f_orig, k, input_params['fastperiod'], input_params['slowperiod']))

            if 'timeperiod1' in input_params and 'timeperiod2' in input_params and 'timeperiod3' in input_params:
                for k in getattr(abstract, f_orig).output_names:
                    out.append('{}_{}_tp1{}_tp2{}_tp3{}'.format(f_orig, k, input_params['timeperiod1'],
                                                                input_params['timeperiod2'], input_params['timeperiod3']))
            # for the next talib
            # if 'timeperiod1' in input_params and 'timeperiod2' in input_params and 'timeperiod3' in input_params:
            #     for k in getattr(abstract, f_orig).output_names:
            #         out.append('{}_{}_tpa{}_tpb{}_tpc{}'.format(f_orig, k, input_params['timeperiod1'],
            #                                                     input_params['timeperiod2'], input_params['timeperiod3']))

            if 'fastk_period' in input_params and 'fastd_period' in input_params:
                for k in getattr(abstract, f_orig).output_names:
                    out.append(
                        '{}_{}_fk{}_fd{}'.format(f_orig, k, input_params['fastk_period'], input_params['fastd_period']))

            if set(out).issubset(self.finalCols):
                return
            # checking for structured array
            if len(df.dtype) > 0:
                df_talib = getattr(abstract, f_orig)({'open': df['open'], 'high': df['high'], 'low': df['low'], 'close': df['close'], 'volume': df['volume']}, **input_params)
            else:  # assuming it's an ndarray, not recarray. used for normalizing
                df_talib = getattr(abstract, f_orig)(
                    {'open': df[:, 0], 'high': df[:, 1], 'low': df[:, 2], 'close': df[:, 3], 'volume': df[:, 4]}, **input_params)
        else:
            # reset active states from previous calls to this function. like input parameters!
            defpar = {}
            for k, v in getattr(abstract, f_orig)._Function__opt_inputs.items():
                defpar[k] = v['default_value']
            getattr(abstract, f_orig).parameters = defpar
            for k in getattr(abstract, f_orig).output_names:
                out.append('{}_{}'.format(f_orig, k))
            if set(out).issubset(self.finalCols):
                return
            if len(df.dtype) > 0:
                df_talib = getattr(abstract, f_orig)({'open': df['open'], 'high': df['high'], 'low': df['low'], 'close': df['close'], 'volume': df['volume']})
            else:  # assuming it's an ndarray, not recarray
                df_talib = getattr(abstract, f_orig)(
                    {'open': df[:, 0], 'high': df[:, 1], 'low': df[:, 2], 'close': df[:, 3], 'volume': df[:, 4]})

        self.finalCols += out
        if type(df_talib) != np.ndarray:
            if type(df_talib) == list:  # multiple outputs by talib function
                pass
            elif type(df_talib) == tuple:  # probably multiple pandas..not sure
                df_talib = np.asarray(df_talib)
            else:  # pandas frame
                df_talib = df_talib.values

        # merge new data with previous outputs
        if self.initTalibConcat:
            self.talibConcat = df_talib
            self.initTalibConcat = False
        # elif only_last_row and self.talibConcat.ndim > 1:
        #     # fill up df_talib up to talibConcats size. this is when Talibs have been calculated for smaller input sizes
        #     if (type(df_talib) == list and df_talib[0].shape[0] < self.talibConcat.shape[1]):
        #         df_talib = self.pad_arr_vstack(df_talib, length=self.talibConcat.shape[1], in_type=1)
        #     elif (type(df_talib) == np.ndarray and df_talib.shape[0] < self.talibConcat.shape[1]):
        #         df_talib = self.pad_arr_vstack(df_talib, length=self.talibConcat.shape[1], in_type=2)
        #     self.talibConcat = np.vstack((self.talibConcat, df_talib))
        else:
            self.talibConcat = np.vstack((self.talibConcat, df_talib))
        return

    @staticmethod
    def pad_arr_vstack(df_talib, length, in_type):
        if in_type == 3:
            # works only with 1 columns so far
            col = df_talib.dtype.names[0]
            t = df_talib[col]
            try:
                t.dtype = None
            except:
                pass
            t = np.hstack((
                np.full(length - df_talib[col].shape[0], np.nan),
                t
            ))
            return np.array(t, dtype=df_talib.dtype)
        elif in_type == 1:
            for i in range(len(df_talib)):
                df_talib[i] = np.hstack((
                    np.full(length - df_talib[i].shape[0], np.nan),
                    df_talib[i]
                ))
        elif in_type == 2:
            df_talib = np.hstack((
                np.full(length - df_talib.shape[0], np.nan),
                df_talib
            ))
        return df_talib

    @staticmethod
    def max_tp_in_strings(out):
        max_timeperiod = []
        for col in out:
            match_tp = re.findall(r'(?<=tp1)\d+|(?<=tp2)\d+|(?<=tp3)\d+', col)
            # match_tp = re.findall(r'(?<=tpa)\d+|(?<=tpb)\d+|(?<=tpc)\d+', col)
            if match_tp:
                max_timeperiod.append(max([int(v) for v in match_tp]))
            else:
                match_d = re.findall(r'\d+', col)
                if match_d:
                    max_timeperiod.append(max([int(v) for v in match_d]))
        return max(max_timeperiod)

    @staticmethod
    def talib_ordered_dict2list(inp):
        out = []
        for val in list(inp):
            if type(val) == list:
                out.extend(val)
            else:
                out.append(val)
        return out

    def get_potential_top_x_bin(self, percthreshold, look_back_period, req_feat):
        if type(req_feat) == str:
            req_feat = [req_feat]
        if type(percthreshold) == str:
            percthreshold = [percthreshold]
        for p in percthreshold:
            for f in req_feat:
                cols = [k for k, v in self.featureLib.items() if self.featureLib[k]['rootname'] == f]
                for c in cols:
                    self.reg_func_lib('{}-top{}pc'.format(c, p),
                                      'gen_top_x_bin',
                                      c,  # input function
                                      tal_params={'percthreshold': p,
                                                  'look_back_period': look_back_period},
                                      )
        return

    def gen_top_x_bin(self, df, req_feat, only_last_row=True):
        inp = [self.featureLib[req_feat]['inputs']]
        percthreshold = self.featureLib[req_feat]['talParams']['percthreshold']
        look_back_period = self.featureLib[req_feat]['talParams']['look_back_period']
        if only_last_row:
            # removing nans from input. otherwise min, max & range equal np.nan
            d = df[inp][np.isnan(df[inp].view(df[inp].dtype[0])) == False].view(df[inp].dtype[0])
            if d.__len__() == 0:
                return np.full((1,), np.nan, dtype=[(req_feat, np.int)])
            max_ = d.max()
            min_ = d.min()
            range_ = max_ - min_
            val_threshold = max_ - range_ * percthreshold / 100
        else:
            strides = rolling_window(df[inp].view(df[inp].dtype[0]), look_back_period)
            max_ = np.max(strides, axis=1)
            min_ = np.min(strides, axis=1)
            range_ = np.subtract(max_, min_)
            val_threshold = np.hstack((
                np.full(look_back_period - 1, np.nan),
                np.add(min_, (range_ * percthreshold / 100))
            ))

        if only_last_row:
            res = 1 if np.greater(df[inp][-1:], val_threshold) else 0
            df1 = np.full((1,), res, dtype=[(req_feat, np.int)])
            return self.pad_arr_vstack(df1, length=df.shape[0], in_type=3)
        elif type(df) == np.ndarray:
            df1 = np.where(np.greater(df[inp].view(df[inp].dtype[0]), val_threshold), 1, 0)
            df1.dtype = [(req_feat, df1.dtype.type)]
        elif type(df) in [pd.DataFrame, pd.Series]:
            df1 = np.where(df.loc[:, inp] > val_threshold, 1, 0)
            df1 = pd.DataFrame(df1, index=df.index, columns=[req_feat])
        return df1

    def get_potential_bottom_x_bin(self, percthreshold, look_back_period, req_feat):
        if type(req_feat) == str:
            req_feat = [req_feat]
        for p in percthreshold:
            for f in req_feat:
                cols = [k for k, v in self.featureLib.items() if self.featureLib[k]['rootname'] == f]
                for c in cols:
                    self.reg_func_lib('{}-bottom{}pc'.format(c, p),
                                      'gen_bottom_x_bin',
                                      c,
                                      tal_params={'percthreshold': p,
                                                  'look_back_period': look_back_period},
                                      )
        return

    def gen_bottom_x_bin(self, df, req_feat, only_last_row=True):
        inp = [self.featureLib[req_feat]['inputs']]
        # this wont work with a list of feats as imput. single ok
        percthreshold = self.featureLib[req_feat]['talParams']['percthreshold']
        look_back_period = self.featureLib[req_feat]['talParams']['look_back_period']
        # removing nans from input. otherwise min, max & range equal np.nan
        if only_last_row:
            d = df[inp][np.isnan(df[inp].view(df[inp].dtype[0])) == False].view(df[inp].dtype[0])
            if d.__len__() == 0:
                return np.full((1,), np.nan, dtype=[(req_feat, np.int)])
            max_ = d.max()
            min_ = d.min()
            range_ = max_ - min_
            val_threshold = min_ + range_ * percthreshold / 100
        else:
            strides = rolling_window(df[inp].view(df[inp].dtype[0]), look_back_period)
            max_ = np.max(strides, axis=1)
            min_ = np.min(strides, axis=1)
            range_ = np.subtract(max_, min_)
            val_threshold = np.hstack((
                np.full(look_back_period - 1, np.nan),
                np.add(min_, (range_ * percthreshold / 100))
            ))

        if only_last_row:
            res = 1 if np.less(df[inp][-1:], val_threshold) else 0
            df1 = np.full((1,), res, dtype=[(req_feat, np.int)])
            return self.pad_arr_vstack(df1, length=df.shape[0], in_type=3)
        elif type(df) == np.ndarray:
            df1 = np.where(np.less(df[inp].view(df[inp].dtype[0]), val_threshold), 1, 0)
            df1.dtype = [(req_feat, df1.dtype.type)]
            return df1
        elif type(df) in [pd.DataFrame, pd.Series]:
            df1 = np.where(df.loc[:, inp] < val_threshold, 1, 0)
            df1 = pd.DataFrame(df1, index=df.index, columns=[req_feat])
        return df1

    def get_potential_d1(self, req_feat):
        if type(req_feat) == str:
            req_feat = [req_feat]
        for f in req_feat:
            cols = [k for k, v in self.featureLib.items() if self.featureLib[k]['rootname'] == f]
            for c in cols:
                self.reg_func_lib('{}-D1'.format(c),  # lib key
                                  'generate_d1',  # gen function rootname
                                  c,  # inputs
                                  )
        return

    def generate_d1(self, df, req_feat, only_last_row=True):
        # if type(req_feat) == str:
        #     req_feat = [req_feat]
        inp = [self.featureLib[req_feat]['inputs']]
        if only_last_row:
            df1 = df[inp]
            df1.dtype = None
            res_old = df1[-2] - df1[-3]
            res_new = df1[-1] - df1[-2]
            df1 = np.array([res_old, res_new], dtype=[(req_feat, np.float)])
            return self.pad_arr_vstack(df1, length=df.shape[0], in_type=3)
        elif type(df) == np.ndarray:
            df1 = self.subtract_str_arr(df[inp], shift5(df[inp], 1))
            df1.dtype = [(req_feat, df1.dtype.type)]
        elif type(df) in [pd.DataFrame, pd.Series]:
            df1 = df.loc[:, inp] - df.loc[:, inp].shift(1)
            df1.columns = [req_feat]
        return df1

    def get_potential_d2(self, req_feat):
        if type(req_feat) == str:
            req_feat = [req_feat]
        for f in req_feat:
            cols = [k for k, v in self.featureLib.items() if self.featureLib[k]['rootname'] == f]
            for c in cols:
                self.reg_func_lib('{}-D2'.format(c),  # lib key
                                  'generate_d2',  # gen function name
                                  '{}-D1'.format(c),  # inputs
                                  )
        return

    def generate_d2(self, df, req_feat, only_last_row=True):
        # if type(req_feat) == str:
        #     req_feat = [req_feat]
        inp = [self.featureLib[req_feat]['inputs']]
        if type(inp[0]) == list:
            inp = inp[0]

        if only_last_row:
            df1 = df[inp]
            df1.dtype = None
            res = df1[-1] - df1[-2]
            df1 = np.full((1,), res, dtype=[(req_feat, np.float)])
            return self.pad_arr_vstack(df1, length=df.shape[0], in_type=3)
        elif type(df) == np.ndarray:
            df1 = self.subtract_str_arr(df[inp], shift5(df[inp], 1))
            df1.dtype = [(req_feat, df1.dtype.type)]
        elif type(df) in [pd.DataFrame, pd.Series]:
            df1 = df.loc[:, inp] - df.loc[:, inp].shift(1)
            df1.columns = [req_feat]

        return df1

    def get_potential_cross_x_bin(self, req_feat, threshold=0):
        if type(req_feat) == str:
            req_feat = [req_feat]
        for c in req_feat:
            self.reg_func_lib('{}-cross{}'.format(c, threshold),
                              'gen_cross_x_bin',
                              c,
                              tal_params={'threshold': threshold},
                              rootname=c)
        return

    def gen_cross_x_bin(self, df, req_feat, only_last_row=True):
        inp = [self.featureLib[req_feat]['inputs']]
        threshold = self.featureLib[req_feat]['talParams']['threshold']

        if only_last_row:
            if ((np.less(df[inp][-1], threshold) and np.greater(df[inp][-2], threshold)) or \
                    (np.less(df[inp][-1], threshold) and np.greater(df[inp][-2], threshold))):
                res = 1
            else:
                res = 0
            df1 = np.full((1,), res, dtype=[(req_feat, np.int)])
            return df1[-1:]
        elif type(df) == np.ndarray:
            df1 = np.where(
                np.less(df[inp], threshold) and np.greater(shift5(df[inp], 1), threshold), 1,
                np.where(np.less(df[inp], threshold) and np.greater(shift5(df[inp], 1), threshold), 1,
                         0))
            df1.dtype = [(req_feat, df1.dtype.type)]
        elif type(df) in [pd.DataFrame, pd.Series]:
            df1 = np.where(
                np.less(df.loc[:, inp], threshold) and np.greater(df.loc[:, inp].shift(1), threshold), 1,
                np.where((df.loc[:, inp] < threshold) and (df.loc[:, inp].shift(1) > threshold), 1,
                         0))
            df1 = pd.DataFrame(df1, index=df.index, columns=[req_feat])
        return df1

    def get_potential_cross_mean_bin(self, req_feat, look_back_period):
        if type(req_feat) == str:
            req_feat = [req_feat]
        for f in req_feat:
            cols = [k for k, v in self.featureLib.items() if self.featureLib[k]['rootname'] == f]
            for c in cols:
                self.reg_func_lib('{}-crossMean'.format(c),
                                  'genCrossMeanBin',
                                  c,  # inputs
                                  tal_params={'look_back_period': look_back_period},
                                  )
        return

    def genCrossMeanBin(self, df, req_feat, only_last_row=True):
        inp = [self.featureLib[req_feat]['inputs']]
        look_back_period = self.featureLib[req_feat]['talParams']['look_back_period']
        ## removing nans from input. otherwise min, max & range equal np.nan
        if only_last_row:
            mean = np.mean(
                df[inp][np.isnan(df[inp].view(df[inp].dtype[0])) == False].view(df[inp].dtype[0])
            )
        else:
            mean = np.hstack((
                np.full(look_back_period - 1, np.nan),
                np.mean(rolling_window(df[inp].view(df[inp].dtype[0]), look_back_period), axis=1)
            ))

        if only_last_row:
            if ((np.less(df[inp][-1], mean) and np.greater(df[inp][-2], mean)) or \
                    (np.less(df[inp][-1], mean) and np.greater(df[inp][-2], mean))):
                res = 1
            else:
                res = 0
            df1 = np.full((1,), res, dtype=[(req_feat, np.int)])
            return df1[-1:]
        else:
            df1 = np.where(np.less(df[inp].view(df[inp].dtype[0]), mean) & np.greater(shift5(df[inp], 1).view(df[inp].dtype[0]), mean), 1,
                           np.where(np.greater(df[inp].view(df[inp].dtype[0]), mean) & np.less(shift5(df[inp], 1).view(df[inp].dtype[0]), mean), 1,
                                    0))
            df1.dtype = [(req_feat, df1.dtype.type)]
        return df1

    def get_potential_around_x_bin(self, req_feat, aroundx=[0], thresholdperc=[5], look_back_period=150):
        if type(req_feat) == str:
            req_feat = [req_feat]
        if type(thresholdperc) == str:
            thresholdperc = [thresholdperc]
        for p in thresholdperc:
            for ar in aroundx:
                for f in req_feat:
                    cols = [k for k, v in self.featureLib.items() if self.featureLib[k]['rootname'] == f]
                    for c in cols:
                        self.reg_func_lib('{}-around{}-{}pc'.format(c, ar, p),
                                          'gen_around_x_bin',
                                          c,
                                          tal_params={'aroundx': ar,
                                                      'thresholdperc': p,
                                                      'look_back_period': look_back_period},
                                          rootname=c)
        return

    def gen_around_x_bin(self, df, req_feat, only_last_row=True):
        inp = [self.featureLib[req_feat]['inputs']]
        aroundx = self.featureLib[req_feat]['talParams']['aroundx']
        thresholdperc = self.featureLib[req_feat]['talParams']['thresholdperc']
        look_back_period = self.featureLib[req_feat]['talParams']['look_back_period']
        # removing nans from input. otherwise min, max & range equal np.nan

        if only_last_row:
            d = df[inp][np.isnan(df[inp].view(df[inp].dtype[0])) == False].view(df[inp].dtype[0])
            if d.__len__() == 0:
                return np.full((1,), np.nan, dtype=[(req_feat, np.int)])
            step = (d.max() - d.min()) * (thresholdperc / 100)
            upper = np.hstack((
                np.full(look_back_period - 1, np.nan),
                aroundx + step))
            lower = np.hstack((
                np.full(look_back_period - 1, np.nan),
                aroundx - step))
        else:
            strides = rolling_window(df[inp].view(df[inp].dtype[0]), look_back_period)
            max_ = np.max(strides, axis=1)
            min_ = np.min(strides, axis=1)
            step = (max_ - min_) * (thresholdperc / 100)
            upper = np.hstack((
                np.full(look_back_period - 1, np.nan),
                aroundx + step))
            lower = np.hstack((
                np.full(look_back_period - 1, np.nan),
                aroundx - step))

        if only_last_row:
            res = 1 if np.less(df[inp][-1], upper) and np.greater(df[inp][-1], lower) else 0
            df1 = np.full((1,), res, dtype=[(req_feat, np.int)])
            return df1[-1:]
        else:
            df1 = np.where(np.less(df[inp].view(df[inp].dtype[0]), upper) & np.greater(df[inp].view(df[inp].dtype[0]), lower), 1, 0)
            df1.dtype = [(req_feat, df1.dtype.type)]
        return df1

    def get_potential_above_band_bin(self, req_feat):
        if type(req_feat) == str:
            req_feat = [req_feat]
        for f in req_feat:
            band = getattr(abstract, f).output_names
            # tupels of line and band, but only where params are the same
            # str search the function?
            bands = [k for k, v in self.featureLib.items() if self.featureLib[k]['rootname'] == f]

            middle = []
            upper = []
            lower = []
            for c in bands:
                if re.search('middleband', c):
                    middle.append(c)
                elif re.search('upperband', c):
                    upper.append(c)
                elif re.search('lowerband', c):
                    lower.append(c)
            match = []
            for c in middle:
                for b in upper:
                    if self.featureLib[c]['talParams'] == self.featureLib[b]['talParams']:
                        match.append((c, b))
            for c in middle:
                for b in lower:
                    if self.featureLib[c]['talParams'] == self.featureLib[b]['talParams']:
                        match.append((c, b))
            for m in match:
                self.reg_func_lib('{}-above-{}-Ind'.format(m[0], m[1]),
                                  'gen_above_band_bin',
                                  m,  # inputs
                                  tal_params=None, )
        return

    def gen_above_band_bin(self, df, req_feat, only_last_row=True):
        # if type(req_feat) == str:
        #     req_feat = [req_feat]
        tup = self.featureLib[req_feat]['inputs']
        # band = [self.featureLib[req_feat]['talParams']['band']]
        if only_last_row:
            res = np.where(np.greater(df[tup[0]][-1:], df[tup[1]][-1:]), 1, 0)
            df1 = np.full((1,), res, dtype=[(req_feat, np.int)])
            return df1[-1:]
        elif type(df) == np.ndarray:
            df1 = np.where(np.greater(df[tup[0]], df[tup[1]]), 1, 0)
            df1.dtype = [(req_feat, df1.dtype.type)]
        elif type(df) in [pd.DataFrame, pd.Series]:
            df1 = np.where(df.loc[:, tup[0]] > df.loc[:, tup[1]], 1, 0)
            df1 = pd.DataFrame(df1, index=df.index, columns=[req_feat])
        return df1

    def get_potential_feats_crossed_bin(self, req_feat, recomb_inp_params=False):
        if type(req_feat) == str:
            req_feat = [req_feat]
        for f in req_feat:
            cols = [k for k, v in self.featureLib.items() if self.featureLib[k]['rootname'] == f]
            tupels = list(combinations(cols, 2))
            for tup in tupels:
                self.reg_func_lib('{}-crossed-{}-Bin'.format(tup[0], tup[1]),
                                  'gen_feats_crossed_bin',
                                  tup,  # inputs
                                  tal_params=None, )
                if recomb_inp_params:
                    self.reg_func_lib('{}-crossed-{}-Bin'.format(tup[1], tup[0]),
                                      'gen_feats_crossed_bin',
                                      tup,  # inputs
                                      tal_params=None, )
        return

    def gen_feats_crossed_bin(self, df, req_feat, only_last_row=True):
        tup = self.featureLib[req_feat]['inputs']
        if only_last_row:
            res = np.where(np.less(df[tup[0]][-1], df[tup[1]][-2]), 1,
                           np.where(np.less(df[tup[1]][-1], df[tup[0]][-2]), 1,
                                    0))
            df1 = np.full((1,), res, dtype=[(req_feat, np.int)])
            return df1[-1:]
        elif type(df) == np.ndarray:
            df1 = np.where(np.less(df[tup[0]], shift5(df[tup[1]], 1)), 1,
                           np.where(df[tup[1]] < shift5(df[tup[0]], 1), 1,
                                    0))
            df1.dtype = [(req_feat, df1.dtype.type)]
        elif type(df) in [pd.DataFrame, pd.Series]:
            df1 = np.where(df.loc[:, tup[0]] < (df.loc[:, tup[1]].shift(1)), 1,
                           np.where(df.loc[:, tup[1]] < (df.loc[:, tup[0]].shift(1)), 1,
                                    0))
            df1 = pd.DataFrame(df1, index=df.index, columns=[req_feat])
        return df1

    def get_potential_subtract_close(self, df, lessCloseF=0.8):
        sel = []
        for col in df.columns:
            # print(col)
            if df.loc[:, col].all().mean() > df.loc[:, 'close'].all().mean() * lessCloseF:
                sel.append(col)
                self.reg_func_lib('{}-close'.format(col), 'gen_subtract_close', [col, 'close'])
        return

    def gen_subtract_close(self, df, req_feat, only_last_row=True):
        # print('Creating Subtract close feats: {}'.format(req_feat))
        if isinstance(req_feat, str):
            col_tuple = [tuple(self.featureLib[req_feat].inputs)]
            col_names = [req_feat]
        elif isinstance(req_feat, list):
            col_tuple = [tuple(self.featureLib[f].inputs) for f in req_feat]
            col_names = req_feat
        if col_tuple:
            df1 = pd.concat([(df[tup[0]] - df[tup[1]]) for tup in col_tuple], axis=1, ignore_index=True)
            df1.columns = col_names
        else:
            return df
        return df1

    def get_potential_subtractions(self, df, cols):
        sel = []
        for col in cols:
            if df[col].max() > self.m_params.maxV:
                sel.append(col)
        col_tuple = combinations(sel, 2)
        for tup in col_tuple:
            self.reg_func_lib('{}-sub-{}'.format(tup[0], tup[1]),
                              'gen_subtractions',
                              tup)
        return

    def gen_subtractions(self, df, req_feat, only_last_row=True):
        if isinstance(req_feat, str):
            col_tuple = [tuple(self.featureLib[req_feat].inputs)]
        elif isinstance(req_feat, list):
            col_tuple = [tuple(self.featureLib[f].inputs) for f in req_feat]
        df1 = pd.concat([(df[tup[0]] - df[tup[-1]]) for tup in col_tuple], axis=1, ignore_index=True)
        # df1 = pd.DataFrame(df1)
        # df2 = pd.concat([(df[tup[0]] - df[tup[-1]]) for tup in combos2], axis=1, ignore_index=True)
        # df2 = pd.DataFrame(df2)
        # df2.columns = combos2
        return df1

    def get_potential_gt_bin(self, req_feat):
        if isinstance(req_feat, str):
            req_feat = [req_feat]
        req_feat = [c for c in req_feat if c not in self.exclude_in_gt_bin]
        col_tuple = list(combinations(req_feat, 2))
        for tup in col_tuple:
            self.reg_func_lib('{}-gt-{}'.format(tup[0], tup[1]),
                              'gen_gt_bin',
                              tup)
        return

    def gen_gt_bin(self, df, req_feat, only_last_row=True):
        if isinstance(req_feat, str):
            col_tuple = [tuple(self.featureLib[req_feat].inputs)]
            req_feat = [req_feat]
        elif isinstance(req_feat, list):
            col_tuple = [tuple(self.featureLib[col].inputs) for col in req_feat]
            col_names = req_feat
        df1 = np.vstack([
            np.where(df[tup[0]] > df[tup[1]], 1, np.where(df[tup[0]] < df[tup[1]], -1, 0)) for tup in col_tuple
        ]).transpose()
        df1 = pd.DataFrame(df1, index=df.index, columns=req_feat)
        return df1

    @staticmethod
    def get_potential_poly_interactions(df):
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
        col_names = poly.get_feature_names(df.columns)
        col_names = [str(x).replace(' ', '-') for x in col_names]
        return poly, col_names

    @staticmethod
    def gen_poly_interactions(df, poly, col_names, only_last_row=True):
        # need to refines this here as well. not selecting features currently. useless
        d = poly.fit_transform(df)
        return d

    def get_required_dependency_feats(self, available_feats, dep_feats):
        dep_feats = self.input_list2unique_list(dep_feats)
        return [col for col in dep_feats if col not in available_feats]

    @staticmethod
    def input_list2unique_list(inp):
        out = []
        for el in inp:
            if el is None:
                continue
            elif type(el) == str:
                out.append(el)
            else:
                out.extend(el)
        return list(set(out))

    def gen_requested_feats_fast(self, nd, colnames=None, only_last_row=True, ohlc_normalized=True):
        # colnames = ['WILLR_real_19-bottom10pc']
        # list all functions required to generate each requested feature
        if isinstance(colnames, str):
            colnames = [colnames]
        available_feats = list(nd.dtype.names)
        gen_feat = [c for c in colnames if c not in available_feats]
        available_feats.append('periods')
        dependency_feats_a = [self.featureLib[col].inputs for col in gen_feat]
        dependency_feats_a = self.get_required_dependency_feats(available_feats, dependency_feats_a)
        available_feats = available_feats + dependency_feats_a
        if dependency_feats_a:
            dependency_feats_b = [self.featureLib[col].inputs for col in dependency_feats_a if col not in available_feats]
            dependency_feats_b = self.get_required_dependency_feats(available_feats, dependency_feats_b)
        else:
            dependency_feats_b = []
        available_feats = available_feats + dependency_feats_b
        if dependency_feats_b:
            dependency_feats_c = [self.featureLib[col].inputs for col in dependency_feats_b if col not in available_feats]
            dependency_feats_c = self.get_required_dependency_feats(available_feats, dependency_feats_c)
            available_feats = available_feats + dependency_feats_c
        else:
            dependency_feats_c = []
        if dependency_feats_c:
            dependency_feats_d = [self.featureLib[col].inputs for col in dependency_feats_c if col not in available_feats]
            dependency_feats_d = self.get_required_dependency_feats(available_feats, dependency_feats_d)
        else:
            dependency_feats_d = []

        total_required = list(np.unique(gen_feat + dependency_feats_a + dependency_feats_b + dependency_feats_c + dependency_feats_d))
        req_talibs = [col for col in total_required if self.featureLib[col].function == self.talibGenSingle and col not in nd.dtype.names]
        if req_talibs:
            if ohlc_normalized:
                gen_col, nd_talib = self.gen_talibs_fast(nd, req_feat=req_talibs)
                nd_talib = np.reshape(nd_talib, (-1, 1)) if nd_talib.ndim == 1 else nd_talib
                req_t_ix = [gen_col.index(v) for v in req_talibs]
                gen_col = list(np.array(gen_col)[req_t_ix])
                nd_talib = nd_talib[:, req_t_ix]
                nd_talib = make_struct_nda(nd_talib, cols=gen_col)
                nd = join_struct_arrays([nd, nd_talib])
                available_feats = available_feats + gen_col
            else:
                print('Warning: the requested feats require additional Talibs, but the data is normalized already')
                raise ValueError

        need_a = [col for col in dependency_feats_a if col not in nd.dtype.names]
        if len(need_a) > 0:
            nd = join_struct_arrays([nd] + [getattr(EngineerFeatures, self.featureLib[col].function)(self, nd, col, only_last_row=only_last_row) for col in need_a])
            # print('need_a generated: {}'.format(len(need_a)))
        need_b = [col for col in dependency_feats_b if col not in nd.dtype.names]
        if len(need_b) > 0:
            nd = join_struct_arrays(
                [nd] + [getattr(EngineerFeatures, self.featureLib[col].function)(self, nd, col, only_last_row=only_last_row) for col in
                        need_b])
            print('need_b generated: {}'.format(len(need_b)))
        need_c = [col for col in dependency_feats_c if col not in nd.dtype.names]
        if len(need_c) > 0:
            nd = join_struct_arrays(
                [nd] +
                [getattr(EngineerFeatures, self.featureLib[col].function)(self, nd, col, only_last_row=only_last_row) for col in need_c])
            print('need_c generated: {}'.format(len(need_c)))
        need_d = [col for col in dependency_feats_d if col not in nd.dtype.names]
        if need_d:
            nd = join_struct_arrays(
                [nd] +
                [getattr(EngineerFeatures, self.featureLib[col].function)(self, nd, col, only_last_row=only_last_row) for col in need_d])
            print('need_d generated: {}'.format(len(need_d)))
        rest_feats = [col for col in total_required if col not in nd.dtype.names]
        # print('Generating remaining features: {}...'.format(len(rest_feats)))

        nd = join_struct_arrays(
            [nd] +
            [getattr(EngineerFeatures, self.featureLib[col].function)(self, nd, col, only_last_row=only_last_row) for col in rest_feats],
            def_type=np.float)
        # print('generate features b4 drop: {}'.format(len(df.columns)))

        # extra = [col for col in total_required if col not in colnames]
        # df.drop(extra, axis=1, inplace=True)
        return nd

    def reg_func_lib(self, col, function, inputs, tal_params=None, rootname=None, n_out=None):
        if isinstance(col, list):
            for key in col:
                self.featureLib[key] = Dotdict([
                    ('function', function),
                    ('inputs', inputs),
                    ('talParams', tal_params),
                    ('rootname', rootname),
                    ('nOut', n_out),
                ])
        else:
            self.featureLib[col] = Dotdict([
                ('function', function),
                ('inputs', inputs),
                ('talParams', tal_params),
                ('rootname', rootname),
                ('nOut', n_out),
            ])

    @staticmethod
    def get_max(d, inp):
        if isinstance(d, np.ndarray):
            max_ = np.max(d[inp].view(d[inp].dtype[0]))
        elif isinstance(d, (pd.DataFrame, pd.Series)):
            max_ = d.loc[:, inp].max()
        else:
            raise TypeError
        return max_

    @staticmethod
    def get_min(d, inp):
        if isinstance(d, np.ndarray):
            min_ = np.min(d[inp].view(d[inp].dtype[0]))
        elif isinstance(d, (pd.DataFrame, pd.Series)):
            min_ = d.loc[:, inp].min()
        else:
            raise TypeError('')
        return min_

    @staticmethod
    def subtract_str_arr(a, b):
        a.dtype = None
        b.dtype = None
        return a - b


def binning(df, cnt_v, bin_down_f):
    for col in df.columns:
        # nun = df[col].nunique()
        if df[col].nunique() > cnt_v:
            print(col)
            df[col] = pd.cut(df[col], bin_down_f)
    return df


def poly_feat_eng(df):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
    d = poly.fit_transform(df)
    feature_names = poly.get_feature_names(df.columns)
    feature_names = [str(x).replace(' ', '-') for x in feature_names]
    return d, feature_names
