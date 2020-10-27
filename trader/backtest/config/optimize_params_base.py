from common.utils.util_func import Dotdict
from common.modules import timing


class HpTup:
    def __init__(s, label, def_val, hp_function, hp_range, use_def_val):
        s.label = label
        s.def_val = def_val
        s.hp_function = hp_function
        s.hp_range = hp_range
        s.use_def_val = use_def_val

    def get_tup(s, override_use_def=False):
        if s.use_def_val or override_use_def:
            return (s.label, s.def_val)
        elif not s.use_def_val:
            return (s.label, s.hp_function(s.label, s.hp_range))
        else:
            raise ('Unhandled scenario - fix!')


class OptParamsBase:
    def deactivate_entry(s):
        for item in s.p_opt:
            if item.label == 'preds_net_thresh':
                item.def_val *= 1000

    def hyperopt_aspect_only(s, timing):
        if timing == timing.entry:
            for item in s.p_opt:
                if item.label in [
                    'preds_net_thresh',
                    'bull_bear_stretch',
                    # 'd_net_cancel_entry'
                ]:
                    item.use_def_val = False
                else:
                    item.use_def_val = True
        elif timing == timing.exit:
            for item in s.p_opt:
                if item.label in ['profit_target',
                                  'trailing_stop_a',
                                  'max_trailing_stop_a',
                                  'trail_profit_stop',
                                  'preds_net_exit',
                                  'veto_stop_ema_d300',
                                  # 'twin_peak_trailing_profit'
                                  ]:
                    item.use_def_val = False
                else:
                    item.use_def_val = True

    def get_popt(s, max_evals):
        return s.select_hp_space(max_evals)

    def select_hp_space(s, max_evals):
        if max_evals == 1:
            return Dotdict(
                [item.get_tup(override_use_def=True) for item in s.p_opt]
            )
        else:
            return Dotdict(
                [item.get_tup() for item in s.p_opt]
            )

    def degrees_of_freedom(s, max_evals):
        if max_evals < 2:
            return 1
        else:
            p = 1
            for i in [len(list(hpTup.hp_range)) for hpTup in s.p_opt if hpTup.use_def_val is False]:
                p *= i
            return p

    def select_hp_space_deprecated(s, max_evals):
        ret = {}
        if max_evals == 1:
            for k, v in s.p_opt.items():
                # if type(v2) == list and len(v2) > 1:
                ret[k] = v[0]

        elif max_evals > 1:
            for k, v in s.p_opt.items():
                if type(v) == list and len(v) > 1:
                    ret[k] = v[1]
                elif type(v) == list and len(v) == 1:
                    ret[k] = v[0]
                else:
                    ret[k] = v
        return ret

    def prefix_hp_tup_label(s, prefix):
        for item in s.p_opt:
            item.label = prefix + item.label
