from common.modules import dotdict


class Strategy:

    def __init__(s, id, asset, direction, hp_opt=None):
        s.id = id
        s.asset = asset
        s.direction = direction
        if hp_opt is not None:
            s.set_p_opt(hp_opt)

    def set_p_opt(s, hp_opt):
        for k, v in hp_opt.items():
            if str(s.id)[0] == k[0]:
                k = s.rm_prefix(k)
            s.__setattr__(k, v)

    def rm_prefix(s, k):
        return '_'.join(
            k.split('_')[1:]
        )

    def set_trailing_stop_b(s):
        s.trailing_stop_b = s.trailing_stop_a / s.profit_target

    def add_opt_params(s, hp_opt):
        s.set_p_opt(hp_opt)


class StrategyLib(object):
    def __init__(s, strats: list):
        # assert len(strats) == len(opt_p), 'Need do define p-opt for each strat'
        # s.asset = asset
        # s.direction = direction
        s.lib = dotdict()
        s.store(strats)
        # s.p_opt_map = s.set_hp_opt(opt_p)
        # s.override_p_opt_labels()

    def store(s, strats: list):
        for strat in strats:
            s.lib[strat.id] = strat
        # return dotdict({i: (strats[i][0], strats[i][1]) for i in range(0, len(strats))})

    def deactivate_all_directions_but(s, direction):
        for strategy_id, strategy in s.lib.items():
            if strategy.direction != direction:
                strategy.p_opt.deactivate_entry()

    def set_p_opt_map(s):
        s.p_opt_map = {}
        for strategy_id, strategy in s.lib.items():
            s.p_opt_map[strategy_id] = strategy.p_opt

    # def set_hp_opt(s, opt_p):
    #     return {i: opt_p[i] for i in range(0, len(opt_p))}

    def override_p_opt_labels(s):
        for i, p_opt in s.p_opt_map.items():
            p_opt.prefix_hp_tup_label('{}_'.format(i))

    def merge_p_opts(s, max_evals):
        s.merged = {}
        degrees_of_freedom = []
        for d in s.p_opt_map.values():
            degrees_of_freedom.append(d.degrees_of_freedom(max_evals))
            s.merged = {**s.merged, **d.get_popt(max_evals)}
        total_degrees_of_freedom = 1
        for i in degrees_of_freedom:
            total_degrees_of_freedom *= i
        print('Total degrees of freedom: {} - by {}'.format(total_degrees_of_freedom, degrees_of_freedom))
        return s.merged

    def get_all_strategies(s):
        return s.lib.values()

    def get_strategy(s, id):
        return s.lib[id]
        # return Strategy(id, s.id_map.id[])

    def update_p_opt(s, p_opt):
        p_opt_by_id = dotdict()
        for key, val in p_opt.items():
            l = key.split('_')
            k = '_'.join(l[1:])
            id = int(l[0])
            if int(id) not in p_opt_by_id.keys():
                p_opt_by_id[id] = dotdict()
            p_opt_by_id[id][k] = val
        for id, p_opt in p_opt_by_id.items():
            s.lib[id].set_p_opt(p_opt)

    def set_trailing_stop_b(s):
        for i in s.lib.keys():
            s.lib[i].set_trailing_stop_b()
