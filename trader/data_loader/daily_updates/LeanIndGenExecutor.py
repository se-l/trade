import sys
import os
import json
import datetime
import subprocess
from utils.utilFunc import date_day_range
import time
import math


class LeanIndGenExecutor:
    def __init__(s, sym, start_date, end_date, step_size, processes=None):
        s.sym = sym
        s.start_date = start_date
        s.end_date = end_date
        s.processes = processes
        if processes is not None:
            s.set_step_and_dates()
        else:
            s.step_size = step_size
        s.days_lookback = s.step_size + 1
        s.json_path_fn = r'C:\repos\trade\ind_gen_config.json'
        s.lean_exe_path_fn = r'C:\repos\trade\qc\daily_updates\batGenExec.bat'

    def run(s):
        s.save_json()
        s.start_lean()

    def set_step_and_dates(s):
        n_days = int(round((s.end_date - s.start_date).total_seconds() / 86400, 0))
        s.step_size = math.ceil(n_days / s.processes)
        s.start_date += datetime.timedelta(days=s.step_size)
        s.end_date += datetime.timedelta(days=s.step_size)

    def run_multiple(s):
        date_range = list(date_day_range(s.start_date, s.end_date))
        print(f'{len(date_range[::s.step_size])} processes:')
        print(date_range[::s.step_size])
        for date in date_range[::s.step_size]:
            s.save_json(str(date)[:10], str(date + datetime.timedelta(days=1))[:10])
            s.start_lean()
            time.sleep(60)

    def save_json(s, start_date=None, end_date=None):
        start_date = str(s.start_date)[:10] if start_date is None else start_date
        end_date = str(s.end_date)[:10] if start_date is None else end_date
        with open(s.json_path_fn, 'w') as f:
            json.dump(
                dict(symbol=s.sym, start_date=start_date, end_date=end_date, days_lookback=s.days_lookback), f
            )

    def start_lean(s):
        # print('Start Lean at {}'.format(time.time()))
        subprocess.Popen(s.lean_exe_path_fn, shell=True)


if __name__ == '__main__':
    leanIndGenExecutor = LeanIndGenExecutor(
        'ethusd',
        datetime.datetime(2019, 5, 19),  # date that should be written to from 0 second
        datetime.datetime(2019, 6, 1),   # date before OnData() starts. this date does not get started
        step_size=None,
        processes=2
    )
    leanIndGenExecutor.run_multiple()