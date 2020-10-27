#from clr import AddReference
#AddReference("System")
#//from System import *
#from QuantConnect import *
import pickle, json, sys, os, getpass
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import time
import json

user = getpass.getuser()
if os.name == 'posix':
    qc_py_algo_path = '/home/{}/repos/quantconnect/Lean3/Algorithm.Python/seb2'.format(user)
else:
    qc_py_algo_path = r'C:\repos\quantconnect\Lean3\Algorithm.Python\seb2'
sys.path.append(qc_py_algo_path)
from utilFunc import *

class SlackHook(object):
    def __init__(s):
        s.args = {}
        s.live_mode = False

    def set_args(s, **kwargs):
        s.args = dict(zip(kwargs['keys'], kwargs['values']))
        for k in ['time', 'live', 'symbol']:
            s.args[k] = kwargs[k]

    def test_exec(s):
        return 'Exectued SlackHook Module'

    def build_msg(s, fsk):
        print(fsk)
        s.fsk = fsk.copy()
        if s.fsk[0] == 'long':
            s.fsk[2] = ":arrow_upper_right:"
        else:
            s.fsk[2] = ":arrow_lower_right:"

        return '''{{
            "channel": "{3}",
			 "username": "New Limit Order",
			 "text": "Sent {0} Limit Order for {12} {1} at {13}. #{4}",
			 "icon_emoji": "{2}",
            "attachments": [
                {{
                    "text": "Update:",
                    "fallback": "",
                    "callback_id": "",
                    "color": "#3AA3E3",
                    "attachment_type": "default",
                    "actions": [
                        {{
                            "name": "game",
                            "text": "Limit Price",
                            "type": "button",
                            "value": "chess"
                        }},
                        {{
                            "name": "game",
                            "text": "Quantity",
                            "type": "button",
                            "value": "maze"
                        }},
                        {{
                            "name": "game",
                            "text": "Cancel Order",
                            "type": "button",
                            "value": "cancel",
                            "url": "{14}"
                        }},
                        {{
                            "name": "game",
                            "text": "Auto-Trader<Status>",
                            "style": "<status-style>",
                            "type": "button",
                            "value": "war",
                            "confirm": {{
                                "title": "Are you sure?",
                                "text": "Auto-Trader will switched <on or off>?",
                                "ok_text": "Yes",
                                "dismiss_text": "No"
                            }}
                        }}
                    ]
                }}
            ]
        }}'''.format(*s.fsk)

slackHook = SlackHook()


def unit_test():
    fsk = [str(i) for i in range(15)]
    msg = slackHook.build_msg(fsk)
    print(msg)

if __name__ == '__main__':
    unit_test()