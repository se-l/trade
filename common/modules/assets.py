from .enum_utils import EnumStr
from enum import Enum


class Assets(EnumStr, Enum):
    xbtusd = 'xbtusd'
    ethusd = 'ethusd'
    xrpz18 = 'xrpz18'
    xrpusd = 'xrpusd'
    xrpxbt = 'xrpxbt'

    gbpusd = 'gbpusd'
    eurusd = 'eurusd'
    usdjpy = 'usdjpy'
