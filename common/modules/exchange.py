from .enum_utils import EnumStr
from enum import Enum


class Exchange(EnumStr, Enum):
    bitmex = 'bitmex'
    bitfinex = 'bitfinex'
    fxcm = 'fxcm'
    ib = 'ib'
