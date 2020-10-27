from .enum_utils import EnumStr
from enum import Enum


class Exchanges(EnumStr, Enum):
    bitmex = 'bitmex'
    fxcm = 'fxcm'
    ib = 'ib'
