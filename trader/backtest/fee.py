from common.modules import assets
from common.modules import order_type
from common.modules import exchanges


class Fee:

    @staticmethod
    def fee(exchange, order_type, asset):
        if exchange == exchanges.bitmex:
            return Fee.fee_bitmex(order_type, asset)
        elif exchange in [exchanges.fxcm, exchanges.ib]:
            return Fee.fee_ib(order_type, asset)
        else:
            raise ValueError('Unknown Broker')

    @staticmethod
    def fee_ib(order_type=None, asset=None):
        return 0.00002

    @staticmethod
    def fee_bitmex(order_type, asset):
        affiliate_scheme_factor = 0
        if asset in [assets.xbtusd, assets.ethusd]:
            if order_type == order_type.limit:
                return -0.00025 * (1 - affiliate_scheme_factor)
            elif order_type == order_type.market:
                return 0.00075 * (1 - 2 * affiliate_scheme_factor)
        elif asset in [assets.xrpusd, assets.xrpz18, assets.xrpxbt]:
            if order_type == order_type.limit:
                return -0.00050 * (1 - affiliate_scheme_factor)
            elif order_type == order_type.market:
                return 0.0025 * (1 - 2 * affiliate_scheme_factor)
