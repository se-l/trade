import collections
import datetime
from ibapi.contract import Contract

from trader.raw_data_fetcher.interactive_brokers.Program import printWhenExecuting, TestWrapper, TestClient, TestApp, BarData
from trader.raw_data_fetcher.interactive_brokers.ContractSamples import ContractSamples
from threading import Thread
from collections import deque
import datetime
import pickle


class IbDataFetcher(TestApp):

    @printWhenExecuting
    def historicalDataOperations_req(self):
        # Requesting historical data
        # ! [reqHeadTimeStamp]
        # self.reqHeadTimeStamp(4101, ContractSamples.USStockAtSmart(), "TRADES", 0, 1)
        # ! [reqHeadTimeStamp]

        # ! [reqhistoricaldata]
        queryTime = (datetime.datetime.today() - datetime.timedelta(days=100)).strftime("%Y%m%d %H:%M:%S")
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.currency = "USD"
        contract.exchange = "IDEALPRO"
        self.reqHistoricalData(4102, contract, queryTime,
                               "1 M", "1 min", "BID", 0, 1, False, [])

        # self.reqHistoricalData(4103, ContractSamples.EuropeanStock(), queryTime,
        #                        "10 D", "1 min", "TRADES", 1, 1, False, [])
        # self.reqHistoricalData(4104, ContractSamples.EurGbpFx(), "",
        #                        "1 M", "1 day", "MIDPOINT", 1, 1, True, [])
        # ! [reqhistoricaldata]

    # ! [historicaldata]
    def historicalData(self, reqId: int, bar: BarData):
        self.my_historic_data.append(bar)
        print("HistoricalData. ReqId:", reqId, "BarData.", bar)
    # ! [historicaldata]

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        super().historicalDataEnd(reqId, start, end)
        print("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)
        with open(r'/trader/data_loader/raw_data_fetcher/interactive_brokers\fx_data.pickle', 'wb') as f:
            pickle.dump(list(self.my_historic_data), f)
        self.disconnect()


def main():

    try:
        app = IbDataFetcher()
        # ! [connect]
        app.connect("127.0.0.1", 7497, clientId=0)
        # ! [connect]
        print("serverVersion:%s connectionTime:%s" % (app.serverVersion(),
                                                      app.twsConnectionTime()))

        # ! [clientrun]
        # app.globalCancelOnly = True
        app.run()
        # ! [clientrun]
    except:
        raise
    finally:
        pass


if __name__ == "__main__":
    main()
