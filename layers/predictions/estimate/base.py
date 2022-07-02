import pickle
import os

from common.interfaces.iestimate import IEstimate
from common.paths import Paths
from connector.influxdb.influxdb_wrapper import influx


class EstimateBase(IEstimate):

    def save(self):
        """
        Models
        Feat importance !!!!!!!!!!!!!!!!!
        Influx
        """
        try:
            os.mkdir(os.path.join(Paths.trade_model, self.ex))
        except FileExistsError:
            pass
        with open(os.path.join(Paths.trade_model, self.ex, 'boosters.p'), 'wb') as f:
            pickle.dump(self.boosters, f)
        with open(os.path.join(Paths.trade_model, self.ex, 'pred_label_val.p'), 'wb') as f:
            pickle.dump(self.pred_label_val, f)
        with open(os.path.join(Paths.trade_model, self.ex, 'pred_label_ho.p'), 'wb') as f:
            pickle.dump(self.pred_label_ho, f)
        with open(os.path.join(Paths.trade_model, self.ex, 'label_val.p'), 'wb') as f:
            pickle.dump(self.ps_label, f)
        with open(os.path.join(Paths.trade_model, self.ex, 'label_ho.p'), 'wb') as f:
            pickle.dump(self.ps_label_ho, f)

    def to_influx(self):
        assert len(self.preds_val.index.unique()) == len(self.preds_val), 'Timestamp is not unique. Group By time first before uploading to influx.'
        self.preds_val = self.preds_val.rename(columns={0: 'predictions'})
        influx.write(
            record=self.preds_val,
            data_frame_measurement_name='predictions',
            data_frame_tag_columns={**{
                'exchange': self.exchange.name,
                'asset': self.sym.name,
                'information': 'CV',
                'ex': self.ex
            }, **self.tags},
        )
        self.preds_ho = self.preds_ho.rename(columns={0: 'predictions'})
        influx.write(
            record=self.preds_ho,
            data_frame_measurement_name='predictions',
            data_frame_tag_columns={**{
                'exchange': self.exchange.name,
                'asset': self.sym.name,
                'information': 'HO',
                'ex': self.ex
            }, **self.tags},
        )
