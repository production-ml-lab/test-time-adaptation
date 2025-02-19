import math
from yacs.config import CfgNode

from tta.method import BaseMethod
from tta.misc.registry import DATASET_REGISTRY
from tta.utils.data import build_test_loader

dataset_registry = DATASET_REGISTRY

class Runner():
    def __init__(self, config: CfgNode, method: BaseMethod):
        self.config = config
        self.method = method

        self.shift_type = config.SHIFT.TYPE
        self.shift_serverity = config.SHIFT.SEVERITY

        self.results = {}

    
    def run(self):
        for shift_name in self.shift_type:
            for severity_level in self.shift_serverity:
                dataset = dataset_registry.get(self.config.DATA.NAME)(
                    corrupt_domain_orders = [shift_name],
                    severity = severity_level,
                    num_samples = self.config.DATA.BATCH_SIZE
                )
                test_loader = build_test_loader(dataset, batch_size=self.config.DATA.BATCH_SIZE)

                preds = []
                gts = []

                for x, y, _ in test_loader:
                    self.method.forward_and_adapt(x)
                    y_pred = self.method.predict(x)

                    preds.extend(y_pred.tolist())
                    gts.extend(y.tolist())

                acc, err = self.get_accuracy(preds, gts)
                self.results[f'{shift_name}_{severity_level}'] = math.floor(err * 1000) / 1000
        
        return self.results

    
    def get_accuracy(self, preds, gts):
        assert len(preds) == len(gts)
        num_total = len(preds)
        num_correct = sum([1 if y_pred == y else 0 for y_pred, y in zip(preds, gts)])
        
        acc = num_correct / num_total
        err = 1 - acc

        return acc, err


