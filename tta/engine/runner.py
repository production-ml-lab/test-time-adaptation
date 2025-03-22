import math
from yacs.config import CfgNode

from tta.method import BaseMethod
from tta.misc.registry import DATASET_REGISTRY
from tta.data import build_test_loader
from tta.utils.metrics import get_accuracy


dataset_registry = DATASET_REGISTRY


class Runner:
    def __init__(self, config: CfgNode, method: BaseMethod):
        self.config = config
        self.method = method

        self.shift_type = config.SHIFT.TYPE
        self.shift_severity = config.SHIFT.SEVERITY

        self.results = {}

    def run(self):
        num_exp = len(self.shift_type) * len(self.shift_severity)
        cnt = 0
        for shift_name in self.shift_type:
            for severity_level in self.shift_severity:
                dataset_name = self.config.DATA.NAME
                batch_size = self.config.DATA.BATCH_SIZE

                dataset = dataset_registry.get(dataset_name)(
                    corrupt_domain_orders=[shift_name],
                    severity=severity_level,
                )
                test_loader = build_test_loader(dataset, batch_size=batch_size)

                preds = []
                gts = []

                for x, y, _ in test_loader:
                    self.method.forward_and_adapt(x)
                    y_pred = self.method.predict(x)

                    preds.extend(y_pred.tolist())
                    gts.extend(y.tolist())

                acc, err = get_accuracy(preds, gts)
                self.results[f"{shift_name}_{severity_level}"] = (
                    math.floor(err * 1000) / 1000
                )

                self.method.reset()

                cnt += 1
                print(
                    f"[{cnt}/{num_exp}] TTA Evaluation is done on {shift_name}_{severity_level}"
                )

        return self.results
