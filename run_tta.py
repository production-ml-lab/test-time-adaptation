import logging

from tta.engine.runner import Runner
from tta.misc.registry import ADAPTATION_REGISTRY
from tta.config.utils import parse_args, load_config

logger = logging.getLogger(__name__)

adapt_registry = ADAPTATION_REGISTRY

def evaluate():
    # Load configs
    args = parse_args()
    config = load_config(args.config, args.opts)
    logger.info(f"Experiment config: {args.config}\n", config)

    # Load TTA method
    method = adapt_registry.get(config.MODEL.ADAPTATION)(config=config)    
    logger.info(method)

    # Run TTA engine
    runner = Runner(config=config, method=method)
    results = runner.run()
    print(results)


if __name__ == "__main__":
    evaluate()
