import logging

from tta.engine.runner import Runner
from tta.misc.registry import ADAPTATION_REGISTRY
from tta.config.tools import (
    parse_args,
    load_config,
    setup_logger,
    format_experiment_results,
)

logger = logging.getLogger(__name__)

adapt_registry = ADAPTATION_REGISTRY


def evaluate():
    # Load configs
    args = parse_args()
    config = load_config(args.config, args.opts)
    config_name = args.config.split("/")[-2] + "_" + args.config.split("/")[-1]
    logger = setup_logger(config_name, config)
    logger.info(f"Experiment config: {args.config}\n", config)

    # Load TTA method
    method = adapt_registry.get(config.MODEL.ADAPTATION)(config=config)

    # Run TTA engine
    runner = Runner(config=config, method=method)
    results = runner.run()
    formatted_results = format_experiment_results(results)
    print(formatted_results)
    logger.info("\n" + formatted_results)


if __name__ == "__main__":
    evaluate()
