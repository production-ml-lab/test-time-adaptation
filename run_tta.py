import logging

from tta.engine.runner import Runner
from tta.misc.registry import ADAPTATION_REGISTRY
from tta.misc.utils import cfg_node_to_dict
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
    print("Experiment config:\n", config)

    # Load TTA method
    method_name = config.METHOD.NAME
    method_kwargs = cfg_node_to_dict(config.METHOD)
    model_name = config.MODEL.NAME
    model_backend = config.MODEL.BACKEND
    model_pretrain = config.MODEL.PRETRAIN
    method = adapt_registry.get(method_name)(
        model_name=model_name,
        model_backend=model_backend,
        model_pretrain=model_pretrain,
        **method_kwargs,
    )

    # # Run TTA engine
    data_kwargs = cfg_node_to_dict(config.DATA)
    runner = Runner(
        method=method,
        **data_kwargs,
    )
    results = runner.run()
    formatted_results = format_experiment_results(results)
    print(formatted_results)
    logger.info("\n" + formatted_results)


if __name__ == "__main__":
    evaluate()
