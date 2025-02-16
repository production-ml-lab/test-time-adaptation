import logging

import method
from config.utils import parse_args, load_config
from misc.registry import ADAPTATION_REGISTRY

logger = logging.getLogger(__name__)

def evaluate():
    # Load configs
    args = parse_args()
    config = load_config(args.config, args.opts)
    logger.info(f"Experiment config: {args.config}\n", config)

    # Load TTA method
    model = adaptations.get(config.MODEL.ADAPTATION)(config=config)
    logger.info(model)
    

if __name__ == '__main__':
    adaptations = ADAPTATION_REGISTRY
    logger.info(adaptations)

    evaluate()