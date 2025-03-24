from yacs.config import CfgNode


def cfg_node_to_dict(cfg_node):
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    return {k.lower(): cfg_node_to_dict(v) for k, v in cfg_node.items()}
