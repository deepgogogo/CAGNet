from data.bit import build_bit
from data.tvhi import build_tvhi

def build_dataset(cfg):
    if cfg.dataset_name=='bit':return build_bit(cfg)
    if cfg.dataset_name == 'tvhi': return build_tvhi(cfg)
    raise NotImplementedError
