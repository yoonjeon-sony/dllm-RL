import sys, os
# torchrun adds the script's directory (train_sft/train/) to sys.path[0],
# which shadows the 'train' package with train.py module. Fix by removing
# it and ensuring train_sft/ is at the front instead.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
_root = os.path.dirname(_script_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

from train.train import train
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256
# torch._dynamo.config.suppress_errors = True

if __name__ == "__main__":
    train()

    
