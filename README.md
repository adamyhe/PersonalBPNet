# PersonalBPNet

A small modification to bpnetlite's BPNet to accomodate large validation datasets.

Redid the validation loop to work with a PyTorch DataLoader, rather than having to load the whole validation set into memory at once. Also, the model checkpoints save the optimizer state dict and epoch number in addition to the model state dict, so that training can be resumed from a checkpoint.

Additionally, we include a Pytorch implementation of CLIPNET, which is essentially BPNet with added batch norm and maxpool layers, similar to what was done with the original CLIPNET implementation in tensorflow.

## Installation

Clone and install github repo:

```
git clone git@github.com:adamyhe/PersonalBPNet.git
cd PersonalBPNet
pip install -e . # for editable mode.
```

Then the `PersonalBPNet` and `CLIPNET` classes can be directly imported:

```
from personal_bpnet import PersonalBPNet, CLIPNET
```

This package is currently in active dev and may change drastically. Models have not been extensively benchmarked yet.
