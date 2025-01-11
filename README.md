# PersonalBPNet

A small modification to bpnetlite's BPNet to accomodate large validation datasets.

Redid the validation loop to work with a PyTorch DataLoader (e.g., one generated by [GenVarLoader](https://genvarloader.readthedocs.io/en/latest/)), rather than having to load the whole validation set into memory at once. Also, the model checkpoints save the optimizer state dict, epoch number, and number of steps since last improvement in addition to the model state dict, so that training can be resumed from a checkpoint w/ the correct optimizer and early stopping/epoch states.

Additionally, we include a Pytorch implementation of CLIPNET, which is essentially BPNet with added batch norm layers, similar to what was done with the original CLIPNET implementation in tensorflow.

## Installation

Clone and install github repo:

```sh
git clone git@github.com:adamyhe/PersonalBPNet.git
cd PersonalBPNet
pip install -e . # for editable mode.
```

Then the `PersonalBPNet` and `CLIPNET` classes can be directly imported:

```python
from personal_bpnet import PersonalBPNet, CLIPNET
```

This package is currently in active dev and may change drastically. Models have not been extensively benchmarked yet. May be lots of typos/copy paste errors. A personalized ChromBPNet fitting method has not been included, as I personally have not had success training such models.

## Note about lazy layers

Because I was lazy, I decided to use lazy layers in the CLIPNET pytorch implementation. These are in early development for pytorch, so their APIs/functionality may be unstable. CLIPNET was developed with torch v2.3.1, so if those layers are misbehaving consider downgrading your torch version.

## Command line interface

For convenience, prediction and attribution (DeepLIFT/SHAP) methods can be accessed via a CLI:

```bash
clipnet predict -h
clipnet attribute -h
```
