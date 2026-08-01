# PauseNet

> This model is in active development, no published weights yet, and has not been extensively benchmarked.

See [installation](installation.md) first.

`PauseNet` is a transfer-learning wrapper (a subclass of `bpnetlite.bpnet.CountWrapper`) around `bpnetlite.bpnet.BPNet`, `PersonalBPNet`, or `CLIPNET` models that transforms them to predict a single scalar output per input sequence. This is designed for fine-tuning the base-resolution models to predict regulatory phenotypes that can only be represented as a single scalar value per region (e.g., pausing index, for which this class is named).

`PauseNet` replaces the base model's counts-head `linear` layer (and its batch normalization layer, if present) with freshly initialized ones, which remain trainable regardless of `base_trainable`. By default, the rest of the network is also trainable; set `base_trainable=False` to fine-tune only the new head.

```python
from personal_bpnet import CLIPNET, PauseNet
import torch

# This is for loading from a weights dictionary.
# If you saved the full model, just directly use pretrain=torch.load("weights.torch")
pretrain = CLIPNET(**init_args)
pretrain.load_state_dict(torch.load("weights.torch"))

model = PauseNet(pretrain)
model.fit(**params)
```

The new head's input width defaults to the base model's existing `linear.in_features`, so it doesn't need to be passed explicitly unless you want to override it.

A personalized ChromBPNet fitting method has not been included, as we have not yet had success training such models.
