# PersonalBPNet and CLIPNET

See [the README](../README.md#install) for installation.

## PersonalBPNet

`PersonalBPNet` is identical to the `BPNet` class from [bpnetlite](https://github.com/jmschrei/bpnet-lite) (same architecture, same `forward`), but its `fit` method has been modified to accept a PyTorch `DataLoader` for validation data, rather than fixed tensors. This significantly improves the memory footprint of the validation step. Do note that random data augmentations (jittering, reverse complement) should be turned off to return a fixed validation dataset.

```python
from personal_bpnet import PersonalBPNet

model = PersonalBPNet(n_filters=64, n_layers=8, n_outputs=1, n_control_tracks=0)
model.fit(
    training_data,   # DataLoader yielding (X, y, labels) or (X, X_ctl, y, labels)
    optimizer,
    valid_data=valid_data,  # DataLoader, not fixed tensors
    max_epochs=100,
    device="cuda",
)
```

Because its architecture is unmodified from `bpnetlite.bpnet.BPNet`, `PersonalBPNet` weights can be loaded directly into a standard `BPNet` instance (and vice versa), as long as the same init arguments are used.

## CLIPNET

`CLIPNET` extends `PersonalBPNet` to include batch normalization layers after each convolutional and linear layer, which we've found to improve prediction accuracy. It's a subclass of `PersonalBPNet`, so it inherits the same `fit()` method.

```python
from personal_bpnet import CLIPNET

model = CLIPNET(n_filters=512, n_outputs=2, n_control_tracks=0)
```

### Pretrained weights

We've deposited pre-trained CLIPNET PyTorch weights on [Zenodo](https://zenodo.org/records/15258030). These were trained using the same multi-individual LCL PRO-cap dataset as the original CLIPNET models (training data also available on Zenodo). We've only saved the model weights, so you'll need to initialize the models, then use `load_state_dict`:

```python
import os

import torch

from personal_bpnet import CLIPNET

os.system("wget https://zenodo.org/records/14632152/files/lcl_procap_models.tar --quiet")
os.system("tar -xvf lcl_procap_models.tar")

model = CLIPNET(
    n_filters=512, n_outputs=2, n_control_tracks=0, n_layers=8, trimming=(2114 - 1000) // 2
)
model.load_state_dict(torch.load("lcl_procap_models/f1.torch"))  # 9 model replicates.
```

**IMPORTANT:** The pretrained CLIPNET PyTorch models have been trained on half two-hot encoded sequences. That is, homozygous positions are represented with one-hot encodings of the 4 nucleotides and the heterozygous positions are represented as `[0.5, 0.5, 0, 0], [0.5, 0, 0.5, 0], ...`. See `personal_bpnet.utils.twohot_encode`.

See also: [porting the original TensorFlow CLIPNET weights](clipnet-tf.md), [the `clipnet`/`pausenet` CLIs](cli.md).
