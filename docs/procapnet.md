# ProCapNet

See [installation](installation.md) first.

`ProCapNet` is a subclass of `bpnetlite.bpnet.BPNet` that implements the masked profile loss from the [ProCapNet paper](https://www.biorxiv.org/content/10.1101/2024.05.28.596138v2). The masked loss allows specific positions (e.g., those overlapping peaks from other assays) to be excluded from the MNLL profile loss, which improves model attributions.

```python
from personal_bpnet import ProCapNet

model = ProCapNet(n_filters=512, n_outputs=2, n_control_tracks=0, count_loss_weight=100)
model.fit(
    training_data,   # DataLoader yielding (X, y, labels) or (X, X_ctl, y, labels)
    optimizer,
    X_valid=X_valid,
    y_valid=y_valid,
    y_has_mask=True,
    max_epochs=50,
    device="cuda",
)
```

When `y_has_mask=True` (the default), training batches must provide `y` with shape `(N, n_outputs+1, L)` where the final channel is a boolean mask — positions where the mask is `True` are excluded from the profile loss. `y_valid` should have shape `(N, n_outputs, L)` with no mask channel.

Because the forward pass is identical to `bpnetlite.bpnet.BPNet`, `ProCapNet` weights can be loaded directly into a standard `BPNet` instance (and vice versa), as long as the same init arguments are used.

A verbatim copy of the original ProCapNet implementation is included at `src/personal_bpnet/procapnet_orig.py` for full reproducibility of the original project's results.
