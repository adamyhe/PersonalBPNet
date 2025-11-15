# Adapted from bpnetlite
# Copied here because we need these newer losses but the newer bpnetlite
# versions are not super stable. Will delete when bpnetlite is updated.

import torch
from bpnetlite.losses import MNLLLoss, log1pMSELoss


def _mixture_loss(y, y_hat_logits, y_hat_logcounts, count_loss_weight, labels=None):
    """A function that takes in predictions and truth and returns the loss.

    This function takes in the observed integer read counts, the predicted logits,
    and the predicted logcounts, and returns the total loss. Importantly, this
    calculates a single multinomial over all strands in the tracks and a single
    count loss across all tracks.

    The logits do not have to be normalized.


    Parameters
    ----------
    y: torch.Tensor, shape=(n,
    """

    y_hat_logits = y_hat_logits.reshape(y_hat_logits.shape[0], -1)
    y_hat_logits = torch.nn.functional.log_softmax(y_hat_logits, dim=-1)

    y = y.reshape(y.shape[0], -1)
    y_ = y.sum(dim=-1).reshape(y.shape[0], 1)

    # Calculate the profile and count losses
    if labels is not None:
        profile_loss = MNLLLoss(y_hat_logits[labels == 1], y[labels == 1]).mean()
    else:
        profile_loss = MNLLLoss(y_hat_logits, y).mean()

    count_loss = log1pMSELoss(y_hat_logcounts, y_).mean()

    # Extract the profile loss for logging
    profile_loss_ = profile_loss.item()
    count_loss_ = count_loss.item()

    # Mix losses together
    loss = profile_loss + count_loss_weight * count_loss

    return profile_loss_, count_loss_, loss
