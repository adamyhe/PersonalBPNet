# losses.py
# Authors: adamyhe <adamyhe@gmail.com>

"""
This module contains the losses used by BNBPNet for training.
"""

import torch
from bpnetlite.losses import MNLLLoss, log1pMSELoss


def _mixture_loss_masked(
    y, y_hat_logits, y_hat_logcounts, count_loss_weight, labels=None, mask=None
):
    """A modified version of the _mixture_loss function from bpnetlite to allow
    for masking out specified positions from the loss calculation.

    Parameters
    ----------
    y: torch.Tensor, shape=(n, ..., L)
            A tensor with `n` examples and `L` possible categories.
            True labels.

    y_hat_logits: torch.Tensor, shape=(n, ..., L)
            A tensor with `n` examples and `L` possible categories.
            Predicted logits from the profile head.

    y_hat_logcounts: torch.Tensor, shape=(n, ...)
            A tensor with `n` examples and `L` possible categories.
            Predicted log counts from the count head.

    count_loss_weight: float
            The weight of the count loss component of the mixture loss.

    labels: torch.Tensor, shape=(n, ..., L)
            A tensor with `n` examples and `L` possible categories.

    mask: torch.Tensor, shape=(n, L)
            A boolean tensor with `n` examples and `L` possible categories.
            Positions where `mask[i, j] == True` will be masked out.

    Returns
    -------
    profile_loss: torch.Tensor, shape=(n, ...)
            The profile loss component of the mixture loss.

    count_loss: torch.Tensor, shape=(n, ...)
            The count loss component of the mixture loss.

    loss: torch.Tensor, shape=(n, ...)
            The mixture loss.
    """

    y_hat_logits = y_hat_logits.reshape(y_hat_logits.shape[0], -1)
    y_hat_logits = torch.nn.functional.log_softmax(y_hat_logits, dim=-1)

    y = y.reshape(y.shape[0], -1)
    y_ = y.sum(dim=-1).reshape(y.shape[0], 1)

    # Calculate the profile and count losses
    if labels is not None:
        profile_loss = MNLLLoss_masked(
            y_hat_logits[labels == 1], y[labels == 1], mask
        ).mean()
    else:
        profile_loss = MNLLLoss_masked(y_hat_logits, y, mask).mean()

    count_loss = log1pMSELoss(y_hat_logcounts, y_).mean()

    # Mix losses together
    loss = profile_loss + count_loss_weight * count_loss

    return profile_loss, count_loss, loss


def MNLLLoss_masked(logps, true_counts, mask=None):
    """A wrapper function for MNLLLoss that allows for masking out specified
    positions from the loss calculation.

    Parameters
    ----------
    logps: torch.Tensor, shape=(n, ..., L)
            A tensor with `n` examples and `L` possible categories.

    true_counts: torch.Tensor, shape=(n, ..., L)
            A tensor with `n` examples and `L` possible categories.

    mask: torch.Tensor, shape=(n, L)
            A boolean tensor with `n` examples and `L` possible categories.
            Positions where `mask[i, j] == True` will be masked out.

    Returns
    -------
    loss: torch.Tensor, shape=(n, ...)
            The multinomial log likelihood loss of the true counts given the
            predicted probabilities
    """
    if mask is not None:
        loss = 0
        for logps_i, true_counts_i, mask_i in zip(logps, true_counts, mask):
            logps_i = torch.masked_select(logps_i, mask_i)[None, ...]
            true_counts_i = torch.masked_select(true_counts_i, mask_i)[None, ...]
            # Ensure correct shape, since MNLLLoss expects N, ..., L
            if logps_i.shape < logps.shape:
                logps_i = logps_i.unsqueeze(dim=-1)
                true_counts_i = true_counts_i.unsqueeze(dim=-1)
            loss += MNLLLoss(logps_i, true_counts_i).mean()
        return loss / len(logps)
    else:
        return MNLLLoss(logps, true_counts)
