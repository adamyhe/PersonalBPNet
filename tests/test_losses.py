import torch
from bpnetlite.losses import MNLLLoss, _mixture_loss

from personal_bpnet.losses import MNLLLoss_masked, _mixture_loss_masked


def _random_profile_batch(batch=3, strands=2, length=20, seed=0):
    generator = torch.Generator().manual_seed(seed)
    y = torch.rand(batch, strands, length, generator=generator) * 10
    y_hat_logits = torch.randn(batch, strands, length, generator=generator)
    y_hat_logcounts = torch.randn(batch, 1, generator=generator)
    return y, y_hat_logits, y_hat_logcounts


class TestMixtureLossMasked:
    def test_delegates_to_unmasked_mixture_loss_when_mask_is_none(self):
        y, y_hat_logits, y_hat_logcounts = _random_profile_batch()

        expected = _mixture_loss(y, y_hat_logits, y_hat_logcounts, 100)
        actual = _mixture_loss_masked(y, y_hat_logits, y_hat_logcounts, 100)

        for expected_component, actual_component in zip(expected, actual):
            assert torch.allclose(expected_component, actual_component)

    def test_mask_of_all_true_matches_unmasked_loss(self):
        y, y_hat_logits, y_hat_logcounts = _random_profile_batch()
        mask = torch.ones(y.shape[0], y.shape[-1], dtype=torch.bool)

        expected = _mixture_loss(y, y_hat_logits, y_hat_logcounts, 100)
        actual = _mixture_loss_masked(y, y_hat_logits, y_hat_logcounts, 100, mask=mask)

        for expected_component, actual_component in zip(expected, actual):
            assert torch.allclose(expected_component, actual_component, atol=1e-5)

    def test_masking_changes_the_loss(self):
        y, y_hat_logits, y_hat_logcounts = _random_profile_batch()
        mask = torch.ones(y.shape[0], y.shape[-1], dtype=torch.bool)
        mask[:, : y.shape[-1] // 2] = False

        full_loss = _mixture_loss_masked(y, y_hat_logits, y_hat_logcounts, 100)
        partial_loss = _mixture_loss_masked(
            y, y_hat_logits, y_hat_logcounts, 100, mask=mask
        )

        assert not torch.allclose(full_loss[0], partial_loss[0])

    def test_labels_filter_examples_before_masking(self):
        y, y_hat_logits, y_hat_logcounts = _random_profile_batch(batch=4)
        mask = torch.ones(y.shape[0], y.shape[-1], dtype=torch.bool)
        labels = torch.tensor([1, 0, 1, 0])

        profile_loss, count_loss, loss = _mixture_loss_masked(
            y, y_hat_logits, y_hat_logcounts, 100, labels=labels, mask=mask
        )

        assert profile_loss.ndim == 0
        assert loss.ndim == 0


class TestMNLLLossMasked:
    def test_all_true_mask_matches_plain_mnll_loss(self):
        # Matches bpnetlite's _mixture_loss convention: a single multinomial over
        # all strands and positions flattened together, not a per-strand one.
        batch, strands, length = 3, 2, 10
        logits = torch.randn(batch, strands, length)
        true_counts = torch.rand(batch, strands, length) * 5
        mask = torch.ones(batch, length, dtype=torch.bool)

        logps = torch.nn.functional.log_softmax(logits.reshape(batch, -1), dim=-1)
        expected = MNLLLoss(logps, true_counts.reshape(batch, -1)).mean()

        actual = MNLLLoss_masked(logits, true_counts, mask)
        assert torch.allclose(actual, expected, atol=1e-5)
