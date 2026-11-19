import os

import torch

from cli._common import average_profile_and_counts, resolve_model_paths


class TestResolveModelPaths:
    def test_single_file_passthrough(self):
        assert resolve_model_paths("model.torch") == ["model.torch"]

    def test_directory_expands_to_replicate_paths(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        paths = resolve_model_paths(str(model_dir), pattern="f{i}.torch", n=9)
        assert paths == [
            os.path.join(str(model_dir), f"f{i}.torch") for i in range(1, 10)
        ]

    def test_custom_pattern_and_count(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        paths = resolve_model_paths(str(model_dir), pattern="fold_{i}.h5", n=3)
        assert paths == [
            os.path.join(str(model_dir), f"fold_{i}.h5") for i in range(1, 4)
        ]


class TestAverageProfileAndCounts:
    def test_matches_manual_average_before_nonlinearity(self):
        torch.manual_seed(0)
        predictions = [(torch.randn(2, 2, 5), torch.randn(2, 1)) for _ in range(3)]

        actual = average_profile_and_counts(predictions)

        profiles, counts = zip(*predictions)
        mean_profile = torch.stack([p.reshape(2, -1) for p in profiles]).mean(dim=0)
        mean_count = torch.stack(list(counts)).mean(dim=0)
        expected = (
            torch.nn.functional.softmax(mean_profile, dim=-1)
            * (torch.exp(mean_count) - 1)
        ).reshape(2, 2, 5)

        assert torch.allclose(actual, expected)

    def test_differs_from_averaging_after_softmax(self):
        # Regression check: ensembling must average logits/log-counts *before* the
        # softmax/exp nonlinearity, not average already-rescaled per-replicate
        # tracks (which biases the ensemble toward uniformity).
        torch.manual_seed(1)
        predictions = [(torch.randn(2, 2, 5), torch.randn(2, 1)) for _ in range(3)]

        correct = average_profile_and_counts(predictions)

        post_hoc_tracks = [
            torch.nn.functional.softmax(p.reshape(2, -1), dim=-1) * (torch.exp(c) - 1)
            for p, c in predictions
        ]
        incorrect = torch.stack(post_hoc_tracks).mean(dim=0).reshape(2, 2, 5)

        assert not torch.allclose(correct, incorrect)

    def test_single_replicate_reduces_to_direct_computation(self):
        torch.manual_seed(2)
        profile = torch.randn(2, 2, 5)
        count = torch.randn(2, 1)

        actual = average_profile_and_counts([(profile, count)])
        expected = (
            torch.nn.functional.softmax(profile.reshape(2, -1), dim=-1)
            * (torch.exp(count) - 1)
        ).reshape(2, 2, 5)

        assert torch.allclose(actual, expected)


def test_clipnet_tf_batch_size_defaults(monkeypatch):
    """`clipnet_tf attribute` leaves -bs to the fast engine's planner (None),
    `ism` defaults to 1024 edits a step, and the other subcommands keep the
    shared default of 16."""

    import argparse
    import sys

    import cli.tf as tf

    seen = {}
    parse_args = argparse.ArgumentParser.parse_args

    def spy(self, *args, **kwargs):
        ns = parse_args(self, *args, **kwargs)
        seen["batch_size"] = ns.batch_size
        seen["tf32"] = ns.tf32
        raise SystemExit  # stop before any model or data is loaded

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
    required = ["-f", "a.fa", "-b", "a.bed", "-o", "o.npz", "-m", "m.h5"]
    for argv, expected in [
        (["attribute"], None),
        (["predict"], 16),
        (["predict_tss"], 16),
        (["attribute", "-bs", "4"], 4),
        (["ism"], 1024),
    ]:
        monkeypatch.setattr(sys, "argv", ["clipnet_tf", *argv, *required])
        try:
            tf.cli()
        except SystemExit:
            pass
        assert seen["batch_size"] == expected, argv
        assert seen["tf32"] is False, argv
    monkeypatch.setattr(sys, "argv", ["clipnet_tf", "ism", "--tf32", *required])
    try:
        tf.cli()
    except SystemExit:
        pass
    assert seen["tf32"] is True
