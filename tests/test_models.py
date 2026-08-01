import torch
from bpnetlite.bpnet import BPNet, CountWrapper

from personal_bpnet import CLIPNET, PauseNet, PersonalBPNet, ProCapNet


def _random_input(batch=2, length=200):
    return torch.randn(batch, 4, length)


class TestPersonalBPNet:
    def test_forward_shapes(self):
        model = PersonalBPNet(n_filters=8, n_layers=2, trimming=4)
        profile, counts = model(_random_input())
        assert profile.shape == (2, 1, 192)
        assert counts.shape == (2, 1)

    def test_state_dict_matches_plain_bpnet(self):
        # Architecture (and therefore state_dict keys/shapes) must stay identical
        # to bpnetlite.bpnet.BPNet so weights are interchangeable in both directions.
        personal = PersonalBPNet(n_filters=8, n_layers=2, trimming=4)
        bpnet = BPNet(
            n_filters=8, n_layers=2, n_outputs=1, n_control_tracks=0, trimming=4
        )
        assert set(personal.state_dict().keys()) == set(bpnet.state_dict().keys())
        for key, value in bpnet.state_dict().items():
            assert value.shape == personal.state_dict()[key].shape

    def test_default_trimming(self):
        model = PersonalBPNet(n_layers=3)
        assert model.trimming == 2**3


class TestCLIPNET:
    def test_forward_shapes(self):
        model = CLIPNET(n_filters=8, n_layers=2, trimming=4)
        profile, counts = model(_random_input())
        assert profile.shape == (2, 2, 192)
        assert counts.shape == (2, 1)

    def test_state_dict_superset_of_bpnet_with_batchnorm_layers(self):
        clipnet = CLIPNET(n_filters=8, n_layers=2, trimming=4)
        bpnet = BPNet(
            n_filters=8, n_layers=2, n_outputs=2, n_control_tracks=0, trimming=4
        )
        bpnet_keys = set(bpnet.state_dict().keys())
        clipnet_keys = set(clipnet.state_dict().keys())

        assert bpnet_keys <= clipnet_keys
        bn_keys = clipnet_keys - bpnet_keys
        for prefix in ("ibn", "rbn.0", "rbn.1", "pbn", "cbn"):
            assert any(key.startswith(prefix) for key in bn_keys)

    def test_default_config(self):
        model = CLIPNET()
        assert model.n_filters == 512
        assert model.n_layers == 8
        assert model.n_outputs == 2
        assert model.n_control_tracks == 0
        assert model.trimming == (2114 - 1000) // 2


class TestPauseNet:
    def test_replaces_base_model_counts_head(self):
        base = CLIPNET(n_filters=8, n_layers=2, trimming=4)
        model = PauseNet(base)
        assert model.model.linear.in_features == 8
        output = model(_random_input())
        assert output.shape == (2, 1)

    def test_infers_n_filters_from_base_model(self):
        # A base model with control tracks widens the linear layer's input by 1;
        # PauseNet's new head must match that width automatically.
        base = CLIPNET(n_filters=8, n_layers=2, trimming=4, n_control_tracks=2)
        model = PauseNet(base)
        assert model.model.linear.in_features == base.linear.in_features

    def test_base_trainable_false_freezes_base_but_not_new_head(self):
        base = CLIPNET(n_filters=8, n_layers=2, trimming=4)
        model = PauseNet(base, base_trainable=False)

        for name, param in model.named_parameters():
            if "linear" in name or "cbn" in name:
                assert param.requires_grad
            else:
                assert not param.requires_grad

    def test_works_without_batchnorm_in_base_model(self):
        base = PersonalBPNet(n_filters=8, n_layers=2, trimming=4, n_outputs=1)
        assert not hasattr(base, "cbn")
        model = PauseNet(base)
        assert not hasattr(model.model, "cbn")
        output = model(_random_input())
        assert output.shape == (2, 1)

    def test_is_a_count_wrapper(self):
        base = CLIPNET(n_filters=8, n_layers=2, trimming=4)
        assert isinstance(PauseNet(base), CountWrapper)


class TestProCapNet:
    def test_forward_shapes(self):
        model = ProCapNet(n_filters=8, n_layers=2, trimming=4)
        profile, counts = model(_random_input())
        assert profile.shape == (2, 2, 192)
        assert counts.shape == (2, 1)

    def test_default_config(self):
        model = ProCapNet()
        assert model.n_filters == 512
        assert model.count_loss_weight == 100
