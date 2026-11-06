import pytest
import torch
import torch.nn as nn

from fl4hma.models.unet import UNetCNN


def _has(model, cls):
    return any(isinstance(m, cls) for m in model.modules())


class TestNorm:
    def test_default_is_group_norm(self):
        model = UNetCNN(in_channels=4, out_channels=1, base_filters=16)
        assert _has(model, nn.GroupNorm)
        assert not _has(model, nn.BatchNorm2d)

    def test_group_norm_model_has_no_buffers(self):
        model = UNetCNN(in_channels=4, out_channels=1, base_filters=16)
        assert list(model.buffers()) == []
        assert len(model.state_dict()) == len(list(model.parameters()))

    def test_batch_norm_still_available(self):
        model = UNetCNN(in_channels=4, out_channels=1, base_filters=16, norm="batch")
        assert _has(model, nn.BatchNorm2d)
        assert not _has(model, nn.GroupNorm)

    def test_invalid_norm_raises(self):
        with pytest.raises(ValueError):
            UNetCNN(norm="layer")

    @pytest.mark.parametrize("base_filters", [2, 4, 16, 32])
    @pytest.mark.parametrize("use_attention", [True, False])
    def test_forward_shape(self, base_filters, use_attention):
        model = UNetCNN(
            in_channels=4,
            out_channels=1,
            base_filters=base_filters,
            use_attention=use_attention,
        )
        out = model(torch.randn(2, 4, 32, 32))
        assert out.shape == (2, 1, 32, 32)

    def test_group_norm_output_independent_of_batch(self):
        torch.manual_seed(0)
        model = UNetCNN(in_channels=4, out_channels=1, base_filters=8).train()
        x = torch.randn(4, 4, 32, 32)
        batched = model(x)[:1]
        alone = model(x[:1])
        torch.testing.assert_close(batched, alone)
