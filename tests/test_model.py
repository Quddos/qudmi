import torch

from qudmi.data.amass import INPUT_DIM, OUTPUT_DIM
from qudmi.models.transformer import QudmiTransformer


def test_forward_shape():
    model = QudmiTransformer()
    x = torch.randn(4, 40, INPUT_DIM)
    out = model(x)
    assert out.shape == (4, OUTPUT_DIM)


def test_parameter_count_within_target_range():
    model = QudmiTransformer()
    n = model.num_parameters()
    assert 1e6 < n < 20e6, f"expected 1M-20M params, got {n:,}"


def test_gradients_flow():
    model = QudmiTransformer()
    x = torch.randn(2, 40, INPUT_DIM)
    target = torch.randn(2, OUTPUT_DIM)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_variable_window_length():
    # windows shorter than the configured max_len should still work (e.g. at stream startup
    # before `window` frames of history have accumulated).
    model = QudmiTransformer()
    x = torch.randn(1, 10, INPUT_DIM)
    out = model(x)
    assert out.shape == (1, OUTPUT_DIM)
