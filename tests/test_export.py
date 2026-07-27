import torch

from qudmi.export.onnx_export import export, verify
from qudmi.models.transformer import QudmiTransformer


def test_onnx_export_matches_pytorch(tmp_path):
    model = QudmiTransformer()
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    out_path = tmp_path / "model.onnx"
    dummy_input = export(str(checkpoint_path), str(out_path))

    verify(str(out_path), dummy_input, model)  # raises AssertionError on mismatch
