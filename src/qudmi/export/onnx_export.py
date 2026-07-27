"""Export a trained checkpoint to ONNX for engine-agnostic (Unity/Unreal/etc.) inference.

Usage:
    python -m qudmi.export.onnx_export --checkpoint checkpoints/best.pt --out qudmi.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from qudmi.data.amass import INPUT_DIM
from qudmi.models.transformer import QudmiTransformer


def export(checkpoint_path: str, out_path: str, window: int = 40):
    model = QudmiTransformer()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, window, INPUT_DIM)
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["tracker_window"],
        output_names=["pose"],
        opset_version=18,
        dynamic_axes=None,  # fixed shape: the Unity plugin always feeds a full `window`-frame buffer
    )

    # torch's exporter defaults to writing weights to a separate "<name>.onnx.data" file
    # ("external data") rather than embedding them in the .onnx graph file. That's a footgun
    # for distribution -- e.g. the Unity package -- since the two files must always travel
    # together and nothing enforces that. Re-save as a single self-contained file.
    model_proto = onnx.load(out_path, load_external_data=True)
    onnx.save(model_proto, out_path, save_as_external_data=False)
    external_data_file = Path(out_path).with_name(Path(out_path).name + ".data")
    if external_data_file.exists():
        external_data_file.unlink()

    return dummy_input


def verify(out_path: str, dummy_input: torch.Tensor, model: torch.nn.Module):
    """Sanity check: ONNX Runtime output must match the original PyTorch model's output.

    model.eval() is mandatory here, not just a caller convention: with dropout active, the
    PyTorch forward pass is stochastic and will never match the (deterministically exported)
    ONNX graph, regardless of whether the export itself is correct.
    """
    model.eval()
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    session = onnxruntime.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"tracker_window": dummy_input.numpy()})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"Max abs difference between PyTorch and ONNX Runtime outputs: {max_diff:.2e}")
    assert max_diff < 1e-4, "ONNX export does not numerically match the PyTorch model"
    print("ONNX export verified: outputs match.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out", default="qudmi.onnx")
    parser.add_argument("--window", type=int, default=40)
    args = parser.parse_args()

    model = QudmiTransformer()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy_input = export(args.checkpoint, args.out, args.window)
    print(f"Exported {args.checkpoint} -> {args.out}")
    verify(args.out, dummy_input, model)


if __name__ == "__main__":
    main()
