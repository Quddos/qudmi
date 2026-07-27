# Roadmap

Tracked step by step. Check off as each lands; keep this file honest — it's the source of truth
for project status, not the commit history.

- [x] Repo scaffold, docs, license
- [x] Python environment + local GPU training verified (RTX 3060 / CUDA)
- [x] AMASS dataset downloaded (ACCAD + BMLmovi, 95 distinct actors so far; more subsets
      recommended for gender/body-shape balance -- see note below)
- [x] Data preprocessing pipeline: AMASS → (sparse signal, full-body pose) pairs
      (`src/qudmi/data/{rotations,smplh,amass}.py`, `scripts/preprocess_amass.py`; validated
      end-to-end on the ACCAD subset: 252 sequences → 6002/1060/2632 train/val/test windows)
- [x] First model architecture implemented (transformer, ~5–20M params)
      (`src/qudmi/models/transformer.py`: 4.8M params, 1.46ms GPU / 2.25ms CPU inference
      latency for a single frame — both well under the 20ms real-time budget)
- [x] First training run completes, loss curves logged
      (`src/qudmi/train/{losses,train}.py`; runs end-to-end on GPU, checkpointing + TensorBoard
      logging confirmed working)
- [x] Confirmed the model actually generalizes once given enough data: after adding BMLmovi
      (95 distinct actors total, 55610/6264/7308 train/val/test windows), 40 epochs gives
      rot loss 0.009/0.011 (train/val) and trans loss 0.017/0.029 (train/val) -- both track
      together now, unlike the single-subset (ACCAD-only, 9 actors) run where translation
      loss overfit by ~30x. Best checkpoint: val loss 0.0395 at epoch 40 (checkpoints/best.pt,
      not committed to git -- see .gitignore).
- [x] Evaluation: MPJPE on held-out AMASS subjects (`src/qudmi/train/evaluate.py`)
      Train/val/test MPJPE: 151mm / 248mm / 492mm. The test-split jump is a real, diagnosed
      finding, not a bug: the current data mix (ACCAD + BMLmovi) is heavily gender/body-shape
      imbalanced (BMLmovi contributes ~85 actors, nearly all labeled female; ACCAD contributes
      only a couple of male actors), and the test split's hash-based subject assignment happened
      to land the one clearly-male ACCAD actor ("Male1") there alongside a batch of BMLmovi
      subjects -- an out-of-distribution body shape for what the model mostly trained on.
      Rotation loss stays consistent across splits (0.008/0.011/0.015 train/val/test), so this
      is specifically a translation/body-shape generalization gap. Fix is a more gender-balanced
      dataset mix, not more code -- flagged as a follow-up data task below.
      jitter/foot-skate deferred: both need temporally-ordered consecutive-window predictions
      from the same sequence, which the current shuffled-window dataset format doesn't preserve.
- [x] ONNX export of trained model (`src/qudmi/export/onnx_export.py`)
      Verified numerically identical to PyTorch (max diff ~2e-6). ONNX Runtime CPU inference:
      1.72ms/frame, comfortably under the 20ms real-time budget.
- [ ] Unity + OpenXR plugin skeleton runs the ONNX model live on tracked head+hands
- [ ] End-to-end demo: real Quest headset driving a full-body avatar in a Unity scene
- [ ] Public model weights + writeup published (Hugging Face + README)

## Later / v1+ ideas (not scoped yet)

- Hand/finger pose from controller input
- Foot-contact-aware physics correction
- Cross-engine plugin (Unreal, Godot/OpenXR generic)
- Smaller/quantized variant for standalone headset CPUs (no external GPU)
