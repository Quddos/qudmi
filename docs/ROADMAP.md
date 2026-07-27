# Roadmap

Tracked step by step. Check off as each lands; keep this file honest — it's the source of truth
for project status, not the commit history.

- [x] Repo scaffold, docs, license
- [ ] Python environment + local GPU training verified (RTX 3060 / CUDA)
- [ ] AMASS dataset downloaded (requires free account — manual step, see [DATA.md](DATA.md))
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
- [ ] Evaluation: MPJPE / jitter / foot-skate on held-out AMASS subjects
- [ ] ONNX export of trained model
- [ ] Unity + OpenXR plugin skeleton runs the ONNX model live on tracked head+hands
- [ ] End-to-end demo: real Quest headset driving a full-body avatar in a Unity scene
- [ ] Public model weights + writeup published (Hugging Face + README)

## Later / v1+ ideas (not scoped yet)

- Hand/finger pose from controller input
- Foot-contact-aware physics correction
- Cross-engine plugin (Unreal, Godot/OpenXR generic)
- Smaller/quantized variant for standalone headset CPUs (no external GPU)
