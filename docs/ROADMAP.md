# Roadmap

Tracked step by step. Check off as each lands; keep this file honest — it's the source of truth
for project status, not the commit history.

- [x] Repo scaffold, docs, license
- [ ] Python environment + local GPU training verified (RTX 3060 / CUDA)
- [ ] AMASS dataset downloaded (requires free account — manual step, see [DATA.md](DATA.md))
- [x] Data preprocessing pipeline: AMASS → (sparse signal, full-body pose) pairs
      (`src/qudmi/data/{rotations,smplh,amass}.py`, `scripts/preprocess_amass.py`; validated
      end-to-end on the ACCAD subset: 252 sequences → 6002/1060/2632 train/val/test windows)
- [ ] First model architecture implemented (transformer or TCN, ~5–20M params)
- [ ] First training run completes, loss curves logged
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
