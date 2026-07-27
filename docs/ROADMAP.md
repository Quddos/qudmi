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
      logging confirmed working. Rotation loss generalizes reasonably (train 0.021 / val 0.043
      after 15 epochs). Translation loss overfits badly and validation numbers are noisy --
      **this is a data-scale problem, not a training-loop bug**: this single AMASS subset
      (ACCAD) has only 9 distinct actors, so a subject-held-out val split is either empty or,
      after fixing the split logic, just 4 windows from 1 actor. Real training needs more AMASS
      subsets downloaded (see docs/DATA.md) before these loss numbers mean anything.)
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
