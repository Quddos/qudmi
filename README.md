# Qudmi

**Qudmi** is an open, lightweight imitation-learning model that turns sparse VR/XR/AR tracker
signals (headset + two hand controllers) into believable full-body avatar motion in real time —
so any VR/XR/AR application can drop in realistic full-body avatars without needing extra body
trackers.

Every consumer headset gives you three tracked points: the head and two hands. Full-body
avatars still need to fake the rest (spine, hips, legs, feet), and most apps either skip legs
entirely or use hand-tuned inverse-kinematics that looks stiff. Qudmi learns that mapping from
motion-capture data instead of hand-coding it, and ships it as a small (~5–20M parameter) model
any engine can run locally at real-time latency.

> **Status: working end-to-end on hardware.** A Quest 3S drives a Mixamo Humanoid avatar in
> Unity: standing, correctly grounded, tracking head and hands. Root-aligned MPJPE **76mm val /
> 94mm test** on the official AMASS split — for reference, published three-point work reports
> ~60-70mm. Full numbers and honest limitations in [docs/RESULTS.md](docs/RESULTS.md).
>
> **Weights:** [huggingface.co/quddusr/qudmi](https://huggingface.co/quddusr/qudmi) ·
> [GitHub release v0.1.0](https://github.com/Quddos/qudmi/releases/tag/v0.1.0)
>
> ⚠️ **The trained weights are research-only.** AMASS's license prohibits training networks on it
> for commercial use of any kind, so the weights cannot ship in a commercial product. The code is
> MIT and freely reusable. See [docs/RESULTS.md#licensing-important](docs/RESULTS.md).
>
> See
> [docs/ROADMAP.md](docs/ROADMAP.md) for what's done and what's next, and
> [docs/VISION.md](docs/VISION.md) for the longer-term mission and why this aims to be more
> than a single VR plugin.

## Why this exists

- **The gap**: sparse-tracker → full-body pose is a known hard problem in VR (see AvatarPoser,
  AGRoL, Meta's "Generative Legs" research) but there's no simple open SDK an indie or mid-size
  studio can just drop into Unity/Unreal.
- **The approach**: behavior cloning (supervised imitation learning) on motion-capture data —
  learn to reproduce full-body motion from the same sparse signal a headset actually provides.
- **The constraint**: it has to run on-device, in real time (target <20ms per frame), on
  hardware an indie dev already owns.

## Project layout

```
qudmi/
├── docs/                 Design docs, spec, roadmap
├── src/qudmi/
│   ├── data/             Dataset loading & preprocessing (AMASS → sparse/full-body pairs)
│   ├── models/           Model architectures
│   ├── train/            Training loop, losses, checkpointing
│   └── export/           ONNX export for engine-agnostic inference
├── unity_plugin/         Unity + OpenXR integration plugin (consumes exported ONNX model)
├── scripts/              One-off utility / setup scripts
└── tests/                Unit tests
```

## Running everything for free

This project is designed so a solo developer can build and train it at zero cash cost:

- **Compute**: trains locally on a single consumer GPU (developed against an RTX 3060, 12GB
  VRAM). No cloud training required. If you don't have a GPU, free tiers on Google Colab or
  Kaggle Notebooks both work.
- **Data**: [AMASS](https://amass.is.tue.mpg.de/) is free for non-commercial/research use
  (requires a free account — see [docs/DATA.md](docs/DATA.md)).
- **Code hosting**: GitHub (free, public repo).
- **Model hosting**: Hugging Face Hub has free hosting for open model weights.
- **Engine**: Unity Personal is free; OpenXR is an open standard.

## Getting started

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for environment setup, and
[docs/SPEC.md](docs/SPEC.md) for the exact problem definition (inputs, outputs, metrics).

## License

MIT — see [LICENSE](LICENSE).
