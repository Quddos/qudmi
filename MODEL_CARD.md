# Qudmi v0.1 — model card

Full-body human pose from sparse VR tracking: head + two hand controllers → 22 SMPL body joints,
in real time, on device.

## ⚠️ License: research and non-commercial use only

**The weights may not be used in a commercial product or service.**

They are trained on [AMASS](https://amass.is.tue.mpg.de/), whose license permits non-commercial
scientific research, education and artistic projects only, and states explicitly that it
"prohibits the use of the Dataset to train methods/algorithms/neural networks/etc. for commercial
use of any kind". That restriction passes to anything trained on it, including these weights.

The **code** in [github.com/Quddos/qudmi](https://github.com/Quddos/qudmi) is MIT and carries no
such restriction — the architecture, training pipeline and Unity runtime are freely reusable, and
you can retrain on your own data for any purpose.

## What it does

| | |
|---|---|
| Input | 40 frames × 36 features — head and both wrists, each as position (3) + 6D rotation (6) + velocity (3), at 30 Hz |
| Output | 135 values — 22 joint rotations in 6D form + root translation |
| Parameters | 4.8 M |
| Inference | ~2 ms CPU (ONNX Runtime), ~1.5 ms GPU |
| Format | ONNX opset 18, single self-contained file |

Input is canonicalized against the head's horizontal position and yaw, which is the only
reference frame a headset can actually supply. Rotations use the 6D continuous representation
(Zhou et al. 2019) rather than quaternions or Euler angles, which are discontinuous and hurt
regression.

## Performance

Root-aligned MPJPE, the standard metric in this literature.

| Split | Source | MPJPE |
|---|---|---|
| val | HumanEva | 76 mm |
| test | TotalCapture | 94 mm |

Published three-point work (AvatarPoser, AGRoL) reports roughly 60–70 mm, so this sits about
1.4× behind the state of the art. Six-sensor systems such as DTP report ~10 mm, but with the
pelvis and both ankles tracked the root and legs are measured rather than inferred — a different
and considerably easier problem.

### Where the error is

| Joint group | MPJPE |
|---|---|
| feet / ankles | 173–196 mm |
| knees | ~111 mm |
| wrists | ~107 mm |

**Legs dominate, and that is structural.** Nothing observes the lower body at three tracked
points, so it is a plausible inference rather than a measurement. Every three-point method shares
this limit.

**The wrist number is misleading on its own.** Wrist position is a model *input*, but the network
predicts rotations and nothing constrains forward kinematics to return the wrist to the observed
position — error accumulates along spine → collar → shoulder → elbow → wrist. The Unity runtime
corrects this with two-bone IK onto the tracked controllers, so live hand error is effectively
zero. The root is handled the same way: the runtime anchors to the measured headset position
rather than the predicted root translation, which generalizes poorly across datasets.

## Training

- **Data**: official AMASS split — ACCAD, BMLmovi, CMU, KIT (589,761 windows) for training,
  HumanEva for validation, TotalCapture for test.
- **Losses**: rotation-matrix MSE (1.0), FK joint-position (0.5), root translation (0.1). The low
  translation weight is deliberate: it generalizes ~10× worse than the other terms and the runtime
  discards it, so it should not dominate the gradients.
- **Schedule**: AdamW, lr 1e-4 halving every 15 epochs, early stopping on validation pose quality.
  Stopped at epoch 27; best checkpoint epoch 17.
- **Hardware**: single RTX 3060, a few hours.

## Known limitations

- **Legs are inferred, not measured** — see above. Expect plausible rather than accurate.
- **Foot sliding.** DTP's foot-velocity loss is not implemented, because it differences
  consecutive *predicted* frames and this model predicts a single frame from a window. It needs
  sequence-level training to be meaningful.
- **Body-size range.** AMASS subjects top out near 1.65 m head height. The runtime scales input
  toward the training distribution at calibration, but users far outside it may see degraded
  quality.
- **Not profiled on standalone Quest.** Developed over Quest Link.

## Intended use

Research, education, prototyping and non-commercial projects: social VR presence, full-body
avatars, movement analysis, animation authoring.

Not suitable as-is for clinical, diagnostic or safety-critical use — accuracy has not been
validated for any of those, and the lower body is inferred.

## Citation

If AMASS-derived work like this is useful to you, cite AMASS (Mahmood et al. 2019) and the
sparse-tracking methods this builds on: AvatarPoser (Jiang et al. 2022), AGRoL (Du et al. 2023),
and DTP (Liu et al., *Virtual Reality* 28:116, 2024).
