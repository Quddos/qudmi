# Results

## Benchmark

Root-aligned MPJPE, the standard metric in this literature (DTP states it as "with the position
and rotation of the root joint aligned"; AvatarPoser and AGRoL report the same way).

| Split | Source | Windows | MPJPE |
|---|---|---|---|
| val | HumanEva | 3,552 | **76 mm** |
| test | TotalCapture | 18,151 | **94 mm** |

Trained on the official AMASS split (ACCAD, BMLmovi, CMU, KIT → 589,761 windows), 4.8M-parameter
transformer, early-stopped at epoch 17 of a maximum 80.

### Context

| Method | Sensors | Reported error |
|---|---|---|
| DTP (Liu et al. 2024) | 6 (head, pelvis, 2 hands, 2 feet) | 10 mm |
| AvatarPoser / AGRoL | 3 (head, 2 hands) | ~60-70 mm |
| **Qudmi** | **3 (head, 2 hands)** | **94 mm** |

DTP's figure is not comparable: with the pelvis and both ankles tracked, the root and legs are
measured rather than inferred, which is a substantially easier problem. The three-point numbers
are the fair comparison, and Qudmi currently sits about 1.4x behind them.

### Per-joint breakdown (test split)

| Joint group | MPJPE |
|---|---|
| feet / ankles | 173-196 mm |
| knees | ~111 mm |
| wrists | ~107 mm |

**The legs dominate, and that is structural.** With three tracked points nothing observes the
lower body; the model produces a plausible lower body consistent with upper-body motion rather
than a measurement. This is the shared limitation of all three-point methods and no amount of
training removes it.

**The wrist figure is misleading in isolation** and is addressed at runtime rather than in the
model. Wrist position is a model *input*, but the network predicts rotations and nothing
constrains forward kinematics to return the wrist to the observed position -- error accumulates
along spine → collar → shoulder → elbow → wrist. The Unity runtime corrects this with two-bone
IK onto the tracked controllers, so live hand error is effectively zero. Same reasoning applies
to the root: the runtime anchors the rig to the measured headset position instead of using the
predicted root translation, which generalizes poorly across datasets (val 0.30 vs train 0.08).

## What is not measured yet

- **Jitter and foot skate.** Both require predictions on temporally consecutive frames, and the
  dataset stores independently shuffled windows. Same reason DTP's foot-velocity loss is not
  implemented: single-frame prediction cannot express it. Needs sequence-level training.
- **Standalone Quest performance.** Developed over Quest Link; CPU inference is ~2ms on desktop
  but has not been profiled on-device.

## Licensing: important

The code in this repository is MIT. **The trained weights are not, and cannot be.**

AMASS is licensed for non-commercial scientific research only, and the license is explicit that
it "prohibits the use of the Dataset to train methods/algorithms/neural networks/etc. for
commercial use of any kind." Any weights trained on AMASS therefore inherit that restriction:
they may be used for research, education, and non-commercial projects, but not shipped in a
commercial product or service.

This is a genuine constraint on the project's goal of a drop-in component for any developer,
since most game studios - including indies - ship commercially. Published work in this area
(AvatarPoser, AGRoL, DTP) is released on the same non-commercial basis, so this is normal for
research artifacts, but it is not what a general-purpose SDK needs.

Resolving it means retraining on motion data whose license permits commercial use. The original
CMU Graphics Lab Motion Capture Database is the most promising candidate ("free for all uses"),
but it is distributed as raw marker/skeleton data rather than SMPL parameters, so it would need
fitting to the SMPL body model first - a real piece of work, not a config change.

Until then: **the architecture, training pipeline, and Unity runtime are freely reusable; the
released weights are research-only.**
