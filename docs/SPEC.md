# Qudmi v0 — Problem Specification

This document pins down exactly what the first working version of Qudmi does, so the data
pipeline, model, and evaluation all target the same thing. Scope is deliberately narrow —
generalize later, ship something small and correct first.

## Task

**Input**: a short time window (e.g. last 40 frames at 30–90Hz) of sparse tracker signal:

| Signal          | Representation                                   |
|------------------|---------------------------------------------------|
| Head pose        | position (3) + rotation (6D continuous, see below) |
| Left hand pose   | position (3) + rotation (6D)                       |
| Right hand pose  | position (3) + rotation (6D)                       |
| Linear velocities | finite difference of the three positions above    |

Rotations are encoded in the 6D continuous representation (Zhou et al., 2019) rather than
quaternions or Euler angles — it avoids the discontinuities that hurt regression training.

**Output**: full-body pose for the current frame, as local joint rotations for a fixed skeleton
(SMPL body model joint set, 21–24 joints depending on final skeleton choice), plus the root
(pelvis) position and orientation.

**Latency target**: inference for one frame must run in well under 20ms on a mid-range consumer
GPU (or a modern CPU, as a stretch goal), so it fits inside a VR frame budget alongside
rendering.

## Why behavior cloning, not RL

This is a sequence regression problem: for every frame in a motion-capture recording we know
both the sparse signal (derived from the recording) and the ground-truth full-body pose. That's
a direct supervised target — no reward function, no environment to interact with. "Imitation
learning" here specifically means **behavior cloning**: train a model to reproduce the
demonstrated mapping from sparse input to full pose. This is simpler and more sample-efficient
than adversarial IL (GAIL) or RL-from-demonstrations, and it's the right first step because we
have complete, dense demonstrations rather than partial trajectories.

## Data

- **Primary dataset**: [AMASS](https://amass.is.tue.mpg.de/) — a large unified collection of
  motion-capture datasets retargeted to the SMPL body model, which is exactly the joint
  representation we need. Free for non-commercial/research use.
- **Derivation of sparse signal**: since AMASS doesn't record headset/controller data directly,
  the head and hand tracker signal is *derived* from the SMPL joint positions/orientations at
  the head and wrist joints for each frame. This is the standard approach used by AvatarPoser,
  AGRoL, and similar prior work.
- **Split**: standard AMASS train/val/test split by subject (not by frame), so the model is
  evaluated on motion from people it never trained on.

## Model (first version)

- Small transformer encoder over the input window, or a causal temporal convolutional network
  — both are cheap to run per-frame with a sliding window / KV-cache.
- Target size: 5–20M parameters. This is a regression head over a short window, not a language
  model — it does not need billions of parameters to do this well (prior work gets strong
  results well under 50M params).

## Losses

- Joint rotation loss (geodesic distance or 6D regression loss) — primary signal.
- Joint position loss (forward-kinematics the predicted rotations through the skeleton, compare
  3D joint positions) — keeps errors physically meaningful rather than purely rotational.
- Temporal smoothness penalty (difference between consecutive predicted frames) — motion-capture
  ground truth is smooth; naive per-frame regression tends to jitter.
- Optional foot-contact/sliding penalty once legs are in scope — foot sliding is the most
  visually obvious failure mode in this class of model.

## Evaluation metrics

- **MPJPE** (Mean Per-Joint Position Error, mm) — standard pose-estimation metric, lets us
  compare against published numbers from AvatarPoser/AGRoL as a sanity check.
- **Jitter** — average acceleration of predicted joint positions over time, to catch
  jerky/unnatural motion that position error alone won't reveal.
- **Foot skate** — average foot displacement during predicted stance/contact frames.

## Explicit non-goals for v0

- Hand/finger articulation (out of scope — assume tracked hand pose is only wrist pose).
- Physical interaction with the environment (contact-aware motion, object manipulation).
- Multiple body shapes/heights beyond what AMASS's SMPL shape parameters already cover.
- Networked/multiplayer motion prediction.

These are reasonable v1+ directions once the core sparse-to-full-body model works.
