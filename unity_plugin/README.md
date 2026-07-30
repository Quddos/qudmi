# Qudmi Full Body (Unity package)

Drop-in full-body avatar motion from a VR headset + two hand controllers. No ML code required.

## Install

Unity 6000.0 or newer. In your project: **Window > Package Manager > + > Install package from git
URL**, and paste:

```
https://github.com/Quddos/qudmi.git?path=/unity_plugin
```

That pulls the code *and* the trained model, so it works immediately with nothing else to
download. `com.unity.ai.inference` (Unity's inference package, formerly "Sentis" -- Sentis-era
class names still work, under the `Unity.InferenceEngine` namespace) resolves automatically.

To pin a version, append a tag: `...?path=/unity_plugin#v0.1.0`.

> **Why not searchable in Package Manager?** Unity's built-in registry is curated by Unity; you
> cannot publish to it directly. The searchable route for open-source packages is
> [OpenUPM](https://openupm.com), a community scoped registry -- planned, and it needs the git
> URL install above to be proven first. The Asset Store is not an option here: its content is
> used commercially, and these weights are licensed for non-commercial research only.

## Try it in 30 seconds, no headset needed

**Window > Qudmi > Create Demo Scene**, then press Play.

You get a ground plane, a generated Humanoid avatar, and simulated head/hand motion driving it —
so you can see what the model actually does before setting up an XR rig or even owning a headset.
Useful for evaluating output too: the simulated motion repeats exactly, which live tracking can't.

The avatar is deliberately a set of primitives. A package can't redistribute a real character
(Mixamo rigs are Adobe's, Unity's sample rigs are Unity's), so **Window > Qudmi > Create Demo
Avatar** generates one instead — ugly, but unambiguously ours to ship.

## Quick start with your own avatar

**Window > Qudmi > Setup Wizard** does the whole setup and checks for the things that commonly go
wrong silently — a leftover `Main Camera` making `Camera.main` ambiguous, an Animator Controller
fighting the driver, a rig imported as Generic rather than Humanoid. It can also drop a preview
clone in front of you, which matters more than it sounds: you are *inside* the avatar, so it is
otherwise the one thing you cannot see.

The manual equivalent is below if you would rather wire it yourself.

## Manual setup

1. The model ships with the package at `Runtime/Model/qudmi_v0.onnx` — Inference Engine imports
   `.onnx` files automatically as a `ModelAsset`. (To use your own, export one with the main
   repo's `src/qudmi/export/onnx_export.py`.)
2. On any GameObject with a **Humanoid**-rigged `Animator` (check the model's import settings:
   Rig > Animation Type > Humanoid), add the **Qudmi Full Body Driver** component
   (`Qudmi/Qudmi Full Body Driver` in the Add Component menu).
3. Assign the `ModelAsset` in the Inspector. Assign Head/Left Hand/Right Hand transforms if you
   have a specific XR rig hierarchy, or leave them empty -- the component falls back to
   `Camera.main` for the head and a best-effort search for common XR Interaction Toolkit
   controller object names for the hands. If you're using an XR rig prefab (e.g. XR Interaction
   Toolkit's "XR Origin (XR Rig)"), make sure the scene doesn't *also* have Unity's default
   `Main Camera` left over from a new-scene template -- two GameObjects tagged `MainCamera` makes
   `Camera.main` ambiguous, and auto-detect may grab the wrong (untracked) one. Delete the
   leftover default camera.
4. Leave the Animator's **Controller** empty (`None`). This component writes bone transforms
   directly and does not need a Controller or an animation clip; anything the Controller plays
   will simply fight it.
5. Press Play, then **stand upright facing forward with your arms hanging down for ~5 seconds.**
   Calibration fits itself automatically at that point (see below) and logs when it's done.

That's the entire integration. No inference code, no per-joint wiring.

### Calibration -- why it matters

A tracker's pose is not the body joint's pose, and the difference is not small. Measured over
10,187 standing frames of the training set, the orientation the model expects for each tracked
joint sits 165 degrees (head), 112 (left wrist) and 87 (right wrist) away from what a Unity XR
rig reports for the same physical pose. Feeding device rotations in unmodified gives a
confidently wrong pose rather than a slightly noisy one, and no amount of retraining fixes it.
DTP (Liu et al., *Virtual Reality* 28:116, sec 3.2) handles the same problem with a T-pose
calibration step.

The component ships with baked-in defaults so it works with no setup, and refits them to the
specific wearer and controller convention on calibration. Calibration also scales the input to
the wearer's height, since AMASS subjects top out near 1.65 m and a taller user would otherwise
sit outside the distribution the model was trained on.

By default this happens automatically 5 seconds after tracking starts (`Auto Calibrate Delay`).
There is also a **Calibrate** button in the Inspector during Play, though standing in the
calibration pose while reaching for a mouse is awkward -- the delay exists for that reason.

### Seeing your own avatar

The camera sits inside the avatar's head, so the wearer cannot see their own body from the
outside -- which makes judging pose quality surprisingly hard. Add **Qudmi Avatar Preview** to a
second copy of the character, placed a couple of metres in front and turned to face the play
area, and it will mirror the driven pose in real time.

The driven avatar's head bone is shrunk to nothing by default (`Hide Head In First Person`) so
the wearer isn't looking at the inside of their own skull. That only affects the driven rig; the
preview clone keeps a complete body.

## Why Unity's Inference Engine instead of raw ONNX Runtime

Quest/Android is the actual deployment target for a VR full-body driver, and ONNX Runtime's
native-plugin story there is considerably more fragile than Unity's own cross-platform inference
package (asset-pipeline integration, no separate native binary wrangling per platform).

## Verifying the math (for contributors)

The trickiest part of this package is that the training pipeline (Python/AMASS) works in a
right-handed Y-up coordinate convention, while Unity is left-handed Y-up. `CoordinateConversion.cs`
handles this; getting it wrong would silently produce a plausible-looking but incorrect pose
rather than an obvious crash, so it's checked with an actual numeric test rather than trusted by
derivation alone:

- `scripts/generate_unity_parity_fixture.py` (repo root) computes reference input/output values
  in Python and writes `Tests/Resources/parity_case_1.json`.
- `Tests/ParityTests.cs` (Unity EditMode test) feeds the same raw inputs through the C# code and
  asserts the results match the Python reference to within 1e-3.

Run it via Unity's Test Runner (`Window > General > Test Runner > EditMode`), or in CI/batchmode:

```bash
Unity -batchmode -nographics -runTests -testPlatform EditMode -projectPath <project> -testResults results.xml
```

## What's verified vs. not (as of this writing)

- **Verified working on hardware**: a Quest 3S over Quest Link driving a Mixamo Humanoid avatar.
  The avatar stands upright, is correctly placed on the ground, and follows head and hand
  movement. Compiles against Unity 6000.3 / 6000.4 with Inference Engine 2.2-2.6, and the three
  Python-parity tests pass.
- **Not yet production quality**: pose accuracy is limited by the model, not the integration.
  Arms and torso track recognizably; the legs are the weak point and always will be to some
  degree -- with only three tracked points nothing observes them, so the lower body is a
  plausible inference rather than a measurement. Published three-point work lands around 6-7cm
  mean joint error against roughly 1cm for six-tracker systems.
- **Not yet done**: standalone Android/Quest build (developed over Link), quantized on-device
  variant, and support for optional extra trackers for users who own them.

Getting to this point turned up four real bugs the unit tests could not have caught, all of
which produced plausible-looking wrong output rather than errors: applying poses outside
`OnAnimatorIK`, retargeting from SMPL's rest pose instead of a standing one, mixing left- and
right-handed rotations in one product, and treating a controller's pose as a wrist joint's pose.
Each was found by measuring live values against the training distribution -- worth knowing if you
extend this.
