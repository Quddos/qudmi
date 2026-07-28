# Qudmi Full Body (Unity package)

Drop-in full-body avatar motion from a VR headset + two hand controllers. No ML code required.

## Install

1. Open Package Manager (`Window > Package Manager`) in your Unity project (Unity 6000.0+).
2. `+` > "Install package from disk..." (or add a git/local dependency to `Packages/manifest.json`
   pointing at this `unity_plugin/` folder).
3. This package depends on `com.unity.ai.inference` (Unity's inference package, formerly named
   "Sentis" -- the class names Sentis-era tutorials reference still work, just under the
   `Unity.InferenceEngine` namespace now). Package Manager resolves it automatically.

## Use

1. Import your exported model (`checkpoints/qudmi_v0.onnx`, see the main repo's
   `src/qudmi/export/onnx_export.py`) into the project as a Unity asset -- Inference Engine
   recognizes `.onnx` files automatically as a `ModelAsset`.
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
4. The Animator needs a **Controller assigned (not "None")**, with **IK Pass enabled on its base
   layer** -- an empty Controller with no states/clips is fine, since this component fully drives
   the pose every frame. This is required, not optional: `Animator.SetBoneLocalRotation` (what
   this component uses to apply the pose) only works when called from Unity's `OnAnimatorIK`
   callback, and Unity only invokes that callback when IK Pass is on. Skipping this doesn't fail
   loudly -- the symptom is Console spam ("should only be done in OnAnimatorIK or OnStateIK")
   escalating into NaN bone positions within a few seconds, found the hard way during an actual
   headset test.
5. Press Play.

That's the entire integration. No inference code, no per-joint wiring.

### Creating the empty IK-pass Controller

1. In the Project window: right-click → Create → Animator Controller. Name it anything (e.g.
   `QudmiIKPass`).
2. Double-click it to open the Animator window. Select the **Base Layer** (left panel), click the
   gear icon next to it, and check **IK Pass**.
3. Assign this Controller to your character's Animator component. No states or clips needed.

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

- Verified: the package compiles cleanly against Unity 6000.4 + Inference Engine 2.2, and all
  three parity tests (input encoding, per-joint local rotation decode, root recomposition) pass.
- In progress: a live end-to-end run with a real Quest headset driving a real Mixamo Humanoid
  avatar. A real bug (SetBoneLocalRotation called outside OnAnimatorIK, cascading into NaN bone
  positions) was found and fixed via this testing, not caught by the parity tests -- those check
  the math functions in isolation, not the MonoBehaviour's actual Unity lifecycle usage. Still to
  confirm: the avatar visibly tracking head/hand movement correctly once IK Pass is enabled.
