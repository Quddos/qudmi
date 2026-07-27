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
   controller object names for the hands.
4. Make sure the Animator's Controller is empty/idle -- this component fully drives the
   humanoid pose every frame, and fighting with an animation clip will look wrong.
5. Press Play.

That's the entire integration. No inference code, no per-joint wiring.

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
- Not yet verified: a live end-to-end run with a real headset driving a real avatar in Play
  mode -- that's a separate, later roadmap item (docs/ROADMAP.md) requiring a physical XR
  device, distinct from this package's math/compile correctness.
