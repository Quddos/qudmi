using Unity.InferenceEngine;
using UnityEngine;

namespace Qudmi
{
    /// <summary>
    /// Drop this on any GameObject with a Humanoid-rigged Animator to drive full-body motion
    /// from just head + two hand transforms -- no ML code required. Assign the exported Qudmi
    /// ONNX model and, optionally, the head/hand Transforms to track (left unassigned, it
    /// falls back to Camera.main for the head and a best-effort search for common XR
    /// Interaction Toolkit controller names for the hands).
    ///
    /// The Animator's Controller can be left as None -- this component writes bone transforms
    /// directly (the approach procedural full-body IK solutions generally use) rather than going
    /// through Animator.SetBoneLocalRotation, which would additionally require an IK-Pass-enabled
    /// Controller and only works inside OnAnimatorIK.
    /// </summary>
    [AddComponentMenu("Qudmi/Qudmi Full Body Driver")]
    [RequireComponent(typeof(Animator))]
    public class QudmiFullBodyDriver : MonoBehaviour
    {
        [SerializeField] private ModelAsset modelAsset;
        [Tooltip("Defaults to CPU: the GPUCompute backend was observed returning all-NaN output " +
            "for this model (identical input through the CPU backend is fine), and inference is " +
            "~2ms/frame on CPU anyway, so GPU isn't needed to hit the frame budget.")]
        [SerializeField] private BackendType backend = BackendType.CPU;

        [Header("Tracked transforms (leave empty to auto-detect)")]
        [SerializeField] private Transform headTransform;
        [SerializeField] private Transform leftHandTransform;
        [SerializeField] private Transform rightHandTransform;

        [Header("Output smoothing")]
        [Range(0f, 0.95f)]
        [SerializeField] private float smoothing = 0.3f;

        [Header("Sampling")]
        [Tooltip("Training data was resampled to this rate (see docs/SPEC.md). Pushing a new " +
            "frame every render frame instead -- e.g. 90Hz on Quest -- would make the model's " +
            "40-frame window cover a much shorter real-time span than it was trained on, " +
            "skewing velocity and temporal context even though nothing would crash.")]
        [SerializeField] private float targetSampleFps = 30f;

        [Header("Placement")]
        [Tooltip("Positions the rig so its head bone coincides with the tracked headset, instead " +
            "of trusting the model's predicted root height. Strongly recommended: the headset " +
            "position is measured, whereas the predicted root height carries model error and " +
            "assumes the avatar's proportions match the training bodies -- either of which sinks " +
            "the avatar into the floor or floats it above. Also what makes first-person view " +
            "line up, since the camera then sits in the avatar's own head.")]
        [SerializeField] private bool anchorHeadToTracker = true;

        [Tooltip("Shrinks the head bone to nothing so the wearer isn't looking at the inside of " +
            "their own head. Only affects this rig, so a QudmiAvatarPreview clone still shows a " +
            "complete body. Turn off for a third-person or spectator camera.")]
        [SerializeField] private bool hideHeadInFirstPerson = true;

        [Header("Calibration")]
        [Tooltip("Seconds after tracking starts before calibration is fitted automatically. " +
            "Exists because the wearer cannot be standing in the calibration pose and clicking a " +
            "mouse button at the same time. Set to 0 to calibrate only manually.")]
        [SerializeField] private float autoCalibrateDelay = 5f;

        [Header("Diagnostics")]
        [Tooltip("Logs the canonicalized model input once per second, so live values can be " +
            "compared against the training distribution. A mismatch here means the model is " +
            "being fed out-of-distribution input, which no amount of retraining fixes.")]
        [SerializeField] private bool logInputDiagnostics;

        private Animator _animator;
        private TrackerWindowBuffer _buffer;
        private IInferenceEngine _engine;
        private Quaternion[] _smoothedRotations;
        private Vector3 _smoothedRootPosition;
        private bool _hasPreviousPose;
        private bool _hasPendingPose;
        private bool _loggedRawPoseDiagnostic;
        private bool _loggedAutoDetectWarning;
        private float _diagnosticTimer;
        private bool _calibrated;
        private float _trackingStartTime = -1f;
        private float _timeSinceLastSample;
        private PoseDecoder.DecodedPose _pendingPose;
        private Transform[] _boneTransforms;
        private Quaternion[] _restWorldRotations;
        private Quaternion[] _globalRotations;

        private void Awake()
        {
            _animator = GetComponent<Animator>();
            _buffer = new TrackerWindowBuffer
            {
                // Baked defaults so the component does something sensible with no setup; press
                // the Inspector's Calibrate button (or call Calibrate()) for a rig-specific fit.
                RotationOffsets = TrackerCalibration.DefaultOffsets(),
            };
            CaptureRestPose();

            if (modelAsset != null)
            {
                _engine = new SentisInferenceEngine(modelAsset, backend);
            }
            else
            {
                Debug.LogWarning("QudmiFullBodyDriver: no model assigned, this component will do nothing.", this);
            }

            // Deliberately not resolving tracked transforms here -- see AutoDetectTransformsIfNeeded.
        }

        private void OnDestroy()
        {
            (_engine as System.IDisposable)?.Dispose();
        }

        /// <summary>
        /// Fits the tracker corrections to this wearer and rig. Call it (or press Calibrate in
        /// the Inspector during Play) while standing upright, facing forward, arms hanging down --
        /// the pose the reference values in TrackerCalibration were measured from.
        ///
        /// Solves both corrections at once: the per-tracker rotation offsets, and a uniform
        /// position scale mapping this wearer's height onto the training set's.
        /// </summary>
        public void Calibrate()
        {
            if (_buffer == null || !_buffer.IsFull)
            {
                Debug.LogWarning("QudmiFullBodyDriver: can't calibrate yet -- still filling the " +
                    "initial tracking window. Wait a second in Play mode and try again.", this);
                return;
            }

            _buffer.RotationOffsets = TrackerCalibration.ComputeOffsets(_buffer.CurrentCanonicalRotations());

            float headHeight = _buffer.AnchorHeadPosition.y;
            if (headHeight > 0.5f)
            {
                _buffer.PositionScale = TrackerCalibration.ReferenceHeadHeight / headHeight;
            }

            _calibrated = true;
            Debug.Log($"QudmiFullBodyDriver: calibrated. Head height {headHeight:F2}m -> " +
                $"position scale {_buffer.PositionScale:F3}, rotation offsets fitted to this rig.", this);
        }

        /// <summary>
        /// Records each mapped bone's bind-pose world orientation, which retargeting needs (see
        /// ApplyPose). Must run before anything poses the rig, hence Awake.
        /// </summary>
        private void CaptureRestPose()
        {
            _boneTransforms = new Transform[QudmiConstants.NumBodyJoints];
            _restWorldRotations = new Quaternion[QudmiConstants.NumBodyJoints];
            _globalRotations = new Quaternion[QudmiConstants.NumBodyJoints];

            for (int j = 0; j < QudmiConstants.NumBodyJoints; j++)
            {
                // Null for bones this rig doesn't have -- UpperChest and the toe bones are
                // genuinely optional in Unity's humanoid definition, so this isn't an error.
                Transform bone = _animator.GetBoneTransform(QudmiConstants.HumanBodyBoneForJoint[j]);
                _boneTransforms[j] = bone;
                _restWorldRotations[j] = bone != null ? bone.rotation : Quaternion.identity;
            }

            // Scale rather than disable: the head bone still has to drive its children's
            // transforms, and hiding the renderer would take the whole body with it since the
            // avatar is a single skinned mesh.
            if (hideHeadInFirstPerson && _boneTransforms[QudmiConstants.HeadJoint] != null)
            {
                _boneTransforms[QudmiConstants.HeadJoint].localScale = Vector3.one * 0.0001f;
            }
        }

        private void LateUpdate()
        {
            if (_engine == null)
            {
                return;
            }

            if (headTransform == null || leftHandTransform == null || rightHandTransform == null)
            {
                AutoDetectTransformsIfNeeded();
                if (headTransform == null || leftHandTransform == null || rightHandTransform == null)
                {
                    return; // try again next frame
                }
            }

            _timeSinceLastSample += Time.deltaTime;
            float samplePeriod = 1f / targetSampleFps;
            if (_timeSinceLastSample >= samplePeriod)
            {
                _timeSinceLastSample %= samplePeriod; // carry over remainder rather than resetting to 0,
                                                       // so the average sampled rate stays accurate even
                                                       // when render rate isn't a clean multiple of it
                SampleAndPredict();
            }

            // Applied every rendered frame, not just on sample frames: inference runs at the
            // training rate (~30Hz) but the avatar should still move smoothly at headset refresh
            // rate, which the smoothing below interpolates toward.
            if (_hasPendingPose)
            {
                ApplyPose();
            }
        }

        private void SampleAndPredict()
        {
            _buffer.PushFrame(
                headTransform.position, headTransform.rotation,
                leftHandTransform.position, leftHandTransform.rotation,
                rightHandTransform.position, rightHandTransform.rotation);

            if (!_buffer.IsFull)
            {
                return; // still filling the initial window of history
            }

            if (!_calibrated && autoCalibrateDelay > 0f)
            {
                if (_trackingStartTime < 0f)
                {
                    _trackingStartTime = Time.time;
                    Debug.Log($"QudmiFullBodyDriver: auto-calibrating in {autoCalibrateDelay:F0}s -- " +
                        "stand upright, face forward, arms hanging down.", this);
                }
                else if (Time.time - _trackingStartTime >= autoCalibrateDelay)
                {
                    Calibrate();
                }
            }

            float[] features = _buffer.ComputeFeatures();
            if (ContainsNonFinite(features))
            {
                // Distinguishes "bad tracking data went in" from "the model/decode produced
                // something bad" -- e.g. if the XR device hasn't started delivering real poses
                // yet, this fires instead of the pose-side check below.
                Debug.LogWarning("QudmiFullBodyDriver: tracker input contains NaN/Infinity, skipping this frame.", this);
                return;
            }

            if (logInputDiagnostics)
            {
                LogInputDiagnostics(features);
            }

            float[] pose = _engine.Predict(features);

            if (!_loggedRawPoseDiagnostic && ContainsNonFinite(pose))
            {
                // One-time detailed dump: distinguishes "the model itself output NaN" from "the
                // decode math produced NaN from an otherwise-valid model output" -- the generic
                // warning below can't tell those apart, and they point at very different fixes
                // (inference backend/model import vs. PoseDecoder.cs).
                int nanCount = 0;
                foreach (float v in pose)
                {
                    if (float.IsNaN(v) || float.IsInfinity(v)) nanCount++;
                }
                int previewCount = System.Math.Min(6, pose.Length);
                var preview = new float[previewCount];
                System.Array.Copy(pose, preview, previewCount);
                Debug.LogWarning(
                    $"QudmiFullBodyDriver: raw model output already contains {nanCount}/{pose.Length} " +
                    $"non-finite values (first {previewCount}: {string.Join(", ", preview)}). " +
                    "This points at the model/inference backend, not PoseDecoder.", this);
                _loggedRawPoseDiagnostic = true;
            }

            PoseDecoder.DecodedPose decoded = PoseDecoder.Decode(
                pose, _buffer.AnchorHeadPosition, _buffer.AnchorHeadRotation, _buffer.PositionScale);

            if (!IsValid(decoded))
            {
                // Seen in practice before the XR device is actually delivering tracked poses
                // (e.g. mid Quest Link handshake): held-static/uninitialized input can push the
                // model into a degenerate output. Skip this frame rather than write NaN into
                // the rig -- silently freezing on the last good pose is far less destructive.
                Debug.LogWarning("QudmiFullBodyDriver: discarding a non-finite pose prediction this frame.", this);
                return;
            }

            _pendingPose = decoded;
            _hasPendingPose = true;
        }

        /// <summary>
        /// Writes the predicted pose onto the rig, retargeting from SMPL's skeleton convention to
        /// this particular rig's.
        ///
        /// The retargeting is not optional bookkeeping -- it's required for correctness. SMPL's
        /// FK (see src/qudmi/data/smplh.py) builds the rest pose purely from joint *offsets* with
        /// no rest rotations, so every SMPL joint's rest frame is the identity, i.e. aligned with
        /// the world axes. A typical rig (Mixamo, most DCC exports) instead has each bone's rest
        /// frame aligned along the bone. Feeding SMPL local rotations straight in therefore
        /// applies each rotation in the wrong frame; the visible symptom is a plausibly-articulated
        /// but completely wrong pose (a curled-up body that bends oddly on small head movements),
        /// not an obvious crash.
        ///
        /// So: accumulate SMPL locals into global rotations G_j, then place each bone at
        /// G_j * inverse(S_j) * B_j, where B_j is that bone's captured bind-pose orientation and
        /// S_j is SMPL's orientation for the same physical pose the rig is bound in (standing --
        /// see SmplStandingReference). The inverse(S_j) term is what makes the two rest poses
        /// comparable; without it the avatar ends up ~165 degrees over-rotated, i.e. lying down.
        /// At the standing pose G_j equals S_j and the rig lands exactly on its bind pose.
        ///
        /// World rotations are assigned (rather than local ones) so that rigs with extra
        /// intermediate bones -- Mixamo's Spine/Spine1/Spine2 vs. SMPL's three spine joints --
        /// don't need the two hierarchies to correspond one-to-one. QudmiConstants.Parents is
        /// ordered parents-before-children, so writing a parent first and then overwriting the
        /// child's world rotation converges to the right result.
        /// </summary>
        private void ApplyPose()
        {
            if (!_hasPreviousPose)
            {
                _smoothedRotations = _pendingPose.LocalRotations;
                _smoothedRootPosition = _pendingPose.RootPositionUnity;
                _hasPreviousPose = true;
            }
            else
            {
                for (int j = 0; j < QudmiConstants.NumBodyJoints; j++)
                {
                    _smoothedRotations[j] = Quaternion.Slerp(_pendingPose.LocalRotations[j], _smoothedRotations[j], smoothing);
                }
                _smoothedRootPosition = Vector3.Lerp(_pendingPose.RootPositionUnity, _smoothedRootPosition, smoothing);
            }

            // Index 0 already carries the world-recomposed root rotation from PoseDecoder.
            _globalRotations[0] = _smoothedRotations[0];
            for (int j = 1; j < QudmiConstants.NumBodyJoints; j++)
            {
                _globalRotations[j] = _globalRotations[QudmiConstants.Parents[j]] * _smoothedRotations[j];
            }

            if (_boneTransforms[0] != null)
            {
                _boneTransforms[0].position = _smoothedRootPosition;
            }

            for (int j = 0; j < QudmiConstants.NumBodyJoints; j++)
            {
                Transform bone = _boneTransforms[j];
                if (bone != null)
                {
                    // Relative to SMPL's *standing* reference, not its identity rest pose -- see
                    // SmplStandingReference for why those differ by ~165 degrees and why using
                    // the latter laid the avatar on the floor.
                    bone.rotation = _globalRotations[j]
                        * Quaternion.Inverse(SmplStandingReference.Rotations[j])
                        * _restWorldRotations[j];
                }
            }

            // Re-anchor only after the rotations are in, since they determine where the head bone
            // actually ends up. Translating the root is rigid, so this moves the whole rig without
            // disturbing the pose.
            if (anchorHeadToTracker && headTransform != null)
            {
                Transform root = _boneTransforms[QudmiConstants.RootJoint];
                Transform head = _boneTransforms[QudmiConstants.HeadJoint];
                if (root != null && head != null)
                {
                    root.position += headTransform.position - head.position;
                }
            }
        }

        /// <summary>
        /// Dumps the anchor (most recent) frame of the canonicalized input. Compare against the
        /// training distribution: head should sit at (0, ~1.3-1.7, 0) by construction, and with
        /// arms hanging down the wrists sit near (-0.14, 0.84, 0.05) and (+0.13, 0.86, -0.01).
        /// Live values far outside that mean the model is receiving input it never saw in
        /// training -- the most likely reason being that a VR controller's pose is not the same
        /// thing as the SMPL wrist joint pose the model was trained on.
        /// </summary>
        private void LogInputDiagnostics(float[] features)
        {
            _diagnosticTimer += Time.deltaTime;
            if (_diagnosticTimer < 1f)
            {
                return;
            }
            _diagnosticTimer = 0f;

            int anchor = (QudmiConstants.WindowLength - 1) * QudmiConstants.InputDim;
            string[] names = { "head  ", "Lwrist", "Rwrist" };
            var sb = new System.Text.StringBuilder("QudmiFullBodyDriver live input (canonicalized, metres):");
            for (int t = 0; t < QudmiConstants.NumTrackers; t++)
            {
                int b = anchor + t * QudmiConstants.FeaturesPerTracker;
                sb.Append($"\n  {names[t]} pos=({features[b]:F3}, {features[b + 1]:F3}, {features[b + 2]:F3})")
                  .Append($"  rot6d=[{features[b + 3]:F2}, {features[b + 4]:F2}, {features[b + 5]:F2}, ")
                  .Append($"{features[b + 6]:F2}, {features[b + 7]:F2}, {features[b + 8]:F2}]");
            }
            Debug.Log(sb.ToString(), this);
        }

        private static bool IsValid(PoseDecoder.DecodedPose pose)
        {
            if (!IsFinite(pose.RootPositionUnity))
            {
                return false;
            }
            foreach (Quaternion q in pose.LocalRotations)
            {
                if (float.IsNaN(q.x) || float.IsNaN(q.y) || float.IsNaN(q.z) || float.IsNaN(q.w) ||
                    float.IsInfinity(q.x) || float.IsInfinity(q.y) || float.IsInfinity(q.z) || float.IsInfinity(q.w))
                {
                    return false;
                }
            }
            return true;
        }

        private static bool IsFinite(Vector3 v) =>
            !float.IsNaN(v.x) && !float.IsNaN(v.y) && !float.IsNaN(v.z) &&
            !float.IsInfinity(v.x) && !float.IsInfinity(v.y) && !float.IsInfinity(v.z);

        private static bool ContainsNonFinite(float[] values)
        {
            foreach (float v in values)
            {
                if (float.IsNaN(v) || float.IsInfinity(v))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Resolves the tracked transforms by name, retried every frame until it succeeds rather
        /// than running once at startup.
        ///
        /// The retry matters: XRI's XRInputModalityManager deactivates the controller GameObjects
        /// on startup and only re-enables whichever set matches the detected input mode
        /// (controllers vs. hand tracking). GameObject.Find ignores inactive objects, so a
        /// one-shot lookup in Awake is a race against that manager -- it succeeds or silently
        /// fails depending on script execution order, leaving the avatar frozen with only the
        /// head resolved (Camera.main is always active, the controllers are not). Retrying picks
        /// them up the moment they're enabled, and also covers controllers powered on late.
        /// </summary>
        private void AutoDetectTransformsIfNeeded()
        {
            if (headTransform == null && Camera.main != null)
            {
                headTransform = Camera.main.transform;
            }

            if (leftHandTransform == null)
            {
                leftHandTransform = FindByCommonName("LeftHand Controller", "Left Controller", "LeftHand");
            }

            if (rightHandTransform == null)
            {
                rightHandTransform = FindByCommonName("RightHand Controller", "Right Controller", "RightHand");
            }

            bool resolved = headTransform != null && leftHandTransform != null && rightHandTransform != null;

            if (resolved && _loggedAutoDetectWarning)
            {
                Debug.Log("QudmiFullBodyDriver: tracked transforms resolved, driving avatar now.", this);
                _loggedAutoDetectWarning = false;
            }
            else if (!resolved && !_loggedAutoDetectWarning)
            {
                // Logged once, not every frame, since this is an expected transient state while
                // the XR rig starts up rather than necessarily a misconfiguration.
                Debug.LogWarning(
                    "QudmiFullBodyDriver: waiting on tracked transforms " +
                    $"(head={(headTransform != null ? "ok" : "missing")}, " +
                    $"left={(leftHandTransform != null ? "ok" : "missing")}, " +
                    $"right={(rightHandTransform != null ? "ok" : "missing")}). " +
                    "Retrying each frame; assign them manually in the Inspector if this persists.", this);
                _loggedAutoDetectWarning = true;
            }
        }

        private static Transform FindByCommonName(params string[] names)
        {
            foreach (string name in names)
            {
                GameObject found = GameObject.Find(name);
                if (found != null)
                {
                    return found.transform;
                }
            }
            return null;
        }
    }
}
