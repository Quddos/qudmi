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

        private Animator _animator;
        private TrackerWindowBuffer _buffer;
        private IInferenceEngine _engine;
        private Quaternion[] _smoothedRotations;
        private Vector3 _smoothedRootPosition;
        private bool _hasPreviousPose;
        private bool _hasPendingPose;
        private bool _loggedRawPoseDiagnostic;
        private bool _loggedAutoDetectWarning;
        private float _timeSinceLastSample;
        private PoseDecoder.DecodedPose _pendingPose;
        private Transform[] _boneTransforms;
        private Quaternion[] _restWorldRotations;
        private Quaternion[] _globalRotations;

        private void Awake()
        {
            _animator = GetComponent<Animator>();
            _buffer = new TrackerWindowBuffer();
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

            float[] features = _buffer.ComputeFeatures();
            if (ContainsNonFinite(features))
            {
                // Distinguishes "bad tracking data went in" from "the model/decode produced
                // something bad" -- e.g. if the XR device hasn't started delivering real poses
                // yet, this fires instead of the pose-side check below.
                Debug.LogWarning("QudmiFullBodyDriver: tracker input contains NaN/Infinity, skipping this frame.", this);
                return;
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

            PoseDecoder.DecodedPose decoded = PoseDecoder.Decode(pose, _buffer.AnchorHeadPosition, _buffer.AnchorHeadRotation);

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
        /// G_j * B_j, where B_j is that bone's captured bind-pose orientation. Left-multiplying by
        /// G_j applies the model's rotation in world space, about the bone's own rest orientation.
        /// With G_j = identity this reduces to the untouched bind pose, as it should.
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
                    bone.rotation = _globalRotations[j] * _restWorldRotations[j];
                }
            }
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
