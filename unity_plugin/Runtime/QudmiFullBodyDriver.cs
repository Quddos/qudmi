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
    /// The Animator needs a Controller assigned -- not None -- with "IK Pass" enabled on its
    /// base layer. That's what makes Unity actually invoke OnAnimatorIK below, which is the
    /// *only* context Animator.SetBoneLocalRotation is supported in (confirmed the hard way:
    /// calling it from LateUpdate produces Unity's own "should only be done in OnAnimatorIK or
    /// OnStateIK" warning and, worse, NaN bone positions after a few frames). An empty
    /// Controller with just IK Pass on is enough; it doesn't need any actual states/clips since
    /// this component fully drives the pose.
    /// </summary>
    [AddComponentMenu("Qudmi/Qudmi Full Body Driver")]
    [RequireComponent(typeof(Animator))]
    public class QudmiFullBodyDriver : MonoBehaviour
    {
        [SerializeField] private ModelAsset modelAsset;
        [SerializeField] private BackendType backend = BackendType.GPUCompute;

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
        private float _timeSinceLastSample;
        private PoseDecoder.DecodedPose _pendingPose;

        private void Awake()
        {
            _animator = GetComponent<Animator>();
            _buffer = new TrackerWindowBuffer();

            if (modelAsset != null)
            {
                _engine = new SentisInferenceEngine(modelAsset, backend);
            }
            else
            {
                Debug.LogWarning("QudmiFullBodyDriver: no model assigned, this component will do nothing.", this);
            }

            AutoDetectTransformsIfNeeded();
        }

        private void OnDestroy()
        {
            (_engine as System.IDisposable)?.Dispose();
        }

        private void LateUpdate()
        {
            if (_engine == null || headTransform == null || leftHandTransform == null || rightHandTransform == null)
            {
                return;
            }

            _timeSinceLastSample += Time.deltaTime;
            float samplePeriod = 1f / targetSampleFps;
            if (_timeSinceLastSample < samplePeriod)
            {
                return; // not yet time for the next ~30fps sample
            }
            _timeSinceLastSample %= samplePeriod; // carry over remainder rather than resetting to 0,
                                                   // so the average sampled rate stays accurate even
                                                   // when render rate isn't a clean multiple of it

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
        /// The only place Animator.SetBoneLocalRotation is actually supported -- see the class
        /// remarks above. Only called at all if the Animator's Controller has IK Pass enabled.
        /// </summary>
        private void OnAnimatorIK(int layerIndex)
        {
            if (!_hasPendingPose)
            {
                return;
            }

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

            for (int j = 0; j < QudmiConstants.NumBodyJoints; j++)
            {
                _animator.SetBoneLocalRotation(QudmiConstants.HumanBodyBoneForJoint[j], _smoothedRotations[j]);
            }

            Transform hips = _animator.GetBoneTransform(HumanBodyBones.Hips);
            if (hips != null)
            {
                hips.position = _smoothedRootPosition;
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

            if (headTransform == null || leftHandTransform == null || rightHandTransform == null)
            {
                Debug.LogWarning(
                    "QudmiFullBodyDriver: couldn't auto-detect all tracked transforms. " +
                    "Assign Head/Left Hand/Right Hand manually in the Inspector.", this);
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
