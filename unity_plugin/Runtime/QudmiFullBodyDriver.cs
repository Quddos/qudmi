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
    /// The Animator's Controller should be empty/idle -- this component fully drives the
    /// humanoid pose every frame via Animator.SetBoneLocalRotation, which only takes effect
    /// reliably when called after the Animator's own update, hence LateUpdate below.
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
        private float _timeSinceLastSample;

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
            float[] pose = _engine.Predict(features);
            PoseDecoder.DecodedPose decoded = PoseDecoder.Decode(pose, _buffer.AnchorHeadPosition, _buffer.AnchorHeadRotation);

            ApplyPose(decoded);
        }

        private void ApplyPose(PoseDecoder.DecodedPose decoded)
        {
            if (!_hasPreviousPose)
            {
                _smoothedRotations = decoded.LocalRotations;
                _smoothedRootPosition = decoded.RootPositionUnity;
                _hasPreviousPose = true;
            }
            else
            {
                for (int j = 0; j < QudmiConstants.NumBodyJoints; j++)
                {
                    _smoothedRotations[j] = Quaternion.Slerp(decoded.LocalRotations[j], _smoothedRotations[j], smoothing);
                }
                _smoothedRootPosition = Vector3.Lerp(decoded.RootPositionUnity, _smoothedRootPosition, smoothing);
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
