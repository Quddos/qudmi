using UnityEngine;

namespace Qudmi
{
    /// <summary>
    /// Copies a driven avatar's pose onto a second rig, so the wearer can watch their own body
    /// from the outside.
    ///
    /// This is not only a demo nicety. A first-person VR avatar is the one thing its wearer
    /// cannot see: the camera sits inside the head, so judging whether the pose is any good
    /// otherwise means taking the headset off and looking at the Scene view. Placing a clone in
    /// front of the wearer makes the output continuously visible, which is what any real
    /// evaluation of pose quality needs.
    ///
    /// Copies local rotations rather than re-running the retargeting, which is exact when both
    /// rigs share a bind pose (i.e. the same character model, the normal case for a preview) and
    /// costs nothing. For a *different* character, drive it with its own QudmiFullBodyDriver
    /// instead, so it gets retargeting fitted to its own rest pose.
    ///
    /// Note this is a clone, not a mirror: it reproduces the pose rather than reflecting it, so
    /// raising your right hand raises the clone's right hand. Facing the clone toward the wearer
    /// then makes that appear on the wearer's left, exactly as it would with another person
    /// standing opposite -- which is what you want when checking how the avatar reads to others.
    /// </summary>
    [AddComponentMenu("Qudmi/Qudmi Avatar Preview")]
    [RequireComponent(typeof(Animator))]
    [DefaultExecutionOrder(200)] // after QudmiFullBodyDriver's LateUpdate has posed the source
    public class QudmiAvatarPreview : MonoBehaviour
    {
        [Tooltip("The avatar being driven by a QudmiFullBodyDriver. Left empty, the first one in " +
            "the scene is used.")]
        [SerializeField] private Animator sourceAvatar;

        [Tooltip("Keeps the clone's lowest foot on the ground plane it was placed at. The driven " +
            "avatar solves the equivalent problem by anchoring to the tracked headset, but a " +
            "preview has no tracked target, so without this it sinks or floats by however much " +
            "the predicted hip height is off.")]
        [SerializeField] private bool keepFeetOnGround = true;

        // Ankles and toes. Toes are optional in Unity's humanoid rig, hence the null checks.
        private static readonly int[] FootJoints = { 7, 8, 10, 11 };

        private Animator _animator;
        private Transform[] _sourceBones;
        private Transform[] _targetBones;
        private Vector3 _hipsBindLocalPosition;
        private float _groundY;
        private bool _warned;

        private void Awake()
        {
            _animator = GetComponent<Animator>();

            if (sourceAvatar == null)
            {
                QudmiFullBodyDriver driver = FindAnyObjectByType<QudmiFullBodyDriver>();
                if (driver != null)
                {
                    sourceAvatar = driver.GetComponent<Animator>();
                }
            }

            if (sourceAvatar == null)
            {
                Debug.LogWarning("QudmiAvatarPreview: no source avatar found or assigned; " +
                    "this component will do nothing.", this);
                return;
            }

            _sourceBones = new Transform[QudmiConstants.NumBodyJoints];
            _targetBones = new Transform[QudmiConstants.NumBodyJoints];
            for (int j = 0; j < QudmiConstants.NumBodyJoints; j++)
            {
                HumanBodyBones bone = QudmiConstants.HumanBodyBoneForJoint[j];
                _sourceBones[j] = sourceAvatar.GetBoneTransform(bone);
                _targetBones[j] = _animator.GetBoneTransform(bone);
            }

            _groundY = transform.position.y;
            Transform hips = _targetBones[QudmiConstants.RootJoint];
            if (hips != null)
            {
                _hipsBindLocalPosition = hips.localPosition;
            }
        }

        private void LateUpdate()
        {
            if (_sourceBones == null)
            {
                return;
            }

            if (!_animator.isHuman && !_warned)
            {
                Debug.LogWarning("QudmiAvatarPreview: this rig is not Humanoid, so its bones " +
                    "can't be matched to the source.", this);
                _warned = true;
                return;
            }

            Transform hips = _targetBones[QudmiConstants.RootJoint];

            // Reset before re-grounding so the offset is recomputed from a known state each frame
            // rather than accumulated on top of the previous frame's correction.
            if (hips != null)
            {
                hips.localPosition = _hipsBindLocalPosition;
            }

            for (int j = 0; j < QudmiConstants.NumBodyJoints; j++)
            {
                if (_sourceBones[j] != null && _targetBones[j] != null)
                {
                    // Local, not world: this keeps the clone's pose relative to its own root, so
                    // it stays where it was placed and faces whichever way it was turned.
                    _targetBones[j].localRotation = _sourceBones[j].localRotation;
                }
            }

            if (keepFeetOnGround && hips != null)
            {
                // Measured after the rotations are applied, since those decide where the feet
                // actually end up.
                float lowest = float.MaxValue;
                foreach (int j in FootJoints)
                {
                    if (_targetBones[j] != null)
                    {
                        lowest = Mathf.Min(lowest, _targetBones[j].position.y);
                    }
                }

                if (lowest < float.MaxValue)
                {
                    hips.position += Vector3.up * (_groundY - lowest);
                }
            }
        }
    }
}
