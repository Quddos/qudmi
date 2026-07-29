namespace Qudmi
{
    /// <summary>
    /// Mirrors constants from the Python training pipeline (src/qudmi/data/amass.py,
    /// src/qudmi/data/smplh.py) exactly. If these drift from the Python side, the model's
    /// input/output shapes silently stop matching what it was trained on.
    /// </summary>
    public static class QudmiConstants
    {
        public const int NumBodyJoints = 22; // SMPL+H body joints only (0-21), no fingers.
        public const int WindowLength = 40;  // frames of history the model expects per inference.
        public const int FeaturesPerTracker = 12; // position(3) + 6D rotation(6) + velocity(3)
        public const int NumTrackers = 3; // head, left wrist, right wrist (default config)
        public const int InputDim = NumTrackers * FeaturesPerTracker; // 36
        public const int OutputDim = NumBodyJoints * 6 + 3; // 22 * 6D rotation + root translation = 135
        public const int RootJoint = 0;
        public const int HeadJoint = 15;

        // Joint index layout, standard SMPL/SMPL+H order (docs/SPEC.md / smplh.py):
        // 0 pelvis(root), 1 L_hip, 2 R_hip, 3 spine1, 4 L_knee, 5 R_knee, 6 spine2, 7 L_ankle,
        // 8 R_ankle, 9 spine3, 10 L_foot, 11 R_foot, 12 neck, 13 L_collar, 14 R_collar, 15 head,
        // 16 L_shoulder, 17 R_shoulder, 18 L_elbow, 19 R_elbow, 20 L_wrist, 21 R_wrist
        public static readonly int[] Parents =
        {
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
        };

        /// <summary>
        /// SMPL+H joint index -> Unity HumanBodyBones, for applying predicted local rotations
        /// to a Humanoid-rigged Animator. Fingers/toes are intentionally left unmapped (SPEC.md:
        /// hand articulation is out of scope for v0).
        /// </summary>
        public static readonly UnityEngine.HumanBodyBones[] HumanBodyBoneForJoint =
        {
            UnityEngine.HumanBodyBones.Hips,
            UnityEngine.HumanBodyBones.LeftUpperLeg,
            UnityEngine.HumanBodyBones.RightUpperLeg,
            UnityEngine.HumanBodyBones.Spine,
            UnityEngine.HumanBodyBones.LeftLowerLeg,
            UnityEngine.HumanBodyBones.RightLowerLeg,
            UnityEngine.HumanBodyBones.Chest,
            UnityEngine.HumanBodyBones.LeftFoot,
            UnityEngine.HumanBodyBones.RightFoot,
            UnityEngine.HumanBodyBones.UpperChest,
            UnityEngine.HumanBodyBones.LeftToes,
            UnityEngine.HumanBodyBones.RightToes,
            UnityEngine.HumanBodyBones.Neck,
            UnityEngine.HumanBodyBones.LeftShoulder,
            UnityEngine.HumanBodyBones.RightShoulder,
            UnityEngine.HumanBodyBones.Head,
            UnityEngine.HumanBodyBones.LeftUpperArm,
            UnityEngine.HumanBodyBones.RightUpperArm,
            UnityEngine.HumanBodyBones.LeftLowerArm,
            UnityEngine.HumanBodyBones.RightLowerArm,
            UnityEngine.HumanBodyBones.LeftHand,
            UnityEngine.HumanBodyBones.RightHand,
        };
    }
}
