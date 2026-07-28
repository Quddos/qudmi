using UnityEngine;

namespace Qudmi
{
    /// <summary>
    /// Decodes the model's 135-dim output (22 joints x 6D rotation + 3D root translation, all
    /// in the canonicalized/head-yaw-removed frame -- see docs/SPEC.md) into Unity-space local
    /// rotations ready to apply to a Humanoid rig, plus the root's world-space position.
    ///
    /// Joints 1-21 are parent-relative (local) rotations: frame-invariant by construction (they
    /// come straight from the AMASS pose array untouched by canonicalization in amass.py), so
    /// decoding them is just "6D -> quaternion -> Unity space". Joint 0 (root) is different: the
    /// model predicts it *in the canonicalized frame*, so recovering the true world-space root
    /// pose requires composing the current anchor's yaw back in. Getting this step wrong would
    /// leave the avatar facing/positioned relative to some fixed arbitrary direction rather than
    /// tracking the real headset -- verified against Tests/Fixtures/parity_case_1.json's
    /// expected_unity_root_quaternion_xyzw / expected_unity_root_translation.
    /// </summary>
    public static class PoseDecoder
    {
        public struct DecodedPose
        {
            public Quaternion[] LocalRotations; // length NumBodyJoints; index 0 is world-recomposed root
            public Vector3 RootPositionUnity;
        }

        /// <param name="positionScale">The scale TrackerWindowBuffer applied to the input; the
        /// predicted root translation comes back in that same scaled space and has to be divided
        /// out before it is combined with the (unscaled) real-world anchor position.</param>
        public static DecodedPose Decode(float[] pose135, Vector3 anchorHeadPositionPython,
            Quaternion anchorHeadRotationPython, float positionScale = 1f)
        {
            Vector3 anchorHeadPosHoriz = anchorHeadPositionPython;
            anchorHeadPosHoriz.y = 0f;
            Quaternion yaw = ForwardYaw(anchorHeadRotationPython);

            var rotations = new Quaternion[QudmiConstants.NumBodyJoints];
            for (int j = 0; j < QudmiConstants.NumBodyJoints; j++)
            {
                Quaternion canonicalRot = Rotation6D.Decode(pose135, j * 6);
                Quaternion worldRotPython = (j == 0) ? yaw * canonicalRot : canonicalRot;
                rotations[j] = CoordinateConversion.ConjugateRotation(worldRotPython);
            }

            Vector3 rootTransCanonical = new Vector3(pose135[132], pose135[133], pose135[134]);
            if (positionScale > 0f)
            {
                rootTransCanonical /= positionScale;
            }
            Vector3 worldTransPython = yaw * rootTransCanonical + anchorHeadPosHoriz;

            return new DecodedPose
            {
                LocalRotations = rotations,
                RootPositionUnity = CoordinateConversion.FlipZ(worldTransPython),
            };
        }

        /// <summary>R_y(yaw) -- the forward (non-inverse) counterpart of
        /// TrackerWindowBuffer.InverseYaw, used to recompose the root's world rotation.</summary>
        private static Quaternion ForwardYaw(Quaternion rot)
        {
            Vector3 forward = rot * Vector3.forward;
            float yaw = Mathf.Atan2(forward.x, forward.z);
            Vector3 fwd = new Vector3(Mathf.Sin(yaw), 0f, Mathf.Cos(yaw));
            return Quaternion.LookRotation(fwd, Vector3.up);
        }
    }
}
