using UnityEngine;

namespace Qudmi
{
    /// <summary>
    /// Maps VR device orientations onto the SMPL joint orientations the model was actually
    /// trained on. A tracker's pose is not the body joint's pose, and the gap is not small.
    ///
    /// Measured over 10,187 standing/arms-down frames of the training set, the canonical
    /// orientation of each tracked joint sits 165 degrees (head), 112 (left wrist) and 87 (right
    /// wrist) away from identity -- while a Unity XR rig reports very nearly identity for the
    /// same physical pose. Feeding device rotations in unmodified therefore misstates every
    /// orientation input by 90-165 degrees on every single frame, which yields a confidently
    /// wrong pose rather than a noisy one, and which no amount of retraining can fix. DTP (Liu
    /// et al., Virtual Reality 28:116, sec 3.2) addresses the same problem with a T-pose
    /// calibration step; this is the equivalent.
    ///
    /// The correction is a constant per-tracker rotation applied on the *right* (in the sensor's
    /// own frame), so it commutes with the left-multiplied yaw canonicalization in
    /// TrackerWindowBuffer rather than fighting it.
    /// </summary>
    public static class TrackerCalibration
    {
        /// <summary>
        /// The canonical-frame orientation each tracker should report when the wearer stands
        /// upright with arms down: the training set's mean rotation for those frames, projected
        /// back onto SO(3). Given as matrix columns in the pipeline's (right-handed) space.
        /// </summary>
        public static readonly Quaternion[] StandingReference =
        {
            // head
            FromColumns(new Vector3(-0.942f, +0.070f, +0.329f),
                        new Vector3(-0.336f, -0.244f, -0.910f),
                        new Vector3(+0.016f, -0.967f, +0.254f)),
            // left wrist
            FromColumns(new Vector3(-0.202f, -0.904f, -0.378f),
                        new Vector3(+0.221f, +0.334f, -0.916f),
                        new Vector3(+0.954f, -0.268f, +0.132f)),
            // right wrist
            FromColumns(new Vector3(+0.096f, +0.926f, +0.365f),
                        new Vector3(-0.699f, +0.324f, -0.637f),
                        new Vector3(-0.709f, -0.194f, +0.678f)),
        };

        /// <summary>
        /// Median head height of the training set (metres). Live input is scaled toward this so
        /// that users taller or shorter than the mocap subjects still land inside the
        /// distribution the model saw -- AMASS subjects in the current training mix top out
        /// around 1.65m, so a 1.8m user is otherwise off the end of it.
        /// </summary>
        public const float ReferenceHeadHeight = 1.42f;

        /// <summary>
        /// Offsets used before the wearer calibrates. Assumes the rig reports ~identity canonical
        /// rotations while standing, which is what a Unity XR head camera does; controllers whose
        /// grip convention differs will still want an explicit Calibrate().
        /// </summary>
        public static Quaternion[] DefaultOffsets() => (Quaternion[])StandingReference.Clone();

        /// <summary>
        /// offset_j = measured_j^-1 * reference_j, so that applying the offset at the calibration
        /// instant reproduces the training reference exactly.
        /// </summary>
        public static Quaternion[] ComputeOffsets(Quaternion[] canonicalMeasured)
        {
            var offsets = new Quaternion[StandingReference.Length];
            for (int i = 0; i < offsets.Length; i++)
            {
                offsets[i] = Quaternion.Inverse(canonicalMeasured[i]) * StandingReference[i];
            }
            return offsets;
        }

        /// <summary>Column convention matches Rotation6D.Decode: forward = column 2, up = column 1.</summary>
        private static Quaternion FromColumns(Vector3 column0, Vector3 column1, Vector3 column2)
        {
            return Quaternion.LookRotation(column2, column1);
        }
    }
}
