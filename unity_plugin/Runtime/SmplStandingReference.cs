using UnityEngine;

namespace Qudmi
{
    /// <summary>
    /// The global orientation of each SMPL joint for a standing, arms-down pose, averaged over
    /// 10,187 such frames of the training set and projected back onto SO(3).
    ///
    /// This exists because SMPL's own rest pose is *not* the same physical configuration as a
    /// Unity rig's bind pose, so retargeting cannot treat the two as interchangeable.
    /// src/qudmi/data/smplh.py rotates rest joint *positions* by the Z-up to Y-up matrix while
    /// leaving rest *orientations* at identity. That is self-consistent for AMASS -- each
    /// sequence's root orientation absorbs the difference -- but it means the identity pose in
    /// the pipeline's frame is a body lying on its back. A Mixamo-style rig's bind pose is
    /// standing upright. Retargeting straight off SMPL's identity rest therefore applies an
    /// extra ~165 degree "stand the body up" rotation to a rig that is already standing, and
    /// tips the avatar onto the floor -- which is exactly what it did.
    ///
    /// Measured deviation from identity per joint runs 124-178 degrees, so this is not a
    /// rounding detail; every joint needs it.
    ///
    /// Used as: worldRotation_j = G_j * inverse(StandingReference_j) * B_j, where G_j is the
    /// model's accumulated global rotation and B_j the rig's captured bind orientation. At the
    /// standing pose G_j equals StandingReference_j, so the rig lands exactly on its bind pose.
    /// </summary>
    public static class SmplStandingReference
    {
        public static readonly Quaternion[] Rotations =
        {
            FromColumns(new Vector3(-0.9011f, -0.0004f, +0.4337f), new Vector3(-0.4336f, +0.0253f, -0.9008f), new Vector3(-0.0106f, -0.9997f, -0.0230f)),  // 0  pelvis
            FromColumns(new Vector3(-0.9261f, +0.0174f, +0.3769f), new Vector3(-0.3732f, +0.1050f, -0.9218f), new Vector3(-0.0556f, -0.9943f, -0.0908f)),  // 1  L_hip
            FromColumns(new Vector3(-0.8791f, -0.0612f, +0.4727f), new Vector3(-0.4765f, +0.0962f, -0.8739f), new Vector3(+0.0080f, -0.9935f, -0.1138f)),  // 2  R_hip
            FromColumns(new Vector3(-0.9025f, +0.0237f, +0.4301f), new Vector3(-0.4296f, -0.1205f, -0.8949f), new Vector3(+0.0306f, -0.9924f, +0.1189f)),  // 3  spine1
            FromColumns(new Vector3(-0.8995f, -0.0597f, +0.4327f), new Vector3(-0.4261f, -0.0984f, -0.8993f), new Vector3(+0.0963f, -0.9934f, +0.0630f)),  // 4  L_knee
            FromColumns(new Vector3(-0.8951f, -0.0406f, +0.4441f), new Vector3(-0.4359f, -0.1305f, -0.8905f), new Vector3(+0.0941f, -0.9906f, +0.0991f)),  // 5  R_knee
            FromColumns(new Vector3(-0.9072f, -0.0220f, +0.4201f), new Vector3(-0.4144f, -0.1247f, -0.9015f), new Vector3(+0.0722f, -0.9920f, +0.1040f)),  // 6  spine2
            FromColumns(new Vector3(-0.8973f, +0.1051f, +0.4288f), new Vector3(-0.4315f, -0.0028f, -0.9021f), new Vector3(-0.0936f, -0.9945f, +0.0479f)),  // 7  L_ankle
            FromColumns(new Vector3(-0.9099f, -0.2066f, +0.3597f), new Vector3(-0.3636f, -0.0203f, -0.9313f), new Vector3(+0.1997f, -0.9782f, -0.0567f)),  // 8  R_ankle
            FromColumns(new Vector3(-0.9067f, -0.0370f, +0.4201f), new Vector3(-0.4139f, -0.1130f, -0.9033f), new Vector3(+0.0808f, -0.9929f, +0.0871f)),  // 9  spine3
            FromColumns(new Vector3(-0.9127f, +0.1728f, +0.3702f), new Vector3(-0.3559f, +0.1086f, -0.9282f), new Vector3(-0.2006f, -0.9790f, -0.0377f)),  // 10 L_foot
            FromColumns(new Vector3(-0.8677f, -0.1804f, +0.4632f), new Vector3(-0.4626f, -0.0476f, -0.8853f), new Vector3(+0.1817f, -0.9824f, -0.0421f)),  // 11 R_foot
            FromColumns(new Vector3(-0.9141f, -0.0453f, +0.4030f), new Vector3(-0.3960f, -0.1149f, -0.9110f), new Vector3(+0.0876f, -0.9923f, +0.0870f)),  // 12 neck
            FromColumns(new Vector3(-0.6916f, +0.0033f, +0.7223f), new Vector3(-0.7119f, -0.1723f, -0.6808f), new Vector3(+0.1222f, -0.9850f, +0.1215f)),  // 13 L_collar
            FromColumns(new Vector3(-0.9983f, -0.0030f, +0.0581f), new Vector3(-0.0571f, -0.1356f, -0.9891f), new Vector3(+0.0108f, -0.9908f, +0.1352f)),  // 14 R_collar
            FromColumns(new Vector3(-0.9246f, -0.0681f, +0.3748f), new Vector3(-0.3591f, -0.1724f, -0.9172f), new Vector3(+0.1271f, -0.9827f, +0.1350f)),  // 15 head
            FromColumns(new Vector3(+0.2142f, +0.0479f, +0.9756f), new Vector3(-0.9235f, -0.3153f, +0.2183f), new Vector3(+0.3181f, -0.9478f, -0.0233f)),  // 16 L_shoulder
            FromColumns(new Vector3(-0.6058f, -0.0555f, -0.7937f), new Vector3(+0.7646f, -0.3164f, -0.5615f), new Vector3(-0.2200f, -0.9470f, +0.2341f)),  // 17 R_shoulder
            FromColumns(new Vector3(+0.2732f, -0.3392f, +0.9002f), new Vector3(-0.8651f, -0.4959f, +0.0757f), new Vector3(+0.4207f, -0.7994f, -0.4289f)),  // 18 L_elbow
            FromColumns(new Vector3(-0.5396f, +0.3764f, -0.7531f), new Vector3(+0.6598f, -0.3667f, -0.6559f), new Vector3(-0.5230f, -0.8508f, -0.0504f)),  // 19 R_elbow
            FromColumns(new Vector3(+0.3111f, -0.4010f, +0.8616f), new Vector3(-0.8749f, -0.4749f, +0.0949f), new Vector3(+0.3712f, -0.7834f, -0.4986f)),  // 20 L_wrist
            FromColumns(new Vector3(-0.4433f, +0.4313f, -0.7858f), new Vector3(+0.7343f, -0.3281f, -0.5943f), new Vector3(-0.5142f, -0.8404f, -0.1713f)),  // 21 R_wrist
        };

        /// <summary>
        /// Column convention matches Rotation6D.Decode: forward = column 2, up = column 1.
        ///
        /// The conjugation is required, not cosmetic. These values were measured in the Python
        /// pipeline's right-handed space, but they are consumed in ApplyPose alongside
        /// PoseDecoder's output, which has already been converted to Unity's left-handed space.
        /// Leaving them unconverted mixes the two handednesses in a single product and flips the
        /// avatar upside down -- legs up, head through the floor.
        /// </summary>
        private static Quaternion FromColumns(Vector3 column0, Vector3 column1, Vector3 column2)
        {
            return CoordinateConversion.ConjugateRotation(Quaternion.LookRotation(column2, column1));
        }
    }
}
