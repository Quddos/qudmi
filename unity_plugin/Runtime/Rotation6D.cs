using UnityEngine;

namespace Qudmi
{
    /// <summary>
    /// Mirrors src/qudmi/data/rotations.py's matrix_to_rot6d / rot6d_to_matrix exactly: the
    /// model is trained on the 6D continuous rotation representation (Zhou et al., 2019), not
    /// quaternions or Euler angles, because those have discontinuities that hurt regression.
    /// Column convention matches the Python side: column 0 and column 1 of the rotation matrix
    /// (i.e. "right" and "up"), concatenated.
    /// </summary>
    public static class Rotation6D
    {
        public static void Encode(Quaternion q, float[] outSixValues, int offset)
        {
            Matrix4x4 r = Matrix4x4.Rotate(q);
            Vector3 col0 = r.GetColumn(0);
            Vector3 col1 = r.GetColumn(1);
            outSixValues[offset + 0] = col0.x;
            outSixValues[offset + 1] = col0.y;
            outSixValues[offset + 2] = col0.z;
            outSixValues[offset + 3] = col1.x;
            outSixValues[offset + 4] = col1.y;
            outSixValues[offset + 5] = col1.z;
        }

        public static Quaternion Decode(float[] sixValues, int offset)
        {
            Vector3 a1 = new Vector3(sixValues[offset + 0], sixValues[offset + 1], sixValues[offset + 2]);
            Vector3 a2 = new Vector3(sixValues[offset + 3], sixValues[offset + 4], sixValues[offset + 5]);

            Vector3 b1 = a1.normalized;
            Vector3 b2 = (a2 - Vector3.Dot(b1, a2) * b1).normalized;
            Vector3 b3 = Vector3.Cross(b1, b2);

            // Columns are [b1, b2, b3] = [right, up, forward] -- reconstruct via forward/up
            // rather than a hand-rolled matrix->quaternion algorithm, same reasoning as
            // CoordinateConversion.ConjugateRotation.
            return Quaternion.LookRotation(b3, b2);
        }
    }
}
