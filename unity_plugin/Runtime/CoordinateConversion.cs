using UnityEngine;

namespace Qudmi
{
    /// <summary>
    /// AMASS/this project's training pipeline uses a right-handed Y-up convention (see the
    /// Z-up -> Y-up fix in src/qudmi/data/smplh.py); Unity is left-handed Y-up. Every position
    /// and rotation crossing the Python/Unity boundary needs this conversion, in either
    /// direction -- both operations below are self-inverse (applying them twice is a no-op),
    /// so the same functions work whether converting Unity input into the model's space or the
    /// model's predicted output into Unity's space.
    ///
    /// Derivation: position conversion is a single-axis flip (negate Z). Rotation conversion is
    /// conjugation by F = diag(1, 1, -1): R_other = F * R * F. Both are worked out in full,
    /// with the algebra checked against a concrete yaw-rotation example, in the module
    /// docstring of scripts/generate_unity_parity_fixture.py -- that script is also what
    /// generates Tests/Fixtures/parity_case_1.json, which ParityTests.cs checks this file's
    /// functions against numerically rather than trusting the derivation alone.
    /// </summary>
    public static class CoordinateConversion
    {
        public static Vector3 FlipZ(Vector3 v) => new Vector3(v.x, v.y, -v.z);

        public static Quaternion ConjugateRotation(Quaternion q)
        {
            Matrix4x4 r = Matrix4x4.Rotate(q);
            Vector3 col1 = r.GetColumn(1); // "up"
            Vector3 col2 = r.GetColumn(2); // "forward"

            Vector3 newCol1 = new Vector3(col1.x, col1.y, -col1.z);
            Vector3 newCol2 = new Vector3(-col2.x, -col2.y, col2.z);

            // Reconstruct via forward/up rather than hand-rolling a matrix->quaternion
            // (trace-based) algorithm -- the conjugated matrix is a valid proper rotation
            // (det = det(F)*det(R)*det(F) = (-1)(1)(-1) = +1), so its columns are consistent
            // with what LookRotation expects.
            return Quaternion.LookRotation(newCol2, newCol1);
        }
    }
}
