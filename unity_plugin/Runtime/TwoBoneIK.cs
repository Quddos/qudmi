using UnityEngine;

namespace Qudmi
{
    /// <summary>
    /// Analytic two-bone IK, used to place the avatar's hands exactly on the tracked controllers.
    ///
    /// Worth doing because the hand position is *known*: it is a model input. The network
    /// predicts joint rotations, though, and nothing constrains forward kinematics to put the
    /// wrist back where the controller actually was -- error accumulates along
    /// spine -> collar -> shoulder -> elbow -> wrist. Measured on the TotalCapture test split,
    /// reconstructed wrists land ~107mm from ground truth, which is plainly visible when you
    /// tuck a hand to your chest and the avatar's arm doesn't follow.
    ///
    /// So: let the model decide the pose, then correct the arms to agree with what is measured.
    /// The current elbow position is used as the pole hint, which keeps the elbow swing the model
    /// chose (that part genuinely is a prediction) and only fixes the wrist.
    /// </summary>
    public static class TwoBoneIK
    {
        /// <param name="upper">shoulder/upper-arm bone</param>
        /// <param name="lower">elbow/forearm bone</param>
        /// <param name="end">wrist bone</param>
        /// <param name="target">world position the wrist should reach</param>
        public static void Solve(Transform upper, Transform lower, Transform end, Vector3 target)
        {
            Vector3 a = upper.position;
            Vector3 b = lower.position;
            Vector3 c = end.position;

            float upperLength = Vector3.Distance(a, b);
            float lowerLength = Vector3.Distance(b, c);
            if (upperLength <= 1e-5f || lowerLength <= 1e-5f)
            {
                return;
            }

            // Clamped just inside full extension and full fold: at exactly those limits the
            // triangle degenerates and the rotation axis below becomes undefined.
            float reach = Vector3.Distance(a, target);
            reach = Mathf.Clamp(reach, Mathf.Abs(upperLength - lowerLength) + 1e-4f, upperLength + lowerLength - 1e-4f);

            float currentSpan = Vector3.Distance(a, c);
            if (currentSpan <= 1e-5f)
            {
                return;
            }

            float currentElbow = LawOfCosines(upperLength, lowerLength, currentSpan);
            float targetElbow = LawOfCosines(upperLength, lowerLength, reach);
            float currentShoulder = LawOfCosines(upperLength, currentSpan, lowerLength);
            float targetShoulder = LawOfCosines(upperLength, reach, lowerLength);

            // Rotate within the arm's existing plane, so the elbow keeps pointing where the model
            // put it rather than snapping to some arbitrary default.
            Vector3 axis = Vector3.Cross(c - a, b - a);
            if (axis.sqrMagnitude < 1e-10f)
            {
                axis = Vector3.Cross(c - a, Vector3.up);
                if (axis.sqrMagnitude < 1e-10f)
                {
                    return;
                }
            }
            axis = axis.normalized;

            upper.rotation = Quaternion.AngleAxis((targetShoulder - currentShoulder) * Mathf.Rad2Deg, axis) * upper.rotation;
            lower.rotation = Quaternion.AngleAxis((targetElbow - currentElbow) * Mathf.Rad2Deg, axis) * lower.rotation;

            // Bend is now right but the arm may point the wrong way; swing the whole limb onto
            // the target.
            upper.rotation = Quaternion.FromToRotation(end.position - a, target - a) * upper.rotation;
        }

        /// <summary>Angle (radians) opposite <paramref name="opposite"/> in a triangle.</summary>
        private static float LawOfCosines(float adjacentA, float adjacentB, float opposite)
        {
            float cos = (adjacentA * adjacentA + adjacentB * adjacentB - opposite * opposite)
                        / (2f * adjacentA * adjacentB);
            return Mathf.Acos(Mathf.Clamp(cos, -1f, 1f));
        }
    }
}
