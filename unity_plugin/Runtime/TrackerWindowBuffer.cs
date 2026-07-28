using UnityEngine;

namespace Qudmi
{
    /// <summary>
    /// Maintains a rolling window of tracker frames and produces the canonicalized
    /// (window * InputDim) feature buffer the model expects -- mirrors
    /// src/qudmi/data/window_features.py's canonicalize_window() exactly. Input is raw
    /// Unity-space (left-handed) transforms, e.g. straight from OpenXR head/hand tracking;
    /// conversion to the model's right-handed training convention (CoordinateConversion)
    /// happens internally, once, per frame pushed.
    /// </summary>
    public class TrackerWindowBuffer
    {
        private readonly int _window;
        private readonly Vector3[][] _positions; // [frameIndex][trackerIndex], python-space
        private readonly Quaternion[][] _rotations; // same indexing, python-space
        private int _count;

        public TrackerWindowBuffer(int window = QudmiConstants.WindowLength)
        {
            _window = window;
            _positions = new Vector3[window][];
            _rotations = new Quaternion[window][];
            for (int i = 0; i < window; i++)
            {
                _positions[i] = new Vector3[QudmiConstants.NumTrackers];
                _rotations[i] = new Quaternion[QudmiConstants.NumTrackers];
            }
            _count = 0;
        }

        public bool IsFull => _count >= _window;

        /// <summary>
        /// Per-tracker rotation corrections, applied on the right so they compose cleanly with the
        /// left-multiplied yaw canonicalization. Null (the default) means no correction, which is
        /// what the Python-parity tests exercise; QudmiFullBodyDriver supplies real values from
        /// TrackerCalibration. See TrackerCalibration for why this is needed at all.
        /// </summary>
        public Quaternion[] RotationOffsets { get; set; }

        /// <summary>
        /// Uniform scale applied to canonicalized positions (and hence velocities), used to map a
        /// wearer's body scale into the range the model was trained on.
        /// </summary>
        public float PositionScale { get; set; } = 1f;

        /// <summary>
        /// The uncorrected canonical rotation of each tracker at the current frame -- the input
        /// TrackerCalibration.ComputeOffsets needs in order to solve for the offsets.
        /// </summary>
        public Quaternion[] CurrentCanonicalRotations()
        {
            Quaternion invYaw = InverseYaw(AnchorHeadRotation);
            var result = new Quaternion[QudmiConstants.NumTrackers];
            for (int tr = 0; tr < QudmiConstants.NumTrackers; tr++)
            {
                result[tr] = invYaw * _rotations[_window - 1][tr];
            }
            return result;
        }

        /// <summary>The current (most recent) frame's head position/rotation, in python-space
        /// (post-conversion). Exposed so the pose decoder can recompose the root joint's
        /// world-space transform using the same anchor this window was canonicalized against.</summary>
        public Vector3 AnchorHeadPosition => _positions[_window - 1][0];
        public Quaternion AnchorHeadRotation => _rotations[_window - 1][0];

        /// <summary>Push one frame of raw Unity-space tracker transforms (head, left hand, right hand).</summary>
        public void PushFrame(Vector3 headPos, Quaternion headRot, Vector3 lHandPos, Quaternion lHandRot,
            Vector3 rHandPos, Quaternion rHandRot)
        {
            // Shift older frames back (index 0 = oldest, index window-1 = current), matching
            // the ascending-time ordering process_sequence builds its windows in.
            for (int i = 0; i < _window - 1; i++)
            {
                _positions[i] = _positions[i + 1];
                _rotations[i] = _rotations[i + 1];
            }

            var pos = new Vector3[QudmiConstants.NumTrackers];
            var rot = new Quaternion[QudmiConstants.NumTrackers];
            pos[0] = CoordinateConversion.FlipZ(headPos);
            rot[0] = CoordinateConversion.ConjugateRotation(headRot);
            pos[1] = CoordinateConversion.FlipZ(lHandPos);
            rot[1] = CoordinateConversion.ConjugateRotation(lHandRot);
            pos[2] = CoordinateConversion.FlipZ(rHandPos);
            rot[2] = CoordinateConversion.ConjugateRotation(rHandRot);

            _positions[_window - 1] = pos;
            _rotations[_window - 1] = rot;

            if (_count < _window) _count++;
        }

        /// <summary>
        /// Computes the canonicalized (window * InputDim) feature buffer, flattened in
        /// [frame][tracker][feature] order matching the Python side's reshape.
        /// </summary>
        public float[] ComputeFeatures()
        {
            Vector3 anchorHeadPosHoriz = AnchorHeadPosition;
            anchorHeadPosHoriz.y = 0f;

            Quaternion invYaw = InverseYaw(AnchorHeadRotation);

            var canonPos = new Vector3[_window][];
            var canonRot6D = new float[_window][];
            for (int t = 0; t < _window; t++)
            {
                canonPos[t] = new Vector3[QudmiConstants.NumTrackers];
                canonRot6D[t] = new float[QudmiConstants.NumTrackers * 6];
                for (int tr = 0; tr < QudmiConstants.NumTrackers; tr++)
                {
                    Vector3 shifted = _positions[t][tr] - anchorHeadPosHoriz;
                    // Scaling here also scales the velocities below, since those are differences
                    // of these same canonical positions.
                    canonPos[t][tr] = (invYaw * shifted) * PositionScale;

                    Quaternion canonRot = invYaw * _rotations[t][tr];
                    if (RotationOffsets != null)
                    {
                        canonRot *= RotationOffsets[tr];
                    }
                    Rotation6D.Encode(canonRot, canonRot6D[t], tr * 6);
                }
            }

            var features = new float[_window * QudmiConstants.InputDim];
            for (int t = 0; t < _window; t++)
            {
                for (int tr = 0; tr < QudmiConstants.NumTrackers; tr++)
                {
                    Vector3 vel = (t == 0)
                        // No earlier frame to diff against; backward-fill from frame 1's
                        // velocity, matching process_sequence/canonicalize_window exactly.
                        ? canonPos[System.Math.Min(1, _window - 1)][tr] - canonPos[0][tr]
                        : canonPos[t][tr] - canonPos[t - 1][tr];

                    int baseIdx = t * QudmiConstants.InputDim + tr * QudmiConstants.FeaturesPerTracker;
                    features[baseIdx + 0] = canonPos[t][tr].x;
                    features[baseIdx + 1] = canonPos[t][tr].y;
                    features[baseIdx + 2] = canonPos[t][tr].z;
                    features[baseIdx + 3] = canonRot6D[t][tr * 6 + 0];
                    features[baseIdx + 4] = canonRot6D[t][tr * 6 + 1];
                    features[baseIdx + 5] = canonRot6D[t][tr * 6 + 2];
                    features[baseIdx + 6] = canonRot6D[t][tr * 6 + 3];
                    features[baseIdx + 7] = canonRot6D[t][tr * 6 + 4];
                    features[baseIdx + 8] = canonRot6D[t][tr * 6 + 5];
                    features[baseIdx + 9] = vel.x;
                    features[baseIdx + 10] = vel.y;
                    features[baseIdx + 11] = vel.z;
                }
            }
            return features;
        }

        /// <summary>
        /// inv_yaw = transpose(R_y(yaw)) = R_y(-yaw), where yaw = atan2(forward.x, forward.z)
        /// -- mirrors rotations.py's yaw_only_matrix exactly. Built explicitly from R_y's
        /// known column values (forward = (sin θ, 0, cos θ), up = (0, 1, 0)) rather than via
        /// Quaternion.AngleAxis, to avoid depending on whether Unity's angle-sign convention
        /// happens to match the right-handed formula this mirrors -- see CoordinateConversion.
        /// </summary>
        private static Quaternion InverseYaw(Quaternion rot)
        {
            Vector3 forward = rot * Vector3.forward; // column 2 of the rotation matrix
            float yaw = Mathf.Atan2(forward.x, forward.z);
            float negYaw = -yaw;
            Vector3 invForward = new Vector3(Mathf.Sin(negYaw), 0f, Mathf.Cos(negYaw));
            return Quaternion.LookRotation(invForward, Vector3.up);
        }
    }
}
