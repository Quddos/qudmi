using NUnit.Framework;
using UnityEngine;

namespace Qudmi.Tests
{
    /// <summary>
    /// Verifies the C# runtime math against reference values computed in Python (see
    /// scripts/generate_unity_parity_fixture.py) rather than trusting the C# port "by
    /// construction" -- this is specifically to catch handedness-conversion mistakes
    /// (AMASS/Python is right-handed Y-up, Unity is left-handed Y-up), which would silently
    /// produce a plausible-looking but wrong pose rather than an obvious crash.
    /// </summary>
    public class ParityTests
    {
        [System.Serializable]
        private class Fixture
        {
            public int window;
            public float[] headPosFlat, headQuatFlat;
            public float[] leftHandPosFlat, leftHandQuatFlat;
            public float[] rightHandPosFlat, rightHandQuatFlat;
            public float[] expectedFeaturesFlat;
            public float[] syntheticPose135;
            public float[] expectedLocalQuatsFlat;
            public float[] expectedRootQuat;
            public float[] expectedRootTranslation;
        }

        private const float Tolerance = 1e-3f;

        private static Fixture Load()
        {
            TextAsset json = Resources.Load<TextAsset>("parity_case_1");
            Assert.IsNotNull(json, "Tests/Resources/parity_case_1.json not found -- run " +
                "scripts/generate_unity_parity_fixture.py");
            return JsonUtility.FromJson<Fixture>(json.text);
        }

        private static Vector3 ReadVec3(float[] flat, int frame) =>
            new Vector3(flat[frame * 3], flat[frame * 3 + 1], flat[frame * 3 + 2]);

        private static Quaternion ReadQuat(float[] flat, int frame) =>
            new Quaternion(flat[frame * 4], flat[frame * 4 + 1], flat[frame * 4 + 2], flat[frame * 4 + 3]);

        [Test]
        public void InputEncoding_MatchesPythonReference()
        {
            Fixture f = Load();
            var buffer = new TrackerWindowBuffer(f.window);

            for (int t = 0; t < f.window; t++)
            {
                buffer.PushFrame(
                    ReadVec3(f.headPosFlat, t), ReadQuat(f.headQuatFlat, t),
                    ReadVec3(f.leftHandPosFlat, t), ReadQuat(f.leftHandQuatFlat, t),
                    ReadVec3(f.rightHandPosFlat, t), ReadQuat(f.rightHandQuatFlat, t));
            }

            Assert.IsTrue(buffer.IsFull);
            float[] computed = buffer.ComputeFeatures();

            Assert.AreEqual(f.expectedFeaturesFlat.Length, computed.Length);
            for (int i = 0; i < computed.Length; i++)
            {
                Assert.AreEqual(f.expectedFeaturesFlat[i], computed[i], Tolerance,
                    $"feature mismatch at flat index {i} (frame {i / QudmiConstants.InputDim}, " +
                    $"offset {i % QudmiConstants.InputDim})");
            }
        }

        [Test]
        public void OutputDecode_LocalJointRotations_MatchPythonReference()
        {
            Fixture f = Load();
            // Anchor doesn't matter for joints 1-21 (frame-invariant, see PoseDecoder), so any
            // fixed anchor works here -- use the fixture's actual anchor for realism.
            Vector3 anchorPos = ReadVec3(f.headPosFlat, f.window - 1);
            // headPosFlat is Unity-space; PoseDecoder expects python-space, so convert first.
            anchorPos = CoordinateConversion.FlipZ(anchorPos);
            Quaternion anchorRot = CoordinateConversion.ConjugateRotation(ReadQuat(f.headQuatFlat, f.window - 1));

            PoseDecoder.DecodedPose decoded = PoseDecoder.Decode(f.syntheticPose135, anchorPos, anchorRot);

            for (int j = 1; j < QudmiConstants.NumBodyJoints; j++)
            {
                Quaternion expected = ReadQuat(f.expectedLocalQuatsFlat, j - 1);
                AssertQuaternionsMatch(expected, decoded.LocalRotations[j], $"joint {j}");
            }
        }

        [Test]
        public void OutputDecode_RootRecomposition_MatchesPythonReference()
        {
            Fixture f = Load();
            Vector3 anchorPos = CoordinateConversion.FlipZ(ReadVec3(f.headPosFlat, f.window - 1));
            Quaternion anchorRot = CoordinateConversion.ConjugateRotation(ReadQuat(f.headQuatFlat, f.window - 1));

            PoseDecoder.DecodedPose decoded = PoseDecoder.Decode(f.syntheticPose135, anchorPos, anchorRot);

            Quaternion expectedRootQuat = new Quaternion(
                f.expectedRootQuat[0], f.expectedRootQuat[1], f.expectedRootQuat[2], f.expectedRootQuat[3]);
            AssertQuaternionsMatch(expectedRootQuat, decoded.LocalRotations[0], "root rotation");

            Vector3 expectedRootPos = new Vector3(
                f.expectedRootTranslation[0], f.expectedRootTranslation[1], f.expectedRootTranslation[2]);
            Assert.AreEqual(expectedRootPos.x, decoded.RootPositionUnity.x, Tolerance);
            Assert.AreEqual(expectedRootPos.y, decoded.RootPositionUnity.y, Tolerance);
            Assert.AreEqual(expectedRootPos.z, decoded.RootPositionUnity.z, Tolerance);
        }

        private static void AssertQuaternionsMatch(Quaternion expected, Quaternion actual, string label)
        {
            // Quaternions q and -q represent the same rotation; Dot ~ +-1 both mean "match".
            float dot = Mathf.Abs(Quaternion.Dot(expected, actual));
            Assert.Greater(dot, 1f - 1e-3f, $"{label}: expected {expected}, got {actual} (dot={dot})");
        }
    }
}
