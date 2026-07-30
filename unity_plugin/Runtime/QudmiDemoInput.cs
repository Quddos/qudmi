using UnityEngine;

namespace Qudmi
{
    /// <summary>
    /// Moves stand-in head and hand transforms through a looping motion, so the demo shows the
    /// model working with no VR hardware attached.
    ///
    /// Worth having beyond convenience: it lets someone evaluate Qudmi's output before owning a
    /// headset or setting up an XR rig, and it gives a deterministic input signal for eyeballing
    /// changes to the runtime, which live tracking cannot (you cannot repeat a hand movement
    /// exactly). Not a substitute for real tracking -- the motion here is invented, not measured.
    /// </summary>
    [AddComponentMenu("Qudmi/Qudmi Demo Input (simulated tracking)")]
    public class QudmiDemoInput : MonoBehaviour
    {
        [SerializeField] private Transform head;
        [SerializeField] private Transform leftHand;
        [SerializeField] private Transform rightHand;

        [Tooltip("Standing head height in metres. AMASS subjects sit around 1.4-1.65m, so values " +
            "far outside that are outside what the model was trained on.")]
        [SerializeField] private float headHeight = 1.6f;

        [SerializeField] private float speed = 0.6f;

        private void Update()
        {
            float t = Time.time * speed;

            if (head != null)
            {
                // Slight bob and sway, plus a slow look left and right.
                head.position = new Vector3(Mathf.Sin(t * 0.5f) * 0.05f,
                                            headHeight + Mathf.Sin(t * 2f) * 0.02f,
                                            0f);
                head.rotation = Quaternion.Euler(Mathf.Sin(t * 0.7f) * 8f, Mathf.Sin(t * 0.4f) * 35f, 0f);
            }

            // Arms swing between hanging down and raised in front, which is the range that shows
            // whether the shoulder/elbow chain is behaving.
            float swing = (Mathf.Sin(t) + 1f) * 0.5f;   // 0..1

            if (leftHand != null)
            {
                leftHand.position = new Vector3(-0.25f - swing * 0.05f,
                                                headHeight - 0.75f + swing * 0.55f,
                                                swing * 0.35f);
                leftHand.rotation = Quaternion.Euler(0f, 0f, swing * 40f);
            }

            if (rightHand != null)
            {
                float offswing = 1f - swing;   // opposite phase, as in a natural walk
                rightHand.position = new Vector3(0.25f + offswing * 0.05f,
                                                 headHeight - 0.75f + offswing * 0.55f,
                                                 offswing * 0.35f);
                rightHand.rotation = Quaternion.Euler(0f, 0f, -offswing * 40f);
            }
        }
    }
}
