using UnityEditor;
using UnityEngine;

namespace Qudmi.Editor
{
    /// <summary>
    /// Adds setup validation on top of the default Inspector so integration mistakes (non-
    /// Humanoid rig, missing model) are caught at edit time rather than as a silent no-op at
    /// runtime -- everything here is configuration, no code required from the integrator.
    /// </summary>
    [CustomEditor(typeof(QudmiFullBodyDriver))]
    public class QudmiFullBodyDriverEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var driver = (QudmiFullBodyDriver)target;
            var animator = driver.GetComponent<Animator>();

            if (animator == null)
            {
                EditorGUILayout.HelpBox(
                    "No Animator found on this GameObject. QudmiFullBodyDriver requires one with a Humanoid avatar.",
                    MessageType.Error);
            }
            else if (!animator.isHuman)
            {
                EditorGUILayout.HelpBox(
                    "This Animator's avatar is not configured as Humanoid. Set the model's Rig type to " +
                    "Humanoid in its import settings, or full-body pose can't be applied.",
                    MessageType.Error);
            }

            DrawDefaultInspector();

            EditorGUILayout.Space();
            using (new EditorGUI.DisabledScope(!Application.isPlaying))
            {
                if (GUILayout.Button("Calibrate (stand upright, arms down)", GUILayout.Height(28)))
                {
                    driver.Calibrate();
                }
            }

            if (!Application.isPlaying)
            {
                EditorGUILayout.HelpBox(
                    "Calibration runs against live tracking, so it's only available in Play mode. " +
                    "Baked defaults are used until then -- see TrackerCalibration.",
                    MessageType.Info);
            }
        }
    }
}
