using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace Qudmi.Editor
{
    /// <summary>
    /// One-click scene setup with validation, opened from Window > Qudmi > Setup Wizard.
    ///
    /// Exists because every problem encountered bringing this up on real hardware was a *setup*
    /// problem, not a code problem: a leftover Main Camera making Camera.main ambiguous, an
    /// Animator Controller fighting the driver, a rig imported as Generic instead of Humanoid.
    /// None of them failed loudly. A checklist that detects each one and offers to fix it is
    /// worth more to a new user than any amount of documentation telling them not to make those
    /// mistakes.
    /// </summary>
    public class QudmiSetupWizard : EditorWindow
    {
        private GameObject _avatar;
        private Vector2 _scroll;

        [MenuItem("Window/Qudmi/Setup Wizard")]
        public static void Open()
        {
            GetWindow<QudmiSetupWizard>("Qudmi Setup").minSize = new Vector2(420, 460);
        }

        private void OnGUI()
        {
            _scroll = EditorGUILayout.BeginScrollView(_scroll);

            EditorGUILayout.LabelField("Qudmi Full Body — scene setup", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Drives a Humanoid avatar's whole body from a VR headset and two controllers. " +
                "Pick the character to drive, fix anything flagged below, then apply.",
                MessageType.None);
            EditorGUILayout.Space();

            _avatar = (GameObject)EditorGUILayout.ObjectField("Avatar", _avatar, typeof(GameObject), true);
            EditorGUILayout.Space();

            List<string> blockers = new List<string>();
            DrawModelCheck(blockers);
            DrawAvatarChecks(blockers);
            DrawSceneChecks();

            EditorGUILayout.Space();
            using (new EditorGUI.DisabledScope(blockers.Count > 0))
            {
                if (GUILayout.Button("Set up avatar", GUILayout.Height(30)))
                {
                    ApplySetup();
                }
            }

            if (blockers.Count > 0)
            {
                EditorGUILayout.HelpBox("Resolve the errors above first:\n• " + string.Join("\n• ", blockers),
                    MessageType.Warning);
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("After setup", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Press Play, then stand upright facing forward with your arms hanging down for " +
                "about 5 seconds. Calibration fits itself automatically and logs when it's done.\n\n" +
                "You are inside the avatar, so you won't see it — add a preview clone below to " +
                "watch it from the outside.",
                MessageType.Info);

            using (new EditorGUI.DisabledScope(_avatar == null))
            {
                if (GUILayout.Button("Add a preview clone in front of the player"))
                {
                    AddPreviewClone();
                }
            }

            EditorGUILayout.EndScrollView();
        }

        private void DrawModelCheck(List<string> blockers)
        {
            if (QudmiModelInstaller.FindInstalledModel() != null)
            {
                return;
            }

            EditorGUILayout.HelpBox(
                "The model weights aren't in this project yet. They're distributed separately from " +
                "the package because they're licensed for non-commercial research use only, while " +
                "the package code is MIT.",
                MessageType.Warning);
            if (GUILayout.Button("Download model weights (~19.6 MB)"))
            {
                QudmiModelInstaller.DownloadWithConsent();
            }
            blockers.Add("Model weights not installed.");
            EditorGUILayout.Space();
        }

        private void DrawAvatarChecks(List<string> blockers)
        {
            if (_avatar == null)
            {
                EditorGUILayout.HelpBox("Drag in the character you want driven.", MessageType.Info);
                blockers.Add("No avatar selected.");
                return;
            }

            var animator = _avatar.GetComponent<Animator>();
            if (animator == null)
            {
                EditorGUILayout.HelpBox("This object has no Animator component.", MessageType.Error);
                blockers.Add("Avatar has no Animator.");
            }
            else if (!animator.isHuman)
            {
                EditorGUILayout.HelpBox(
                    "This rig is not Humanoid. Select the model file, then Rig > Animation Type > " +
                    "Humanoid > Apply. Qudmi maps SMPL joints onto Unity's humanoid bones, so a " +
                    "Generic rig cannot be driven.",
                    MessageType.Error);
                blockers.Add("Rig is not Humanoid.");
            }
            else
            {
                EditorGUILayout.HelpBox("Humanoid rig detected.", MessageType.Info);

                if (animator.runtimeAnimatorController != null)
                {
                    EditorGUILayout.HelpBox(
                        "An Animator Controller is assigned. Qudmi writes bone transforms directly, " +
                        "so anything the Controller plays will fight it. Setup will clear it.",
                        MessageType.Warning);
                }
            }
        }

        private void DrawSceneChecks()
        {
            int mainCameras = 0;
            foreach (Camera cam in FindObjectsByType<Camera>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            {
                if (cam.CompareTag("MainCamera"))
                {
                    mainCameras++;
                }
            }

            if (mainCameras > 1)
            {
                EditorGUILayout.HelpBox(
                    $"{mainCameras} cameras are tagged MainCamera. Head auto-detection uses " +
                    "Camera.main, which is ambiguous with more than one — it may pick a camera " +
                    "that isn't tracked. Delete the leftover default Main Camera if you added an " +
                    "XR rig prefab.",
                    MessageType.Warning);
            }
            else if (mainCameras == 0)
            {
                EditorGUILayout.HelpBox(
                    "No camera tagged MainCamera. Add an XR rig (e.g. XR Interaction Toolkit's " +
                    "\"XR Origin (XR Rig)\"), or assign the head transform manually on the component.",
                    MessageType.Warning);
            }
        }

        private void ApplySetup()
        {
            var animator = _avatar.GetComponent<Animator>();
            Undo.RecordObject(animator, "Qudmi setup");
            animator.runtimeAnimatorController = null;

            var driver = _avatar.GetComponent<QudmiFullBodyDriver>();
            if (driver == null)
            {
                driver = Undo.AddComponent<QudmiFullBodyDriver>(_avatar);
            }

            SerializedObject so = new SerializedObject(driver);
            SerializedProperty modelProp = so.FindProperty("modelAsset");
            if (modelProp != null && modelProp.objectReferenceValue == null)
            {
                Object model = QudmiModelInstaller.FindInstalledModel();
                if (model != null)
                {
                    modelProp.objectReferenceValue = model;
                    so.ApplyModifiedProperties();
                }
                else
                {
                    Debug.LogWarning("Qudmi: model weights not found. Use Window > Qudmi > " +
                        "Download Model Weights, then assign it on the component.", driver);
                }
            }

            EditorUtility.SetDirty(_avatar);
            Selection.activeGameObject = _avatar;
            Debug.Log("Qudmi: avatar set up. Press Play and stand still, arms down, for ~5s to calibrate.", _avatar);
        }


        private void AddPreviewClone()
        {
            GameObject clone = Instantiate(_avatar);
            clone.name = _avatar.name + " (Qudmi Preview)";
            Undo.RegisterCreatedObjectUndo(clone, "Add Qudmi preview clone");

            if (clone.TryGetComponent(out QudmiFullBodyDriver cloneDriver))
            {
                DestroyImmediate(cloneDriver);
            }
            clone.AddComponent<QudmiAvatarPreview>();

            // Two metres ahead of the player, turned to face them.
            Camera head = Camera.main;
            Vector3 origin = head != null ? head.transform.position : Vector3.zero;
            clone.transform.position = new Vector3(origin.x, 0f, origin.z) + Vector3.forward * 2f;
            clone.transform.rotation = Quaternion.Euler(0f, 180f, 0f);

            Selection.activeGameObject = clone;
            Debug.Log("Qudmi: preview clone added in front of the player.", clone);
        }
    }
}
