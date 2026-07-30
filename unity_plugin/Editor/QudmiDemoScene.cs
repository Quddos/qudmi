using UnityEditor;
using UnityEngine;

namespace Qudmi.Editor
{
    /// <summary>
    /// Builds a working demo scene on demand, from Window > Qudmi > Create Demo Scene.
    ///
    /// Generated rather than shipped as a committed .unity file, for two reasons. A scene asset
    /// hard-codes GUID references to the package's own assets, which is fragile across installs and
    /// silently produces missing references rather than an error. And a scene cannot be shipped
    /// with a character in it at all: Mixamo rigs are Adobe's and Unity's sample rigs are Unity's,
    /// so neither is ours to redistribute -- hence QudmiDemoAvatar generating one instead.
    ///
    /// Building it in code also means the demo cannot drift out of date the way a committed scene
    /// would: it is constructed from the same components a user would add by hand.
    /// </summary>
    public static class QudmiDemoScene
    {
        [MenuItem("Window/Qudmi/Create Demo Scene")]
        public static void Create()
        {
            var root = new GameObject("Qudmi Demo");
            Undo.RegisterCreatedObjectUndo(root, "Create Qudmi demo scene");

            CreateGround(root.transform);
            CreateLight(root.transform);

            // Stand-in tracked transforms. A real integration points the driver at an XR rig's
            // camera and controllers instead; these exist so the demo runs with no hardware.
            var rig = new GameObject("Simulated Tracking");
            rig.transform.SetParent(root.transform, false);
            Transform head = CreateMarker("Head (simulated)", rig.transform, 0.1f);
            Transform leftHand = CreateMarker("Left Hand (simulated)", rig.transform, 0.07f);
            Transform rightHand = CreateMarker("Right Hand (simulated)", rig.transform, 0.07f);

            var input = rig.AddComponent<QudmiDemoInput>();
            SetPrivate(input, "head", head);
            SetPrivate(input, "leftHand", leftHand);
            SetPrivate(input, "rightHand", rightHand);

            GameObject avatar = QudmiDemoAvatar.Create();
            avatar.transform.SetParent(root.transform, true);

            var driver = avatar.AddComponent<QudmiFullBodyDriver>();
            SetPrivate(driver, "headTransform", head);
            SetPrivate(driver, "leftHandTransform", leftHand);
            SetPrivate(driver, "rightHandTransform", rightHand);
            SetPrivate(driver, "modelAsset", FindModel());
            // The wearer isn't literally inside this avatar, so keep its head visible.
            SetPrivate(driver, "hideHeadInFirstPerson", false);

            CreateViewingCamera(root.transform);

            Selection.activeGameObject = avatar;
            Debug.Log("Qudmi: demo scene created. Press Play -- the avatar is driven by simulated " +
                "head/hand motion, so no headset is needed. Save the scene if you want to keep it.", avatar);
        }

        private static void CreateGround(Transform parent)
        {
            GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.SetParent(parent, false);
            ground.transform.localScale = Vector3.one * 2f;
        }

        private static void CreateLight(Transform parent)
        {
            var go = new GameObject("Directional Light");
            go.transform.SetParent(parent, false);
            go.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
            Light light = go.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.1f;
        }

        private static void CreateViewingCamera(Transform parent)
        {
            // Only added when the scene has no camera, so this never fights an XR rig the user
            // has already set up.
            if (Camera.main != null)
            {
                return;
            }

            var go = new GameObject("Demo Camera");
            go.tag = "MainCamera";
            go.transform.SetParent(parent, false);
            go.transform.position = new Vector3(0f, 1.4f, 3f);
            go.transform.rotation = Quaternion.Euler(5f, 180f, 0f);
            go.AddComponent<Camera>();
        }

        private static Transform CreateMarker(string name, Transform parent, float size)
        {
            GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localScale = Vector3.one * size;
            Object.DestroyImmediate(go.GetComponent<Collider>());
            return go.transform;
        }

        private static Object FindModel()
        {
            foreach (string guid in AssetDatabase.FindAssets("qudmi_v0"))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                if (path.EndsWith(".onnx"))
                {
                    return AssetDatabase.LoadMainAssetAtPath(path);
                }
            }
            Debug.LogWarning("Qudmi: bundled model not found; assign it manually on the driver.");
            return null;
        }

        /// <summary>
        /// Assigns a [SerializeField] private field. Used instead of widening those fields to
        /// public just so the editor can reach them -- the inspector-facing API stays as intended.
        /// </summary>
        private static void SetPrivate(Object target, string field, Object value)
        {
            var so = new SerializedObject(target);
            SerializedProperty prop = so.FindProperty(field);
            if (prop == null)
            {
                Debug.LogWarning($"Qudmi: no serialized field '{field}' on {target.GetType().Name}.");
                return;
            }
            prop.objectReferenceValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void SetPrivate(Object target, string field, bool value)
        {
            var so = new SerializedObject(target);
            SerializedProperty prop = so.FindProperty(field);
            if (prop != null)
            {
                prop.boolValue = value;
                so.ApplyModifiedPropertiesWithoutUndo();
            }
        }
    }
}
