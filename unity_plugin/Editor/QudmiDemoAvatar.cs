using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace Qudmi.Editor
{
    /// <summary>
    /// Builds a simple primitive-shaped Humanoid avatar at runtime, so the package can demonstrate
    /// itself with no external assets.
    ///
    /// This exists for a licensing reason, not an aesthetic one. A sample scene needs a
    /// Humanoid-rigged character, but a package cannot redistribute one: Mixamo characters are
    /// Adobe's, Unity's sample rigs are Unity's, and shipping either would be handing on assets
    /// that are not ours to hand on. Generating the rig sidesteps that entirely -- it is ugly, but
    /// it is unambiguously distributable, and it lets someone see Qudmi working within a minute of
    /// installing rather than after sourcing and importing a character.
    /// </summary>
    public static class QudmiDemoAvatar
    {
        // Rough adult proportions in metres, T-pose (arms out along X). Unity's humanoid rig needs
        // 15 required bones; shoulders/chest/neck/toes are optional but included since they make
        // the retargeting visibly better.
        private struct BoneSpec
        {
            public string Name;
            public HumanBodyBones Bone;
            public string Parent;
            public Vector3 Position;   // world-space T-pose position
            public float Thickness;

            public BoneSpec(string name, HumanBodyBones bone, string parent, Vector3 position, float thickness)
            {
                Name = name; Bone = bone; Parent = parent; Position = position; Thickness = thickness;
            }
        }

        private static readonly BoneSpec[] Skeleton =
        {
            new BoneSpec("Hips",          HumanBodyBones.Hips,          null,           new Vector3(0f,     0.95f, 0f), 0.16f),
            new BoneSpec("Spine",         HumanBodyBones.Spine,         "Hips",         new Vector3(0f,     1.08f, 0f), 0.15f),
            new BoneSpec("Chest",         HumanBodyBones.Chest,         "Spine",        new Vector3(0f,     1.24f, 0f), 0.17f),
            new BoneSpec("Neck",          HumanBodyBones.Neck,          "Chest",        new Vector3(0f,     1.42f, 0f), 0.07f),
            new BoneSpec("Head",          HumanBodyBones.Head,          "Neck",         new Vector3(0f,     1.54f, 0f), 0.14f),

            new BoneSpec("LeftShoulder",  HumanBodyBones.LeftShoulder,  "Chest",        new Vector3(-0.06f, 1.38f, 0f), 0.07f),
            new BoneSpec("LeftUpperArm",  HumanBodyBones.LeftUpperArm,  "LeftShoulder", new Vector3(-0.18f, 1.38f, 0f), 0.08f),
            new BoneSpec("LeftLowerArm",  HumanBodyBones.LeftLowerArm,  "LeftUpperArm", new Vector3(-0.45f, 1.38f, 0f), 0.07f),
            new BoneSpec("LeftHand",      HumanBodyBones.LeftHand,      "LeftLowerArm", new Vector3(-0.70f, 1.38f, 0f), 0.06f),

            new BoneSpec("RightShoulder", HumanBodyBones.RightShoulder, "Chest",        new Vector3(0.06f,  1.38f, 0f), 0.07f),
            new BoneSpec("RightUpperArm", HumanBodyBones.RightUpperArm, "RightShoulder",new Vector3(0.18f,  1.38f, 0f), 0.08f),
            new BoneSpec("RightLowerArm", HumanBodyBones.RightLowerArm, "RightUpperArm",new Vector3(0.45f,  1.38f, 0f), 0.07f),
            new BoneSpec("RightHand",     HumanBodyBones.RightHand,     "RightLowerArm",new Vector3(0.70f,  1.38f, 0f), 0.06f),

            new BoneSpec("LeftUpperLeg",  HumanBodyBones.LeftUpperLeg,  "Hips",         new Vector3(-0.09f, 0.90f, 0f), 0.11f),
            new BoneSpec("LeftLowerLeg",  HumanBodyBones.LeftLowerLeg,  "LeftUpperLeg", new Vector3(-0.09f, 0.50f, 0f), 0.09f),
            new BoneSpec("LeftFoot",      HumanBodyBones.LeftFoot,      "LeftLowerLeg", new Vector3(-0.09f, 0.08f, 0f), 0.08f),
            new BoneSpec("LeftToes",      HumanBodyBones.LeftToes,      "LeftFoot",     new Vector3(-0.09f, 0.03f, 0.10f), 0.05f),

            new BoneSpec("RightUpperLeg", HumanBodyBones.RightUpperLeg, "Hips",         new Vector3(0.09f,  0.90f, 0f), 0.11f),
            new BoneSpec("RightLowerLeg", HumanBodyBones.RightLowerLeg, "RightUpperLeg",new Vector3(0.09f,  0.50f, 0f), 0.09f),
            new BoneSpec("RightFoot",     HumanBodyBones.RightFoot,     "RightLowerLeg",new Vector3(0.09f,  0.08f, 0f), 0.08f),
            new BoneSpec("RightToes",     HumanBodyBones.RightToes,     "RightFoot",    new Vector3(0.09f,  0.03f, 0.10f), 0.05f),
        };

        [MenuItem("Window/Qudmi/Create Demo Avatar")]
        public static GameObject CreateMenuItem()
        {
            GameObject avatar = Create();
            Selection.activeGameObject = avatar;
            Undo.RegisterCreatedObjectUndo(avatar, "Create Qudmi demo avatar");
            return avatar;
        }

        /// <summary>Creates the rig and returns it, with a valid Humanoid Avatar on its Animator.</summary>
        public static GameObject Create()
        {
            var root = new GameObject("Qudmi Demo Avatar");
            var transforms = new Dictionary<string, Transform>();

            foreach (BoneSpec spec in Skeleton)
            {
                var joint = new GameObject(spec.Name);
                joint.transform.SetParent(spec.Parent == null ? root.transform : transforms[spec.Parent], true);
                joint.transform.position = spec.Position;
                // Identity rotation on every bone: it makes the T-pose the bind pose and keeps the
                // HumanDescription below trivial, which is what Unity's avatar validation wants.
                joint.transform.rotation = Quaternion.identity;
                transforms[spec.Name] = joint.transform;

                AddVisual(joint.transform, spec.Thickness);
            }

            var animator = root.AddComponent<Animator>();
            animator.avatar = BuildAvatar(root, transforms);
            return root;
        }

        private static void AddVisual(Transform joint, float thickness)
        {
            GameObject visual = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            visual.name = joint.name + " Visual";
            visual.transform.SetParent(joint, false);
            visual.transform.localScale = Vector3.one * thickness;
            // Colliders would fight the XR rig's character controller and serve no purpose here.
            Object.DestroyImmediate(visual.GetComponent<Collider>());
        }

        private static Avatar BuildAvatar(GameObject root, Dictionary<string, Transform> transforms)
        {
            var humanBones = new List<HumanBone>();
            var skeletonBones = new List<SkeletonBone>
            {
                new SkeletonBone
                {
                    name = root.name,
                    position = root.transform.localPosition,
                    rotation = root.transform.localRotation,
                    scale = root.transform.localScale,
                },
            };

            string[] humanNames = HumanTrait.BoneName;
            foreach (BoneSpec spec in Skeleton)
            {
                humanBones.Add(new HumanBone
                {
                    boneName = spec.Name,
                    humanName = humanNames[(int)spec.Bone],
                    limit = new HumanLimit { useDefaultValues = true },
                });

                Transform t = transforms[spec.Name];
                skeletonBones.Add(new SkeletonBone
                {
                    name = spec.Name,
                    position = t.localPosition,
                    rotation = t.localRotation,
                    scale = t.localScale,
                });
            }

            var description = new HumanDescription
            {
                human = humanBones.ToArray(),
                skeleton = skeletonBones.ToArray(),
                upperArmTwist = 0.5f, lowerArmTwist = 0.5f,
                upperLegTwist = 0.5f, lowerLegTwist = 0.5f,
                armStretch = 0.05f, legStretch = 0.05f,
                feetSpacing = 0f,
            };

            Avatar avatar = AvatarBuilder.BuildHumanAvatar(root, description);
            avatar.name = "QudmiDemoAvatar";
            if (!avatar.isValid || !avatar.isHuman)
            {
                Debug.LogError("Qudmi: failed to build a valid Humanoid avatar for the demo rig.");
            }
            return avatar;
        }
    }
}
