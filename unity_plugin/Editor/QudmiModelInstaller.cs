using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Networking;

namespace Qudmi.Editor
{
    /// <summary>
    /// Fetches the trained weights into the user's project, after showing them the licence.
    ///
    /// The weights are deliberately not bundled in the package. The package code is MIT; the
    /// weights are trained on AMASS, whose licence forbids commercial use of anything trained on
    /// it. Shipping both under one MIT package would misstate the licence of what is being
    /// distributed, and someone would reasonably ship it in a paid game and be in breach without
    /// ever knowing. Downloading them separately also means the person installing actively sees
    /// and accepts those terms, which is the point.
    /// </summary>
    public static class QudmiModelInstaller
    {
        public const string ModelUrl = "https://huggingface.co/quddusr/qudmi/resolve/main/qudmi_v0.onnx";
        public const string ModelPageUrl = "https://huggingface.co/quddusr/qudmi";
        private const string InstallFolder = "Assets/Qudmi";
        private const string InstallPath = InstallFolder + "/qudmi_v0.onnx";

        /// <summary>The model asset in this project, or null if it hasn't been installed yet.</summary>
        public static Object FindInstalledModel()
        {
            foreach (string guid in AssetDatabase.FindAssets("qudmi_v0"))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                if (path.EndsWith(".onnx"))
                {
                    return AssetDatabase.LoadMainAssetAtPath(path);
                }
            }
            return null;
        }

        [MenuItem("Window/Qudmi/Download Model Weights")]
        public static void DownloadWithConsent()
        {
            if (FindInstalledModel() != null &&
                !EditorUtility.DisplayDialog("Qudmi",
                    "The model is already in this project. Download it again?", "Download again", "Cancel"))
            {
                return;
            }

            bool accepted = EditorUtility.DisplayDialog(
                "Qudmi model weights — licence",
                "The weights are trained on AMASS, which is licensed for non-commercial " +
                "scientific research, education and non-commercial projects only.\n\n" +
                "AMASS explicitly prohibits using it to train networks for commercial use of any " +
                "kind, so these weights MAY NOT be shipped in a commercial product or service.\n\n" +
                "The Qudmi package code is MIT and carries no such restriction — you may retrain " +
                "on your own data for any purpose.\n\n" +
                "Download the weights on these terms?",
                "I understand — download", "Cancel");

            if (!accepted)
            {
                return;
            }

            Download();
        }

        private static void Download()
        {
            Directory.CreateDirectory(InstallFolder);

            using UnityWebRequest request = UnityWebRequest.Get(ModelUrl);
            request.downloadHandler = new DownloadHandlerFile(InstallPath);
            UnityWebRequestAsyncOperation operation = request.SendWebRequest();

            try
            {
                while (!operation.isDone)
                {
                    if (EditorUtility.DisplayCancelableProgressBar(
                            "Downloading Qudmi weights",
                            $"{request.downloadedBytes / (1024f * 1024f):F1} MB of ~19.6 MB",
                            request.downloadProgress))
                    {
                        request.Abort();
                        // Partial file left behind would import as a corrupt ModelAsset, which
                        // fails far less clearly than simply not having it.
                        SafeDelete(InstallPath);
                        return;
                    }
                }
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }

            if (request.result != UnityWebRequest.Result.Success)
            {
                SafeDelete(InstallPath);
                Debug.LogError($"Qudmi: model download failed ({request.error}). " +
                    $"Download it manually from {ModelPageUrl} and place it anywhere under Assets.");
                return;
            }

            AssetDatabase.Refresh();
            Debug.Log($"Qudmi: weights installed to {InstallPath} (research/non-commercial use only).");
        }

        private static void SafeDelete(string path)
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }
}
