using Unity.InferenceEngine;
using UnityEngine;

namespace Qudmi
{
    /// <summary>
    /// Runs the exported Qudmi ONNX model via Unity's Inference Engine (formerly "Sentis" --
    /// the package was renamed to com.unity.ai.inference/Unity.InferenceEngine, but "Sentis" is
    /// kept in this class's name since it's still the widely-recognized term). Not raw ONNX
    /// Runtime C# bindings: this is Unity's own cross-platform inference package, notably
    /// including Android/Quest, which is the actual deployment target for a VR full-body
    /// driver; ONNX Runtime's native plugin story on Android is considerably more fragile for a
    /// "just works" package.
    /// </summary>
    public class SentisInferenceEngine : IInferenceEngine, System.IDisposable
    {
        private readonly Worker _worker;
        private readonly int _window;
        private readonly int _inputDim;

        public SentisInferenceEngine(ModelAsset modelAsset, BackendType backend = BackendType.GPUCompute,
            int window = QudmiConstants.WindowLength, int inputDim = QudmiConstants.InputDim)
        {
            Model model = ModelLoader.Load(modelAsset);
            _worker = new Worker(model, backend);
            _window = window;
            _inputDim = inputDim;
        }

        public float[] Predict(float[] input)
        {
            using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, _window, _inputDim), input);
            _worker.Schedule(inputTensor);
            using Tensor<float> output = _worker.PeekOutput() as Tensor<float>;
            return output.DownloadToArray();
        }

        public void Dispose()
        {
            _worker?.Dispose();
        }
    }
}
