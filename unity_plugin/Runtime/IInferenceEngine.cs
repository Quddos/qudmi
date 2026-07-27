namespace Qudmi
{
    /// <summary>
    /// Abstraction over whatever actually runs the ONNX model, so the driver's buffer
    /// management / pose decoding / bone application logic (the parts verified against
    /// Tests/Fixtures/parity_case_1.json) doesn't depend on a specific inference backend's API
    /// surface. SentisInferenceEngine is the shipped implementation.
    /// </summary>
    public interface IInferenceEngine
    {
        /// <param name="input">Flattened (window * InputDim) feature buffer.</param>
        /// <returns>Flattened (OutputDim) pose prediction.</returns>
        float[] Predict(float[] input);
    }
}
