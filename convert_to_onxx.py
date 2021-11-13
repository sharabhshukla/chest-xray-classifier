import coremltools
import onnxmltools

# Load a Core ML model
coreml_model = coremltools.utils.load_spec('ml_model/chestimageclassifier.mlmodel')

# Convert the Core ML model into ONNX
onnx_model = onnxmltools.convert_coreml(coreml_model, 'chextxrayonnxmodel.onnx')

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, 'example.onnx')