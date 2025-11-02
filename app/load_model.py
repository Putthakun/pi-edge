import onnxruntime as ort

session = ort.InferenceSession("face_final.onnx", providers=["CPUExecutionProvider"])