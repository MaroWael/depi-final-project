from ultralytics import YOLO

model = YOLO("cleaning_surface.pt")
# Export with opset 17 for better compatibility with onnxruntime
model.export(format="onnx", opset=17, simplify=True)
print("✓ Model exported with opset 17")
