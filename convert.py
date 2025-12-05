from ultralytics import YOLO

model = YOLO("cleaning_surface.pt")
model.export(format="onnx")