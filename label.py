from ultralytics import YOLO

model = YOLO("cleaning_surface.pt")
print(model.names)
