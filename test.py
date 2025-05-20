import torch
import cv2 as cv
# Model loading
#model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Can be 'yolov5n' - 'yolov5x6', or 'custom'
from ultralytics import YOLO

model = YOLO('runs/detect/train2/weights/best.pt')
# Inference on images

img = cv.imread('images/arrows.jpg')
# Run inference
results = model(img)

# Display results
results[0].show()  # Other options: .show(), .save(), .crop(), .pandas(), etc. Explore these in the Predict mode documentation.