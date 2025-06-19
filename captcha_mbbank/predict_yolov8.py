from ultralytics import YOLO
import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Load classes
with open('classes.txt', 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f if line.strip()]

def find_latest_yolov8_model():
    """Tìm model YOLOv8 mới nhất"""
    model_dirs = list(Path('yolov8_runs').glob('exp*'))
    if not model_dirs:
        raise FileNotFoundError("Không tìm thấy model YOLOv8. Hãy train trước.")
    
    latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    model_path = latest_dir / 'weights' / 'best.pt'
    
    if not model_path.exists():
        model_path = latest_dir / 'weights' / 'last.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy weights trong {latest_dir}")
    
    print(f"Loading YOLOv8 model from: {model_path}")
    return str(model_path)

def predict_captcha(image_path):
    # Load model
    model_path = find_latest_yolov8_model()
    model = YOLO(model_path)
    
    # Load và resize ảnh
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh!")
        return "", []
    
    print(f"Original image size: {img.shape}")
    img_resized = cv2.resize(img, (160, 50))
    print(f"Resized image size: {img_resized.shape}")
    
    # Dự đoán
    results = model(img_resized, conf=0.01)  # Confidence threshold thấp
    
    # Lấy kết quả
    detections = []
    if len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                detections.append([x1, y1, x2, y2, conf, cls])
    
    # Sắp xếp theo vị trí x (từ trái sang phải)
    detections = sorted(detections, key=lambda x: x[0])
    
    # Ghép chuỗi captcha
    captcha_text = ""
    for det in detections:
        if det[4] > 0.01:  # Confidence threshold
            class_id = int(det[5])
            if class_id < len(classes):
                captcha_text += classes[class_id]
    
    return captcha_text, detections

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_yolov8.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)
    
    captcha_text, detections = predict_captcha(image_path)
    print(f"Predicted captcha: {captcha_text}")
    print(f"Number of detections: {len(detections)}")
    
    # In chi tiết từng ký tự
    for i, det in enumerate(detections):
        if det[4] > 0.01:
            class_id = int(det[5])
            if class_id < len(classes):
                print(f"Char {i+1}: {classes[class_id]} (conf: {det[4]:.3f})")
    
    # In tất cả detections
    print(f"\nAll detections:")
    for i, det in enumerate(detections):
        class_id = int(det[5])
        if class_id < len(classes):
            print(f"Detection {i+1}: {classes[class_id]} (conf: {det[4]:.3f}, bbox: {det[:4]})") 