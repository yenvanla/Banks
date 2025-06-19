try:
    from fastapi import FastAPI, File, UploadFile, Body
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import tempfile
    from pathlib import Path
    import base64
except Exception as e:
    import os
    os.system('pip install fastapi pydantic ultralytics opencv-python numpy python-multipart')
    from fastapi import FastAPI, File, UploadFile, Body
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import tempfile
    from pathlib import Path
    import base64

app = FastAPI()

# Load classes
with open('classes.txt', 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f if line.strip()]

def find_latest_yolov8_model():
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

# Load model 1 lần khi start server
model_path = find_latest_yolov8_model()
model = YOLO(model_path)

@app.post("/predict")
async def predict_captcha(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Không thể đọc ảnh!"}, status_code=400)
    if img.shape[1] != 160 or img.shape[0] != 50:
        img = cv2.resize(img, (160, 50))
    results = model(img, conf=0.01)
    captcha_text = ''
    if len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            boxes = result.boxes
            # Sắp xếp theo x1 (trái sang phải)
            sorted_boxes = sorted(boxes, key=lambda box: box.xyxy[0][0].item())
            for box in sorted_boxes:
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                if conf > 0.01 and cls < len(classes):
                    captcha_text += classes[cls]
    return {"captcha": captcha_text}

class Base64ImageRequest(BaseModel):
    image_base64: str

@app.post("/predict_base64")
async def predict_captcha_base64(data: Base64ImageRequest):
    try:
        img_bytes = base64.b64decode(data.image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "Không thể đọc ảnh!"}, status_code=400)
        if img.shape[1] != 160 or img.shape[0] != 50:
            img = cv2.resize(img, (160, 50))
        results = model(img, conf=0.01)
        captcha_text = ''
        if len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes
                sorted_boxes = sorted(boxes, key=lambda box: box.xyxy[0][0].item())
                for box in sorted_boxes:
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    if conf > 0.01 and cls < len(classes):
                        captcha_text += classes[cls]
        return {"captcha": captcha_text}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/")
def root():
    return {"message": "YOLOv8 Captcha API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_yolov8:app", host="0.0.0.0", port=8000, reload=True)