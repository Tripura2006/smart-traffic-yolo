import base64
import time
from typing import Dict, List, Any
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Smart Traffic and Road Monitoring API",
    description="YOLO-powered object detection API for road traffic monitoring",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Target road & traffic classes from COCO dataset
ROAD_CLASSES = {
    "person": "Pedestrian",
    "bicycle": "Bicycle",
    "car": "Car / Auto",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
    "traffic light": "Traffic Light",
    "stop sign": "Stop Sign",
}

# Color palette (BGR format for OpenCV) for distinct visual identification
CLASS_COLORS = {
    "person": (76, 217, 100),       # Green
    "bicycle": (255, 149, 0),       # Orange
    "car": (0, 122, 255),           # Blue
    "motorcycle": (255, 204, 0),    # Cyan/Yellow
    "bus": (255, 45, 85),           # Bright Red/Pink
    "truck": (175, 82, 222),        # Purple
    "traffic light": (255, 59, 48), # Coral Red
    "stop sign": (88, 86, 214),     # Indigo
}
DEFAULT_COLOR = (142, 142, 147)

# Global YOLO model holder
_yolo_model = None

def get_yolo_model():
    """Lazily load YOLO model to ensure instant server startup."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            print("Loading YOLOv8 model (yolov8n.pt)...")
            _yolo_model = YOLO("yolov8n.pt")
            print("YOLOv8 model loaded successfully!")
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="YOLO (ultralytics/torch) package installation is still completing. Please retry in 10-20 seconds."
            )
    return _yolo_model


@app.get("/")
def read_root():
    """Health check and API status endpoint."""
    model_ready = False
    try:
        from ultralytics import YOLO
        model_ready = True
    except ImportError:
        model_ready = False

    return {
        "status": "online",
        "system": "Smart Traffic and Road Monitoring System",
        "model": "YOLOv8n",
        "model_engine_ready": model_ready,
        "supported_classes": list(ROAD_CLASSES.keys()),
        "endpoints": {
            "detect": "POST /detect (multipart/form-data with image file)"
        }
    }


@app.post("/detect")
async def detect_traffic_objects(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.25, ge=0.05, le=1.0)
) -> Dict[str, Any]:
    """
    Accepts an uploaded traffic image, executes YOLO inference, draws 
    styled bounding boxes using OpenCV, and returns detection metrics.
    """
    # 1. Validate MIME type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload an image (JPG, PNG, WebP)."
        )

    # 2. Get YOLO model
    model = get_yolo_model()

    # 3. Read image bytes into OpenCV format
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Corrupted or unsupported image file: {str(e)}"
        )

    height, width, _ = img.shape
    start_time = time.time()

    # 4. Run YOLO inference
    results = model(img, conf=confidence_threshold)
    inference_time_ms = round((time.time() - start_time) * 1000, 2)

    detections: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {}
    annotated_img = img.copy()

    # 5. Process YOLO detections and draw with OpenCV
    det_id = 1
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]
            conf = float(box.conf[0].item())

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Count occurrences per class
            display_name = ROAD_CLASSES.get(class_name, class_name.capitalize())
            class_counts[display_name] = class_counts.get(display_name, 0) + 1

            # Get class-specific color
            color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness=2)

            # Draw styled label badge
            label_text = f"{display_name} {int(conf * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.45, min(0.7, width / 1800))
            font_thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )

            # Ensure label stays within image bounds
            badge_y1 = max(0, y1 - text_h - 8)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = x1 + text_w + 10

            # Draw filled header background for label
            cv2.rectangle(
                annotated_img,
                (badge_x1, badge_y1),
                (badge_x2, badge_y2),
                color,
                thickness=-1
            )

            # Put white or dark text on badge
            text_color = (0, 0, 0) if (color[0]*0.299 + color[1]*0.587 + color[2]*0.114) > 160 else (255, 255, 255)
            cv2.putText(
                annotated_img,
                label_text,
                (badge_x1 + 5, badge_y2 - 4),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA
            )

            detections.append({
                "id": det_id,
                "class_name": display_name,
                "raw_class": class_name,
                "confidence": round(conf, 4),
                "confidence_percent": f"{round(conf * 100, 1)}%",
                "color_rgb": f"rgb({color[2]},{color[1]},{color[0]})",
                "box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })
            det_id += 1

    # 6. Encode the OpenCV annotated image to Base64
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, buffer = cv2.imencode(".jpg", annotated_img, encode_param)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    image_data_uri = f"data:image/jpeg;base64,{img_base64}"

    total_count = len(detections)

    # Congestion & Traffic Density Analytics
    if total_count == 0:
        congestion_level = "No Traffic"
        congestion_color = "#64748b"
    elif total_count <= 4:
        congestion_level = "Low (Smooth Flow)"
        congestion_color = "#10b981"
    elif total_count <= 10:
        congestion_level = "Moderate (Normal Traffic)"
        congestion_color = "#f59e0b"
    else:
        congestion_level = "High (Heavy Congestion)"
        congestion_color = "#ef4444"

    pedestrians_count = class_counts.get("Pedestrian", 0)
    bicycles_count = class_counts.get("Bicycle", 0)
    has_pedestrian_warning = (pedestrians_count + bicycles_count) > 0

    return {
        "success": True,
        "filename": file.filename,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_width": width,
        "image_height": height,
        "inference_time_ms": inference_time_ms,
        "total_count": total_count,
        "congestion_level": congestion_level,
        "congestion_color": congestion_color,
        "pedestrians_count": pedestrians_count,
        "has_pedestrian_warning": has_pedestrian_warning,
        "class_counts": class_counts,
        "detections": detections,
        "image_base64": image_data_uri
    }

