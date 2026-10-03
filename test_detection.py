"""
Test script for Smart Traffic and Road Monitoring System.
Tests YOLOv8 model loading and inference on a synthesized traffic scene.
"""
import cv2
import numpy as np

def create_sample_traffic_image():
    # 640x480 test image with road markings
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Sky
    img[0:200, :] = [230, 200, 150]
    # Road
    img[200:480, :] = [60, 60, 60]
    # Yellow lane markings
    for y in range(220, 480, 50):
        cv2.rectangle(img, (315, y), (325, y + 30), (0, 220, 255), -1)
    
    # Save temporary image
    cv2.imwrite("test_traffic.jpg", img)
    print("Created sample test image: test_traffic.jpg")

if __name__ == "__main__":
    create_sample_traffic_image()
    try:
        from ultralytics import YOLO
        print("Loading YOLOv8n model...")
        model = YOLO("yolov8n.pt")
        print("Running inference on test_traffic.jpg...")
        results = model("test_traffic.jpg")
        print("YOLO Inference Successful! Model detected:", len(results[0].boxes), "objects.")
    except ImportError as e:
        print("YOLO packages still installing or not found:", e)
    except Exception as e:
        print("Inference test result:", e)
