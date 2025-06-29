from ultralytics import YOLO

if __name__ == "__main__":
    # Load pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Train
    model.train(
        data='C:/Users/piyus/Desktop/Task drone/drone_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0
    )
