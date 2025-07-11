# Drone Detection, Tracking, and Camera Control with YOLOv8

## Overview

This project provides a robust, real-time system for **drone detection**, **trajectory tracking**, and **camera auto-centering** using [YOLOv8](https://github.com/ultralytics/ultralytics).

- It detects drones in video feeds using a trained YOLOv8 model.
- The system **visualizes drone trajectories** by drawing their flight path directly on the video frames, making it easy to see where the drone has traveled.
- Camera movement commands are generated for pan/tilt servos, allowing the camera to automatically track and keep the drone centered in the view at all times.

**Key Visuals:**
- The **drone’s bounding box** is shown on each video frame.
- The **drone's center** is marked (typically with a green dot).
- The **center of the video frame** is marked (typically with a blue dot), indicating the target the camera tries to keep the drone at.
- The **trajectory of the drone** is displayed as a red line tracing its movement history across frames.

---

## Features

- **YOLOv8-based drone detection**
- **Trajectory visualization** on video
- **Camera movement logic** for real-time auto-centering (pan & tilt)
- **Easy hardware integration** (Raspberry Pi, Arduino, etc.)
- Flexible for single/multiple drones and extensible for tracking

---

## Dataset

This project uses a public drone detection dataset, which can be downloaded from Kaggle:

**[YOLO Drone Detection Dataset on Kaggle](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset)**

- The dataset is formatted for YOLO and contains images and corresponding label files for drone detection tasks.
- It is split into training and validation sets, each with `images/` and `labels/` subfolders.
- The label files use standard YOLO format:  
  `<class_id> <x_center> <y_center> <width> <height>`  
  (All values are normalized to [0,1].)

Please download and extract the dataset before training or running inference.

## Requirements

- Python 3.8+
- Ultralytics YOLO: `pip install ultralytics`
- OpenCV: `pip install opencv-python`
- (Optional) Servo control libraries:
    - For Raspberry Pi: `gpiozero`, `adafruit-circuitpython-servokit`
    - For Arduino: `pyserial`

---

## Dataset Preparation

Organize your dataset as follows:

    drone_dataset/
    ├── train/
    │ ├── images/
    │ └── labels/
    ├── valid/
    │ ├── images/
    │ └── labels/
    └── data.yaml

### Dataset Structure

- Each `images/` folder contains training or validation images.
- Each `labels/` folder contains the corresponding YOLO-format `.txt` files.

### Yolo Label format

Each .txt file contains lines like 

    <class_id> <x_center> <y_center> <width> <height>

All values are normalized to [0, 1] relative to the image size.

---

## Training YOLOv8

    from ultralytics import YOLO

    if __name__ == "__main__":
        model = YOLO('yolov8n.pt')
        model.train(
            data='C:/Users/yourname/path/to/drone_dataset/data.yaml',
            epochs=100,
            imgsz=640,
            batch=16,
            device=0  # set to 'cpu' if no GPU
        )

Trained weights are saved to runs/detect/train/weights/best.pt

---

## Inference & Trajectory Visualization
Run YOLOv8 on a video, print detection info, and visualize drone trajectory:

    from ultralytics import YOLO
    import cv2

    video_path = r'path\to\your\video.mp4'
    model_path = r'path\to\runs\detect\train\weights\best.pt'

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    trajectory_points = []

    ret, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            print(f"Class: {cls}, Confidence: {conf:.2f}, Coordinates: ({x1},{y1},{x2},{y2}), Center: ({cx},{cy})")
            if conf > 0.4:
                trajectory_points.append((cx, cy))

        # Draw trajectory
        for i in range(1, len(trajectory_points)):
            cv2.line(annotated_frame, trajectory_points[i - 1], trajectory_points[i], (0, 0, 255), 2)

        # Draw center and detected point
        cv2.circle(annotated_frame, (frame_center_x, frame_center_y), 8, (255,0,0), -1)  # Blue: frame center
        if trajectory_points:
            cv2.circle(annotated_frame, trajectory_points[-1], 8, (0,255,0), -1)  # Green: drone center

        cv2.imshow("YOLOv8 Drone Detection with Trajectory", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
- Blue dot: Frame center (target)

- Green dot: Drone center

- Red line: Drone's trajectory

---

## Camera Movement Logic (Pan/Tilt Servos)
The camera uses two servos (pan & tilt) to keep the drone centered:

    def move_servo_pan(delta):
        print(f"Pan servo by {delta} degrees")  # Replace with real servo code

    def move_servo_tilt(delta):
        print(f"Tilt servo by {delta} degrees")  # Replace with real servo code

    # In your detection loop:
    error_x = cx - frame_center_x
    error_y = cy - frame_center_y
    pan_delta = pan_gain * error_x
    tilt_delta = tilt_gain * error_y

    move_servo_pan(pan_delta)
    move_servo_tilt(tilt_delta)
- Tune pan_gain and tilt_gain for smoothness.

- Integrate with hardware via GPIO/serial as needed.

## Troubleshooting
- Multiprocessing errors (Windows):
    Wrap your main code in:

        if __name__ == "__main__":
            # your code
- Path issues:
    Use forward slashes (/), avoid spaces in folder names.