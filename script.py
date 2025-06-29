from ultralytics import YOLO
import cv2

video_path = r'C:\Users\piyus\Desktop\Task drone\Data\Videos\V_DRONE_080.mp4'
model_path = r'C:\Users\piyus\Desktop\Task drone\drone_dataset\runs\detect\train\weights\best.pt'

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

# For trajectory (list of center points)
trajectory_points = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Draw results and print info
    annotated_frame = results[0].plot()

    # Assume one drone per frame for simplicity (if multiple, can extend logic)
    for box in results[0].boxes:
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        print(f"Class: {cls}, Confidence: {conf:.2f}, Coordinates: ({x1},{y1},{x2},{y2}), Center: ({cx},{cy})")

        # Add center point to trajectory (for drones above confidence threshold, e.g. 0.4)
        if conf > 0.4:
            trajectory_points.append((cx, cy))

    # Draw trajectory
    for i in range(1, len(trajectory_points)):
        cv2.line(annotated_frame, trajectory_points[i - 1], trajectory_points[i], (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Drone Detection with Trajectory", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
