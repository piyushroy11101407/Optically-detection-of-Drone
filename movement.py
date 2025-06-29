import cv2
from ultralytics import YOLO

# Pseudo-servo functions
def move_servo_pan(delta):
    print(f"Pan servo by {delta} degrees")  # Replace with real command

def move_servo_tilt(delta):
    print(f"Tilt servo by {delta} degrees")  # Replace with real command

video_path = r'C:\Users\piyus\Desktop\Task drone\Data\Videos\V_DRONE_080.mp4'
model_path = r'C:\Users\piyus\Desktop\Task drone\drone_dataset\runs\detect\train\weights\best.pt'

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

# Get frame size
ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]
frame_center_x = frame_width // 2
frame_center_y = frame_height // 2

# Control sensitivity (tune as needed)
pan_gain = 0.1  # how many degrees per pixel offset
tilt_gain = 0.1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    if len(results[0].boxes) > 0:
        # Assume one drone; pick highest confidence
        box = max(results[0].boxes, key=lambda b: b.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        # Compute error
        error_x = cx - frame_center_x  # +ve: right, -ve: left
        error_y = cy - frame_center_y  # +ve: down, -ve: up

        # Convert error to servo movement
        pan_delta = pan_gain * error_x
        tilt_delta = tilt_gain * error_y

        move_servo_pan(pan_delta)
        move_servo_tilt(tilt_delta)
        
        # (Optional) Draw the center and box for debugging
        annotated = results[0].plot()
        cv2.circle(annotated, (frame_center_x, frame_center_y), 8, (255,0,0), -1)
        cv2.circle(annotated, (cx, cy), 8, (0,255,0), -1)
        cv2.imshow('Camera Tracking', annotated)
    else:
        cv2.imshow('Camera Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
