import cv2
import os
from datetime import datetime
from visionPreprocess import crop_hand

from settings import TRAIN_PREPROCESS_METHOD, COLLECT_DIR as DATA_DIR, JETSON_CAMERA
os.makedirs(DATA_DIR, exist_ok=True)


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Create class folders 0–9
for i in range(10):
    os.makedirs(os.path.join(DATA_DIR, str(i)), exist_ok=True)

# Open webcam (force V4L2 backend for Linux)
pipeline = gstreamer_pipeline(**JETSON_CAMERA)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

print("Press keys 0–9 to save a processed image to that class.")
print("Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # --- Apply your preprocessing ---
    crop = crop_hand(frame, method=TRAIN_PREPROCESS_METHOD)
    processed = crop.image if crop is not None else None

    # Show raw camera feed
    cv2.imshow("Camera", frame)

    # Show processed crop
    if processed is not None:
        cv2.imshow("Processed", processed)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # If user presses a digit key
    if ord('0') <= key <= ord('9'):
        label = chr(key)
        folder = os.path.join(DATA_DIR, label)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{label}_{timestamp}.png"
        filepath = os.path.join(folder, filename)

        # Save processed image (scaled back to 0–255)
        if processed is None:
            print("No hand detected; nothing saved")
            continue
        cv2.imwrite(filepath, processed)

        print(f"Saved {filepath}")

cap.release()
cv2.destroyAllWindows()

