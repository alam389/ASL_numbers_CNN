import cv2
import os
from datetime import datetime
from visionPreprocess import crop_hand

from settings import COLLECT_DIR as DATA_DIR, WEBCAM_INDEX
os.makedirs(DATA_DIR, exist_ok=True)

# Create class folders 0–9
for i in range(10):
    os.makedirs(os.path.join(DATA_DIR, str(i)), exist_ok=True)

# --- Windows webcam capture ---
# CAP_DSHOW avoids long startup delays on Windows
cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)

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

    # Apply your preprocessing
    crop = crop_hand(frame, method="skin")
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
        save_img = (frame).astype("uint8")
        cv2.imwrite(filepath, save_img)

        print(f"Saved {filepath}")

cap.release()
cv2.destroyAllWindows()