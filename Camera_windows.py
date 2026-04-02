import cv2
import torch
import torchvision.transforms as transforms
from models.model_CNN import SimpleCNN
from models.complex_CNN import ASLNet
from visionPreprocess import crop_hand
import trained_model

from settings import MODEL_PATH, LEGACY_ARCH, LEGACY_NUM_CLASSES, ROI_SIZE, QUIT_KEY, WEBCAM_INDEX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
trained = trained_model.load(MODEL_PATH, device=device, arch=LEGACY_ARCH, num_classes=LEGACY_NUM_CLASSES)

def main():

    # --- USE NORMAL WEBCAM ---
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    print("Camera opened:", cap.isOpened())
    if not cap.isOpened():
        print("Camera failed to open")
        exit()

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Frame grab failed")
                break

            H, W, _ = frame.shape

            # Define centered 400×400 ROI
            roi_size = ROI_SIZE
            x1 = W//2 - roi_size//2
            y1 = H//2 - roi_size//2
            x2 = x1 + roi_size
            y2 = y1 + roi_size

            roi = frame[y1:y2, x1:x2]

            # Process the ROI (or full frame depending on your pipeline)
            crop = crop_hand(roi, method="yolo")
            if crop is None:
                continue
            tensor = crop.tensor.unsqueeze(0).to(device)
            bbox = crop.bbox
            hand_region = crop.image

            pred = trained.predict(tensor).item()

            # Draw prediction
            cv2.putText(frame, f"Prediction: {pred}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw ROI box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw bounding box inside ROI if found
            if bbox is not None:
                bx, by, bw, bh = bbox
                cv2.rectangle(roi, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

            # Show processed image (scaled back to 0–255)
            #cv2.imshow("Processed", processed[0])
            cv2.imshow("Camera", frame)
            #cv2.imshow("Live", live)
            if hand_region is not None:
                cv2.imshow("Hand_region", hand_region)
               # cv2.imshow("Gray", )
            print(f"Prediction: {pred}")

            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                break

    finally:
        print("Releasing camera...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()