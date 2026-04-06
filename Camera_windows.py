import cv2
import torch
from device import pick_device
import torchvision.transforms as transforms
from models.model_CNN import SimpleCNN
from models.complex_CNN import ASLNet
from visionPreprocess import crop_hand
import trained_model

from settings import MODEL_PATH, LEGACY_ARCH, LEGACY_NUM_CLASSES, QUIT_KEY, WEBCAM_INDEX, FLIP_DISPLAY, TRAIN_PREPROCESS_METHOD

device = pick_device()

# Load Model
trained = trained_model.load(MODEL_PATH, device=device, legacy_arch=LEGACY_ARCH, legacy_num_classes=LEGACY_NUM_CLASSES)
# Crop the same way the model was trained (stored in the checkpoint)
METHOD = trained.preprocess_method or TRAIN_PREPROCESS_METHOD

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

            crop = crop_hand(frame, method=METHOD)
            if crop is None:
                # no detection: keep the window responsive instead of freezing on the last frame
                cv2.imshow("Camera", cv2.flip(frame, 1) if FLIP_DISPLAY else frame)
                if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                    break
                continue
            tensor = crop.tensor.unsqueeze(0).to(device)
            bbox = crop.bbox
            hand_region = crop.image

            pred, conf = trained.predict_with_confidence(tensor)[0]

            # Draw the detected hand box
            if bbox is not None:
                bx, by, bw, bh = bbox
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

            # Mirror only what's shown; the model sees the unflipped frame (as in the dataset)
            display = cv2.flip(frame, 1) if FLIP_DISPLAY else frame
            cv2.putText(display, f"Prediction: {pred} ({conf:.0%})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Camera", display)
            #cv2.imshow("Live", live)
            if hand_region is not None:
                cv2.imshow("Hand_region", hand_region)
               # cv2.imshow("Gray", )
            print(f"Prediction: {pred} ({conf:.0%})")

            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                break

    finally:
        print("Releasing camera...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()