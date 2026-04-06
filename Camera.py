import cv2
import torch
from device import pick_device
import torchvision.transforms as transforms
from models.model_CNN import SimpleCNN
from models.complex_CNN import ASLNet
from visionPreprocess import crop_hand
import trained_model

from settings import MODEL_PATH, LEGACY_ARCH, LEGACY_NUM_CLASSES, QUIT_KEY, JETSON_CAMERA, TRAIN_PREPROCESS_METHOD

device = pick_device()

#Load Model
# LEGACY_* are only used if MODEL_PATH is a legacy bare state_dict
trained = trained_model.load(MODEL_PATH, device=device, legacy_arch=LEGACY_ARCH, legacy_num_classes=LEGACY_NUM_CLASSES)
# Crop the same way the model was trained (stored in the checkpoint)
METHOD = trained.preprocess_method or TRAIN_PREPROCESS_METHOD

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
def main():

    cap = cv2.VideoCapture(gstreamer_pipeline(**JETSON_CAMERA), cv2.CAP_GSTREAMER)


    print("Camera opened:", cap.isOpened())
    if not cap.isOpened():
        print("Camera failed to open")
        exit()

    try:
        while True:
            ret, frame = cap.read()
            #print("ret:", ret, "frame:", None if frame is None else frame.shape)

            if not ret:
                break

            H, W, _ = frame.shape

            # Define a centered 400×400 ROI
            roi_size = ROI_SIZE
            x1 = W//2 - roi_size//2
            y1 = H//2 - roi_size//2
            x2 = x1 + roi_size
            y2 = y1 + roi_size

            roi = frame[y1:y2, x1:x2]
            
            #processed, bbox = preprocess_live(roi)
            crop = crop_hand(frame, method=METHOD)
            if crop is None:
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                    break
                continue
            processed, bbox = crop.tensor.numpy(), crop.bbox

            tensor = crop.tensor.float().unsqueeze(0).to(device)
 
            pred = trained.predict_names(tensor)[0]

            cv2.putText(frame, f"Prediction: {pred}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

		# Draw bounding box if found
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if bbox is not None:
                bx, by, bw, bh = bbox
                cv2.rectangle(roi, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)

            #cv2.imshow("Skin Mask", skin_mask)
            cv2.imshow("Final Processed", processed[0] * 255)

            cv2.imshow("Camera", frame)
            print(f"Prediction: {pred}")
            if cv2.waitKey(1) & 0xFF == QUIT_KEY:
                break

    finally:
        print("Releasing camera...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
	main()

