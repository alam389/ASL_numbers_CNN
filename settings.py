"""Central configuration. Plain constants only (no heavy imports) so any script can import it."""
import os

# --- Paths ---
DATA_DIR = "data"                          # training images, one subfolder per class
EVAL_DIR = "Test_Data"                     # evaluation images, same layout
CACHE_DIR = "cache"                        # preprocessed crop cache (gitignored)
COLLECT_DIR = "my_dataset_processed"       # where collect*.py saves camera captures
MODEL_PATH = "asl_net_model.pth"           # written by train.py, read by test.py and the camera scripts

# --- Model ---
ARCH = "ASLNet"                            # architecture train.py builds; key in trained_model.ARCHITECTURES
# Only used when MODEL_PATH is a legacy bare state_dict (no metadata inside)
LEGACY_ARCH = "SimpleCNN"
LEGACY_NUM_CLASSES = 2


def class_names(data_dir=DATA_DIR):
    """Class order = sorted subfolder names (matches dataset.py)."""
    if not os.path.isdir(data_dir):
        return []
    return sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))


# --- Preprocessing ---
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
INPUT_SIZE = (128, 128)                    # model input (H, W)
YOLO_WEIGHTS = "yolov8n.pt"
TRAIN_PREPROCESS_METHOD = "mediapipe"      # "mediapipe", "yolo" or "skin"; stored in the checkpoint
HAND_MODEL_PATH = "hand_landmarker.task"   # MediaPipe model, downloaded on first use (gitignored)
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_CROP_PADDING = 0.25                   # extra margin around the landmarks, as a fraction of hand size
HAND_MIN_CONFIDENCE = 0.3
SKIN_CROP_PADDING = 0.05                   # fraction of crop side added around the skin-mask box

# --- Training ---
DEVICE = "auto"                            # "auto" (cuda > mps > cpu), "cuda", "mps" or "cpu"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_WORKERS = 0                            # crops are cached in memory, workers only add overhead
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.05                           # remainder is the test split
EVAL_BATCH_SIZE = 32
BACKGROUND_SWAP_PROB = 0.0                 # chance to paste hands onto random backgrounds; only for black-background data (e.g. 0.85)
AUGMENT = True                             # random shift/rotate/scale/brightness on training batches only

# --- Camera ---
ROI_SIZE = 400                             # centered square drawn on the live view
QUIT_KEY = 27                              # Esc
FLIP_DISPLAY = True                        # mirror the preview (Camera_windows.py); the model input is not flipped
WEBCAM_INDEX = 0                           # Camera_windows.py / collect_windows.py
JETSON_CAMERA = dict(                      # Camera.py / collect.py GStreamer pipeline
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
)
