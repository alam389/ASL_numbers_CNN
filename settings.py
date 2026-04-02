"""Central configuration. Plain constants only (no heavy imports) so any script can import it."""
import os

# --- Paths ---
DATA_DIR = "data"                          # training images, one subfolder per class
EVAL_DIR = "Test_Data"                     # evaluation images, same layout
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
INPUT_SIZE = (128, 128)                    # model input (H, W)
YOLO_WEIGHTS = "yolov8n.pt"
TRAIN_PREPROCESS_METHOD = "yolo"           # "yolo" or "skin"; stored in the checkpoint
SKIN_CROP_PADDING = 0.05                   # fraction of crop side added around the skin-mask box

# --- Training ---
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.05                           # remainder is the test split
EVAL_BATCH_SIZE = 32

# --- Camera ---
ROI_SIZE = 400                             # centered square drawn on the live view
QUIT_KEY = 27                              # Esc
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
