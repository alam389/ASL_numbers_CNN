"""Self-describing model checkpoints.

A checkpoint is a dict: {"format_version", "arch", "class_names", "preprocess_method",
"state_dict"}. `load` rebuilds the architecture from the registry below, so callers
need not hardcode NUM_CLASSES or the model class. Legacy bare state_dict .pth files
load too, if you pass `arch` and `num_classes` (or `class_names`) explicitly.

Deliberately imports only torch and models/ (no dataset, YOLO or train.py).
"""
import torch

from device import pick_device

from models.complex_CNN import ASLNet
from models.model_CNN import SimpleCNN

ARCHITECTURES = {"SimpleCNN": SimpleCNN, "ASLNet": ASLNet}
FORMAT_VERSION = 1


def _arch_name(model_or_name):
    if isinstance(model_or_name, str):
        name = model_or_name
    else:
        name = type(model_or_name).__name__
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {name!r}; registered: {sorted(ARCHITECTURES)}")
    return name


def _resolve_classes(class_names, num_classes):
    if class_names is not None:
        class_names = [str(c) for c in class_names]
        if num_classes is not None and num_classes != len(class_names):
            raise ValueError(f"num_classes={num_classes} != len(class_names)={len(class_names)}")
        return class_names
    if num_classes is None:
        raise ValueError("Provide class_names or num_classes")
    return [str(i) for i in range(num_classes)]


def save(path, model, arch_name=None, class_names=None, num_classes=None, preprocess_method=None):
    """Save `model` with metadata. arch_name defaults to the model's class name."""
    arch = _arch_name(arch_name or model)
    classes = _resolve_classes(class_names, num_classes)
    torch.save({
        "format_version": FORMAT_VERSION,
        "arch": arch,
        "class_names": classes,
        "preprocess_method": preprocess_method,
        "state_dict": model.state_dict(),
    }, path)


class TrainedModel:
    """An eval-mode model plus its metadata."""

    def __init__(self, model, arch, class_names, preprocess_method, device):
        self.model = model
        self.arch = arch
        self.class_names = class_names
        self.preprocess_method = preprocess_method
        self.device = device

    @property
    def num_classes(self):
        return len(self.class_names)

    def logits(self, batch):
        """Raw outputs for a (B,1,H,W) float tensor."""
        with torch.no_grad():
            return self.model(batch.to(self.device))

    def predict(self, batch):
        """Class indices, LongTensor of shape (B,) on CPU."""
        return torch.argmax(self.logits(batch), dim=1).cpu()

    def predict_names(self, batch):
        return [self.class_names[i] for i in self.predict(batch).tolist()]


def load(path, device=None, arch=None, num_classes=None, class_names=None, preprocess_method=None):
    """Load a checkpoint saved by `save`, or a legacy bare state_dict.

    For legacy files `arch` and `num_classes`/`class_names` are required. For new
    files, metadata comes from the file; explicit args are ignored except that a
    mismatching `arch` raises.
    """
    device = torch.device(device) if device is not None else pick_device()
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and "arch" in ckpt:
        if arch is not None and arch != ckpt["arch"]:
            raise ValueError(f"Checkpoint arch is {ckpt['arch']!r}, caller expected {arch!r}")
        arch = ckpt["arch"]
        classes = list(ckpt["class_names"])
        preprocess_method = ckpt.get("preprocess_method")
        state = ckpt["state_dict"]
    else:  # legacy bare state_dict
        if arch is None or (num_classes is None and class_names is None):
            raise ValueError(f"{path} is a legacy state_dict; pass arch and num_classes/class_names")
        classes = _resolve_classes(class_names, num_classes)
        state = ckpt
    model = ARCHITECTURES[_arch_name(arch)](num_classes=len(classes))
    model.load_state_dict(state)
    model.to(device).eval()
    return TrainedModel(model, arch, classes, preprocess_method, device)
