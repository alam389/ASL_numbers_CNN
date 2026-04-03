import torch

from settings import DEVICE


def pick_device(preference=DEVICE):
    """Resolve settings.DEVICE ("auto", "cuda", "mps", "cpu") to a torch.device.

    "auto" prefers CUDA, then Apple's MPS (the Apple GPU), then CPU. PyTorch cannot
    run on the Apple Neural Engine; that needs a Core ML export.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
