import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trained_model
from models.complex_CNN import ASLNet
from models.model_CNN import SimpleCNN


def _batch(n=3):
    return torch.randn(n, 1, 128, 128)  # ASLNet needs 128x128


@pytest.mark.parametrize("cls", [SimpleCNN, ASLNet])
def test_round_trip(tmp_path, cls):
    torch.manual_seed(0)
    model = cls(num_classes=4).eval()
    path = tmp_path / "m.pth"
    trained_model.save(path, model, class_names=["a", "b", "c", "d"], preprocess_method="yolo")
    loaded = trained_model.load(path, device="cpu")
    assert loaded.arch == cls.__name__
    assert loaded.class_names == ["a", "b", "c", "d"]
    assert loaded.num_classes == 4
    assert loaded.preprocess_method == "yolo"
    x = _batch()
    with torch.no_grad():
        expected = torch.argmax(model(x), dim=1)
    assert torch.equal(loaded.predict(x), expected)
    assert loaded.predict_names(x) == [["a", "b", "c", "d"][i] for i in expected.tolist()]
    names_conf = loaded.predict_with_confidence(x)
    assert [n for n, _ in names_conf] == loaded.predict_names(x)
    assert all(0.0 < c <= 1.0 for _, c in names_conf)


def test_legacy_state_dict(tmp_path):
    model = SimpleCNN(num_classes=2).eval()
    path = tmp_path / "legacy.pth"
    torch.save(model.state_dict(), path)
    loaded = trained_model.load(path, device="cpu", arch="SimpleCNN", num_classes=2)
    assert loaded.class_names == ["0", "1"]
    assert loaded.preprocess_method is None
    x = _batch()
    with torch.no_grad():
        assert torch.equal(loaded.predict(x), torch.argmax(model(x), dim=1))


def test_legacy_requires_arch_and_classes(tmp_path):
    path = tmp_path / "legacy.pth"
    torch.save(SimpleCNN(num_classes=2).state_dict(), path)
    with pytest.raises(ValueError):
        trained_model.load(path, device="cpu")


def test_unknown_arch_and_class_mismatch(tmp_path):
    model = SimpleCNN(num_classes=2)
    with pytest.raises(ValueError):
        trained_model.save(tmp_path / "x.pth", model, arch_name="Nope", num_classes=2)
    with pytest.raises(ValueError):
        trained_model.save(tmp_path / "x.pth", model, class_names=["a"], num_classes=2)
    trained_model.save(tmp_path / "ok.pth", model, num_classes=2)
    with pytest.raises(ValueError):
        trained_model.load(tmp_path / "ok.pth", device="cpu", arch="ASLNet")


def test_legacy_fallbacks_do_not_constrain_new_checkpoints(tmp_path):
    new, old = str(tmp_path / "new.pth"), str(tmp_path / "old.pth")
    trained_model.save(new, ASLNet(num_classes=3), num_classes=3)
    torch.save(SimpleCNN(num_classes=2).state_dict(), old)
    kw = dict(legacy_arch="SimpleCNN", legacy_num_classes=2)
    assert trained_model.load(new, **kw).arch == "ASLNet"
    assert trained_model.load(old, **kw).arch == "SimpleCNN"
