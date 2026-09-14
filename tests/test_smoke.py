import subprocess
import sys
import pytest
import ligase

from ligase.utils.seeding import seed_everything

def test_version():
    assert ligase.__version__ == "0.0.1"

def test_seeding_is_deterministic():
    torch = pytest.importorskip("torch")
    seed_everything(7)
    a = torch.rand(8)
    seed_everything(7)
    b = torch.rand(8)
    assert torch.equal(a, b)

def test_import_policy_torch_not_pulled_by_root_import():
    code = "import ligase, sys; assert 'torch' not in sys.modules, 'Hard Rule 1 violated'"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
