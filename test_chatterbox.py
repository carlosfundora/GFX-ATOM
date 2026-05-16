import sys
from unittest.mock import MagicMock
sys.modules['torch'] = MagicMock()
sys.modules['aiter'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()

from atom.audio.chatterbox.engine import ChatterboxEngine

def test_engine_init():
    engine = ChatterboxEngine(model_dir="dummy", device="cpu", dtype="float32")
    print("Engine init successful!")

if __name__ == "__main__":
    test_engine_init()
