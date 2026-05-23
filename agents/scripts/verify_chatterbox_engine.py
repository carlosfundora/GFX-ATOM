import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from unittest.mock import MagicMock
import sys
sys.modules['atom'] = MagicMock()
import numpy as np

# Load engine manually
with open('atom/audio/chatterbox/engine.py') as f:
    code = f.read()
    # remove imports to atom
    lines = [line for line in code.split('\n') if not line.startswith('from atom')]
    code = '\n'.join(lines)
    exec(code, globals())

def test_np_rep_penalty():
    scores = np.array([[-1.0, 1.0, 2.0]], dtype=np.float32)
    input_ids = np.array([[0, 1]])

    res = ChatterboxEngine._np_rep_penalty(input_ids, scores, 1.2)
    assert np.allclose(res, np.array([[-1.2, 0.8333333, 2.0]]))

    scores2 = np.array([[-1.0, 1.0, 2.0], [2.0, -2.0, 3.0]], dtype=np.float32)
    input_ids2 = np.array([[0, 1], [1, 2]])
    res2 = ChatterboxEngine._np_rep_penalty(input_ids2, scores2, 1.2)
    assert np.allclose(res2, np.array([[-1.2, 0.8333333, 2.0], [2.0, -2.4, 2.5]]))

if __name__ == "__main__":
    test_np_rep_penalty()
    print("test_np_rep_penalty passed")
