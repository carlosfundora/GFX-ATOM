import sys
from unittest.mock import MagicMock
sys.modules['aiter'] = MagicMock()
sys.modules['aiter.dist'] = MagicMock()
sys.modules['aiter.dist.shm_broadcast'] = MagicMock()
sys.modules['aiter.utility'] = MagicMock()
sys.modules['aiter.utility.dtypes'] = MagicMock()
sys.modules['vllm'] = MagicMock()

from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
from atom.audio.chatterbox.engine import ChatterboxEngine, RepetitionPenaltyProcessor

print(ChatterboxEngine._np_rep_penalty)
print(RepetitionPenaltyProcessor)
