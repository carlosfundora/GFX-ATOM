import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
from atom.audio.chatterbox.engine import ChatterboxEngine

engine = ChatterboxEngine(
    model_dir="/home/local/Projects/models/huggingface/models--onnx-community--chatterbox-ONNX/snapshots/3cab09af388d3f02bba43443fce88c1f4525ac43",
    backbone_dir="/home/local/Projects/models/huggingface/models--vladislavbro--llama_backbone_0.5/snapshots/a6c48da4d993a2058b95a8c3e2178da29f603f3e",
    device="cuda:0",
    dtype="float16",
)
engine.load()
print("PyTorch Vocab size:", engine._model.get_input_embeddings().weight.shape)
