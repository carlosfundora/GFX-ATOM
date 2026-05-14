import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
from atom.audio.chatterbox.engine import ChatterboxEngine

engine = ChatterboxEngine(
    model_dir="/home/local/Projects/models/huggingface/models--onnx-community--chatterbox-ONNX/snapshots/3cab09af388d3f02bba43443fce88c1f4525ac43",
    backbone_dir=None, # Force ONNX
)
engine.load()
text = "Testing the ONNX fallback to see if it generates clean audio."
wav, metrics = engine.generate(text)
import soundfile as sf
sf.write("test_onnx.wav", wav, engine.service.sample_rate)
print("Done ONNX")
