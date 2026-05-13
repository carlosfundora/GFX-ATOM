import sys
from pathlib import Path
import os
import subprocess

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from atom.audio.chatterbox.engine import ChatterboxEngine
from atom.audio.text_splitter import SentenceSplitter
from atom.audio.utils import audio_to_bytes

def main():
    print("Initializing ChatterboxEngine (GPU, Greedy)...")
    engine = ChatterboxEngine(
        model_dir="/home/local/Projects/models/huggingface/models--onnx-community--chatterbox-ONNX/snapshots/3cab09af388d3f02bba43443fce88c1f4525ac43",
        backbone_dir="/home/local/Projects/models/huggingface/models--vladislavbro--llama_backbone_0.5/snapshots/a6c48da4d993a2058b95a8c3e2178da29f603f3e",
        device="cuda:0",
        dtype="float16",
    )
    
    engine.load()
    
    text = "Testing the GPU generation with greedy decoding. I hope this works properly without any garbling."
    print(f"Generating audio for text: '{text}'")
    
    # We pass temperature=0.0 to force greedy argmax
    wav, metrics = engine.generate(text, temperature=0.0)
    
    print(f"Generation complete! Metrics: {metrics}")
    
    import soundfile as sf
    sf.write("test_gpu_greedy.wav", wav, engine.service.sample_rate)
    
    print(f"Saved audio to test_gpu_greedy.wav")

if __name__ == "__main__":
    main()
