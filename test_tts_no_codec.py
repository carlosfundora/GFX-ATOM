import sys
from pathlib import Path
import os
import subprocess

# Add project root to sys.path to allow importing atom
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from atom.audio.chatterbox.engine import ChatterboxEngine
from atom.audio.text_splitter import SentenceSplitter
from atom.audio.utils import audio_to_bytes

def main():
    print("Testing SentenceSplitter (Rust)...")
    splitter = SentenceSplitter(min_sentence_length=2)
    sentences = []
    for chunk in ["Hello world! This is ", "a test. We will generate speech. And an", "other sentence here."]:
        sentences.extend(splitter.add_text(chunk))
    sentences.append(splitter.flush())
    print("Split sentences:", sentences)
    
    print("Initializing ChatterboxEngine...")
    engine = ChatterboxEngine(
        model_dir="/home/local/Projects/models/huggingface/models--onnx-community--chatterbox-ONNX/snapshots/3cab09af388d3f02bba43443fce88c1f4525ac43",
        backbone_dir="/home/local/Projects/models/huggingface/models--vladislavbro--llama_backbone_0.5/snapshots/a6c48da4d993a2058b95a8c3e2178da29f603f3e",
        device="cpu",
        dtype="float16",
    )
    
    engine.load()
    
    text = "The stars are incredibly bright tonight. I've always loved looking at the night sky. It makes you realize just how vast the universe really is! Sometimes, I wonder if anyone is looking back at us. Either way, it's a beautiful sight to behold. I hope we get a chance to see a shooting star."
    print(f"Generating audio for text: '{text}'")
    
    wav, metrics = engine.generate(text)
    
    print(f"Generation complete! Metrics: {metrics}")
    
    wav_bytes, _ = audio_to_bytes(wav, engine.service.sample_rate, response_format="wav")
    
    out_file = "test_tts_no_codec.wav"
    with open(out_file, "wb") as f:
        f.write(wav_bytes)
    
    print(f"Saved audio to {out_file}")
    
    # Play using aplay if available
    print("Playing audio over speakers...")
    subprocess.run(["aplay", out_file])

if __name__ == "__main__":
    main()
