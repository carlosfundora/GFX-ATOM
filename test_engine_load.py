from atom.audio.chatterbox.engine import ChatterboxEngine

engine = ChatterboxEngine(
    model_dir="/tmp/dummy",
    backbone_dir=None,
)

print(engine.service)
