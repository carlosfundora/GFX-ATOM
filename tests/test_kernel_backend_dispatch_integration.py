from pathlib import Path


def test_linear_wires_selector_hook():
    linear_file = (
        Path(__file__).resolve().parents[1] / "atom" / "model_ops" / "linear.py"
    )
    src = linear_file.read_text(encoding="utf-8")
    assert "select_kernel_backend(" in src
    assert 'op_name="linear"' in src


def test_attention_mha_wires_selector_hook():
    attn_file = (
        Path(__file__).resolve().parents[1]
        / "atom"
        / "model_ops"
        / "attention_mha.py"
    )
    src = attn_file.read_text(encoding="utf-8")
    assert "select_kernel_backend(" in src
    assert 'op_name="attention_mha"' in src
