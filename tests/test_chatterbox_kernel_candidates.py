from atom.audio.chatterbox.kernel_candidates import (
    KERNEL_CANDIDATES,
    allow_experimental_chatterbox_gemm,
    donor_kernel_status,
    rdna2_runtime_detected,
)


def test_donor_kernel_candidates_are_classified_for_chatterbox():
    names = {candidate.name for candidate in KERNEL_CANDIDATES}

    assert "vllm-rdna2-index-fallback" in names
    assert "aiter-triton-gfx1030" in names
    assert "deepspeed-hip-linear" in names
    assert "llama-cpp-tq3-kvcache" in names
    assert all(not candidate.default_enabled for candidate in KERNEL_CANDIDATES)


def test_donor_kernel_status_is_stable_and_contains_local_paths():
    status = donor_kernel_status()

    assert status
    assert all("name" in item and "chatterbox_use" in item for item in status)
    assert any(item["exists"] for item in status)


def test_rdna2_runtime_detection_uses_rocm_arch_env(monkeypatch):
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.delenv("AMDGPU_TARGETS", raising=False)

    assert rdna2_runtime_detected() is False

    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    assert rdna2_runtime_detected() is True

    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1030")
    assert rdna2_runtime_detected() is True


def test_experimental_chatterbox_gemm_requires_opt_in(monkeypatch):
    monkeypatch.delenv("ATOM_CHATTERBOX_EXPERIMENTAL_GEMM", raising=False)
    assert allow_experimental_chatterbox_gemm() is False

    monkeypatch.setenv("ATOM_CHATTERBOX_EXPERIMENTAL_GEMM", "true")
    assert allow_experimental_chatterbox_gemm() is True
