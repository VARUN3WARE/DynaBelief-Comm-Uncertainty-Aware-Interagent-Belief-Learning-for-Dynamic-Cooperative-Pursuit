from __future__ import annotations


def resolve_device(requested: str) -> str:
    """Map ``auto`` to ``cuda`` when available, otherwise ``cpu``; reject unusable CUDA requests."""
    import torch  # lazy: keep env worker processes free of libtorch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {requested!r} requested but CUDA is not available")
    return requested
