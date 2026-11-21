"""AOT target and constraints of the runtime's pointer/scalar launch ABI."""

import os

CAPABILITIES = {"sm_87": (8, 7), "sm_110a": (11, 0)}


def architecture():
    arch = os.environ.get("THORUSTINFER_CUDA_ARCH", "sm_87")
    if arch not in CAPABILITIES:
        raise ValueError(f"Unsupported thorustinfer CUDA target {arch!r}")
    return arch


def options(passes=None):
    arch = architecture()
    settings = dict(passes or {})
    if arch == "sm_110a":
        # TMA descriptors and warp-specialized launch contracts are not in this ABI.
        settings.update(
            {
                "tl.disable_tma_lower": True,
                "tl.disable_warp_specialized": True,
                "tl.disable_vectorize_256": True,
            }
        )
    return dict(target={"kind": "cuda", "arch": arch}, pass_configs=settings)


def require_device(capability):
    arch = architecture()
    if tuple(capability) != CAPABILITIES[arch]:
        raise ValueError(f"Requested {arch} package on CUDA device {capability}")
