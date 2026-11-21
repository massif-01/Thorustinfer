# DeepSeek V4.1 reference

This directory preserves the model implementation, original weight readers, calibration, numerical and vision references, templates, license and former provider from Thorustinfer commit `cd9025bfbb4a2caf908f4ced3146766e70053b31`. File paths below this directory match that revision. The native backend dependencies for historical A2 experiments live in the separate source archive `thorustinfer-0.2.0-source.tar.xz` (SHA-256 `9d79d43b7dbbdc868f1de764c807128d929051604f6b8219a44d66e81aa5716d`). The local Git branch `archive/deepseek-a2-cd9025b` also fixes that complete tree.

The Thor runtime builds CUDA providers. The historical provider and its Python modules are outside the Rust workspace, Python install and test discovery. The HTTP formatter tests read the preserved chat template. The files under `docs/` describe the historical system. This release contains no DeepSeek Thor execution package or model weights.
