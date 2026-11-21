# Source, modifications and third-party notices

## Upstream Orinfer

thorustinfer 0.2.1 is derived from [iMMIQ/orinfer](https://github.com/iMMIQ/orinfer).
The upstream baseline is commit
[`b5ce33d5c04d244b292c5f7866d79b2e0cb1165d`](https://github.com/iMMIQ/orinfer/commit/b5ce33d5c04d244b292c5f7866d79b2e0cb1165d).
The Orinfer-derived engine and its modifications are licensed under
**GNU Lesser General Public License version 3 or later (LGPL-3.0-or-later)**.
The source root and the installed `licenses/` directory contain the LGPL in
`LICENSE` and the accompanying GPL in `COPYING`. Components listed below
retain their own copyright notices and licenses.

- **2026-11-19:** Initial Linux aarch64 bring-up on Jetson Thor T5000, including
  native HTTP/SSE inference with external Qwen3.5-4B E8P weights.
- **2026-11-20:** Ported the SM110a build and execution paths, integrated
  NVFP4 FFN execution with the model loader and added checkpoint-derived
  context capacity and multi-packet prefill support to the Thor distribution.
- **2026-11-21:** Delivered the independent `thorustinfer` CLI, service and
  model-free runtime package. Updated public interfaces, tools and installation
  paths while retaining the declared external model formats and calculation
  ABIs. Added the upstream attribution and corresponding-source instructions
  to the distribution on the same date.

- **2026-11-21:** Added native Responses and Messages HTTP endpoints, token counting and client-side tool replay in the 0.2.0 release.

- **2026-11-21:** Scoped the release to CUDA on Thor and retained the DeepSeek model reference sources in a separate directory.

## Corresponding source for the distributed runtime

Distribute `thorustinfer-0.2.1-source.tar.xz` together with
`thorustinfer-0.2.1-linux-aarch64.tar.xz`, in the same delivery directory,
download location or on the same medium. The source archive contains the
modified project, vendored build dependencies, CUTLASS headers and build
instructions in `BUILD.md`.

If you received only the runtime, obtain **`thorustinfer-0.2.1-source.tar.xz`
from the person or organization that supplied that runtime**, or from the same
delivery location. Give the supplier the exact SHA-256 recorded in your
runtime's `licenses/CORRESPONDING-SOURCE.sha256`. To verify the archive,
run the following from the directory containing the downloaded source file,
using the path to your installed or unpacked runtime.

```sh
sha256sum -c /path/to/thorustinfer/licenses/CORRESPONDING-SOURCE.sha256
```

## Other components

Selected test vectors and adapted test scenarios originate from
[Ollama](https://github.com/ollama/ollama), Copyright (c) Ollama, under the MIT
License. The pinned revision, source test names and adaptations are recorded in
`fixtures/ollama/cases.json`; the full license is in `fixtures/ollama/LICENSE`.

The offline compiler uses PyTorch and TorchVision from the official PyTorch CUDA 13.0 aarch64 wheel index. Their URLs and hashes are recorded in `tools/environments/cuda/uv.lock`. The wheels retain their original licenses and notices.

The Engram tokenizer-normalization and addressing rules in
`reference/deepseek-v41/python/thorustinfer_build/models/deepseek_v41/export_engram.py`, the DSv4.1 chat template and
`fixtures/chat/deepseek-v41.json`, the DeepSeek image geometry and
`fixtures/vision/deepseek-v41.json` are adapted from DeepSeek's V4.1
reference inference implementation, Copyright (c) 2023 DeepSeek, under the MIT
License. The full license is in `reference/deepseek-v41/python/thorustinfer_build/models/deepseek_v41/LICENSE`.

The Thor NVFP4 backend uses NVIDIA CUTLASS headers (4.1.0 as carried in the
pinned TileLang build wheel), BSD-3-Clause. The CUTLASS GEMM configuration in
`kernels/cuda/nvfp4/gemm.cuh` adapts FlashInfer's fp4_gemm_template_sm100.h
(Copyright NVIDIA, Apache-2.0). The offline source kit includes the CUTLASS
headers; the binary kit includes their notices, Apache-2.0 and the NVIDIA CUDA
toolkit license for its statically linked runtime. See
`tools/thor/licenses/`. The Qwen model retains its upstream license.
