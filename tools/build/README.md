# 构建、检查与发布

在线使用 Rust 原生运行时、设备驱动、准备好的模型目录和算子包；Ascend 原生后端同时使用 CANN 运行库。Python、PyTorch、TileLang 均属于离线环境。

## Python 环境与依赖

Python 使用 uv 管理，解释器版本由根目录 `.python-version` 固定。依赖按实际工具用途声明，完整解析结果与 wheel 哈希保存在提交的 `uv.lock`：

| 环境 | 声明和锁文件 | 直接依赖用途 |
| --- | --- | --- |
| CPU | 根目录 `pyproject.toml`、`uv.lock` | NumPy、Safetensors、Tokenizers、Pillow；dev 组的 Ruff |
| CUDA | `tools/environments/cuda/pyproject.toml`、`uv.lock` | 共用 CPU 工具依赖；Torch、TileLang、NVRTC 的 CUDA Python、Torch 原生库需要的 NCCL、Transformers 的视觉和 Chat 模板能力 |

CUDA 项目通过本地虚拟依赖引用根项目，使用自己的独立环境。TVM 来自 TileLang 发布包；FFI、Z3 和其他传递依赖由 uv 按上游约束解析。项目使用的 Jetson Torch wheel 不要求 Triton，当前工具也没有 Triton 调用。

```bash
make python-env
make cuda-env
uv run --locked python -m thorustinfer_build.contracts.catalog --help
uv run --locked --project tools/environments/cuda python -c 'import torch; print(torch.__version__)'
```

`make python-env` 和 `make cuda-env` 先验证发布等待期，再按锁文件同步；`uv run --locked` 执行工具。
依赖须发布满 7 天，两个 uv 项目通过原生 `exclude-newer` 过滤新文件。
新增或更新依赖后先运行对应项目的 `uv lock --upgrade`、`python3 -m tools.dev.dependencies refresh`，
再下载和检查。Cargo 也须先完成发布时间审计，再运行 `cargo fetch --locked`；完整流程见[开发约定](../../docs/development.md)。
常规安装和验收保持锁文件不变。

## 物理机 CUDA 工具链

CUDA 编译与验证直接使用 Jetson AGX Orin 物理机的 CUDA 12.6、驱动及原生 C/C++ 和 Rust 工具链。`nvcc`、Nsight 或 Compute Sanitizer 从机器的 `PATH` 读取；保持设备原有功耗、频率和缓存设置。Python 环境由 uv 单独管理。

Torch 2.14.1+orinfer1、torchvision 0.29.1+orinfer1 使用我们发布的 CPython 3.12 / SM87 / CUDA 12.6 wheel，下载 URL 声明在 CUDA 项目中，内容哈希由锁文件固定。PyPI 通用 CUDA 13 Torch wheel 不适用于这个部署目标。`jetson_wheels.json` 保留源码提交、构建目标和发布资产身份，供更新来源核对。

Torch 与 torchvision 的源码、构建配置和验收入口在 [iMMIQ/pytorch 的 Jetson 分支](https://github.com/iMMIQ/pytorch/tree/jetson-r36.4-cu126-py312-v2.14/jetson) 维护。wheel 验收覆盖 CPU 三角求解/SVD、CUDA 矩阵乘法、torchvision CUDA NMS 和改变输入的 Graph 回放。

```bash
THORUSTINFER_CHECKPOINT_DIR=/path/to/converted \
  bash tools/environments/cuda/run.sh python/thorustinfer_build/models/qwen35/build.py artifacts/build/qwen35 \
  --source /path/to/converted --model-output artifacts/models/qwen35-ready \
  --gpu-lock-held
```

入口先同步锁定的 CPU/CUDA 环境，再获取 `artifacts/gpu-experiment.lock`。模型、参考与编译缓存路径直接引用物理机文件；运行日志、依赖声明与锁文件副本和机器采样写入输出目录。中断时先回收本次工作进程，再释放设备锁；已有服务保持运行。

新环境使用新输出目录执行最小 GPU 编译验收：

```bash
bash tools/environments/cuda/run.sh python/thorustinfer_build/common/smoke.py artifacts/build/compiler-smoke
```

它从空缓存编译生产 norm 与动态行数 greedy 归约，检查非对齐尾部、3/7 行共用 cubin、标量 ABI、输出 guard 和改变输入的 Graph replay，并导出 AOT。算子数值和状态专项检查分别使用 `tests/cuda/operators` 和 `tests/ascend/operators` 中的验收入口。Ascend 使用独立的 CANN 工具链，入口与支持范围见[模型包文档](../../docs/model-packages.md)。

## 自动检查

`make check` 包含 Rust/Ruff 格式、Python 静态检查、编译、clippy、Rust/CPU Python 测试和验收配置校验。GitHub Actions 在 x86_64 和 aarch64 runner 使用相同入口，Python 与 uv 版本固定；Rust 版本由 `rust-toolchain.toml` 固定。

`make fmt` 统一格式，`make lint` 执行格式和静态检查。`make check-offline` 通过 CUDA uv 环境检查离线模块导入，并在 CPU 上执行需要 Torch 的参考数学、前端和动态行数 ABI 测试。

```bash
make check-gpu MODEL=/path/to/prepared-model OUTPUT=artifacts/check-gpu/new-run
```

该入口生成原生 Chat 模板的文本、图片、多图夹具，固定 seed20261002，依次验证 off/decode_only/full 模式的动态 batch、取消/重排/复用隔离、请求丢弃清理和同历史概率。独立 BF16 质量验收另行执行。

## thorustinfer 引擎构建

源码套件含 Rust 源码、Cargo vendor 依赖、固定 CUTLASS 头文件及系统工具链说明。根据套件根目录的 BUILD.md 配置 PATH、CUDA_HOME 和 CUTLASS_ROOT。在源码根目录运行：

```bash
bash tools/thor/build.sh "$PWD/artifacts/thorustinfer-release"
```

输出目录必须是新目录。脚本从 Cargo workspace 读取产品版本，生成 aarch64 CLI、SM110a NVFP4 后端库、安装器、使用说明、许可证与 SHA-256。运行目录不需要 Python。外部模型执行包通过 `thorustinfer package install` 独立安装。
