# 从源码构建 thorustinfer

版本来自源代码内 Cargo workspace 的 package.version。本地独立版本线从 0.1.0 起。源码套件的 source 目录是完整工程，vendor 对应 Cargo.lock，cutlass 是从已固定 SHA 的 TileLang 0.1.15 aarch64 wheel 取得的 CUTLASS 4.1.0 头文件。源码工具的 Python 包和 Cargo 包均使用 thorustinfer 命名。

在 Linux aarch64 上准备 Rust 1.98.1 稳定工具链、Python 3.12、GNU tar/xz、C/C++ 编译工具及兼容 SM110a 的 CUDA 13.3 开发环境。T5000 验收使用 Jetson Linux R38.4、Ubuntu 24.04.4、系统驱动 580.00。工程包含其他后端组件；本次主机为其 C++ 静态依赖使用 Ubuntu Jammy libstdc++-11-dev 11.2.0-19ubuntu1，deb SHA-256 为 4ee878910f92c6ecf70ad61cb4377502713dbb2315bba855d18c5ea751f9ec77。可在用户目录解包该 deb，通过 CXXFLAGS 指向解包后的 GCC11 库目录。

在解出的源码套件根目录执行以下命令，使用新的输出路径。source/.cargo/config.toml 指向随附 vendor。工具链和系统依赖需先备妥。构建无需下载或加载模型。

```bash
export CUDA_HOME=/usr/local/cuda-13.3
export CUTLASS_ROOT="$PWD/cutlass"
export PATH="$(dirname "$(rustup which rustc)"):$CUDA_HOME/bin:$PATH"
# 若使用上述解包的系统 C++ 库
export CXXFLAGS="-B$HOME/build-support/jammy/usr/lib/gcc/aarch64-linux-gnu/11"
SOURCE_ARCHIVE="$(dirname "$PWD")/thorustinfer-0.2.0-source.tar.xz"
cd source
bash tools/thor/build.sh "$HOME/thorustinfer-build-0.2.0" "$SOURCE_ARCHIVE"
```

归档名为 thorustinfer-0.2.0-linux-aarch64.tar.xz。构建器使用锁定的 Cargo 源离线编译，编译引擎自带 NVFP4 库，随后生成安装器、依赖许可证及文件校验和。输出目录内 runtime.sha256 可用于校验归档。从普通 Git checkout 构建时先准备 Cargo.lock 所列依赖或配置 vendor，并将 CUTLASS_ROOT 指向上述头文件。开发检查使用 make check，CPU Python 环境由 uv.lock 管理。

分发时将精确对应的源码归档和运行归档放在一起。构建命令的第二个参数是解包所用的源码归档；脚本把它的 SHA-256 写入运行包内的 licenses/CORRESPONDING-SOURCE.sha256。仅在本地从 Git checkout 开发时，可省略该归档参数。
