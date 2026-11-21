# Thorustinfer

[English](README.md) · [中文](README.zh-CN.md)

Thorustinfer 让你的应用通过 OpenAI Chat Completions API（`POST /v1/chat/completions`）调用 Jetson Thor 上的本地模型。接口支持流式输出；模型支持工具调用时，也可返回工具请求。管理服务可以按需加载模型。在 T5000 上，管理服务空载时占用约 9.45 MiB 常驻内存（RSS，三次启动的中位数）。加载模型后，权重、缓存和计算还会占用内存。

## 快速上手

### 1. 安装引擎

在装有兼容 NVIDIA 驱动的 Jetson Thor（Linux aarch64）上解压运行包，执行安装脚本。安装和运行都不需要 Python，也不用创建 conda 环境。`info` 显示引擎信息，`devices` 列出可用的 GPU。

```bash
tar -xJf thorustinfer-0.1.0-linux-aarch64.tar.xz
bash thorustinfer-0.1.0-linux-aarch64/install.sh
CLI="$HOME/.local/opt/thorustinfer/bin/thorustinfer"
"$CLI" info
"$CLI" devices --backend cuda
```

### 2. 加载模型

向模型提供方取得模型目录和匹配的 Thor（SM110a）执行包。模型目录里应有权重、分词器、`config.json` 和 `cache/model.json`。把下面的示例路径换成这份目录的实际路径。如果执行包是另一个压缩文件，请按[安装指南](tools/thor/README.md)先安装它。

```bash
MODEL_DIR="$HOME/models/my-model"
"$CLI" validate-model "$MODEL_DIR"
"$CLI" serve "$MODEL_DIR" --model local-model --listen 127.0.0.1:8088
```

### 3. 发送第一条请求

看到 `API READY` 后，打开另一个终端。`local-model` 是上一步给模型起的服务名。

```bash
curl -fsS http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"What is 7 plus 5?"}],"temperature":0,"enable_thinking":false,"max_tokens":32}'
```

返回的 JSON 中，`choices[0].message.content` 就是模型的回答。

## 执行方式与性能

Rust 服务可以把多个请求合成一批交给 GPU 处理，每个请求保留自己的状态。GPU 算子在构建执行包时编译。换模型时，CLI 会读取新模型的架构、计算策略和上下文上限。用 `"$CLI" info "$MODEL_DIR"` 可以查看这些信息。

Qwen3.5-4B 在 T5000 上测过两档：`e8p` 使用 E8P Q4A8，`w4a4` 使用原生 NVFP4 加速全部 32 层的前馈投影。若模型提供了两档，在 `serve` 或 `batch` 命令中传入模型集合目录，再用 `--profile e8p` 或 `--profile w4a4` 选择。其他模型可用的档位由适配器和执行包决定。

在 Thor 移植阶段的 Qwen3.5-4B 测试中，同一条请求输入 2048 个 token、输出 48 个 token，E8P 用时 **2.80 秒**，W4A4 用时 **1.76 秒**。数据取 T5000 在 70 W、单请求、关闭 MTP 时预热后的中位数。T4000 使用同一份 SM110a 引擎包，目前只有 T5000 有实测记录。

[服务与客户端](docs/serving.md) · [模型包格式](docs/model-packages.md) · [从源码构建](tools/thor/BUILD.md) · [OpenCode 示例](examples/opencode.json) · [Pi 示例](examples/pi-models.json)
