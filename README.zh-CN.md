# Thorustinfer

[English](README.md) · [中文](README.zh-CN.md)

Thorustinfer 在 Jetson Thor 上运行本地模型。同一个 HTTP 地址提供 Chat Completions、OpenAI Responses 和 Anthropic Messages。客户端可以接收流式回答、在本机执行工具，再将结果交给模型继续回答。管理服务可以按需加载模型。在 T5000 上，管理服务空载时占用约 9.45 MiB 常驻内存（RSS，三次启动的中位数）。加载模型后，权重、缓存和计算还会占用内存。

## 快速上手

### 1. 安装引擎

在装有兼容 NVIDIA 驱动的 Jetson Thor（Linux aarch64）上解压运行包，执行安装脚本。安装和运行都不需要 Python，也不用创建 conda 环境。`info` 显示引擎信息，`devices` 列出可用的 GPU。

```bash
tar -xJf thorustinfer-0.2.1-linux-aarch64.tar.xz
bash thorustinfer-0.2.1-linux-aarch64/install.sh
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

### 3. 连接客户端

看到 `API READY` 后，在运行客户端的机器上打开终端。下面三个示例共用两个值。把 `THOR_URL` 设为服务地址，写到端口即可。把 `THOR_MODEL` 设为启动时 `--model` 指定的名字，或管理器里注册的别名。示例地址适合在 Thor 本机使用。如果在另一台电脑上运行客户端，请在第 2 步将 `--listen` 改为 Thor 的局域网 IP 和端口，在这里填同一地址。

默认值 `local` 是给需要密钥的客户端用的占位值。如果服务启动时设置了 `THORUSTINFER_API_KEY`，请先在客户端将 `THOR_LOCAL_KEY` 环境变量设为同一密钥，再运行示例。

```bash
THOR_URL=http://127.0.0.1:8088
THOR_MODEL=local-model
export THOR_LOCAL_KEY="${THOR_LOCAL_KEY:-local}"

curl -fsS "$THOR_URL/v1/models" -H "Authorization: Bearer $THOR_LOCAL_KEY"
```

在返回的 `data[].id` 中可以看到模型名。

#### OpenAI Chat Completions · curl

```bash
curl -fsS "$THOR_URL/v1/chat/completions" \
  -H "Authorization: Bearer $THOR_LOCAL_KEY" -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{"model":"$THOR_MODEL","messages":[{"role":"user","content":"What is 7 plus 5?"}],"enable_thinking":false,"max_tokens":32}
JSON
```

在返回的 `choices[0].message.content` 中查看回答。

#### OpenAI Responses · Codex CLI

Codex 通过 HTTP 流调用 `POST /v1/responses`，它的 provider 地址要带 `/v1`。下面的配置只对这次运行生效。命令会让 Codex 调用 shell 工具，返回客户端当前目录的路径。

```bash
codex exec --ignore-user-config --skip-git-repo-check --sandbox read-only --model "$THOR_MODEL" \
  -c 'model_provider="thor"' \
  -c 'model_providers.thor.name="Thorustinfer"' \
  -c "model_providers.thor.base_url=\"$THOR_URL/v1\"" \
  -c 'model_providers.thor.wire_api="responses"' \
  -c 'model_providers.thor.env_key="THOR_LOCAL_KEY"' \
  -c 'model_providers.thor.requires_openai_auth=false' \
  -c 'model_providers.thor.supports_websockets=false' \
  -c 'web_search="disabled"' \
  'Use the shell tool to print the current directory. Return the path.'
```

#### Anthropic Messages · Claude Code

Claude Code 会在服务地址后加上 `/v1/messages`。在含有 `README.md` 的目录中运行下面的命令。连通后，Claude Code 会调用客户端的 Read 工具读取文件，并返回第一个标题。

```bash
ANTHROPIC_BASE_URL="$THOR_URL" ANTHROPIC_API_KEY="$THOR_LOCAL_KEY" \
  claude --bare --model "$THOR_MODEL" --tools Read --allowedTools Read \
  -p 'Use Read to open README.md and return its first heading.'
```

Thorustinfer 也提供 `POST /v1/messages/count_tokens`。两套 Agent 接口都由客户端执行工具并传回对话历史。需要保存 Codex 配置或查看接口支持范围，请参阅[安装指南](tools/thor/README.md#连接-codex-和-claude-code)。

## 执行方式与性能

Rust 服务可以把多个请求合成一批交给 GPU 处理，每个请求保留自己的状态。GPU 算子在构建执行包时编译。换模型时，CLI 会读取新模型的架构、计算策略和上下文上限。用 `"$CLI" info "$MODEL_DIR"` 可以查看这些信息。

Qwen3.5-4B 在 T5000 上测过两档：`e8p` 使用 E8P Q4A8，`w4a4` 使用原生 NVFP4 加速全部 32 层的前馈投影。若模型提供了两档，在 `serve` 或 `batch` 命令中传入模型集合目录，再用 `--profile e8p` 或 `--profile w4a4` 选择。其他模型可用的档位由适配器和执行包决定。

在 Thor 移植阶段的 Qwen3.5-4B 测试中，同一条请求输入 2048 个 token、输出 48 个 token，E8P 用时 **2.80 秒**，W4A4 用时 **1.76 秒**。数据取 T5000 在 70 W、单请求、关闭 MTP 时预热后的中位数。T4000 使用同一份 SM110a 引擎包，目前只有 T5000 有实测记录。

[服务与客户端](docs/serving.md) · [模型包格式](docs/model-packages.md) · [从源码构建](tools/thor/BUILD.md) · [OpenCode 示例](examples/opencode.json) · [Pi 示例](examples/pi-models.json)

[独立 Agent 基准](https://github.com/massif-01/Thorustinfer-benchmarks) · [DeepSeek 模型参考](reference/deepseek-v41/README.md)
