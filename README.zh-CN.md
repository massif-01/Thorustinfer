# Thorustinfer

[English](README.md) · [中文](README.zh-CN.md)

Thorustinfer 在 Jetson Thor 上运行本地模型。同一个 HTTP 地址提供 Chat Completions、OpenAI Responses 和 Anthropic Messages。客户端可以接收流式回答、在本机执行工具，再将结果交给模型继续回答。管理服务可以按需加载模型。在 T5000 上，管理服务空载时占用约 9.45 MiB 常驻内存（RSS，三次启动的中位数）。加载模型后，权重、缓存和计算还会占用内存。

## 快速上手

### 1. 安装引擎

在装有兼容 NVIDIA 驱动的 Jetson Thor（Linux aarch64）上解压运行包，执行安装脚本。安装和运行都不需要 Python，也不用创建 conda 环境。`info` 显示引擎信息，`devices` 列出可用的 GPU。

```bash
tar -xJf thorustinfer-0.2.2-linux-aarch64.tar.xz
bash thorustinfer-0.2.2-linux-aarch64/install.sh
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

## Worker 上下文和 KV 内存

`--context N` 限制单请求的输入与输出 token 总数。使用 `--context auto`，worker 在权重加载后根据实际可用内存计算上限。省略以下三个选项时沿用执行包上限，按需映射 KV 页。

| 选项 | 用途 |
| --- | --- |
| `--kv-cache-mib B` | 预留 B MiB 物理页，供设备端请求 KV 和共享前缀使用。实际大小按 CUDA 页粒度向下对齐。只给 B 时自动计算上下文。 |
| `--context N\|auto` | 指定单请求上限，或计算执行包、物理内存和 KV 预算能容纳的最大长度。显式设置无法兑现时在启动阶段报错。 |
| `--context-concurrency C` | 为 C 个同时达到上限的请求规划容量，默认 1，最高为 `--max-active-requests`。短请求仍可利用剩余容量。 |

独立服务、注册模型的管理模式和离线 batch 使用相同参数。管理器会让需要规划容量的 worker 与其他 worker 错开启动。已就绪 worker 的 `planned_growth_bytes` 单列后续按需增长预算，新 worker 计算上限时扣除这部分空间，已预分配的物理池另计。在 worker 的 `/health` 中读取 `context_plan`，可查看包上限、生效上限、规划并发、请求池字节数、对齐后的实际池字节数和当前占用。`--prefix-cache-mib` 控制快照保留容量，`--memory-reserve-mib` 控制系统预留内存。

## 推测解码

带验证器的 W4A4 执行包支持原生 MTP、greedy Prompt Lookup 和独立草稿。原版 W4A4 包继续支持普通解码。W4A4 默认普通解码，已有 E8P 安装沿用执行包默认值。

| `--spec-method` | 候选来源 |
| --- | --- |
| `none` | 主模型普通解码。 |
| `mtp` | 执行包自带的 MTP 头，支持 greedy 和拒绝采样。 |
| `prompt-lookup` | 从本请求已提交的历史中复制重复片段，未命中时走普通解码。 |
| `draft` | 独立的小模型，要求分词规则、token ID 和特殊 token 一致。 |
| `dflash`、`dflash2`、`dspark` | 使用与目标 revision 配对的草稿包，从声明的主模型中间层读取特征。DFlash2 使用已训练的动态卷积和候选选择器，DSpark 使用 Markov 头，可启用置信度头。 |

用已安装的目标模型和小模型目录启动文本草稿模式。

```sh
thorustinfer serve "$TARGET_MODEL" --spec-method draft --draft-model "$SMALL_MODEL"
```

三种块草稿沿用 `--draft-model`。使用 DSpark 时可加 `--draft-confidence P`，P 取 0 到 1，训练得到的置信度低于 P 时缩短候选。省略时在验证器支持的宽度内使用固定长度。在 CUDA 构建环境运行 `python -m thorustinfer_build.models.block_draft.build --source CHECKPOINT --target-model TARGET --output PACKAGE --cache BUILD_CACHE` 可从本地 checkpoint 生成草稿包。

Prompt Lookup 和独立草稿方法适用于单个纯文本请求的无惩罚 greedy 采样。将 `temperature`、`presence_penalty`、`frequency_penalty` 设为 0，`repetition_penalty` 设为 1。其他采样设置由主模型普通解码。独立小模型和块草稿在多个请求同时活跃或需要 token 掩码时也走普通解码。流式接口只发送已经验证的 token。命中主模型前缀时，从真实已提交历史补齐所需草稿状态。worker 在启动时读取本地草稿目录。

显式选择 `--spec-method` 时独立使用该参数；原有 `--mtp-drafts` 和 `--prompt-lookup` 组合继续可用。旧的 lookup 开关允许未命中时由 MTP 提供候选。worker 的 `/health` 报告所选方法和验证宽度，请求结束日志包含候选数、接受数和验证耗时。
