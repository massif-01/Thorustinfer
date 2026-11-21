# Thorustinfer

用 Thorustinfer 在 Jetson Thor 上运行本地模型。服务内置 Chat Completions、OpenAI Responses 和 Anthropic Messages，三者共享同一个模型和 HTTP 地址。模型按需加载。T5000 上，空载管理服务占用约 9.45 MiB 常驻内存（RSS，三次启动的中位数）。推理时的内存占用随模型和上下文长度变化。

## 安装并检查 GPU

在设备上解压运行包，运行安装脚本。默认安装目录为 `$HOME/.local/opt/thorustinfer`。安装和运行引擎都不需要 Python，也不用创建 conda 环境。

```bash
tar -xJf thorustinfer-0.2.1-linux-aarch64.tar.xz
bash thorustinfer-0.2.1-linux-aarch64/install.sh
CLI="$HOME/.local/opt/thorustinfer/bin/thorustinfer"
"$CLI" --version
"$CLI" info
"$CLI" devices --backend cuda
```

T5000 实测环境为 Jetson Linux R38.4、Ubuntu 24.04 aarch64、glibc 2.39 和驱动 580.00。T4000 使用同一份 SM110a 运行包，目前只有 T5000 有实测记录。运行时需要兼容的 NVIDIA 驱动和系统 C/C++ 库。

## 加载外部模型

向模型提供方取得模型目录和匹配的 Thor（SM110a）执行包。把 `MODEL_DIR` 换成你设备上包含权重、分词器、`config.json` 和 `cache/model.json` 的目录。

```bash
MODEL_DIR="$HOME/models/my-model"
```

如果提供方把执行包放在单独的压缩文件里，把 `PACKAGE` 换成该文件的路径，然后安装它。执行包已装好的用户可以跳过这一步。

```bash
PACKAGE="$HOME/Downloads/model-execution.tar.xz"
"$CLI" package install "$PACKAGE" --cache "$MODEL_DIR/cache/packages"
```

核对模型文件，查看模型信息，然后启动服务：

```bash
"$CLI" validate-model "$MODEL_DIR"
"$CLI" info "$MODEL_DIR"
"$CLI" serve "$MODEL_DIR" --model local-model --listen 127.0.0.1:8088
```

看到 `API READY` 后，打开另一个终端。可以用 `curl -fsS http://127.0.0.1:8088/v1/models` 查看服务提供的模型，也可以发送一条聊天请求。请求中的 `local-model` 对应上一步的 `--model`。

```bash
curl -fsS http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"What is 7 plus 5?"}],"temperature":0,"enable_thinking":false,"max_tokens":32}'
```

回答在 JSON 的 `choices[0].message.content` 中。

## 连接 Codex 和 Claude Code

启动服务前，在设备上设置 `THORUSTINFER_API_KEY`。将服务绑定到客户端可达的局域网地址。以下示例中的 `HOST` 替换为设备地址，`local-model` 替换为单模型服务名或管理器内注册的别名。在客户端设置 `THOR_LOCAL_KEY` 为同一密钥。工具在客户端的工作目录和权限范围内执行。

**Codex CLI** 使用 HTTP Responses。在客户端的 `~/.codex/config.toml` 中配置：

```toml
model = "local-model"
model_provider = "thor"
web_search = "disabled"

[model_providers.thor]
name = "Thorustinfer"
base_url = "http://HOST:8088/v1"
wire_api = "responses"
env_key = "THOR_LOCAL_KEY"
requires_openai_auth = false
supports_websockets = false
```

**Claude Code** 使用 Messages，地址填到主机和端口即可。要验证文件工具，可在一个含有 `record.txt` 的工作目录下运行：

```bash
export ANTHROPIC_BASE_URL="http://HOST:8088"
export ANTHROPIC_API_KEY="$THOR_LOCAL_KEY"
claude --bare --model local-model --tools Read --allowedTools Read -p 'Read record.txt and return its contents.'
```

| HTTP 接口 | 支持范围 |
|---|---|
| `POST /v1/chat/completions` | 原有 Chat JSON、SSE、工具调用及 `[DONE]` 结束标记。 |
| `POST /v1/responses` | 前台、无状态请求。传入 `input`、可选 `instructions` 和完整历史。`store` 可省略或设为 `false`。返回 `output` 或具名 SSE 事件。工具输出以 `call_id` 配对，输出项本身另有 `id`。 |
| `POST /v1/messages` | 文本、模型支持的图片、`system`、停止串、工具和流式内容块。指定正整数 `max_tokens`。将 assistant 的完整 `content` 加入历史，再发送含 `tool_result` 的用户消息，`tool_use_id` 对应原调用。 |
| `POST /v1/messages/count_tokens` | 使用实际模型模板及 tokenizer，计入 system、工具定义、历史及图片占位，返回 `input_tokens`。计数不占生成槽位。管理器首次访问该模型时可能初始化 worker。 |

Responses 接受普通函数和自定义文本工具（包括 Codex 的 `apply_patch`）。自定义工具的 grammar 会提供给模型，客户端负责执行并检查实际输入。跨轮请带回完整输出，使用 `function_call_output` 或 `custom_tool_call_output` 回传结果。Messages 的一轮可以同时含有文字和多个工具调用，同名调用也按 ID 匹配。流式客户端应在工具完整结束后执行；输出预算耗尽时，Responses 返回 `incomplete`，Messages 返回 `max_tokens`。

已在 T5000 外部 Qwen3.5-4B 模型上验证 Codex CLI 0.153.4 和 Claude Code 2.1.321 的 HTTP/SSE 文件读取与结果回传。Codex 的自定义补丁工具还需客户端提供可写工作目录并启用相应工具。

Messages 请求设置 `anthropic-version: 2023-06-01`，启用鉴权时使用 `x-api-key` 或 Bearer。Responses 和 Chat 使用 Bearer。0.2.0 的范围为本地模型、HTTP 与客户端工具；服务端保存历史、`previous_response_id`、后台任务、托管搜索、加密推理历史及 WebSocket 均超出此范围。思考开关和预算沿用部署策略，Messages 的 `adaptive` 使用部署默认值。返回给客户端的内容为可见文本和工具，未提供签名思考块。

## 选择执行档位

引擎从模型目录和执行包读取架构、计算策略、上下文上限和每批可处理的输入长度。`info` 显示模型声明的资源需求；设备当前的可用内存决定它能否加载模型和接受请求。

Qwen3.5-4B 在 T5000 上测过 E8P 和 NVFP4 W4A4 两档。要在两档之间选择，将两个模型目录命名为 `e8p` 和 `w4a4`，放在同一个集合目录下。在集合根目录创建 `thorustinfer-profiles.json`：

```json
{"default":"e8p","profiles":["e8p","w4a4"]}
```

把集合目录传给 `serve` 或 `batch`，用 `--profile w4a4` 选用 W4A4，省略该选项则使用清单中的默认档。加载一个模型目录时无需建立集合。其他模型的档位由各自的适配器和执行包决定。

## 批处理、管理和监控

- **离线处理**：准备 JSONL 请求文件，用 `"$CLI" batch "$MODEL_DIR" --model local-model --input requests.jsonl --output results.jsonl` 将回答写入文件。
- **按需加载**：不带模型目录运行 `"$CLI" serve --listen 127.0.0.1:8088` 可启动管理服务。在另一终端设置同样的 `CLI` 和 `MODEL_DIR`，用 `"$CLI" import local-model "$MODEL_DIR"` 注册模型，`"$CLI" list` 查看列表，`"$CLI" load local-model` 加载模型。管理客户端通过本机 Unix socket 联系服务，每个已加载的模型由独立进程运行。
- **鉴权与监控**：设置 `THORUSTINFER_API_KEY` 后，HTTP 请求需携带 `Authorization: Bearer <key>`。管理服务的 `/api/version` 返回产品和版本，模型服务的 `/metrics` 提供前缀为 `thorustinfer_` 的指标。
- **数据与资源**：用 `THORUSTINFER_DATA_DIR` 指定管理数据目录，`THORUSTINFER_EXECUTION_CACHE` 指定执行包缓存。默认路径分别位于用户数据目录和缓存目录的 `thorustinfer` 子目录。`--memory-reserve-mib` 为系统预留内存，`--max-active-requests` 限制并发数。设备已有共享 GPU 锁时，给 `--gpu-lock` 传入该锁的路径。

## 从源码构建

如需自行编译，请从同一交付位置取得 `thorustinfer-0.2.1-source.tar.xz`，按其中单列的 `BUILD.md` 准备 Rust、Python 和 CUDA 开发工具。用运行包内的 `licenses/CORRESPONDING-SOURCE.sha256` 核对源码归档。
