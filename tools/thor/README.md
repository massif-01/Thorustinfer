# Thorustinfer

用 Thorustinfer 在 Jetson Thor 上运行本地模型。客户端请选择 OpenAI Chat Completions 协议（`POST /v1/chat/completions`），可接收流式输出及模型生成的工具调用。模型按需加载。T5000 上，空载管理服务占用约 9.45 MiB 常驻内存（RSS，三次启动的中位数）。推理时的内存占用随模型和上下文长度变化。

## 安装并检查 GPU

在设备上解压运行包，运行安装脚本。默认安装目录为 `$HOME/.local/opt/thorustinfer`。安装和运行引擎都不需要 Python，也不用创建 conda 环境。

```bash
tar -xJf thorustinfer-0.1.0-linux-aarch64.tar.xz
bash thorustinfer-0.1.0-linux-aarch64/install.sh
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

如需自行编译，请从同一交付位置取得 `thorustinfer-0.1.0-source.tar.xz`，按其中单列的 `BUILD.md` 准备 Rust、Python 和 CUDA 开发工具。用运行包内的 `licenses/CORRESPONDING-SOURCE.sha256` 核对源码归档。
