# Thorustinfer

[English](README.md) · [中文](README.zh-CN.md)

Run a local model on Jetson Thor. A single HTTP address serves OpenAI Chat Completions, OpenAI Responses and Anthropic Messages. Clients can stream text, run tools locally and return their results for the next model turn. The model manager can load models on demand. On a T5000, the idle manager used 9.45 MiB of resident memory (median RSS across three starts). Inference needs additional memory for weights, caches and computation.

## Get started

### 1. Install the engine

On a Jetson Thor running Linux aarch64 with a compatible NVIDIA driver, unpack the archive and run the installer. You can install and run the engine without Python or a conda environment. Use `info` to inspect the engine and `devices` to list the GPUs it can see.

```bash
tar -xJf thorustinfer-0.2.0-linux-aarch64.tar.xz
bash thorustinfer-0.2.0-linux-aarch64/install.sh
CLI="$HOME/.local/opt/thorustinfer/bin/thorustinfer"
"$CLI" info
"$CLI" devices --backend cuda
```

### 2. Load a model

Get a model directory and a matching Thor (SM110a) execution package from the model distributor. The directory should contain the weights, tokenizer, `config.json` and `cache/model.json`. Replace the example path below with that directory. If the execution package comes as a separate archive, follow the [installation guide](tools/thor/README.md) to install it first.

```bash
MODEL_DIR="$HOME/models/my-model"
"$CLI" validate-model "$MODEL_DIR"
"$CLI" serve "$MODEL_DIR" --model local-model --listen 127.0.0.1:8088
```

### 3. Send a request

When the server prints `API READY`, open another terminal. Use the name you passed to `--model` (`local-model` in this example).

```bash
curl -fsS http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"What is 7 plus 5?"}],"temperature":0,"enable_thinking":false,"max_tokens":32}'
```

Read the model's answer in the JSON field `choices[0].message.content`.

## Connect an agent

Use the same model name for direct serving or a registered manager alias. Set the server’s `THORUSTINFER_API_KEY` before startup when using authentication, then set `THOR_LOCAL_KEY` to that value on your client.

- **Codex CLI** uses a custom model provider with `base_url = "http://HOST:8088/v1"`, `wire_api = "responses"` and `supports_websockets = false`. Use HTTP streaming and client-supplied history.
- **Claude Code** uses `ANTHROPIC_BASE_URL=http://HOST:8088` and `ANTHROPIC_API_KEY="$THOR_LOCAL_KEY"`. Select the deployed model with `--model local-model`. Its Messages requests use API version `2023-06-01`.

The installation guide includes copyable configuration, tool replay semantics and the supported API subset. [Agent configuration](tools/thor/README.md#连接-codex-和-claude-code)

## Execution and performance

The Rust server batches requests for the GPU and tracks each request's state. GPU kernels are compiled when the execution package is built. When you load a model, the CLI reads its architecture, compute policy and context limit. Run `"$CLI" info "$MODEL_DIR"` to inspect them.

We tested two Qwen3.5-4B profiles on the T5000. The `e8p` profile uses E8P Q4A8. The `w4a4` profile uses native NVFP4 for the feed-forward projections in all 32 layers. If your model collection provides both, pass the collection directory to `serve` or `batch` and choose `--profile e8p` or `--profile w4a4`. Available profiles depend on the model adapter and execution package.

In the Thor port benchmark, a 2,048-token prompt and 48-token response took **2.80 s with E8P** and **1.76 s with W4A4** (Qwen3.5-4B, T5000, 70 W, one request at a time, MTP off, medians after warm-up). The T4000 uses the same SM110a engine package. Hardware measurements are from the T5000.

[Serving and clients](docs/serving.md) · [Model package format](docs/model-packages.md) · [Building from source](tools/thor/BUILD.md) · [OpenCode example](examples/opencode.json) · [Pi example](examples/pi-models.json)
