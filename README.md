# Thorustinfer

[English](README.md) · [中文](README.zh-CN.md)

Run a local model on Jetson Thor. A single HTTP address serves OpenAI Chat Completions, OpenAI Responses and Anthropic Messages. Clients can stream text, run tools locally and return their results for the next model turn. The model manager can load models on demand. On a T5000, the idle manager used 9.45 MiB of resident memory (median RSS across three starts). Inference needs additional memory for weights, caches and computation.

## Get started

### 1. Install the engine

On a Jetson Thor running Linux aarch64 with a compatible NVIDIA driver, unpack the archive and run the installer. You can install and run the engine without Python or a conda environment. Use `info` to inspect the engine and `devices` to list the GPUs it can see.

```bash
tar -xJf thorustinfer-0.2.2-linux-aarch64.tar.xz
bash thorustinfer-0.2.2-linux-aarch64/install.sh
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

### 3. Connect a client

When the server prints `API READY`, open a terminal on the machine where you run your client. Set `THOR_URL` to the server address (scheme, host and port only) and `THOR_MODEL` to the name from `--model` or a registered manager alias. The example below connects from the Thor itself. For a client on another machine, bind `--listen` to the Thor’s LAN address and port in step 2, then use that address here.

The default `local` is a placeholder for clients that require a key. If the server was started with `THORUSTINFER_API_KEY`, export its value as `THOR_LOCAL_KEY` on the client before running the examples.

```bash
THOR_URL=http://127.0.0.1:8088
THOR_MODEL=local-model
export THOR_LOCAL_KEY="${THOR_LOCAL_KEY:-local}"

curl -fsS "$THOR_URL/v1/models" -H "Authorization: Bearer $THOR_LOCAL_KEY"
```

Look for your model name in `data[].id`.

#### OpenAI Chat Completions · curl

```bash
curl -fsS "$THOR_URL/v1/chat/completions" \
  -H "Authorization: Bearer $THOR_LOCAL_KEY" -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{"model":"$THOR_MODEL","messages":[{"role":"user","content":"What is 7 plus 5?"}],"enable_thinking":false,"max_tokens":32}
JSON
```

The reply is in `choices[0].message.content`.

#### OpenAI Responses · Codex CLI

Codex sends `POST /v1/responses` over HTTP streaming. Its provider URL includes `/v1`. This command configures one run and asks Codex to read the working directory path.

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

Claude Code adds `/v1/messages` to the server address. Run this example from a directory containing a readable `README.md`. Claude Code should read the file on your client machine and print its first heading.

```bash
ANTHROPIC_BASE_URL="$THOR_URL" ANTHROPIC_API_KEY="$THOR_LOCAL_KEY" \
  claude --bare --model "$THOR_MODEL" --tools Read --allowedTools Read \
  -p 'Use Read to open README.md and return its first heading.'
```

Thorustinfer also serves `POST /v1/messages/count_tokens`. Both agent protocols support client-run tools and client-supplied conversation history. For a persistent Codex provider and protocol details, see the [installation guide](tools/thor/README.md#连接-codex-和-claude-code).

## Execution and performance

The Rust server batches requests for the GPU and tracks each request's state. GPU kernels are compiled when the execution package is built. When you load a model, the CLI reads its architecture, compute policy and context limit. Run `"$CLI" info "$MODEL_DIR"` to inspect them.

We tested two Qwen3.5-4B profiles on the T5000. The `e8p` profile uses E8P Q4A8. The `w4a4` profile uses native NVFP4 for the feed-forward projections in all 32 layers. If your model collection provides both, pass the collection directory to `serve` or `batch` and choose `--profile e8p` or `--profile w4a4`. Available profiles depend on the model adapter and execution package.

In the Thor port benchmark, a 2,048-token prompt and 48-token response took **2.80 s with E8P** and **1.76 s with W4A4** (Qwen3.5-4B, T5000, 70 W, one request at a time, MTP off, medians after warm-up). The T4000 uses the same SM110a engine package. Hardware measurements are from the T5000.

[Serving and clients](docs/serving.md) · [Model package format](docs/model-packages.md) · [Building from source](tools/thor/BUILD.md) · [OpenCode example](examples/opencode.json) · [Pi example](examples/pi-models.json)

[External agent benchmarks](https://github.com/massif-01/Thorustinfer-benchmarks) · [DeepSeek model reference](reference/deepseek-v41/README.md)

## Worker context and KV memory

`--context N` limits the total prompt and output tokens for each request. To size the limit from available memory after loading the model, use `--context auto`. With all three options omitted, the worker uses the package context and maps KV pages on demand.

| Option | Effect |
| --- | --- |
| `--kv-cache-mib B` | Preallocate B MiB for private device KV and shared prefix pages. The worker rounds down to the CUDA page size. Used alone, this option selects automatic context sizing. |
| `--context N\|auto` | Choose an explicit per-request limit or the largest length that fits the package, physical memory and KV budget. An infeasible explicit setting fails at startup. |
| `--context-concurrency C` | Plan for C simultaneous maximum-length requests. The default is one. C must fit within `--max-active-requests`. Shorter requests can still share the remaining capacity. |

The direct server, registered model workers and offline batch accept the same settings. The manager serializes capacity planning with other worker starts. A ready worker reports its planned on-demand growth separately from its physical pool. Later workers subtract that headroom when sizing their own context. Read `context_plan` from a worker's `/health` for the package limit, effective limit, planning concurrency, requested pool size, aligned pool size and live pool occupancy in bytes. `--prefix-cache-mib` controls snapshot retention and `--memory-reserve-mib` sets aside system memory.

## Speculative decoding

The W4A4 verifier package accepts native MTP, greedy prompt lookup and separate draft models. The original W4A4 package serves ordinary decoding. W4A4 defaults to ordinary decoding. Existing E8P installations retain their package default.

| `--spec-method` | Proposal source |
| --- | --- |
| `none` | Ordinary target decoding. |
| `mtp` | The execution package's native MTP head. Supports greedy and rejection sampling. |
| `prompt-lookup` | Copies a repeated continuation from this request. A miss uses ordinary decoding. |
| `draft` | A separately prepared small model with matching tokenizer and special token IDs. |
| `dflash`, `dflash2`, `dspark` | A checkpoint paired to the target revision. The block drafter reads intermediate target layers. DFlash2 uses its trained convolution and candidate selector. DSpark uses its Markov head and optionally its confidence head. |

For example, start an independent text draft with both installed model directories.

```sh
thorustinfer serve "$TARGET_MODEL" --spec-method draft --draft-model "$SMALL_MODEL"
```

The three block methods use the same `--draft-model` option. With DSpark, add `--draft-confidence P` to shorten the block when trained confidence falls below P (0 to 1). Omitting it uses a fixed proposal length within the verifier's supported width. Prepare a block package offline with `python -m thorustinfer_build.models.block_draft.build --source CHECKPOINT --target-model TARGET --output PACKAGE --cache BUILD_CACHE` in the CUDA build environment.

Prompt lookup and the separate draft methods propose for one text request with unpenalized greedy sampling. Set `temperature`, `presence_penalty` and `frequency_penalty` to 0 and `repetition_penalty` to 1. Other sampling settings use the target decoder. The independent and block methods also use ordinary decoding while multiple requests are active or a token mask is required. Only verified tokens reach the stream. Requests with a prefix hit rebuild missing draft state from committed tokens. Each worker loads its draft from a local directory.

Use `--spec-method` by itself or use the original `--mtp-drafts` and `--prompt-lookup` controls. The original lookup flag allows MTP proposals on misses. The worker's `/health` reports the selected method and verifier width. The request completion log reports proposals, acceptances and verification time.
