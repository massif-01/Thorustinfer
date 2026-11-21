# 模型执行包

CUDA 执行包使用 schema 2，包含编译后的设备资产和执行数据。模型适配器随 `thorustinfer-models` 静态链接进 `thorustinfer-engine`。引擎读取配置和绑定，通过 Rust 接口组装程序、动态 batch、状态绑定和视觉位置。权重保存在模型目录，计算策略声明各路径使用的混合精度。

核心负责 API、调度、CUDA context/stream、内存、Graph 和 prefix 索引。新增模型族通过 Rust 适配器注册，当前 `int8_quality` 策略支持 Qwen3.5、Qwen3-ASR、Flash Next、Nemotron Embed 和 Qwen Reranker。

```text
MODEL_DIR/
  config.json, tokenizer.json, chat_template.jinja, ...
  cache/model.json
  cache/weights/*.safetensors
  cache/cpu/                         # 可选 CPU 查表数据
  cache/packages/<sha256>/
    package.json
    bindings.json                    # 因子化的调用绑定
    assets.safetensors               # cubin、生成源码和 host ABI
    LICENSE
```

`cache/model.json` 使用 schema 1，`execution_package` 记录对应 `package.json` 的 SHA256。`package.json` 使用 schema 2，记录目标、架构、计算策略、配置签名、buffer 契约、batch/prefill profiles、catalog 和 `bindings.json` 的文件身份。默认共享缓存为 `~/.cache/thorustinfer/packages`，可通过 `THORUSTINFER_EXECUTION_CACHE` 覆盖。

`catalog` 保存共享资产、编译实现定义和动态行数 launch 模板。定义包含 cubin、生成源码、host ABI、symbol、block、shared memory、cooperative、参数类型顺序及只读契约。同一 cubin 可对应多个 ABI/launch 契约，定义去重须比较完整契约。

`bindings.json` 保存因子化的张量角色、共享调用模板、逐层参数和特化选择。离线构建检查绑定展开后的实参、字节偏移、标量位模式和动态 launch。在线引擎读取绑定及 catalog 并检查配置签名、buffer dtype/shape/layout 和边界，由静态链接的适配器组装执行计划。

同类层共享调用模板和特化表，每层提供权重、状态、workspace 角色及必要标量。独立 continuation 参数直接查询模板。

Flash Next 的 HC 无损缓存指数和 QSA radix shift 通过已有 ABI 的整数标量绑定，共享相同计算实现；独立请求的 QSA 复用同一私有状态包装器。QSA 条带数离线测量选择。相邻行数区间仅在全部实参、资产及动态 launch 契约相同时合并。整数投影的固定行数和动态行数使用相同模板与候选空间，动态回退在多个实际行数上测量，选择结果随编译器、设备和源码身份失效；发布包仅包含离线选中的实现。

正常加载保留最终程序、局部特化和专家常驻契约引用的 kernel。`plan-model` 和 `validate-model` 保留完整构建契约，供重建和诊断。CUDA Graph replay 直接执行已捕获的程序。

动态 launch 的全行数验证只在离线构建使用的 `plan-model` 和 `validate-model` 执行，相同容量和表达式只验证一次。正常加载检查每个绑定的参数类型和容量 ABI；实际 launch 继续检查真实行数、算术溢出、除零和非零 grid，行数穷举检查限定在离线验证。

Qwen MTP 执行包可声明 `mtp.packet_table`，将验证、提交、草稿、接受后补算和草稿输出中的请求私有操作数放入固定地址表。包装保持原 kernel 的计算体、grid、标量与共享 workspace 字节视图；Rust 每轮更新当前请求地址，完整 packet Graph 按执行阶段和实际执行形状跨请求复用。提交的缓存行数按请求数计，命中仍要求实际 kernel、grid 和标量一致；验证与补算保留各段实际长度、动态参数和混合角色边界。单请求沿用相同 packet；未声明该契约的包继续使用原有执行方式。启用该能力需重新构建匹配的模型数据与执行包，并验证改变输入、替换请求和 Graph replay 后的全部私有状态。

原生 Qwen 构建入口自动生成该地址表。已有 dense Qwen 执行包采用 sequential prefill、dense decode GDN 且尚无地址表时，可通过 `python/thorustinfer_build/models/qwen35/rebuild_mtp.py --model 原模型目录 --target 新模型目录` 增量构建，保留原权重与矩阵实现。该入口由 `bash tools/environments/cuda/run.sh python/thorustinfer_build/models/qwen35/rebuild_mtp.py 输出目录` 调用，并在末尾传入上述模型参数；发布前验证模型计划与实际 host ABI。

地址表的列序、kernel 映射及标量映射作为共享的不可变描述保存，逐轮复制 MTP 策略时只复制其引用。实际请求地址每轮独立刷新。

启动预加载选择实际 packet 使用的私有 kernel 和拷贝实现，并在权重读取的等待间隙由同一 CUDA worker 完成；已完成的读取优先上传，输入槽复用仍等待对应 DMA。

完整 Graph 缓存保留动态 packet 执行计划的不可变引用。同一计划引用可以快速命中；其他计划仍比较完整调用、grid、标量和字节视图。声明地址表的包按阶段和有序段长复用 owner 无关的计划，新请求只生成地址绑定；混合角色边界属于阶段身份。请求状态绑定分别缓存。prefill、prefix 桥接和生成中的 warm 复用同一 packet 实现。

完整 packet Graph 缓存最多保留 16 个条目，并与共享片段一起遵守 12288 个节点预算；满额时淘汰最久未使用的完整图。阶段切换后，连续执行的新形状逐步替换旧形状，历史累计命中次数不参与淘汰。

模型适配器以 Rust 类型向引擎提供指令、buffer views、动态 launch 参数和请求状态绑定。动态 batch 复用已缓存的计划，并按本轮 owner 绑定请求状态。模型实例与设备执行线程绑定。

有限专家常驻模式使用适配器返回的 `expert_banks`，其中包含专家数、按专家连续存储的张量，以及消费 kernel 的 I32 路由计数位置。引擎按实际绑定读取计数，并按模型契约管理全局 SLRU 和 CUDA VMM 页。默认全量常驻使用执行计划所需的绑定。

模型可声明以 SHA256 固定的 `input_assets`。适配器根据程序、当前 token 和请求历史准备命名 buffer 的上传数据，引擎校验访问权限与上传范围。Flash Next 在 CPU 执行 W8 embedding／E8P PLE 查表，多个请求共享有预算的行缓存，各请求提供自己的 n-gram 历史。

MTP 的验证计划可声明始终提交紧凑 recurrent 更新，也可提供 draft 分支的保存/恢复程序；调度器在提出草稿前保存、刷新已提交前缀前恢复。Flash Next 使用这些通用接口保留 GDN、卷积、PLE、HC 和 QSA pending/index 的完整状态。

CUDA context、stream 和显存统一由引擎管理。包声明执行程序，核心负责执行；CUDA Graph 命中时直接重放已有程序。状态绑定缓存最多 16 个槽位/段长组合，仅缓存逻辑名称，GPU 地址仍从当前请求 arena 解析。prefix 身份包含包 digest；快照复用要求执行策略和状态布局一致。

## 构建与更新

Thor 执行包面向 aarch64 和 SM110a。DeepSeek 模型参考及历史依赖见[参考目录](../reference/deepseek-v41/README.md)。

### 源码入口

统一源码入口：

- `crates/thorustinfer-engine`：通用运行设施。
- `crates/thorustinfer-device`：无驱动依赖的设备及执行契约。
- `crates/thorustinfer-model-sdk` 保存 Rust 计划、状态和执行包数据契约。
- `crates/thorustinfer-models`：模型适配器、独立的精度策略选择和注册表。
- `kernels/`：可复用的 TileLang 算子与融合实现。
- `tools/`：离线编译、权重准备、组装与验证。

适配器描述架构语义与执行组合，策略决定可用精度路径；同一策略可被多个适配器支持，融合可以跨执行组件边界。架构/策略组合完成验证后注册。部署时只绑定模型配置实际需要的 kernel 资产。

`make build` 构建静态链接模型适配器的引擎。离线构建器用 Python 和 TileLang 生成匹配的设备资产，发布时将展开后的参数映射写成 `bindings.json`。构建执行包可复用已有引擎检查执行计划。

在保留权重和 cubin 的情况下，可重新生成执行包的数据描述：

```bash
uv run --locked python -m thorustinfer_build.contracts.catalog \
  --model /path/to/prepared-model --output /path/to/rebuilt-model
```

同文件系统的不可变权重通过 hardlink 复用，模型描述与绑定写入独立文件。构建输出经加载器校验后原子发布。`thorustinfer_build.packaging.attach_execution` 校验替换包的 digest、架构和计算策略，发布时再验证模型与包的兼容性。

## 安装执行包

下载引擎、匹配的执行包和 `SHA256SUMS` 后，先检查下载内容。Rust CLI 在 CPU 上完成安装：

```bash
sha256sum --ignore-missing -c SHA256SUMS
thorustinfer package install /path/to/package.tar.xz
thorustinfer plan-model /path/to/new-model
thorustinfer validate-model /path/to/new-model
```

安装器按文件内容识别 `.tar.xz` 和 `.tar.gz`。CUDA 安装校验包 digest、schema、设备目标、绑定和 safetensors 资产的 SHA256，并穷举验证动态行数。Ascend 安装校验原生计划与 AOT 资产。路径越界、链接和重复归档成员会被拒绝。全部校验通过后原子写入缓存，失败时清理临时目录。

默认安装位置与运行时一致：优先 `THORUSTINFER_EXECUTION_CACHE`，其次 `$XDG_CACHE_HOME/thorustinfer/packages`，最后 `~/.cache/thorustinfer/packages`。`--cache /path/to/packages` 只改变本次安装位置；自定义位置需要同时为后续运行设置 `THORUSTINFER_EXECUTION_CACHE`。

`plan-model` 在 CPU 上通过 Rust 适配器展开计划，验证完整动态行数域及 Graph 分支。`validate-model` 还校验权重、CPU 表、前端、绑定和设备资产的内容。CUDA 默认在线检查配置签名、绑定、kernel ABI、权重和 CPU 表结构，跳过内容哈希；`--verify-weights` 显式开启加载校验。

### 安装已发布的 v0.1.3

v0.1.3 使用系统 tar 安装执行包：下载并通过 `SHA256SUMS` 校验后，将执行包的 digest 目录解压到共享缓存，再运行 `validate-model`：

```bash
export THORUSTINFER_EXECUTION_CACHE="$HOME/.cache/thorustinfer/packages"
mkdir -p "$THORUSTINFER_EXECUTION_CACHE"
tar -xf /path/to/package.tar.gz -C "$THORUSTINFER_EXECUTION_CACHE"
thorustinfer validate-model /path/to/prepared-model
```

### 离线归档

模型构建者使用 Python 工具生成 `.tar.xz` 执行包，供部署时安装：

```bash
uv run --locked python -m thorustinfer_build.packaging.package archive /path/to/package /path/to/package.tar.xz
```

## 合并发布布局

离线调优结束后，将 GPU 权重合并成约 2 GiB 的标准 safetensors 分片，CPU 查表按连续编码段合并。张量名称、物理布局、量化位模式及 CPU 行序保持不变；更新索引、CPU 文件偏移与身份 hash。模型目录只包含模型数据，匹配的执行包独立写入指定缓存：

```bash
uv run --locked python -m thorustinfer_build.packaging.compact \
  --model /path/to/prepared-model --output /path/to/compact-model \
  --package-cache ~/.cache/thorustinfer/packages
thorustinfer validate-model /path/to/compact-model
```

Flash Next 的正式权重发布采用稳定的组件和层组名称：主模型专家每三层一片，其余原权重每八层一片，Q4 投影、输出头、HC 无损压缩表示、MTP 和视觉各自独立；CPU 表复用已合并的连续行分片。单片上限默认为 2 GiB，超限会返回发布错误。重打包直接复用每个张量的量化字节、布局、身份和已编译 kernel：

```bash
uv run --locked python -m thorustinfer_build.models.flash_next.repack \
  --model /path/to/optimized-model --output /path/to/published-model
thorustinfer validate-model /path/to/published-model
```

输出目录及 Hugging Face、ModelScope 模型仓库保存配置、tokenizer、模型描述、GPU 权重与 CPU 表；匹配执行包通过 GitHub Release 独立发行，先安装到公共执行缓存再加载模型。模型描述中的 `execution_package` 固定对应包 digest，重打包保持该 digest。

CUDA 的 `assets.safetensors` 用具名 U8 tensor 收纳 cubin、生成源码与 host ABI，去重字节相同的资产。catalog 的资产身份通过 `tensor` 字段绑定对应 tensor 的原始字节。引擎复用 bundle 的 mmap 和头部索引，并按 kernel 加载 CUDA 模块。

执行包单独归档和安装。离线调优工具通过 `plan-model` 读取展开后的执行契约，完成编译和测量后再发布更新的绑定与 catalog。

## 验证

`make check` 统一执行引擎、SDK 及模型 crate 的格式、clippy、单元测试，同时覆盖绑定展开、包校验与安装原子性。更新还应比较完整程序、launch 参数、buffer 与状态契约；GPU 验证需要匹配模型权重，覆盖请求状态、graph 输入变化、prefix 恢复及 MTP 分支。

通用加载链的 GPU 夹具可独立运行：

```bash
bash tools/environments/cuda/run.sh python/thorustinfer_build/validation/validate_model_package_gpu.py artifacts/model-package/check01
```

夹具验证三种 Graph 模式、输入变化和完整 logits；真实模型质量与性能另行验收。

Flash Next 的当前原生包可通过 `python/thorustinfer_build/models/flash_next/build.py` 重新生成整数投影与执行包。该入口读取模型固定的架构绑定契约，按真实矩阵调优，支持内容校验与断点恢复；状态、浮点归约和联合专家组合保持输入中的已验证实现。输入须为当前原生模型并包含匹配的执行包；量化中间权重需先完成模型组装。用法见 [Flash Next 构建](../tools/model/flash_next/README.md)。
