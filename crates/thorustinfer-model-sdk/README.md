# thorustinfer-model-sdk

模型 SDK 声明[数据与算子契约](src/artifact.rs)、[执行计划](src/model.rs)及[调度包绑定](src/packet.rs)。模型的架构、量化策略、目标和执行包摘要由模型描述提供。

[Thor NVFP4 验证器](src/nvfp4.rs)检查版本化计算 ABI、传参和读写几何。数据张量名、量化格式及外部执行包中已声明的 ABI 符号属于模型兼容契约；加载器从引擎安装目录取得受信后端库。
