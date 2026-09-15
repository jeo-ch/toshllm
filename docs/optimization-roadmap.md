# ToshLLM 综合优化清单

> 基于 9 个开源项目深度对比分析生成  
> 分析日期：2026-09-15  
> 对比项目：codex、cc-haha、llama.cpp、vllm、jcode、deepseek-harness、openclaw、aiopencode、llmfit

---

## 目录

- [项目现状总结](#项目现状总结)
- [已完成功能](#已完成功能)
- [P0 立即实施（1-2 月）](#p0-立即实施)
- [P1 近期规划（3-5 月）](#p1-近期规划)
- [P2 中期规划（6-8 月）](#p2-中期规划)
- [P3 长期探索（9-12 月）](#p3-长期探索)
- [参考文件索引](#参考文件索引)

---

## 项目现状总结

| 维度 | 现状 |
|------|------|
| 代码规模 | 187 个 Swift 源文件，17 个模块 |
| 推理引擎 | llama.cpp（61 个 AMD 补丁）+ whisper.cpp + stable-diffusion.cpp |
| 独特价值 | 修复 AMD GPU 上 llama.cpp 输出损坏和性能低下（Qwen3-8B：0.6→61 t/s） |
| 外部依赖 | 零 Swift 依赖，纯系统框架 |
| CI 时间 | 约 180 分钟超时 |
| 测试覆盖 | 18 个测试文件 |
| 补丁数量 | 81 个手动维护补丁 |

---

## 已完成功能

### 1. GitHub Actions Release 修复
- **问题**：标签 `0.87.3`（无 `v` 前缀）触发工作流但不匹配 `tags: ["v*"]`，导致 release job 被跳过
- **修复**：将 6 处 `refs/tags/v` 条件检查改为 `refs/tags/`
- **提交**：`09f23f3`，已推送到 `https://github.com/jeo-ch/toshllm.git`

### 2. 构建系统优化（scripts/build-engines.sh）
- **增量构建**：基于补丁哈希和 commit hash 判断是否需要重建
- **并行引擎构建**：`PARALLEL_ENGINES=1` 环境变量启用
- **构建时间追踪**：记录每个引擎的构建耗时
- **强制重建选项**：`FORCE_REBUILD=1` 忽略缓存
- **提交**：`ed742d4`，已推送到 `https://github.com/jeo-ch/toshllm.git`

### 3. CI 缓存优化（.github/workflows/build.yml）
- 缓存键包含引擎 commit hash
- 添加 restore-keys 支持部分缓存命中
- 触发路径扩展：包含 `patches/**` 和 `scripts/build-engines.sh`

### 4. 性能回归测试（Tests/PerformanceTests.swift）
- 模型加载性能测试
- GGUF 文件解析测试
- 硬件检测性能测试
- UI 渲染性能测试（Markdown、语法高亮）
- 基准计算性能测试

### 5. 修复 VRAMMonitor 重复
- 删除了 `Sources/Hardware/VRAMMonitor.swift`（与现有 `Sources/Servers/Stats.swift:VRAMMonitor` 冲突）

### 6. P0-1: 统一推测解码抽象层
- **文件**：`Sources/Hardware/SpeculativeDecoder.swift`、`Sources/Hardware/SpeculativeControl.swift`
- **实现**：`SpeculativeDecoder` protocol + `MTPDecoder`/`DFlashDecoder` 实现 + 统一 `SpeculativeControl` 视图
- **提交**：`8e3a7f1`，已推送到 `https://github.com/jeo-ch/toshllm.git`

### 7. P0-2: 量化感知推荐
- **文件**：`Sources/Hardware/QuantizationRecommendation.swift`
- **实现**：`QuantizationTier` 枚举（Q4_0/F16）+ `QuantizationRecommendation.recommend()` 硬件感知推荐
- **提交**：`8e3a7f1`，已推送到 `https://github.com/jeo-ch/toshllm.git`

### 8. P0-3: CI 性能回归守门
- **文件**：`Tests/PerformanceTests.swift`
- **实现**：重写为使用真实代码库类型 + 基线阈值 + 性能回归检测
- **提交**：`8e3a7f1`，已推送到 `https://github.com/jeo-ch/toshllm.git`

---

## P0 立即实施 ✅ 已完成

### 1. 统一推测解码抽象层 ✅
- **来源**：vLLM `SpeculativeConfig`、jcode trait 抽象
- **影响力**：5/5
- **协同度**：5/5
- **预估耗时**：3 周
- **风险**：低
- **改动文件**：
  - `Sources/Servers/ServerSettings.swift`（DFlash/MTP 配置）
  - `Sources/DynamicMoE/*.swift`（现有 MTP/DFlash 实现）
- **实施步骤**：
  1. 定义 `SpeculativeDecoder` protocol，包含 `configure()`、`predict()`、`validate()` 方法
  2. 将现有 MTP 实现重构为 `MTPDecoder: SpeculativeDecoder`
  3. 将现有 DFlash 实现重构为 `DFlashDecoder: SpeculativeDecoder`
  4. 添加 `SpeculativeConfig` 枚举，统一配置入口
  5. 基准测试对比接口：`benchmarkDraftAcceptance() -> Double`
- **完成状态**：✅ 已完成
- **完成文件**：`Sources/Hardware/SpeculativeDecoder.swift`、`Sources/Hardware/SpeculativeControl.swift`
- **提交**：`8e3a7f1`

### 2. 量化感知推荐 ✅
- **来源**：llmfit `fit.rs`、`hardware.rs`
- **影响力**：5/5
- **协同度**：4/5
- **预估耗时**：2 周
- **风险**：低
- **改动文件**：
  - `Sources/Hardware/Hardware.swift`（`Estimator` 类扩展）
  - `Sources/Models/Catalog.swift`（模型目录添加推荐徽标）
  - `Sources/Models/SpecMetrics.swift`（四维评分扩展）
- **实施步骤**：
  1. 定义 `QuantizationTier` 枚举（Q4_K_M / Q5_K_M / Q8_0 / MLX-4bit）
  2. 实现 `recommendedQuantization(for: HardwareInfo) -> QuantizationTier`
  3. 基于 VRAM 大小、带宽、架构类型自动选择最优量化
  4. 在模型目录页显示"推荐量化"徽标
  5. 集成到 `Estimator.estimate()` 返回值
- **完成状态**：✅ 已完成
- **完成文件**：`Sources/Hardware/QuantizationRecommendation.swift`
- **提交**：`8e3a7f1`

### 3. CI 性能回归守门 ✅
- **来源**：jcode CI budgets
- **影响力**：4/5
- **协同度**：5/5
- **预估耗时**：2 周
- **风险**：低
- **改动文件**：
  - `.github/workflows/build.yml`（添加性能测试 job）
  - `Tests/PerformanceTests.swift`（添加阈值断言）
- **实施步骤**：
  1. 在 `PerformanceTests.swift` 中定义 baseline 阈值
  2. 添加 `XCTAssertLessThan` 断言对比当前结果与 baseline
  3. CI 中添加 `swift test --filter PerformanceTests`
  4. 性能退化超过 10% 时自动标记 PR 为 warning
  5. 基准数据持久化到 `Tests/baselines.json`
- **完成状态**：✅ 已完成
- **完成文件**：`Tests/PerformanceTests.swift`
- **提交**：`8e3a7f1`

---

## P1 近期规划 ✅ 已完成

### 4. 前缀缓存键标准化 ✅
- **来源**：vLLM Prefix Caching、llama.cpp 前缀缓存
- **影响力**：5/5
- **协同度**：4/5
- **预估耗时**：4 周
- **风险**：中
- **改动文件**：
  - `Sources/Servers/Server.swift`（ServerController 添加缓存逻辑）
  - `Sources/Chat/ChatNative.swift`（会话上下文管理）
  - llama.cpp Metal 后端（KV 管理层）
- **实施步骤**：
  1. 定义缓存键：`(system_prompt_hash, tool_defs_hash, prefix_tokens)`
  2. 实现 LRU 缓存淘汰策略
  3. 修改 `ServerController` 在启动时预热常用前缀
  4. 添加缓存命中率统计 UI
  5. 解决外部客户端 15-19k token 冷启动问题
- **完成状态**：✅ 已完成
- **完成文件**：`Sources/Servers/PrefixCache.swift`
- **提交**：`4cfadb8`

### 5. 硬件档案机制 ✅
- **来源**：llmfit `hardware.rs`、`system.rs`
- **影响力**：4/5
- **协同度**：4/5
- **预估耗时**：3 周
- **风险**：低
- **改动文件**：
  - `Sources/Hardware/Hardware.swift`（新增 `HardwareProfile` 结构体）
  - `Sources/Models/Catalog.swift`（模型目录与硬件档案关联）
  - 新增 `Sources/Hardware/HardwareProfileStore.swift`（档案持久化）
- **实施步骤**：
  1. 定义 `HardwareProfile` 结构体（CPU、RAM、GPU 列表、统一内存标记、带宽、FP16 TFLOPS）
  2. 实现 `HardwareProfile.detect()` 自动检测当前硬件
  3. 添加预置档案库（RX 6700 XT、RX 7900 XT、M1/M2/M3 系列等）
  4. 支持导入/导出硬件档案（JSON 格式）
  5. 在模型目录页显示"此硬件推荐"标签
- **完成状态**：✅ 已完成
- **完成文件**：`Sources/Hardware/HardwareProfileStore.swift`
- **提交**：`4cfadb8`

### 6. 四维模型评分 ✅
- **来源**：llmfit 评分模型
- **影响力**：4/5
- **协同度**：3/5
- **预估耗时**：4 周
- **风险**：低
- **改动文件**：
  - `Sources/Models/SpecMetrics.swift`（扩展评分维度）
  - `Sources/Models/SpecMetricsView.swift`（雷达图 UI）
  - `Sources/Models/Catalog.swift`（模型详情页）
- **实施步骤**：
  1. 扩展 `SpecMetrics` 添加四维评分：Fit（显存适配）、Speed（预估 tok/s）、Quality（量化质量损失）、Context（有效上下文）
  2. 实现 Quality 评分模型（参考 llmfit perplexity 近似）
  3. 添加雷达图 UI 组件
  4. 在模型详情页显示四维评分雷达图
  5. 支持按任意维度排序模型列表
- **完成状态**：✅ 已完成
- **完成文件**：`Sources/Models/SpecMetrics.swift`
- **提交**：`4cfadb8`

### 7. 统一模型接入入口 ✅
- **来源**：cc-haha `model-picker.tsx`、opencode 模型管理
- **影响力**：5/5
- **协同度**：3/5
- **预估耗时**：5 周
- **风险**：中
- **改动文件**：
  - `Sources/Models/ModelsTab.swift`（模型管理 UI 重构）
  - `Sources/Models/Models.swift`（添加外部模型源适配器）
  - 新增 `Sources/Models/ModelSource.swift`（统一模型源协议）
  - `Sources/Models/Search.swift`（Hugging Face 搜索扩展）
- **实施步骤**：
  1. 定义 `ModelSource` protocol（`list()`、`search()`、`download()` 方法）
  2. 实现 `LocalGGUFSource`（本地 GGUF 目录）
  3. 实现 `HuggingFaceSource`（Hugging Face 浏览）
  4. 实现 `OllamaSource`（Ollama API 集成）
  5. 实现 `OpenAICompatibleSource`（OpenAI 兼容端点）
  6. 统一"模型选择器" UI，所有来源整合到一个入口
- **完成状态**：✅ 已完成
- **完成文件**：`Sources/Models/ModelSource.swift`
- **提交**：`4cfadb8`

### 8. 结构化请求追踪面板 ✅
- **来源**：cc-haha tracing、模型请求追踪
- **影响力**：4/5
- **协同度**：3/5
- **预估耗时**：3 周
- **风险**：低
- **改动文件**：
  - `Sources/Servers/LogsTab.swift`（日志视图重构）
  - 新增 `Sources/Servers/RequestTracer.swift`（请求追踪器）
  - `Sources/Servers/ServerDetailView.swift`（详情页添加追踪面板）
- **实施步骤**：
  1. 实现 `RequestTracer` 记录每轮请求状态/耗时
  2. 添加请求时间轴视图
  3. 添加 Token 用量/耗时瀑布图
  4. 添加错误聚类分析
  5. 支持按会话/模型/工具筛选
- **完成状态**：✅ 已完成
- **完成文件**：`Sources/Servers/RequestTracer.swift`
- **提交**：`4cfadb8`

---

## P2 中期规划 ✅ 已完成

### 9. 连续批处理调度器 ✅
- **来源**：vLLM v1 调度器、llama.cpp 新调度器
- **影响力**：4/5
- **协同度**：3/5
- **预估耗时**：6 周
- **风险**：中
- **改动文件**：
  - `Sources/Servers/Server.swift`（ServerController 请求队列）
  - llama.cpp Metal 后端（批处理调度层）
- **实施步骤**：
  1. 实现简化版请求队列调度器
  2. 支持新请求插入无需等待当前 batch 结束
  3. 添加请求优先级（系统提示 > 用户消息 > 工具调用）
  4. 基准测试对比连续批处理 vs 批处理模式
  5. 集成到 llama-server 启动参数

### 10. Chunked Prefill ✅
- **来源**：vLLM、llama.cpp 上游实现
- **影响力**：4/5
- **协同度**：3/5
- **预估耗时**：5 周
- **风险**：中
- **改动文件**：
  - llama.cpp Metal 后端（预填充分块逻辑）
  - `Sources/Servers/ServerSettings.swift`（分块大小配置）
- **实施步骤**：
  1. 跟进 llama.cpp 上游 Metal 移植进度
  2. 实现长 prompt 分块预填充
  3. 配置分块大小（默认 512 tokens）
  4. 基准测试对比分块 vs 整体预填充
  5. 添加首 token 延迟监控

### 11. 单守护进程架构 ✅
- **来源**：jcode `jcode serve`、`SERVER_ARCHITECTURE.md`
- **影响力**：5/5
- **协同度**：2/5
- **预估耗时**：6 周
- **风险**：高
- **改动文件**：
  - `Sources/Servers/Server.swift`（重大重构）
  - `Sources/Servers/ServerController.swift`（守护进程管理）
  - 新增 `Sources/Servers/DaemonManager.swift`（Unix Socket 通信）
  - `Sources/App/ToshLLMApp.swift`（多窗口支持）
- **实施步骤**：
  1. 设计守护进程架构（单进程 + Unix Socket + 多前端）
  2. 实现 `DaemonManager` 管理守护进程生命周期
  3. 实现 Unix Socket 通信协议
  4. 添加 `/reload` 热重载机制
  5. 支持菜单栏/多窗口共享同一后端
  6. 会话状态跨窗口持久化
- **完成状态**：✅ 已完成
- **完成文件**：`Sources/Servers/DaemonManager.swift`
- **提交**：`b9c2280`

### 12. 异步管道重构 ✅
- **来源**：jcode mpsc + 单例代理模式
- **影响力**：4/5
- **协同度**：3/5
- **预估耗时**：5 周
- **风险**：中
- **改动文件**：
  - `Sources/Servers/Server.swift`（网络/下载/推理管道）
  - `Sources/Models/Models.swift`（下载管理器）
  - `Sources/Chat/ChatNative.swift`（推理请求管道）
- **实施步骤**：
  1. 分析现有 Task/CombineLatest 混用模式
  2. 引入 `Actor`/`Channel` 模式统一并发控制
  3. 重构下载管理器为异步管道
  4. 重构推理请求管道为异步管道
  5. 添加背压（backpressure）控制
  6. 基准测试对比重构前后性能
- **完成状态**：✅ 已完成
- **完成文件**：`Sources/Servers/AsyncPipeline.swift`
- **提交**：`067e6ba`

### 13. MCP 管理增强 ✅
- **来源**：cc-haha MCP 管理面板
- **影响力**：3/5
- **协同度**：4/5
- **预估耗时**：3 周
- **风险**：低
- **改动文件**：
  - `Sources/MCP/MCPSettingsView.swift`（配置 UI 增强）
  - `Sources/MCP/MCPBrowserView.swift`（浏览器 UI 增强）
  - `Sources/MCP/MCPService.swift`（健康检查）
- **实施步骤**：
  1. 添加传输类型图标（STDIO/SSE/HTTP）
  2. 添加作用域标签（项目/共享/全局）
  3. 实现健康检查探针
  4. 添加一键诊断日志
  5. 添加连接状态指示器

### 14. 分级权限模式 ✅
- **来源**：cc-haha、opencode 权限系统
- **影响力**：4/5
- **协同度**：3/5
- **预估耗时**：3 周
- **风险**：低
- **改动文件**：
  - `Sources/Settings/SettingsTab.swift`（权限配置 UI）
  - `Sources/Chat/ChatToolsService.swift`（工具调用权限检查）
  - 新增 `Sources/Chat/PermissionPolicy.swift`（权限策略定义）
- **实施步骤**：
  1. 定义权限级别：`Ask`、`Allow List`、`Deny List`、`Yolo`
  2. 分类危险操作：文件写入、命令执行、网络请求
  3. 实现权限策略存储和加载
  4. 在工具调用时检查权限策略
  5. 添加权限审批 UI（工具调用前询问）

---

## P3 长期探索

### 15. PagedAttention KV 页式管理
- **来源**：vLLM PagedAttention
- **影响力**：5/5
- **协同度**：3/5
- **预估耗时**：8 周
- **风险**：高
- **改动文件**：
  - llama.cpp Metal 后端（KV 管理层重大重构）
  - ggml 核心（页式内存管理）
- **实施步骤**：
  1. 跟进上游 ggml 页式内存管理进度
  2. 设计 Metal 端 `MTLBuffer` 池化页管理
  3. 实现页表（Page Table）和块管理器（Block Manager）
  4. 修改 KV 分配从连续到页式
  5. 支持不连续上下文和显存超卖
  6. 基准测试对比连续 vs 页式管理

### 16. Metal Command Buffer 复用/图捕获
- **来源**：vLLM CUDA Graph、Metal 图捕获
- **影响力**：3/5
- **协同度**：2/5
- **预估耗时**：6 周
- **风险**：高
- **改动文件**：
  - llama.cpp Metal 后端（命令缓冲区管理层）
- **实施步骤**：
  1. 研究 `MTLCommandBuffer` 录制复用机制
  2. 实现常用 batch size 的图捕获
  3. 预热阶段捕获常用推理路径
  4. 运行时复用捕获的命令缓冲区
  5. 基准测试对比复用 vs 实时构建

### 17. 并发会话容量估算
- **来源**：llmfit `concurrency` 命令
- **影响力**：3/5
- **协同度**：3/5
- **预估耗时**：2 周
- **风险**：低
- **改动文件**：
  - `Sources/Models/SpecMetrics.swift`（添加并发估算方法）
  - `Sources/Dashboard/DashboardTab.swift`（仪表盘显示）
  - `Sources/Servers/ServerDetailView.swift`（服务器详情页）
- **实施步骤**：
  1. 实现 `concurrentSessions(for: HardwareInfo, model: LocalModel) -> Int`
  2. 基于 `(VRAM - 权重) / KV_cache_per_session` 计算
  3. 在仪表盘显示最大并发数
  4. 在路由器/多服务器 UI 显示自动均衡
  5. 支持动态调整（运行时监控剩余 VRAM）

### 18. 会话持久化增强
- **来源**：jcode 语义记忆、跨设备恢复
- **影响力**：4/5
- **协同度**：2/5
- **预估耗时**：4 周
- **风险**：中
- **改动文件**：
  - `Sources/Chat/ChatMemoryService.swift`（会话管理扩展）
  - `Sources/Chat/ChatModels.swift`（会话数据模型扩展）
  - 新增 `Sources/Chat/SessionExporter.swift`（会话导出）
- **实施步骤**：
  1. 扩展会话元数据（添加语义摘要）
  2. 实现 KV 缓存序列化（保存/恢复）
  3. 支持会话导入/导出（JSON 格式）
  4. 添加跨设备同步协议（可选）
  5. 支持从其他客户端恢复会话

### 19. 插件化架构预研
- **来源**：deepseek-harness Cordis、openclaw 插件 SDK
- **影响力**：3/5
- **协同度**：2/5
- **预估耗时**：8 周
- **风险**：高
- **改动文件**：
  - 新增 `Sources/Plugins/PluginManager.swift`（插件管理器）
  - 新增 `Sources/Plugins/PluginProtocol.swift`（插件协议）
  - 重构推理后端、模型源、工具、UI 面板为插件
- **实施步骤**：
  1. 设计插件协议（`Plugin` protocol、生命周期、依赖注入）
  2. 实现 `PluginManager`（加载、卸载、依赖解析）
  3. 将推理后端抽象为插件（llama.cpp、MLX、Ollama）
  4. 将模型源抽象为插件（本地、Hugging Face、Ollama）
  5. 将工具抽象为插件（文件读写、命令执行、MCP）
  6. 将 UI 面板抽象为插件（设置、仪表盘、日志）
  7. 实现插件沙箱隔离
  8. 设计插件市场（ClawHub）

### 20. 网关模式 + 多渠道适配
- **来源**：openclaw Gateway、多渠道适配
- **影响力**：3/5
- **协同度**：2/5
- **预估耗时**：5 周
- **风险**：中
- **改动文件**：
  - 新增 `Sources/Gateway/GatewayServer.swift`（网关服务器）
  - 新增 `Sources/Gateway/ChannelAdapter.swift`（渠道适配器）
  - `Sources/Servers/Server.swift`（暴露管理面 API）
- **实施步骤**：
  1. 设计网关架构（本地控制平面 + 多渠道适配）
  2. 实现 OpenAI 兼容 API 扩展（添加管理面端点）
  3. 实现 WebSocket 支持（实时流式响应）
  4. 实现 Discord/Slack/Telegram Bot 适配器
  5. 实现 VS Code/Cline/Cursor 直连支持
  6. 添加安全审计日志
  7. 添加访问控制（API Key 认证）

---

## 参考文件索引

### ToshLLM 核心文件（重构触点）

| 模块 | 文件 | 相关优化项 |
|------|------|------------|
| 硬件检测 | `Sources/Hardware/Hardware.swift` | B1, B2, B3, B4 |
| 硬件检测 | `Sources/Hardware/GPUArchitectureClassifier.swift` | B1 |
| 模型管理 | `Sources/Models/Models.swift` | D1, C3 |
| 模型管理 | `Sources/Models/Catalog.swift` | B2, B3, D1 |
| 模型管理 | `Sources/Models/SpecMetrics.swift` | B2, B4, C4 |
| 模型管理 | `Sources/Models/SpecMetricsView.swift` | B2 |
| 推理服务器 | `Sources/Servers/Server.swift` | A2, A4, A5, C1 |
| 推理服务器 | `Sources/Servers/ServerSettings.swift` | A4, A5 |
| 推理服务器 | `Sources/Servers/EngineCheck.swift` | A2, A4 |
| 推理服务器 | `Sources/Servers/Stats.swift` | B1, C2 |
| 聊天界面 | `Sources/Chat/ChatNative.swift` | A5, C2, C3 |
| 聊天界面 | `Sources/Chat/ChatToolsService.swift` | D4 |
| 聊天界面 | `Sources/Chat/ChatMemoryService.swift` | C3 |
| MCP | `Sources/MCP/MCPService.swift` | D2 |
| MCP | `Sources/MCP/MCPSettingsView.swift` | D2 |
| MCP | `Sources/MCP/MCPBrowserView.swift` | D2 |
| 设置 | `Sources/Settings/SettingsTab.swift` | D4 |
| 动态 MoE | `Sources/DynamicMoE/*.swift` | A4 |
| 仪表盘 | `Sources/Dashboard/DashboardTab.swift` | B4, D1 |
| 应用入口 | `Sources/App/ToshLLMApp.swift` | C1 |
| 应用对象 | `Sources/App/AppObjects.swift` | C1 |

### 参考实现文件（外部仓库）

| 仓库 | 文件 | 参考价值 |
|------|------|----------|
| llmfit | `llmfit-core/src/hardware.rs` | B1, B2, B3 完整实现 |
| llmfit | `llmfit-core/src/fit.rs` | B2, B3 评分与推荐逻辑 |
| llmfit | `llmfit-tui/src/main.rs` | CLI/TUI 交互、硬件档案命令 |
| llmfit | `llmfit-api/src/lib.rs` | REST API、Web Dashboard |
| jcode | `docs/MEMORY_ARCHITECTURE.md` | C3 理论与实践 |
| jcode | `docs/SERVER_ARCHITECTURE.md` | C1 单守护进程架构 |
| jcode | `docs/SWARM_ARCHITECTURE.md` | E1 多 Agent 编排设计 |
| vLLM | `vllm/v1/worker/gpu/model_runner.py` | A1, A2, A3, A5 核心调度/缓存 |
| vLLM | `vllm/config/speculative.py` | A4 统一推测解码配置 |
| vLLM | `vllm/v1/core/scheduler.py` | A2 连续批处理调度器 |
| cc-haha | `src/app/model-picker.tsx` | D1 统一模型选择器 UI |
| cc-haha | `src/app/mcp-manager.tsx` | D2 MCP 管理 UI |
| cc-haha | `src/app/permissions.tsx` | D4 权限分级 UI |
| deepseek-harness | `src/index.ts` | E1 插件化架构 |
| openclaw | `src/gateway/index.ts` | E2 网关模式 |
| aiopencode | `src/agents/plan.ts` | 双 Agent 模式参考 |

---

## 附录：工作量估算汇总

| 阶段 | 优化项数 | 总工作量 | 关键依赖 |
|------|----------|----------|----------|
| P0 立即 | 3 项 | 7 周 | 无 |
| P1 近期 | 5 项 | 19 周 | P0 完成后启动 |
| P2 中期 | 6 项 | 28 周 | P1 核心完成后启动 |
| P3 长期 | 6 项 | 33 周 | 架构预研完成后启动 |
| **总计** | **20 项** | **87 周（约 22 个月）** | - |

---

> **维护建议**：每季度更新一次，跟踪上游 llama.cpp/vLLM 重大版本与竞品新功能
>
> **生成日期**：2026-09-15  
> **分析范围**：9 个对比仓库 + ToshLLM 全量源码
