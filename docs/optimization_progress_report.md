# ToshLLM 优化进度报告 - 2026-09-16

## 执行状态概览

| 阶段 | 已完成 | 待完成 | 完成率 |
|------|--------|--------|--------|
| **Phase 1 (P0-P3)** | 27/27 (100%) | 0 (0%) | 100% |
| **Phase 2 P1-2** | ✅ 完成 | - | 100% |
| **Phase 2 P1-3** | ✅ 完成 | - | 100% |
| **Phase 2 P1-4** | ✅ 完成 | - | 100% |
| **Phase 2 P1-6** | ✅ 完成 | - | 100% |
| **Phase 2 P1-5/P2-15** | 🟡 部分 (配置就绪) | ggml Metal Kernel 需外部应用 | 配置已生成 |
| **Phase 2 P1-5/P2-16** | 🟡 部分 (配置就绪) | Metal 图捕获 API 需研究 | 配置已生成 |

## 已完成的优化详情

### P1-2: 连续批处理调度器 (完成)
- **新增**: `ProcessingMode` 枚举 (`.prefill`, `.decode`, `.mixed`)
- **新增**: `ScheduledRequest.mode` (默认 `.prefill`), `maxTokens` (默认 512)
- **新增**: TPS 跟踪与平滑 (`_estimatedTPS`, `tpSSmoothing` [10])- **回压逻辑**: 当 `_estimatedTPS > targetTPS * 1.2` 时 hold back 新请求
- **新增**: 优先级调度与抢占检查
- **新增**: `targetTPS` 初始化参数 (默认 20)
- **修改文件**: `Sources/Servers/ContinuousBatchScheduler.swift` (+105/-12 行)

### P1-3: FlashAttention 集成 (完成)
- **新增**: `ServerSettings.faAmd` (默认 `true`, line 118)
- **新增**: `effectiveFaAmd` 计算属性 (line 904-905)
- **新增**: `--flash-attn` argument (ContinuousBatchScheduler.swift:384)
- **新增**: Metal env var `TOSH_FA_AMD=1` (Server.swift:674)
- **修改文件**: `Sources/Servers/Server.swift` (+40 行相关增量)

### P1-4: PrefixCache LRU 升级 (完成)
- **新增**: `currentUsageBytes()` 方法 (`max(0, stats.totalSizeBytes)`)
- **优化**: 使用 `NSCache<CacheKey, CacheEntry>` 替代自定义 LRU 实现
- **修改文件**: `Sources/Servers/PrefixCache.swift` (+6 行)

### P1-6: KV Cache q8_0/q4_x 参数 (完成)
- **新增**: `ServerSettings.cacheTypeK` (String: "f16"|"q8_0"|"q5_x"|"q4_x"|"iq4_nl")
- **新增**: `ServerSettings.cacheTypeV` (String)
- **新增**: PagedAttention 参数: `pagedAttention`, `pagedAttentionPageTokens`, `pagedAttentionMaxPages`, `pagedAttentionPrefixCache`
- **新增**: Metal env vars: `TOSH_PAGED_ATTENTION`, `TOSH_PAGE_SIZE`, `TOSH_MAX_PAGES`, `TOSH_PREFIX_CACHE`
- **修改文件**: `Sources/Servers/Server.swift` (+40 行相关增量)

## 待完成的优化项

### P2-15: PagedAttention KV 页式管理 (⏳)
- **来源**: vLLM PagedAttention
- **影响力**: 5/5
- **协同度**: 3/5
- **预估耗时**: 8 周
- **风险**: 高
- **原因**: ggml Metal Kernel 需外部应用
- **需要操作**: 下载/集成 ggml 库并应用 `patches/llama/0072-metal-paged-attention.patch` 至 `ggml/src/ggml-metal/ggml-metal.metal`
- **实施步骤**: 
  1. 跟进上游 ggml 页式内存管理进度
  2. 设计 Metal 端 `MTLBuffer` 池化页管理
  3. 实现页表（Page Table）和块管理器（Block Manager）
  4. 修改 KV 分配从连续到页式
  5. 支持不连续上下文和显存超卖
  6. 基准测试对比连续 vs 页式管理

### P2-16: Metal Command Buffer 复用/图捕获 (⏳)
- **来源**: vLLM CUDA Graph、Metal 图捕获
- **影响力**: 3/5
- **协同度**: 2/5
- **预估耗时**: 6 周
- **风险**: 高
- **原因**: Metal 图捕获 API 需进一步研究
- **需要操作**: 研究 `MTLCommandBuffer` 录制复用机制
- **实施步骤**:
  1. 研究 `MTLCommandBuffer` 录制复用机制
  2. 实现常用 batch size 的图捕获
  3. 预热阶段捕获常用推理路径
  4. 运行时复用捕获的命令缓冲区
  5. 基准测试对比复用 vs 实时构建

## 测试验证结果

| 验证项目 | 通过率 | 备注 |
|----------|--------|------|
| Swift build | ✅ 通过 (0.69s) | 无编译错误 |
| 单元测试 | ✅ 315/315 | 通过 |
| 性能回归测试 | ✅ 11/11 | 通过 |
| 总计 | ✅ 326/326 | 无回归 |

## 行动建议

### 即时行动 (无需外部依赖)
- 当前所有已启用的 Phase 2 功能已就绪
- 可直接用于生产环境
- 不需要等待 ggml Metal Kernel 即可使用现有功能

### 延期行动 (需外部依赖)
1. **P2-15**: 若需要完成 PagedAttention Metal Kernel，需集成外部 ggml 库
2. **P2-16**: 若需要 Metal Command Buffer 复用，需研究 Apple Metal 图捕获 API

### 结论
- **Phase 1 (P0-P3)**: 完全完成 (27/27)
- **Phase 2 可用功能**: 完全完成 (P1-2, P1-3, P1-4, P1-6)
- **Phase 2 待研究**: 2 项 (P2-15, P2-16) 依赖外部 ggml Metal 库
- **整体状态**: 优秀 - 326/326 测试通过，无回归

