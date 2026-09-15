# ToshLLM 优化架构图

> 生成日期：2026-09-15

---

## 整体架构

```mermaid
graph TB
    subgraph "UI 层"
        Dashboard[DashboardTab]
        ServerDetail[ServerDetailView]
        Settings[SettingsTab]
    end
    
    subgraph "业务层"
        Chat[ChatNative]
        Models[ModelsTab]
        MCP[MCPService]
    end
    
    subgraph "优化层"
        direction TB
        subgraph "P0 基础"
            SD[SpeculativeDecoder]
            SQ[QuantizationRecommendation]
        end
        
        subgraph "P1 核心"
            PC[PrefixCache]
            HP[HardwareProfileStore]
            MS[ModelSource]
            RT[RequestTracer]
        end
        
        subgraph "P2 增强"
            AP[AsyncPipeline]
            DM[DaemonManager]
            PM[PluginManager]
            GW[GatewayServer]
            PP[PermissionPolicy]
            CE[ConcurrencyEstimator]
        end
        
        subgraph "P3 扩展"
            PE[SessionExporter]
            PL[PluginProtocol]
        end
    end
    
    subgraph "底层引擎"
        llama[llama.cpp]
        Metal[Metal Backend]
        GPU[AMD GPU]
    end
    
    Dashboard --> HP
    Dashboard --> CE
    ServerDetail --> SD
    ServerDetail --> RT
    Settings --> PP
    Settings --> HP
    
    Chat --> PC
    Chat --> RT
    Chat --> PP
    Models --> MS
    
    DM --> AP
    GW --> PM
    PM --> PL
    
    AP --> llama
    DM --> llama
    SD --> Metal
    Metal --> GPU
```

---

## 数据流图

### 请求处理流程

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant GW as GatewayServer
    participant DM as DaemonManager
    participant AP as AsyncPipeline
    participant LLM as llama.cpp
    
    Client->>GW: HTTP Request
    GW->>GW: 验证 API Key
    GW->>DM: 转发请求
    DM->>AP: 管道处理
    AP->>LLM: 推理请求
    LLM-->>AP: 流式响应
    AP-->>DM: 响应流
    DM-->>GW: 响应流
    GW-->>Client: HTTP Response
```

### 推测解码流程

```mermaid
sequenceDiagram
    participant App as 应用
    participant SD as SpeculativeDecoder
    participant MTP as MTPDecoder
    participant DFlash as DFlashDecoder
    
    App->>SD: configure(settings)
    SD->>MTP: predict(context)
    MTP-->>SD: draft tokens
    SD->>MTP: validate(draft, target)
    MTP-->>SD: SpeculativeResult
    SD-->>App: accepted tokens
```

### 并发估算流程

```mermaid
sequenceDiagram
    participant UI as 用户界面
    participant CE as ConcurrencyEstimator
    participant HP as HardwareProfileStore
    participant Model as ModelSpec
    
    UI->>CE: estimate(spec, hw, ctx)
    CE->>HP: 获取硬件信息
    HP-->>CE: HardwareInfo
    CE->>Model: 获取模型信息
    Model-->>CE: ModelSpec
    CE->>CE: 计算 KV 缓存
    CE->>CE: 计算可用 VRAM
    CE-->>UI: Result(maxSessions, t/s)
```

---

## 模块间依赖

### 依赖矩阵

| 模块 | 依赖 | 被依赖 |
|------|------|--------|
| SpeculativeDecoder | - | ServerDetailView |
| QuantizationRecommendation | - | HardwareProfileStore |
| PrefixCache | - | ChatNative |
| HardwareProfileStore | - | ConcurrencyEstimator |
| ModelSource | - | ModelsTab, AsyncPipeline |
| RequestTracer | - | ServerDetailView, SessionExporter |
| AsyncPipeline | - | DaemonManager |
| DaemonManager | AsyncPipeline | GatewayServer |
| PluginManager | PluginProtocol | GatewayServer |
| GatewayServer | PluginManager | External Clients |
| PermissionPolicy | - | ChatNative, SettingsTab |
| ConcurrencyEstimator | HardwareProfileStore | DashboardTab |
| SessionExporter | RequestTracer | - |
| PluginProtocol | - | PluginManager |

---

## 集成点

### 1. Dashboard 集成

```swift
// DashboardTab.swift
struct DashboardMetricsView: View {
    @EnvironmentObject var hardware: HardwareMonitor
    
    var body: some View {
        // 添加并发估算卡片
        ConcurrencyCard(hw: hardware.info)
    }
}

struct ConcurrencyCard: View {
    let hw: HardwareInfo
    
    var body: some View {
        let result = ConcurrencyEstimator.estimate(
            spec: currentModel.spec,
            hw: hw
        )
        // 显示 maxSessions 和 tokensPerSecond
    }
}
```

### 2. ServerDetailView 集成

```swift
// ServerDetailView.swift
struct ServerPerformanceWorkspace: View {
    var body: some View {
        // 添加请求追踪面板
        RequestTracerView(serverID: server.id)
    }
}

struct RequestTracerView: View {
    @StateObject var tracer = RequestTracer()
    let serverID: UUID
    
    var body: some View {
        // 显示请求时间轴和 Token 用量
    }
}
```

### 3. Settings 集成

```swift
// SettingsTab.swift
struct SettingsView: View {
    var body: some View {
        // 添加权限策略设置
        PermissionPolicySection()
    }
}

struct PermissionPolicySection: View {
    @State var policy = PermissionPolicy()
    
    var body: some View {
        // 全局权限级别
        // 分类规则
        // 允许/拒绝模式
    }
}
```

---

## 性能优化点

### 1. PrefixCache

- **优化**: LRU 缓存淘汰策略
- **收益**: 减少 15-19k token 冷启动
- **文件**: `Sources/Servers/PrefixCache.swift`

### 2. AsyncPipeline

- **优化**: URLSession 连接池重用
- **收益**: 减少连接建立开销
- **文件**: `Sources/Servers/AsyncPipeline.swift`

### 3. RequestTracer

- **优化**: 字典替代数组存储
- **收益**: O(1) 查询替代 O(n)
- **文件**: `Sources/Servers/RequestTracer.swift`

### 4. ConcurrencyEstimator

- **优化**: 缓存计算结果
- **收益**: 避免重复计算
- **文件**: `Sources/Models/ConcurrencyEstimator.swift`

---

## 安全考虑

### 1. PermissionPolicy

- **措施**: Glob 模式匹配替代子串匹配
- **文件**: `Sources/Chat/PermissionPolicy.swift:112-116`
- **收益**: 防止路径遍历攻击

### 2. GatewayServer

- **措施**: API Key 验证中间件
- **文件**: `Sources/Gateway/GatewayServer.swift`
- **收益**: 防止未授权访问

### 3. DaemonManager

- **措施**: 重启锁和文件描述符管理
- **文件**: `Sources/Servers/DaemonManager.swift`
- **收益**: 防止资源泄漏和崩溃

---

## 测试覆盖

### 单元测试

| 模块 | 测试文件 | 测试用例数 |
|------|----------|------------|
| ConcurrencyEstimator | `Tests/ConcurrencyEstimatorTests.swift` | 9 |
| QuantizationRecommendation | `Tests/PerformanceTests.swift` | 1 |
| SpeculativeDecoder | `Tests/PerformanceTests.swift` | 1 |

### 集成测试

- GatewayServer → DaemonManager → AsyncPipeline → llama.cpp
- PluginManager → PluginProtocol → GatewayServer

---

## 相关文档

- [API 文档](api-optimization-modules.md)
- [优化路线图](optimization-roadmap.md)
