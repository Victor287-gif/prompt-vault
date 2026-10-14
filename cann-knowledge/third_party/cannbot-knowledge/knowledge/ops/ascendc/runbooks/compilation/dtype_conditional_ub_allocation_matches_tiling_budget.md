---
type: "Runbook"
title: "按 dtype 分配 UB 必须匹配 tiling 预算"
description: "SPLIT_D 的 fp32 路径直接使用输入张量，却仍分配仅低精度转换需要的 xFp32Buf，使实际 UB 占用超过 tiling 预算；按 dtype 守护该缓冲后恢复大 D 路径。R2（950/9.1.0）：布局分支（useRawCopy）切换与 dtype 分支同理——tiling 的 bufferDivisor 必须按 kernel 实际布局同步，且两侧阈值判定式一致。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.memory"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-09-17T02:30:52Z"
platforms: ["a3", "950"]
---

# 按 dtype 分配 UB 必须匹配 tiling 预算

## 现象

RmsNorm 的 NORMAL 路径和小 D 用例已通过，但较长归约维进入 SPLIT_D 后，设备同步报 ACL `507035`，并明确给出 “VEC instruction read/write UB is out of bounds”。故障发生在 fp32 大 D 路径；这与一般的数值 mismatch 或 GM 索引错误不同。

R2（Ascend950PR，arch35，CANN 9.1.0，SwiGlu 算子 BenchNoHarnessSwiGlu）：kernel 侧为小 innerSize 引入 `useRawCopy_` 布局分支（`inQueueX0` 单队列存 `2*ubFormer_` 个 raw 元素，另加 `outQueue` 与 3 个 FP32 `TBuf`）后，host tiling 的 UB 切分仍按**旧布局**估算每输出元素字节占用（FP32：`bufferDivisor = 7 * sizeof(float)`）。修完小 innerSize 混合路径后重跑 ST Real：FP16 的 `[4,8] dim=-1` 用例全部转 PASS，但 `builtin_fp32_2d_dim-1_[4,8]` 仍然 `流同步失败: 507035`——raw 布局 FP32 实占 9 buffer/元素共 294912B > 196608B（192KB）UB。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的编译、链接或运行工程；R2 为 AscendC 自定义算子 host tiling + kernel 协同开发（UB 切分/InitBuffer 布局账目）。
- API 路径：`KernelRmsNormSplitD`、`xFp32Buf_`、`ACL 507035`、`ubFactor` 所在的 kernel、host 或构建接口链；R2 增补 `TilingFunc` 内 `ubFormer` 计算（`ubSize / bufferDivisor` 对齐）、kernel `TPipe::InitBuffer`。
- 验证平台与版本：平台 a3；CANN：unknown。R2 在 Ascend950PR（arch35）+ CANN 9.1.0 实测。
- 适用粒度：`same_operator`，迁移前须重算同算子各 dtype/template 的 live-buffer 预算；R2 表明该规则同样适用于**布局分支**（不限 dtype 分支），粒度推广到 `operator_family`——所有 kernel 内存在多套 UB buffer 布局（按 tiling/shape/布局分支切换）的算子。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断；单布局 kernel（无 useRawCopy 类分支）不涉及 R2 形态。

## 根因

SPLIT_D 的 fp32 计算直接对 `xLocal` 做平方，不使用 `xFp32Buf_`；但 `Init` 仍无条件为它分配 `ubFactor * sizeof(float)`。令 `B_tiling(dtype, tile)` 为 host 预算、`B_live(dtype, tile)` 为 kernel 所有 live queue/TBuf 与固定 scratch 之和，则该分支出现 `B_live = B_tiling + tile * sizeof(float)`，并在 `B_live > UB_available` 时触发 VEC UB 越界。

根因只有一个：kernel 的 dtype 条件分配与 tiling 的逐元素预算不一致。修复不是减小测试规模或改异常处理，而是让 fp32 的实际 live-buffer 集合与 host 预算使用同一公式。

R2 根因（同型，触发维度从 dtype 分支换成布局分支）：kernel 的 `useRawCopy_` 路径改变了 FP32 的 UB 布局账目，host tiling 的 `bufferDivisor` 没有跟着改，导致 `ubFormer_` 偏大、`InitBuffer` 总量超出 192KB UB：

- 旧布局（分块 x0/x1）：3 个 `TQue`（×2 双缓冲）+ 1 个 `TBuf`，全 FP32 → **7 × 4B = 28B/输出元素**。
- raw 布局：`inQueueX0`（×2 双缓冲、每份 `2*ubFormer_` raw 元素，即 4×ubFormer_ 元素 → 16B/元素）+ `outQueue`（×2 → 8B/元素）+ 3 个 `TBuf`（12B/元素）→ **9 × 4B = 36B/输出元素**。

按 `ubFormer_=8192`、UB=196608B（192KB）演算：raw 布局实际需要 `inQueueX0 = 2×16384×4 = 131072B` + `outQueue = 65536B` + 3×`32768B` = **294912B > 196608B** —— `InitBuffer` 超 UB，向量核异常 507035。FP16/BF16 的 raw 布局账目恰与旧布局相同（`(4+2)*dtypeSize + 3*4 = 24B/元素`），所以只有 FP32 暴露该问题——属巧合而非规律。

## 解决配方

### before

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                int64_t numRow, int64_t numCol, int64_t numColAlign,
                                int64_t blockLength, uint32_t ubFactor, float epsilon)
    {
        numCol_ = numCol;
        numColAlign_ = numColAlign;
        ubFactor_ = static_cast<int64_t>(ubFactor);
        epsilon_ = epsilon;
        blockElems_ = 32 / sizeof(T);

        xGm_.SetGlobalBuffer((__gm__ T *)x);
        gammaGm_.SetGlobalBuffer((__gm__ T *)gamma);
        yGm_.SetGlobalBuffer((__gm__ T *)y);

        int64_t blk = AscendC::GetBlockIdx();
        coreStart_ = blockLength * blk;
        coreEnd_ = coreStart_ + blockLength;
        if (coreEnd_ > numRow) coreEnd_ = numRow;
        if (coreStart_ > numRow) coreStart_ = numRow;

        pipe_.InitBuffer(xQueue_, DEPTH, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(gammaQueue_, DEPTH, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(yQueue_, DEPTH, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(xFp32Buf_, ubFactor_ * sizeof(float));
        pipe_.InitBuffer(sqxBuf_, ubFactor_ * sizeof(float));
        pipe_.InitBuffer(reduceBuf_, 256);
        pipe_.InitBuffer(tmpBuf_, 256);
        pipe_.InitBuffer(scalarBuf_, 256);
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(gammaFp32Buf_, ubFactor_ * sizeof(float));
        }
    }
```

### after

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                int64_t numRow, int64_t numCol, int64_t numColAlign,
                                int64_t blockLength, uint32_t ubFactor, float epsilon)
    {
        numCol_ = numCol;
        numColAlign_ = numColAlign;
        ubFactor_ = static_cast<int64_t>(ubFactor);
        epsilon_ = epsilon;
        blockElems_ = 32 / sizeof(T);

        xGm_.SetGlobalBuffer((__gm__ T *)x);
        gammaGm_.SetGlobalBuffer((__gm__ T *)gamma);
        yGm_.SetGlobalBuffer((__gm__ T *)y);

        int64_t blk = AscendC::GetBlockIdx();
        coreStart_ = blockLength * blk;
        coreEnd_ = coreStart_ + blockLength;
        if (coreEnd_ > numRow) coreEnd_ = numRow;
        if (coreStart_ > numRow) coreStart_ = numRow;

        pipe_.InitBuffer(xQueue_, DEPTH, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(gammaQueue_, DEPTH, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(yQueue_, DEPTH, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(sqxBuf_, ubFactor_ * sizeof(float));
        pipe_.InitBuffer(reduceBuf_, 256);
        pipe_.InitBuffer(tmpBuf_, 256);
        pipe_.InitBuffer(scalarBuf_, 256);
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(xFp32Buf_, ubFactor_ * sizeof(float));
            pipe_.InitBuffer(gammaFp32Buf_, ubFactor_ * sizeof(float));
        }
    }
```

### R2 解决配方（布局分支切换的除数同步）

host tiling 用与 kernel **完全相同的阈值**判定 `useRawCopy`，并按布局切换 `bufferDivisor`（verbatim 取自该会话 tiling 写入事件）：

```cpp
    // 5. UB 切分
    // 判断是否使用整块原始数据拷贝策略 (kernel 中 innerSize < 32/sizeof(T) 时启用)
    // useRawCopy 时 UB 布局变化: inQueueX0 存 2*tileSize 元素 (raw data)
    constexpr uint32_t MIN_BURST_BYTES = 32;
    bool useRawCopy = (innerSize * dtypeSize < MIN_BURST_BYTES) && (innerSize > 0);

    // FP32 路径:
    //   非 rawCopy: 3 TQue(×2) + 1 TBuf = 7 buffers, all FP32 → 28 bytes/elem
    //   rawCopy:    1 TQue(×2, 2x size) + 1 TQue(×2) + 3 TBuf = (4+2+3) = 9 → 36 bytes/elem
    // FP16/BF16 路径:
    //   非 rawCopy: 3 TQue(×2, orig) + 3 TBuf(FP32) → 6*dtypeSize + 3*4 = 24 bytes/elem
    //   rawCopy:    1 TQue(×2, 2x orig) + 1 TQue(×2, orig) + 3 TBuf(FP32) → (4+2)*dtypeSize + 3*4 = 24 bytes/elem
    uint64_t bufferDivisor;
    if (dataType == ge::DT_FLOAT) {
        bufferDivisor = useRawCopy ? (9 * sizeof(float)) : (7 * sizeof(float));
    } else {
        bufferDivisor = 6 * dtypeSize + 3 * sizeof(float);  // 24 bytes/elem (FP16/BF16, rawCopy same)
    }
```

要点：`useRawCopy` 判定式（`innerSize * dtypeSize < 32 && innerSize > 0`）必须与 kernel 侧 `useRawCopy_ = (innerSize_ < MIN_BURST_BYTES / sizeof(T)) && (innerSize_ > 0)` 一致，两处阈值漂移会造成 tiling 账目与 kernel 实际布局错位。为什么不削减 kernel buffer 而是改除数：raw 布局是上一卡（小 innerSize 混合路径）的正确性修复，不动；UB 切分本来就该反映真实布局，`ubFormer_` 变小只影响每 tile 元素数，不影响正确性。为什么 FP16/BF16 除数不变：raw 布局下 `(4+2)*2+3*4 = 24B/元素` 与旧布局 `6*2+12 = 24B/元素` 相等，属巧合而非规律——每套布局都应单独记账，不能默认相等。

## 设计决策

优先移除 fp32 不使用的缓冲，而不是把无用项补进 `perElem`。前者使 kernel 的实际资源需求与已有计算路径一致，也保留更大的 `ubFactor`；后者会为从不读取的存储付费并缩小切片。

## 性能影响

修复减少 fp32 SPLIT_D 的 UB 占用，理论上允许保持既有切片宽度；轨迹没有提供同平台、同负载的配对性能数据，因此只证明功能与资源闭环，不宣称加速。R2 同理：`ubFormer_` 变小只影响每 tile 元素数，无配对性能数据。

## 预防

- 为每个模板、每个 dtype 列出所有存活的 `TQue/TBuf`，逐项核对 `depth * bytes`；
- host 的 `perElem` 公式必须与 kernel 的 `if constexpr` 分配条件同源核算；
- 在模板切换点两侧使用参数化归约维 probe，至少覆盖 fp32 与低精度分支；
- 资源估算除逐元素缓冲外，还要预留 reduce、tmp、scalar 等固定开销。
- R2：kernel 内任何改变 UB buffer 数量/大小的分支（useRawCopy、双缓冲切换、额外 workspace buffer）都必须在同一变更里同步 host tiling 的 `bufferDivisor`/UB 切分账目，并在注释里写清"每输出元素字节占用 = Σ(buffer 数 × dtype 字节)"的推导；每套布局单独记账，不能因某 dtype 两套布局恰好相等就默认相等。
- 复查 UB 超额时用乘法显式演算（如 294912B > 196608B），不要只看"元素数没变"。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：大 SPLIT_D 路径触发 UB 越界后的 ACL 507035 vector core exception。（来源平台：a3；证据等级: tool_observed）。
- 修复后：大 SPLIT_D 定向矩阵全部输出 OK。（来源平台：a3；证据等级: tool_observed）。
- 修复后：干净构建 exit=0，随后 32 passed。（来源平台：a3；证据等级: tool_observed）。

R2 轮（Ascend950PR / CANN 9.1.0，SwiGlu 迭代一联调）增量信号：

- R2 修复前（raw 布局 + 旧除数 28B/元素）：`builtin_fp32_2d_dim-1_[4,8]` `流同步失败: 507035`（来源平台：950；证据等级：tool_observed）。
- R2 UB 超额演算：raw 布局 FP32 总占用 131072+65536+32768×3 = 294912B > 196608B（192KB）（来源平台：950；证据等级：tool_observed——数值由 trace 内逐 buffer 推算）。
- R2 除数同步后：`builtin_fp32_2d_dim-1_[4,8]` PASS（`MERE=1.84e-09, MARE=1.32e-08`），12 pass 1 fail（仅剩 bf16，另卡根因）（来源平台：950；证据等级：tool_observed）。
- R2 最终回归：ST Real 13/13 PASS、UT 24/24（来源平台：950；证据等级：tool_observed）。

## 失效边界

本卡只覆盖 `KernelRmsNormSplitD` 的这组缓冲及 tiling 预算。若 `DEPTH`、队列集合、dtype 转换路线、UB 容量或固定 scratch 发生变化，必须重新计算 `B_live(dtype, tile) <= UB_available`，不得复用源任务的逐元素常数。其他平台未在本闭环验证。

R2 补充：UB 总量 196608B 与 `ubFormer_=8192` 是本环境演算值；其他平台/其他 tiling 参数须按实际 UB 大小与对齐因子重算。本卡只解决"布局账目失配"这一根因；`useRawCopy` 阈值本身、32B 约束、bf16 转换分别见独立卡（小 innerSize 混合路径卡、BF16 标量转换卡）。

## 坏实践

- 看到大 D 越界就只把 `ubFactor` 乘固定折扣；这会掩盖 host/kernel 资源模型不一致；
- 为所有 dtype 无条件分配转换缓冲，期待编译器消除；`InitBuffer` 已经占用 UB，不因后续未读而自动安全；
- 只跑 NORMAL 或小 D 用例后宣布全部模板通过。
- R2：kernel 改了 buffer 布局、tiling 除数沿用旧值——FP32 下静默超 UB，报错仍是笼统的 507035，极易误判为"上一修复没修干净"。
- R2：用 dtype 推断布局而不传递统一的 `useRawCopy` 判定——两处阈值一旦漂移即错位。

## 误诊

如果缩小 `ubFactor` 后故障位置不变，或错误日志指向 GM 地址、尾块 DMA、队列同步，应检查地址和生命周期，不要把所有 ACL 507035 都归为容量溢出。只有"模板切换相关 + 实际分配可算出超过 UB + 移除无用缓冲后闭环"才支持本根因。

R2：该失败首轮被当作"混合路径对 FP32 无效"排查（检查 DataCopyPad 参数、CopyOut 大小均正常）；实际混合路径本身有效，超的是 InitBuffer 总量。切换排查视角（算字节账而不是查指令参数）后才定位。

## R3 增量（反向脱节：预算虚高致 ubFactor 缩水，按 kernel 实际 InitBuffer 反推）

R3 并入 Gather 算子（bench_no_harness_gather，Ascend950PR/arch35/CANN 9.1.0，UB=253952B）迭代三全功能审查的实例，补齐本卡契约的**反向**形态——主卡与 R2 是"kernel 实际分配超过 tiling 预算"（超额→507035），R3 是"tiling 预算为 kernel 从未分配的缓冲预留"（虚高→ubFactor 缩水、UB 利用率低下）：

**1. 现象**：kernel Init 阶段只在 UB 分配 index 双缓冲（各 `CeilAlign(ubFactor × indexDtypeSize, 32)` 字节）加一个 32 字节 tempBuf——x 数据逐元素从 GM 直取、从不整体缓存进 UB。但 tiling 的 UB 预算公式按"要把整个 x 缓存进 UB"计算：`xBufSize = CeilAlign(xNumel × xDtypeSize, ...)`（x.numel=2M 的 fp32 case 即 8MB，远超 253952B 的 UB）→ 恒走"回退 ubSize/2"分支 → 再被 bufferNum=4 和 `xDtypeSize + indexDtypeSize` 双重稀释，ubFactor 只有 3968，kernel 实际 UB 占用 31776B/253952B ≈ 12.5%——批量搬运粒度被无谓压小到约 1/7。

**2. 修复（预算 = Σ(kernel 实际 InitBuffer) + 显式余量）**：

```cpp
    // 6、UB 切分（基于 kernel 实际 UB 使用量计算 ubFactor）
    // Kernel 实际 UB 分配（参见 <op>.h::Init）：
    //   - indexBufA: CeilAlign(ubFactor × indexDtypeSize, 32) 字节
    //   - indexBufB (双缓冲): CeilAlign(ubFactor × indexDtypeSize, 32) 字节
    //   - tempBuf: 32 字节（单元素中转缓冲区，MTE2→MTE3 搬运中转）
    // 总 UB = BUFFER_NUM × CeilAlign(ubFactor × idxSize, 32) + 32 ≤ ubSize
    int64_t xNumel = xShape.GetShapeSize();
    uint64_t useDoubleBuffer = (totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufferNum = useDoubleBuffer ? 2 : 1;  // index 缓冲区数量（单/双缓冲）
    // 安全余量：系统/框架 UB 开销 + tempBuf + 每缓冲区 32B 对齐填充
    constexpr int64_t UB_SAFETY_MARGIN = 16384;   // 16KB 系统/框架预留
    constexpr int64_t TEMP_BUF_SIZE = 32;          // 单元素中转缓冲区
    constexpr int64_t ALIGN_OVERHEAD = 32;         // 每缓冲区 32B 对齐预留
    int64_t availableUb = static_cast<int64_t>(ubSize) - UB_SAFETY_MARGIN - TEMP_BUF_SIZE
                          - bufferNum * ALIGN_OVERHEAD;
    int64_t ubFactor = FloorAlign(
        FloorDiv(FloorDiv(availableUb, bufferNum), indexDtypeSize),
        ubBlockSize);
    if (ubFactor < 1) {
        ubFactor = 1;  // 保证至少处理 1 个元素/批
    }
```

配套 UT 断言与 tiling 公式同源（公式漂移在 UT 暴露）：

```cpp
    // 验证 ubFactor 合理性：基于 kernel 实际 UB 分配
    int64_t indexDtypeSize = GetDtypeSize(param.indexDtype);
    int64_t bufferNum = param.expectDoubleBuffer ? 2 : 1;
    int64_t indexUbSize = (tiling->ubFactor * indexDtypeSize + 31) & ~31;  // 32B 对齐
    int64_t ubEstimate = bufferNum * indexUbSize + 32;  // index 缓冲区 + tempBuf
    EXPECT_LE(ubEstimate, static_cast<int64_t>(PLATFORM_UB_SIZE))
        << "ubFactor UB budget exceeded for case: " << param.caseName
        << " (ubEstimate=" << ubEstimate << ", ubSize=" << PLATFORM_UB_SIZE << ")";
```

要点：① 预算公式以 kernel `InitBuffer` 的实际分配为唯一事实源，并把该分配公式以注释契约写进 tiling；② `bufferNum` 语义修正为"index 缓冲区数量（单/双缓冲）= 1/2"——旧公式按"2 queues × BUFFER_NUM"算成 2/4，把队列深度与缓冲个数混为一谈；③ 安全余量取固定 16KB（覆盖系统/框架预留，经验值非实测下界）。

**3. 误诊教训（R3）**：首版修复只留 96 字节余量（UB 用满 99.96%）联调出现 507035 流同步错误时，第一反应是"UB 用太满"，一度想直接加大余量或封顶 ubFactor 了事；回退对照 + 头文件查证后确认真因是 `DataCopyParams.blockLen` uint16 截断（搬运参数截断，另卡）。教训：tiling 优化后出现的运行时错误，先隔离"新参数是否触发 kernel 既有缺陷"，再怀疑预算本身；余量大小与该错误无因果关系，已通过对照实验区分。

**R3 验证信号**（同一次 Gather 迭代三开发会话，修复=改 tiling 公式 + 同步 UT 断言 + 重编译重装 + 重跑 UT/ST）：

- 修复前结构性信号：旧公式下双缓冲 case ubFactor=3968，kernel UB 占用 31776B/253952B ≈ 12.5%（来源平台：950；证据等级：tool_observed）。
- 修复后 UT：38/38 通过（tiling 31 + infershape 7），后扩展到 62/62；单缓冲 int32 case 日志观测 ubFactor=59360、双缓冲 int64 ubFactor=14816（与公式 `FloorAlign(FloorDiv(237472, bufferNum)/indexDtypeSize, ubBlockSize)` 一致）（来源平台：950；证据等级：tool_observed）。
- 修复后 ST：54/54 通过，含三个 ~1M 输出元素大 shape 用例全部 Bitwise Match；Golden 自测 16/16（来源平台：950；证据等级：tool_observed）。
- 数值核验：blockLen 最大 237440 < DataCopyExtParams 上限；availableUb=237472>0 无负值（来源平台：950；证据等级：tool_observed）。

**R3 边界与坏实践**：本配方假设 kernel 不缓存 x；若 kernel 改为整体缓存 x 或新增其他 UB 缓冲，预算公式必须同步重写，否则反向溢出（回到主卡/R2 形态）。`UB_SAFETY_MARGIN=16384` 是经验值，最小安全值未实测；ubSize=253952B 为 Ascend950PR/arch35 的值，其他平台取本平台 `GetCoreMemSize(UB)` 后机制相同（未验证，推断可迁移）。ubFactor 提升约 7.5 倍的结构性批量粒度收益未做 profiling，不能据此宣称具体加速比。坏实践补充：tiling 与 kernel 各写各的 UB 假设互不对照（本例漂移存在两个迭代才被审查发现）；预算公式里混入语义不明的倍数（"2 queues × BUFFER_NUM"）而不回到 kernel 的 `InitBuffer` 分配式反推；用"回退到 ubSize/2"这类粗粒度兜底长期运行（兜底掩盖预算公式本身的错误）。
