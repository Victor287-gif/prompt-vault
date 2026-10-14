---
type: Guide
title: 常见编译/运行错误速查（报错-原因-修复）
description: 17 类高频错误的报错信息、原因与修复代码三元组，含 UB 溢出、coreDim 超限、类型不支持、对齐、advance、stride 转置、精度异常、scalar
  退化、死锁等，附 910_95 平台差异。
tags: [task.debugging, topic.implementation.debugging, topic.compilation, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']
---
# 常见编译/运行错误速查（报错-原因-修复）

触发场景：`MLIRCompilationError`、`ub overflow`、`coreDim > UINT16_MAX`、`illegal memory access`/`misaligned address`、`deadlock detected`/`barrier timeout`、精度差异过大、超时或性能远低于预期。

## 定位方法

**编译错误定阶段**：`TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1` 后查 `~/.triton/dump/`：无 ttir → Python→TTIR 失败；有 ttir 无 ttadapter → TTIR→TTAdapter 失败；有 ttadapter → `MLIR_ENABLE_DUMP=1` 定位失败 Pass，或 `bishengir-compile kernel.ttadapter.mlir --target=Ascend910B3 --enable-auto-multi-buffer=True --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true` 手动编译。

**运行时定位**：解释器 `TRITON_INTERPRET=1`（逻辑/精度）；`TRITON_DEVICE_PRINT=1`（中间结果）；`ENABLE_PRINT_UB_BITS=1`（UB）；`TRITON_MEMORY_DISPLAY=1`（内存 json）；msprof 上板/仿真（性能）。

## 错误三元组

**E1 UB 溢出**：`ub overflow, requires xxx bits while 1572864 bits available!`。原因：单核数据量超 UB。修复：减 BLOCK_SIZE；核内 Tiling（`BLOCK_SIZE_SUB` for 循环）；`triton.Config({..., 'multibuffer': False})`；`ENABLE_PRINT_UB_BITS=1` 查占用。int8 约占 2 倍片上空间，可改 int32。

**E2 coreDim 超限**：`coreDim=524288 can't be greater than UINT16_MAX`（上限 65535）。修复：`TRITON_ALL_BLOCKS_PARALLEL=1`（kernel 须对执行顺序不敏感，有原子/跨核同步会死锁）；或 `BLOCK_SIZE = max(32768, next_power_of_2(cdiv(N, 65535)))`。

**E3 类型不支持**：`unsupported type: uint8/uint64/fp64`。修复：uint→同宽 int、fp64→fp32。NPU 支持 int8/16/32/64、fp16/32、bf16、bool（910_95 加 fp8）。

**E4 对齐错误**：`misaligned memory access`。Vector 要 32B 对齐，CV 融合要 512B。修复：fp32 BLOCK 取 8 倍数、fp16/bf16 取 16 倍数、CV 融合 512B 对齐；`tl.multiple_of(offsets, 128)` 提示对齐。

**E5 dot 降级失败**：`failed to legalize operation 'tt.dot'`。修复：输入转 fp16/fp32/bf16/int8；`al.compile_hint(a, "dot_pad_only_k")`；BLOCK 取 512B 对齐（128/256/256）。

**E6 advance 编译失败**：`failed to legalize operation 'tt.advance'`。修复：循环内重建 `tl.make_block_ptr`（offsets 随迭代更新）替代 `tl.advance`。

**E7 stride 转置不支持**：`unsupported stride pattern for transpose`。修复：stride 写真实内存布局，用 `order=[1,0]` 表达转置。

**E8 精度异常**：排查 = 解释器取基准 → `tl.device_print` 逐步对比。原因表：浮点顺序（1e-6，正常）/ 类型退化（大差异）/ bf16-fp16 损失（关键计算升 fp32）/ mask 差异 / `tl.exp` 近似（1e-7 正常）。通用修法：kernel 内 `.to(tl.float32)` 计算，写回再转。

**E9 Scalar 退化**：`aiv_vec_ratio < 10%`。int64 ADD、int64/int32 CMP 退化为 Scalar。修复：比较前 `cols.to(tl.float32)`。

**E10 device_print 缺 prefix**：`device_print requires a prefix string`。修复：`tl.device_print("x: ", x)`。

**E11 空 tensor**：`all shapes must have size >= 1`。修复：调用前 `if x.numel() == 0: return`。

**E12 bool 不支持**：`unsupported type: bool`。内部自动转 int8；特定操作报错时显式 `mask.to(tl.int8)`。

**E13 atomic_add 多核累加不支持**：修复：`atomic_cas` 自旋锁（`while tl.atomic_cas(lock,0,1)!=0` → 临界区普通 add → `tl.atomic_xchg(lock,0)`），或两阶段归约（各 block 写部分结果 + 单独 kernel 归约）。

**E14 MLIR Pass 失败**：`Failed to run BishengHIR pipeline`。修复：`MLIR_ENABLE_DUMP=1 2>mlir_dump.log` 定位 Pass；bishengir-compile 手动编译。

**E15 运行时死锁**：`deadlock detected`。原因：自旋锁未释放（获取/释放须同路径，避免条件分支取锁）；`TRITON_ALL_BLOCKS_PARALLEL=1` 与同步操作冲突；`sync_block_set/wait` 不配对（参数须一致）。

**E16 NPU 执行超时**：grid 过多分批调度/死锁/数据过大。修复：`grid = (min(num_aicore, cdiv(N, BLOCK)),)` 对齐物理核数；查跨核依赖；缩小数据。

**E17 结果全零/全 NaN**：mask 错未加载 / 类型转换错 / 除零。修复：`tl.device_print` 逐段查；危险操作用 `tl.where` 保护（`tl.where(x > 0, tl.sqrt(x), 0.0)`、`tl.where(d != 0, n/d, 0.0)`）。

## 环境变量组合速查

编译排查：`TRITON_DEBUG=1 + MLIR_ENABLE_DUMP=1 + TRITON_DISABLE_CACHE=1`；精度：`TRITON_INTERPRET=1 + TRITON_DISABLE_CACHE=1`；UB：`ENABLE_PRINT_UB_BITS=1`；coreDim：`TRITON_ALL_BLOCKS_PARALLEL=1`；深度：加 `TRITON_ENABLE_LLVM_DEBUG=1 + LLVM_DEBUG_ONLY="isel"`；编译加速：`TRITON_ASCEND_COMPILE_SPEED_OPT=1`。

## 910_95 平台差异

| 项 | A2/A3 | 910_95 |
|----|-------|--------|
| UB | 192KB（db 96KB×2） | 256KB（db 128KB×2） |
| multibuffer 默认 | 开启 | 关闭 |
| L0C→UB 直通 | 不支持 | 支持（FixPipe，改变同步 Pipe 组合） |
| fp8 | 不支持 | 支持（动态范围小，计算仍走 fp32） |
| SIMT | 不支持 | 支持（减少分支发散影响） |
| 跨核同步 | FFTS | Reg-based SetFlag/WaitFlag（更轻量，Triton API 不变） |
| bishengir 目标 | `--target=Ascend910B3` | `--target=Ascend910_95` |